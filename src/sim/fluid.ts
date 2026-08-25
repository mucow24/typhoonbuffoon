import { KIND_BOUNDARY, KIND_FLUID, KIND_OBJECT, type ParticleStore } from './particles'
import { SpatialHash } from './spatialHash'

const MAX_NEIGHBOURS = 64

/**
 * Position Based Fluids (Macklin & Muller, 2013).
 *
 * PBF is PBD applied to a density constraint, which is why the fluid and the
 * structures share one solver rather than being two worlds bolted together:
 * both are position-based projections run inside the same substep.
 *
 * Kernels are the 2D forms - the usual 3D normalisation constants are wrong
 * here and would put the rest density out by a large factor.
 */
export class FluidSolver {
  /** Particle spacing in metres. The resolution control. */
  spacing = 0.25
  /** Physical density of water. Sets particle mass; not the constraint target. */
  waterDensity = 1000
  /**
   * Constraint target density. NOT 1000: the density a perfect grid at
   * `spacing` actually reports under this kernel differs from the physical
   * density by a fixed factor, and using the physical value put every particle
   * permanently under-dense, so lambda was zero and the fluid behaved like dust.
   * Calibrated from the kernel instead.
   */
  restDensity = 1000
  private calibratedFor = -1
  /** Density projection iterations per substep. */
  iterations = 1
  /**
   * Constraint force mixing added to |gradC|^2 in the lambda denominator.
   * Typical gradSum at rest is order 1-10 in these units, so this is a ~0.1-1%
   * softening - enough to bound lambda if the gradient ever degenerates,
   * small enough to leave the projection stiffness alone.
   */
  relaxation = 1e-2
  /**
   * XSPH viscosity.
   *
   * Modest. An earlier version ran this at 0.4 to smother a residual slosh
   * that turned out to be caused by one-way object coupling; once buoyancy came
   * from the pressure field the slosh went with it, and the water can be thin
   * again.
   */
  viscosity = 0.1
  /**
   * Apply XSPH inside every substep instead of once per frame. The projection
   * pump operates at substep frequency; dissipation sampled at frame rate
   * cannot see it. Costs one extra neighbour pass per substep.
   */
  viscosityEverySubstep = false
  /**
   * Pressure multiplier on the fluid's lambda against DYNAMIC solids; static
   * boundary always gets the single (1x) form.
   *
   * Both extremes measured on probes: full Akinci mirroring (2x) at the
   * static bed pumped 38% transient energy growth (vs 10% at 1x) - the
   * doubled response against a lambda whose denominator assumed unit response
   * resonates the bottom layer - so the bed stays at 1x unconditionally.
   * At a hull, 1x halves interface stiffness and compressed water escapes
   * sideways around a floating box instead of lifting it (a box under 2.5 m
   * of added flood sat at the bottom indefinitely), while 2x over-lifts the
   * same box a third of its height. This dial sets the hull between them.
   */
  hullPressureFactor = 1
  /**
   * Cap on the VELOCITY a density correction may imply, m/s.
   *
   * Position corrections become velocity through (x - x_prev)/h, so the bound
   * is stated in the units of the thing that actually goes wrong. It is a
   * BACKSTOP against teleport-scale corrections from pathological states
   * (overlapping spawns), not a working limit: it must sit ABOVE the fastest
   * legitimate interaction. At 0.5 m/s it capped the pressure solve below
   * every energetic fluid-object contact, so incompressibility switched off
   * exactly when the game got interesting and compression discharged for many
   * frames afterwards as mush.
   *
   * The upper bound matters too: measured on the overlap probe, a cap of 20
   * lets a landing high-speed particle trigger corrections comparable to its
   * own impact speed - a self-sustaining fountain (tail p99 ~35 m/s forever).
   * At 8 the relaunch is always slower than the impact, so splash energy
   * decays geometrically, while every legitimate interaction (waves ~9 m/s
   * relative, splash entry ~8 m/s) still resolves within a frame.
   */
  maxCorrectionSpeed = 8
  /**
   * Cap on the density error C = rho/rho0 - 1 fed to the solve. Normal water
   * never exceeds a few percent; an IMPOSSIBLE state - particles materialised
   * on top of each other - reads 100%+ and the unclamped correction launches
   * the neighbourhood. Saturating C makes such states relax over many frames
   * instead of discharging in one. Purely a robustness backstop: it never
   * engages in reachable states, which the spawn paths guard anyway.
   */
  maxDensityError = 0.25
  /** Artificial pressure - stops particles clumping into strings. ~5% of a
   *  typical compressed-state lambda; 1e-4 was three orders below functional. */
  surfaceTensionK = 2e-3
  surfaceTensionN = 4
  /**
   * Extra neighbour-search radius beyond the kernel support, as a fraction of
   * particle spacing. Lists are built once per FRAME while positions integrate
   * across every substep; with zero margin a pair approaching at speed enters
   * kernel range mid-frame without ever being listed. 0.75 spacing covers
   * ~11 m/s of relative drift at 60 Hz.
   */
  supportMargin = 0.75

  readonly hash = new SpatialHash()
  /** Public so the physics harness can measure compression. Read-only in practice. */
  density = new Float32Array(0)
  private lambda = new Float32Array(0)
  private dpx = new Float32Array(0)
  private dpy = new Float32Array(0)
  private neighbourStart = new Int32Array(0)
  private neighbourCount = new Int32Array(0)
  private neighbours = new Int32Array(0)
  private indices = new Int32Array(0)
  /** particle index -> slot in `indices`. Flat array, not a map: this is read
   *  once per neighbour pair per iteration, which is ~1M lookups a frame. */
  private slot = new Int32Array(0)
  /** Number of live fluid particles in `density`/`lambda`, for the harness. */
  liveCount = 0
  /**
   * Fluid neighbours per SOLID particle, by raw particle index, rebuilt each
   * frame. This is the "is this bit of hull actually in water" signal: the
   * column height field cannot tell a submerged hull from one standing beside
   * a puddle on a slope, and analytic buoyancy applied on that misreading
   * flickered a grounded house into orbit.
   */
  solidWetCount = new Int32Array(0)
  /** Summed fluid velocity per solid particle - divide by the wet count for
   *  the LOCAL water velocity at a hull point. */
  solidFluidVX = new Float32Array(0)
  solidFluidVY = new Float32Array(0)
  /**
   * Viscous exchange coefficient at the hull, per solid-fluid pair per
   * SUBSTEP.
   *
   * This is the drag law for OBJECTS, and it is pairwise and
   * momentum-conserving where every force-based formulation failed: quadratic
   * drag against the column mean misfires in spray, and drag against the
   * measured local flow reads the hull's own entrained wake and vanishes.
   * It runs every substep because the failure it prevents happens INSIDE a
   * frame: a deeply buried buoyant object accelerating at several g outruns
   * the pressure solve's correction cap between frame-level drag samples,
   * the water stops being able to push back, and a 2.7-tonne house breached
   * to 113 m. Per-substep exchange keeps hull and neighbouring water moving
   * together, which is also what entrains the wake that lets a wave carry a
   * crate.
   *
   * The value trades submerged rise speed against interface damping, both
   * swept: 0.06 made a deep-released crate creep up at 0.09 m/s
   * (waterlogged); 0.015 surfaced it fast but left the drifting-house scene
   * churning past its settle bars. 0.02 surfaces the crate in ~35 s and
   * settles the house - per-substep application is what prevents the
   * runaway, not coefficient size.
   */
  hullViscosity = 0.02
  /** Compact solid-fluid pair list for the per-substep hull viscosity. */
  private readonly hullPairFluid: number[] = []
  private readonly hullPairSolid: number[] = []
  /** Reaction displacement owed to solid particles, applied with the fluid's. */
  private solidDx = new Float32Array(0)
  private solidDy = new Float32Array(0)
  private readonly solidTouched: number[] = []
  /** Membership stamp for solidTouched; a zero-sum accumulation is still touched. */
  private solidStamp = new Int32Array(0)
  private solidStampValue = 0

  /** Smoothing radius. Two particle spacings gives ~12 neighbours in 2D. */
  get h(): number {
    return this.spacing * 2
  }

  /** Mass per particle so a grid at `spacing` sits at rest density. */
  get particleMass(): number {
    return this.waterDensity * this.spacing * this.spacing
  }

  /** Density a perfect lattice at `spacing` reports, so C == 0 at rest. */
  calibrate(): void {
    if (this.calibratedFor === this.spacing) return
    const h = this.h
    const h2 = h * h
    const coeff = 4 / (Math.PI * Math.pow(h, 8))
    const d = this.spacing
    const k = Math.ceil(h / d)
    let sum = 0
    for (let i = -k; i <= k; i++) {
      for (let j = -k; j <= k; j++) {
        const r2 = (i * d) * (i * d) + (j * d) * (j * d)
        if (r2 < h2) sum += coeff * Math.pow(h2 - r2, 3)
      }
    }
    this.restDensity = sum * this.particleMass
    this.calibratedFor = this.spacing
  }

  private ensure(n: number): void {
    if (this.density.length >= n) return
    const cap = Math.max(n, 1024)
    this.density = new Float32Array(cap)
    this.lambda = new Float32Array(cap)
    this.dpx = new Float32Array(cap)
    this.dpy = new Float32Array(cap)
    this.neighbourStart = new Int32Array(cap)
    this.neighbourCount = new Int32Array(cap)
    this.neighbours = new Int32Array(cap * MAX_NEIGHBOURS)
    this.indices = new Int32Array(cap)
    this.slot = new Int32Array(cap)
    this.solidDx = new Float32Array(cap)
    this.solidDy = new Float32Array(cap)
    this.solidStamp = new Int32Array(cap)
    this.solidWetCount = new Int32Array(cap)
    this.solidFluidVX = new Float32Array(cap)
    this.solidFluidVY = new Float32Array(cap)
  }

  /** Poly6, 2D: 4/(pi h^8) (h^2 - r^2)^3 */
  private poly6(r2: number, h2: number, coeff: number): number {
    if (r2 >= h2) return 0
    const d = h2 - r2
    return coeff * d * d * d
  }

  /**
   * Rebuild the hash and neighbour lists. Once per FRAME, not per substep:
   * particles move a small fraction of the smoothing radius in one substep, so
   * the lists stay valid across the frame, and rebuilding 12x over was costing
   * an order of magnitude for nothing.
   */
  beginFrame(p: ParticleStore): void {
    this.calibrate()
    const n = p.highWater
    if (n === 0) {
      this.liveCount = 0
      return
    }
    this.ensure(n)

    const h = this.h
    // Neighbour lists must stay valid for a whole frame of motion - search
    // wider than the kernel so pairs that CLOSE to within h mid-frame are
    // already listed. Kernels still cut off at h; the margin only affects
    // candidacy.
    const rq = h + this.spacing * this.supportMargin
    const rq2 = rq * rq

    // Collect live fluid particles.
    const alive = p.slots.alive
    let live = 0
    for (let i = 0; i < n; i++) {
      if (alive[i] === 1 && p.kind[i] === KIND_FLUID) this.indices[live++] = i
    }
    this.liveCount = live
    // Fluid slots for the lambda lookup; solids are guarded at use sites.
    for (let a = 0; a < live; a++) this.slot[this.indices[a]!] = a

    // Cell size covers the widened query radius, so the one-ring bucket walk
    // still sees every candidate.
    this.hash.setCellSize(rq)
    // Solids are in the neighbour search now: an object that displaces water has
    // to be visible to the density estimate, or pressure cannot act on it and
    // buoyancy has to be bolted on from outside. Built even with zero fluid -
    // the world's contact passes (member capsules, object vs object) query
    // this hash too, and two crates on dry land still have to collide.
    this.hash.build(p, (1 << KIND_FLUID) | (1 << KIND_OBJECT) | (1 << KIND_BOUNDARY))
    if (live === 0) return

    // Neighbour lists, built once per frame and reused across projections.
    const posX = p.posX
    const posY = p.posY
    const neighbours = this.neighbours
    let cursor = 0
    for (let a = 0; a < live; a++) {
      const i = this.indices[a]!
      this.neighbourStart[a] = cursor
      let count = 0
      const xi = p.posX[i]!
      const yi = p.posY[i]!
      const buckets = this.hash.collectBuckets(xi, yi)
      const bstarts = this.hash.starts
      const bentries = this.hash.entries
      const scratch = this.hash.scratch
      for (let s = 0; s < buckets && count < MAX_NEIGHBOURS; s++) {
        const b = scratch[s]!
        const end = bstarts[b + 1]!
        for (let k = bstarts[b]!; k < end && count < MAX_NEIGHBOURS; k++) {
          const j = bentries[k]!
          if (j === i) continue
          const dx = xi - posX[j]!
          const dy = yi - posY[j]!
          if (dx * dx + dy * dy < rq2) {
            neighbours[cursor + count] = j
            count++
          }
        }
      }
      this.neighbourCount[a] = count
      cursor += MAX_NEIGHBOURS
    }

    // Wetness census: how many fluid particles sit within support of each
    // solid, and their summed velocity. One walk of the lists just built;
    // consumed by analytic buoyancy. Also collects the compact solid-fluid
    // PAIR list the per-substep hull viscosity iterates - the full lists are
    // mostly fluid-fluid and far too fat to walk twelve times a frame.
    this.solidWetCount.fill(0, 0, n)
    this.solidFluidVX.fill(0, 0, n)
    this.solidFluidVY.fill(0, 0, n)
    this.hullPairFluid.length = 0
    this.hullPairSolid.length = 0
    const kind = p.kind
    const velX = p.velX
    const velY = p.velY
    const invMass = p.invMass
    for (let a = 0; a < live; a++) {
      const i = this.indices[a]!
      const start = this.neighbourStart[a]!
      const count = this.neighbourCount[a]!
      for (let k = 0; k < count; k++) {
        const j = neighbours[start + k]!
        if (kind[j] !== KIND_FLUID) {
          this.solidWetCount[j]!++
          this.solidFluidVX[j]! += velX[i]!
          this.solidFluidVY[j]! += velY[i]!
          if (invMass[j]! > 0) {
            this.hullPairFluid.push(i)
            this.hullPairSolid.push(j)
          }
        }
      }
    }
  }

  /** One density projection pass. Called EVERY substep - see the note in solve order. */
  project(p: ParticleStore, substepH: number): void {
    const live = this.liveCount
    if (live === 0) return

    const h = this.h
    const h2 = h * h
    const poly6Coeff = 4 / (Math.PI * Math.pow(h, 8))
    const spikyCoeff2 = -30 / (Math.PI * Math.pow(h, 5))
    const mass = this.particleMass
    const rho0 = this.restDensity
    const invRho0 = 1 / rho0
    const maxCorr = this.maxCorrectionSpeed * substepH
    const stK = this.surfaceTensionK

    // sCorr reference value at 0.2h, for the artificial pressure term.
    const dq = h2 - 0.04 * h2
    const sCorrDenom = poly6Coeff * dq * dq * dq

    // Hoist every typed array out of the loops - property loads on `this`
    // inside a hot loop are not free.
    const posX = p.posX
    const posY = p.posY
    const invMass = p.invMass
    const kind = p.kind
    const volume = p.volume
    const idx = this.indices
    const nbr = this.neighbours
    const nbrStart = this.neighbourStart
    const nbrCount = this.neighbourCount
    const lambda = this.lambda
    const density = this.density
    const slot = this.slot
    const dpx = this.dpx
    const dpy = this.dpy
    const eps = this.relaxation

    for (let iter = 0; iter < this.iterations; iter++) {
      for (let a = 0; a < live; a++) {
        const i = idx[a]!
        const xi = posX[i]!
        const yi = posY[i]!
        let rho = poly6Coeff * h2 * h2 * h2 * mass
        let gradSum = 0
        let gradIx = 0
        let gradIy = 0

        const start = nbrStart[a]!
        const count = nbrCount[a]!
        for (let k = 0; k < count; k++) {
          const j = nbr[start + k]!
          const dx = xi - posX[j]!
          const dy = yi - posY[j]!
          const r2 = dx * dx + dy * dy
          if (r2 >= h2) continue
          const d = h2 - r2

          // A solid neighbour stands in for the water it displaces: its
          // contribution is rho0 times the area it occupies (Akinci et al.),
          // not the fluid particle mass.
          const mj = kind[j] === KIND_FLUID ? mass : this.waterDensity * volume[j]!
          rho += poly6Coeff * d * d * d * mj

          const r = Math.sqrt(r2)
          if (r > 1e-9) {
            const w = spikyCoeff2 * (h - r) * (h - r) * mj
            const gx = ((w * dx) / r) * invRho0
            const gy = ((w * dy) / r) * invRho0
            gradIx += gx
            gradIy += gy
            gradSum += gx * gx + gy * gy
          }
        }

        density[a] = rho
        let C = rho / rho0 - 1
        // Resist compression only. Pulling particles together to reach rest
        // density would give the free surface a skin and make splashes behave
        // like jelly.
        if (C <= 0) {
          lambda[a] = 0
          continue
        }
        if (C > this.maxDensityError) C = this.maxDensityError
        gradSum += gradIx * gradIx + gradIy * gradIy
        lambda[a] = -C / (gradSum + eps)
      }

      const stamp = ++this.solidStampValue
      for (let a = 0; a < live; a++) {
        const i = idx[a]!
        const xi = posX[i]!
        const yi = posY[i]!
        let dx0 = 0
        let dy0 = 0
        const li = lambda[a]!
        const wi = invMass[i]!

        const start = nbrStart[a]!
        const count = nbrCount[a]!
        for (let k = 0; k < count; k++) {
          const j = nbr[start + k]!
          const dx = xi - posX[j]!
          const dy = yi - posY[j]!
          const r2 = dx * dx + dy * dy
          if (r2 >= h2) continue
          const r = Math.sqrt(r2)
          if (r <= 1e-9) continue

          const solid = kind[j] !== KIND_FLUID
          if (!solid) {
            // Fluid-fluid: the paper's scheme. Each side of the pair applies
            // its own (li + lj) correction when its turn comes, so nothing
            // here needs a mass split - masses are equal by construction.
            const lj = lambda[slot[j]!]!
            let corr = 0
            if (sCorrDenom > 1e-20) {
              const d = h2 - r2
              const ratio = (poly6Coeff * d * d * d) / sCorrDenom
              // surfaceTensionN is 4; a multiply chain beats Math.pow, which
              // was costing more than the rest of the inner loop combined.
              const r2p = ratio * ratio
              corr = -stK * r2p * r2p
            }
            const w = spikyCoeff2 * (h - r) * (h - r) * mass
            const sc = (((li + lj + corr) * w) / r) * invRho0
            dx0 += sc * dx
            dy0 += sc * dy
            continue
          }

          // Fluid-solid. The solid has no lambda of its own; mirror the
          // fluid's (Akinci-style pressure mirroring), which restores the
          // same interface stiffness the fluid-fluid convention has.
          //
          // The PAIR correction is then split by inverse mass, exactly like a
          // PBD distance constraint: a light crate takes most of the motion, a
          // steel block almost none, and m_i*dx_i == m_j*dx_j holds by
          // construction. The old scheme gave the fluid its FULL correction
          // AND the solid a mass-ratio reaction on top - the pair separated by
          // (1 + m_fluid/m_solid) of what the constraint asked, over 2x for
          // anything lighter than water, which is past the over-relaxation
          // stability threshold and was the jitter pump at every hull.
          const wj = invMass[j]!
          const wSum = wi + wj
          if (wSum <= 0) continue
          const mjc = this.waterDensity * volume[j]!
          const w = spikyCoeff2 * (h - r) * (h - r) * mjc
          const li2 = wj > 0 ? this.hullPressureFactor * li : li
          const sc = ((li2 * w) / r) * invRho0
          const shareI = wi / wSum
          dx0 += sc * dx * shareI
          dy0 += sc * dy * shareI

          if (wj > 0) {
            const shareJ = wj / wSum
            if (this.solidStamp[j] !== stamp) {
              this.solidStamp[j] = stamp
              this.solidTouched.push(j)
            }
            this.solidDx[j]! -= sc * dx * shareJ
            this.solidDy[j]! -= sc * dy * shareJ
          }
        }

        const mag = Math.sqrt(dx0 * dx0 + dy0 * dy0)
        if (mag > maxCorr && mag > 1e-12) {
          const k2 = maxCorr / mag
          dx0 *= k2
          dy0 *= k2
        }
        dpx[a] = dx0
        dpy[a] = dy0
      }

      for (let a = 0; a < live; a++) {
        const i = idx[a]!
        if (invMass[i] === 0) continue
        posX[i]! += dpx[a]!
        posY[i]! += dpy[a]!
      }

      for (const j of this.solidTouched) {
        let sx = this.solidDx[j]!
        let sy = this.solidDy[j]!
        const mag = Math.sqrt(sx * sx + sy * sy)
        if (mag > maxCorr) {
          const k2 = maxCorr / mag
          sx *= k2
          sy *= k2
        }
        posX[j]! += sx
        posY[j]! += sy
        this.solidDx[j] = 0
        this.solidDy[j] = 0
      }
      this.solidTouched.length = 0
    }
  }

  /**
   * Pairwise viscous exchange between hull particles and their fluid
   * neighbours - see hullViscosity. Called every substep on the frame's pair
   * list; velocities are current at each call.
   */
  applyHullViscosity(p: ParticleStore): void {
    const c = this.hullViscosity
    if (c <= 0) return
    const n = this.hullPairFluid.length
    if (n === 0) return
    const invMass = p.invMass
    const velX = p.velX
    const velY = p.velY

    for (let k = 0; k < n; k++) {
      const i = this.hullPairFluid[k]!
      const j = this.hullPairSolid[k]!
      const wi = invMass[i]!
      const wj = invMass[j]!
      const wSum = wi + wj
      if (wSum <= 0) continue

      const ux = (velX[j]! - velX[i]!) * c
      const uy = (velY[j]! - velY[i]!) * c
      const shareJ = wj / wSum
      const shareI = wi / wSum
      velX[j]! -= ux * shareJ
      velY[j]! -= uy * shareJ
      velX[i]! += ux * shareI
      velY[i]! += uy * shareI
    }
  }

  /**
   * XSPH viscosity: nudge each particle toward the average velocity of its
   * neighbourhood. This is what stops the surface fizzing, and it doubles as
   * the smoothing that keeps impulsive particle impacts from buzzing structures.
   */
  applyViscosity(p: ParticleStore): void {
    if (this.viscosity <= 0 || this.liveCount === 0) return
    const live = this.liveCount
    const h = this.h
    const h2 = h * h
    const poly6Coeff = 4 / (Math.PI * Math.pow(h, 8))

    for (let a = 0; a < live; a++) {
      const i = this.indices[a]!
      const xi = p.posX[i]!
      const yi = p.posY[i]!
      let vx = 0
      let vy = 0
      let wsum = 0

      const start = this.neighbourStart[a]!
      const count = this.neighbourCount[a]!
      for (let k = 0; k < count; k++) {
        const j = this.neighbours[start + k]!
        const dx = xi - p.posX[j]!
        const dy = yi - p.posY[j]!
        const w = this.poly6(dx * dx + dy * dy, h2, poly6Coeff)
        if (w <= 0) continue
        vx += (p.velX[j]! - p.velX[i]!) * w
        vy += (p.velY[j]! - p.velY[i]!) * w
        wsum += w
      }

      if (wsum > 1e-12) {
        this.dpx[a] = (vx / wsum) * this.viscosity
        this.dpy[a] = (vy / wsum) * this.viscosity
      } else {
        this.dpx[a] = 0
        this.dpy[a] = 0
      }
    }

    for (let a = 0; a < live; a++) {
      const i = this.indices[a]!
      if (p.invMass[i] === 0) continue
      p.velX[i]! += this.dpx[a]!
      p.velY[i]! += this.dpy[a]!
    }
  }
}
