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
  private gradSum = new Float32Array(0)
  private gradIX = new Float32Array(0)
  private gradIY = new Float32Array(0)
  private viscW = new Float32Array(0)
  private indices = new Int32Array(0)
  /** particle index -> slot in `indices`. Flat array, not a map. */
  private slot = new Int32Array(0)

  /**
   * UNIQUE interaction pairs, rebuilt once per frame from the hash.
   *
   * The projection used to walk per-particle neighbour lists, which hold
   * every pair twice - and recomputed the pair geometry (sqrt, both kernels)
   * in the density pass AND again in the correction pass, from both sides.
   * That was 79% of the whole step at 4k particles. Pairs are visited once:
   * geometry and kernels are computed in the accumulate pass and cached for
   * the correction pass, which is then pure multiply-adds.
   */
  private ffA = new Int32Array(0) // fluid slot
  private ffB = new Int32Array(0) // fluid slot
  private ffGX = new Float32Array(0) // spiky gradient x, mass and 1/rho0 folded
  private ffGY = new Float32Array(0)
  private ffCorr = new Float32Array(0) // sCorr artificial pressure
  private ffCount = 0
  private fsA = new Int32Array(0) // fluid slot
  private fsJ = new Int32Array(0) // solid particle index
  private fsMass = new Float32Array(0) // displaced-water mass of the solid
  private fsBoost = new Float32Array(0) // hull pressure factor (1 for static)
  private fsShareI = new Float32Array(0) // inverse-mass split, fluid side
  private fsShareJ = new Float32Array(0)
  private fsGX = new Float32Array(0)
  private fsGY = new Float32Array(0)
  private fsCount = 0
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
   * SPEED-DEPENDENT, like real drag: c = base + rate * |dv|, capped. A single
   * constant cannot serve both regimes - a deep-released crate needs weak
   * coupling to rise at all (strong coupling binds it to a neutrally-buoyant
   * water plug and it crawls), while a floating house slamming in waves needs
   * strong coupling or it careens. Slow steady motion sees ~base; a 3 m/s bob
   * sees several times that; a 7 m/s slam saturates near the cap.
   *
   * NORMALISED PER SOLID PARTICLE: the per-pair coefficient is c divided by
   * the solid's wet count, so a hull particle's total exchange per substep is
   * ~c of its relative velocity whatever the resolution puts around it.
   * Un-normalised, the pair count itself set the drag - every constant
   * silently changed meaning when the neighbour scheme did.
   */
  hullViscosity = 0.05
  /** Extra hull viscosity per m/s of relative speed. */
  hullViscosityRate = 0.06
  /** Ceiling on a solid particle's per-substep exchange fraction. */
  hullViscosityMax = 0.5
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
    this.gradSum = new Float32Array(cap)
    this.gradIX = new Float32Array(cap)
    this.gradIY = new Float32Array(cap)
    this.viscW = new Float32Array(cap)
    this.indices = new Int32Array(cap)
    this.slot = new Int32Array(cap)
    this.solidDx = new Float32Array(cap)
    this.solidDy = new Float32Array(cap)
    this.solidStamp = new Int32Array(cap)
    this.solidWetCount = new Int32Array(cap)
    this.solidFluidVX = new Float32Array(cap)
    this.solidFluidVY = new Float32Array(cap)
  }

  private ensureFFPairs(n: number): void {
    if (this.ffA.length >= n) return
    const cap = Math.max(n, 4096)
    const oldA = this.ffA
    const oldB = this.ffB
    this.ffA = new Int32Array(cap)
    this.ffB = new Int32Array(cap)
    this.ffGX = new Float32Array(cap)
    this.ffGY = new Float32Array(cap)
    this.ffCorr = new Float32Array(cap)
    this.ffA.set(oldA)
    this.ffB.set(oldB)
  }

  private ensureFSPairs(n: number): void {
    if (this.fsA.length >= n) return
    const cap = Math.max(n, 1024)
    const oldA = this.fsA
    const oldJ = this.fsJ
    const oldMass = this.fsMass
    const oldBoost = this.fsBoost
    const oldSI = this.fsShareI
    const oldSJ = this.fsShareJ
    this.fsA = new Int32Array(cap)
    this.fsJ = new Int32Array(cap)
    this.fsMass = new Float32Array(cap)
    this.fsBoost = new Float32Array(cap)
    this.fsShareI = new Float32Array(cap)
    this.fsShareJ = new Float32Array(cap)
    this.fsGX = new Float32Array(cap)
    this.fsGY = new Float32Array(cap)
    this.fsA.set(oldA)
    this.fsJ.set(oldJ)
    this.fsMass.set(oldMass)
    this.fsBoost.set(oldBoost)
    this.fsShareI.set(oldSI)
    this.fsShareJ.set(oldSJ)
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
    // Stale lists are corruption, not staleness: freed indices are recycled
    // by the next create, so pair lists from a previous frame can point at
    // entirely different particles. Empty means empty.
    this.ffCount = 0
    this.fsCount = 0
    this.hullPairFluid.length = 0
    this.hullPairSolid.length = 0
    const n = p.highWater
    if (n === 0) {
      this.liveCount = 0
      return
    }
    this.ensure(n)
    this.solidWetCount.fill(0, 0, n)
    this.solidFluidVX.fill(0, 0, n)
    this.solidFluidVY.fill(0, 0, n)

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

    // Pair lists, built once per frame and reused across every substep. The
    // wetness census and the hull-viscosity list fall out of the same scan.
    const posX = p.posX
    const posY = p.posY
    const kind = p.kind
    const velX = p.velX
    const velY = p.velY
    const invMass = p.invMass
    const volume = p.volume
    const hullFactor = this.hullPressureFactor

    let ff = 0
    let fs = 0
    this.ensureFFPairs(live * 16)
    this.ensureFSPairs(1024)
    for (let a = 0; a < live; a++) {
      const i = this.indices[a]!
      let count = 0
      const xi = posX[i]!
      const yi = posY[i]!
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
          if (dx * dx + dy * dy >= rq2) continue

          if (kind[j] === KIND_FLUID) {
            // Each unordered fluid pair once - the OTHER side's scan skips it.
            if (j < i) continue
            if (ff >= this.ffA.length) this.ensureFFPairs(this.ffA.length * 2)
            this.ffA[ff] = a
            this.ffB[ff] = this.slot[j]!
            ff++
            count++
          } else {
            const wj = invMass[j]!
            if (fs >= this.fsA.length) this.ensureFSPairs(this.fsA.length * 2)
            this.fsA[fs] = a
            this.fsJ[fs] = j
            // A solid neighbour stands in for the water it displaces: its
            // mass in the density estimate is rho0 times the area it occupies
            // (Akinci et al.), not the fluid particle mass.
            this.fsMass[fs] = this.waterDensity * volume[j]!
            this.fsBoost[fs] = wj > 0 ? hullFactor : 1
            // Inverse-mass split of the pair correction, fixed for the frame.
            const wi = invMass[i]!
            const wSum = wi + wj
            this.fsShareI[fs] = wSum > 0 ? wi / wSum : 0
            this.fsShareJ[fs] = wSum > 0 ? wj / wSum : 0
            fs++
            count++

            // Wetness census + hull-viscosity list, same pair.
            this.solidWetCount[j]!++
            this.solidFluidVX[j]! += velX[i]!
            this.solidFluidVY[j]! += velY[i]!
            if (wj > 0) {
              this.hullPairFluid.push(i)
              this.hullPairSolid.push(j)
            }
          }
        }
      }
    }
    this.ffCount = ff
    this.fsCount = fs
  }

  /**
   * One density projection pass. Called EVERY substep - see the note in solve
   * order.
   *
   * Structured over the frame's unique pair lists. The accumulate pass
   * computes each pair's geometry and kernels ONCE, feeding the density and
   * gradient sums of BOTH sides and caching the gradient for the correction
   * pass, which is then pure multiply-adds. Identical maths to the
   * per-particle formulation (each side used to apply (li+lj+corr) from its
   * own visit; the pair form applies the same term with opposite signs), at
   * less than half the pair visits and none of the duplicate sqrt/kernels.
   */
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
    const invSCorrDenom = sCorrDenom > 1e-20 ? 1 / sCorrDenom : 0

    // Hoist every typed array out of the loops - property loads on `this`
    // inside a hot loop are not free.
    const posX = p.posX
    const posY = p.posY
    const invMass = p.invMass
    const idx = this.indices
    const lambda = this.lambda
    const density = this.density
    const dpx = this.dpx
    const dpy = this.dpy
    const gradSum = this.gradSum
    const gradIX = this.gradIX
    const gradIY = this.gradIY
    const eps = this.relaxation
    const maxC = this.maxDensityError
    const ffCount = this.ffCount
    const ffA = this.ffA
    const ffB = this.ffB
    const ffGX = this.ffGX
    const ffGY = this.ffGY
    const ffCorr = this.ffCorr
    const fsCount = this.fsCount
    const fsA = this.fsA
    const fsJ = this.fsJ
    const fsMass = this.fsMass
    const fsBoost = this.fsBoost
    const fsShareI = this.fsShareI
    const fsShareJ = this.fsShareJ
    const fsGX = this.fsGX
    const fsGY = this.fsGY
    const selfRho = poly6Coeff * h2 * h2 * h2 * mass

    for (let iter = 0; iter < this.iterations; iter++) {
      // Accumulate: density, gradient sums, cached pair gradients.
      density.fill(selfRho, 0, live)
      gradSum.fill(0, 0, live)
      gradIX.fill(0, 0, live)
      gradIY.fill(0, 0, live)

      for (let k = 0; k < ffCount; k++) {
        const a = ffA[k]!
        const b = ffB[k]!
        const ia = idx[a]!
        const ib = idx[b]!
        const dx = posX[ia]! - posX[ib]!
        const dy = posY[ia]! - posY[ib]!
        const r2 = dx * dx + dy * dy
        if (r2 >= h2 || r2 <= 1e-18) {
          ffGX[k] = 0
          ffGY[k] = 0
          ffCorr[k] = 0
          continue
        }
        const d = h2 - r2
        const w6 = poly6Coeff * d * d * d
        const rhoC = w6 * mass
        density[a]! += rhoC
        density[b]! += rhoC

        const r = Math.sqrt(r2)
        const w = spikyCoeff2 * (h - r) * (h - r) * mass
        const s = (w / r) * invRho0
        const gx = s * dx
        const gy = s * dy
        ffGX[k] = gx
        ffGY[k] = gy
        // surfaceTensionN is 4; a multiply chain beats Math.pow.
        const ratio = w6 * invSCorrDenom
        const r2p = ratio * ratio
        ffCorr[k] = -stK * r2p * r2p

        gradIX[a]! += gx
        gradIY[a]! += gy
        gradIX[b]! -= gx
        gradIY[b]! -= gy
        const g2 = gx * gx + gy * gy
        gradSum[a]! += g2
        gradSum[b]! += g2
      }

      for (let k = 0; k < fsCount; k++) {
        const a = fsA[k]!
        const j = fsJ[k]!
        const ia = idx[a]!
        const dx = posX[ia]! - posX[j]!
        const dy = posY[ia]! - posY[j]!
        const r2 = dx * dx + dy * dy
        if (r2 >= h2 || r2 <= 1e-18) {
          fsGX[k] = 0
          fsGY[k] = 0
          continue
        }
        const d = h2 - r2
        density[a]! += poly6Coeff * d * d * d * fsMass[k]!

        const r = Math.sqrt(r2)
        const w = spikyCoeff2 * (h - r) * (h - r) * fsMass[k]!
        const s = (w / r) * invRho0
        const gx = s * dx
        const gy = s * dy
        fsGX[k] = gx
        fsGY[k] = gy
        gradIX[a]! += gx
        gradIY[a]! += gy
        gradSum[a]! += gx * gx + gy * gy
      }

      // Lambda. Resist compression only: pulling particles together to reach
      // rest density would give the free surface a skin and make splashes
      // behave like jelly.
      for (let a = 0; a < live; a++) {
        let C = density[a]! * invRho0 - 1
        if (C <= 0) {
          lambda[a] = 0
          continue
        }
        if (C > maxC) C = maxC
        const gs = gradSum[a]! + gradIX[a]! * gradIX[a]! + gradIY[a]! * gradIY[a]!
        lambda[a] = -C / (gs + eps)
      }

      // Correct, from the cached gradients.
      dpx.fill(0, 0, live)
      dpy.fill(0, 0, live)
      for (let k = 0; k < ffCount; k++) {
        const gx = ffGX[k]!
        const gy = ffGY[k]!
        if (gx === 0 && gy === 0) continue
        const a = ffA[k]!
        const b = ffB[k]!
        const f = lambda[a]! + lambda[b]! + ffCorr[k]!
        dpx[a]! += f * gx
        dpy[a]! += f * gy
        dpx[b]! -= f * gx
        dpy[b]! -= f * gy
      }

      // Fluid-solid: the solid has no lambda of its own; the fluid's acts on
      // the pair (boosted at dynamic hulls - see hullPressureFactor) and the
      // correction is split by inverse mass exactly like a PBD distance
      // constraint, so m_i*dx_i == m_j*dx_j holds by construction. The old
      // scheme gave the fluid its FULL correction AND the solid a mass-ratio
      // reaction on top - over-relaxation past 2x for anything lighter than
      // water, the jitter pump at every hull.
      const stamp = ++this.solidStampValue
      for (let k = 0; k < fsCount; k++) {
        const gx = fsGX[k]!
        const gy = fsGY[k]!
        if (gx === 0 && gy === 0) continue
        const a = fsA[k]!
        const f = lambda[a]! * fsBoost[k]!
        dpx[a]! += f * gx * fsShareI[k]!
        dpy[a]! += f * gy * fsShareI[k]!

        const shareJ = fsShareJ[k]!
        if (shareJ > 0) {
          const j = fsJ[k]!
          if (this.solidStamp[j] !== stamp) {
            this.solidStamp[j] = stamp
            this.solidTouched.push(j)
          }
          this.solidDx[j]! -= f * gx * shareJ
          this.solidDy[j]! -= f * gy * shareJ
        }
      }

      // Apply, clamped. See maxCorrectionSpeed for both bounds' rationale.
      for (let a = 0; a < live; a++) {
        const i = idx[a]!
        if (invMass[i] === 0) continue
        let dx0 = dpx[a]!
        let dy0 = dpy[a]!
        const mag = Math.sqrt(dx0 * dx0 + dy0 * dy0)
        if (mag > maxCorr && mag > 1e-12) {
          const k2 = maxCorr / mag
          dx0 *= k2
          dy0 *= k2
        }
        posX[i]! += dx0
        posY[i]! += dy0
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
    const c0 = this.hullViscosity
    const c1 = this.hullViscosityRate
    if (c0 <= 0 && c1 <= 0) return
    const cMax = this.hullViscosityMax
    const n = this.hullPairFluid.length
    if (n === 0) return
    const invMass = p.invMass
    const velX = p.velX
    const velY = p.velY

    const wet = this.solidWetCount
    for (let k = 0; k < n; k++) {
      const i = this.hullPairFluid[k]!
      const j = this.hullPairSolid[k]!
      const wi = invMass[i]!
      const wj = invMass[j]!
      const wSum = wi + wj
      if (wSum <= 0) continue

      const dvx = velX[j]! - velX[i]!
      const dvy = velY[j]! - velY[i]!
      let c = c0 + c1 * Math.sqrt(dvx * dvx + dvy * dvy)
      if (c > cMax) c = cMax
      c /= Math.max(1, wet[j]!)

      const ux = dvx * c
      const uy = dvy * c
      const shareJ = wj / wSum
      const shareI = wi / wSum
      velX[j]! -= ux * shareJ
      velY[j]! -= uy * shareJ
      velX[i]! += ux * shareI
      velY[i]! += uy * shareI
    }
  }

  /**
   * XSPH viscosity: nudge each particle toward the weighted average velocity
   * of its neighbourhood. This is what stops the surface fizzing, and it
   * doubles as the smoothing that keeps impulsive particle impacts from
   * buzzing structures. Solid neighbours contribute to the average (their
   * stillness calms the fluid against them) but are not themselves nudged.
   *
   * Pair-based over the frame's lists, accumulating numerator and weight for
   * both sides of each fluid pair at once - same normalised result as the
   * per-particle walk it replaced.
   */
  applyViscosity(p: ParticleStore): void {
    if (this.viscosity <= 0 || this.liveCount === 0) return
    const live = this.liveCount
    const h2 = this.h * this.h
    const poly6Coeff = 4 / (Math.PI * Math.pow(this.h, 8))
    const posX = p.posX
    const posY = p.posY
    const velX = p.velX
    const velY = p.velY
    const idx = this.indices
    const numX = this.dpx
    const numY = this.dpy
    const wsum = this.viscW

    numX.fill(0, 0, live)
    numY.fill(0, 0, live)
    wsum.fill(0, 0, live)

    const ffCount = this.ffCount
    for (let k = 0; k < ffCount; k++) {
      const a = this.ffA[k]!
      const b = this.ffB[k]!
      const ia = idx[a]!
      const ib = idx[b]!
      const dx = posX[ia]! - posX[ib]!
      const dy = posY[ia]! - posY[ib]!
      const r2 = dx * dx + dy * dy
      if (r2 >= h2) continue
      const d = h2 - r2
      const w = poly6Coeff * d * d * d
      const dvx = velX[ib]! - velX[ia]!
      const dvy = velY[ib]! - velY[ia]!
      numX[a]! += dvx * w
      numY[a]! += dvy * w
      numX[b]! -= dvx * w
      numY[b]! -= dvy * w
      wsum[a]! += w
      wsum[b]! += w
    }

    const fsCount = this.fsCount
    for (let k = 0; k < fsCount; k++) {
      const a = this.fsA[k]!
      const j = this.fsJ[k]!
      const ia = idx[a]!
      const dx = posX[ia]! - posX[j]!
      const dy = posY[ia]! - posY[j]!
      const r2 = dx * dx + dy * dy
      if (r2 >= h2) continue
      const d = h2 - r2
      const w = poly6Coeff * d * d * d
      numX[a]! += (velX[j]! - velX[ia]!) * w
      numY[a]! += (velY[j]! - velY[ia]!) * w
      wsum[a]! += w
    }

    const visc = this.viscosity
    for (let a = 0; a < live; a++) {
      const i = idx[a]!
      if (p.invMass[i] === 0) continue
      const ws = wsum[a]!
      if (ws <= 1e-12) continue
      velX[i]! += (numX[a]! / ws) * visc
      velY[i]! += (numY[a]! / ws) * visc
    }
  }
}
