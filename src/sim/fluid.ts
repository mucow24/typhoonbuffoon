import { KIND_FLUID, type ParticleStore } from './particles'
import { SpatialHash } from './spatialHash'

const MAX_NEIGHBOURS = 48

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
  /** Constraint force mixing, stops lambda blowing up in sparse regions. */
  relaxation = 1e-5
  /**
   * XSPH viscosity.
   *
   * High compared with real water, deliberately. Water is nearly inviscid, so a
   * physically faithful 60 m basin slosh persists for minutes - measured here at
   * 1.6-2.0 m/s still going after 90 seconds, which reads on screen as water
   * that never calms down. Damping the object or the whole system does not help
   * (more hull drag actually made it worse, since a stiffer hull stirs harder).
   * Damping the FLUID does: 0.05 -> 0.4 takes the residual from 2.7 to about 2.0, and 0.7 would take it
   * below 1.2 but is gloopy enough that water no longer finds its own level
   * across a barrier, which is a worse fault than slosh.
   */
  viscosity = 0.4
  /**
   * Cap on the VELOCITY a density correction may imply, m/s.
   *
   * Position corrections become velocity through (x - x_prev)/h, and h is a
   * substep - about 1.4 ms. A cap expressed as a fraction of particle spacing
   * therefore permits absurd speeds: 0.5 * 0.4 m over 1.4 ms is 144 m/s. The
   * bound has to be stated in the units of the thing that actually goes wrong.
   */
  maxCorrectionSpeed = 0.5
  /** Artificial pressure - stops particles clumping into strings. */
  surfaceTensionK = 1e-4
  surfaceTensionN = 4

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
    const h2 = h * h

    // Collect live fluid particles.
    const alive = p.slots.alive
    let live = 0
    for (let i = 0; i < n; i++) {
      if (alive[i] === 1 && p.kind[i] === KIND_FLUID) this.indices[live++] = i
    }
    this.liveCount = live
    if (live === 0) return
    // Every neighbour comes from a fluid-only hash, so each one is guaranteed
    // to have a slot here and the map needs no clearing between substeps.
    for (let a = 0; a < live; a++) this.slot[this.indices[a]!] = a

    this.hash.setCellSize(h)
    this.hash.build(p, 1 << KIND_FLUID)

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
          if (dx * dx + dy * dy < h2) {
            neighbours[cursor + count] = j
            count++
          }
        }
      }
      this.neighbourCount[a] = count
      cursor += MAX_NEIGHBOURS
    }
  }

  /** One density projection pass. Called EVERY substep - see the note in solve order. */
  project(p: ParticleStore, substepH: number): void {
    const live = this.liveCount
    if (live === 0) return

    const h = this.h
    const h2 = h * h
    const poly6Coeff = 4 / (Math.PI * Math.pow(h, 8))
    const spikyCoeff = (-30 / (Math.PI * Math.pow(h, 5))) * this.particleMass
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
          rho += poly6Coeff * d * d * d * mass

          const r = Math.sqrt(r2)
          if (r > 1e-9) {
            const w = spikyCoeff * (h - r) * (h - r)
            const gx = ((w * dx) / r) * invRho0
            const gy = ((w * dy) / r) * invRho0
            gradIx += gx
            gradIy += gy
            gradSum += gx * gx + gy * gy
          }
        }

        density[a] = rho
        const C = rho / rho0 - 1
        // Resist compression only. Pulling particles together to reach rest
        // density would give the free surface a skin and make splashes behave
        // like jelly.
        if (C <= 0) {
          lambda[a] = 0
          continue
        }
        gradSum += gradIx * gradIx + gradIy * gradIy
        lambda[a] = -C / (gradSum + eps)
      }

      for (let a = 0; a < live; a++) {
        const i = idx[a]!
        const xi = posX[i]!
        const yi = posY[i]!
        let dx0 = 0
        let dy0 = 0
        const li = lambda[a]!

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

          const lj = lambda[slot[j]!]!
          let corr = 0
          if (sCorrDenom > 1e-20) {
            const d = h2 - r2
            const ratio = (poly6Coeff * d * d * d) / sCorrDenom
            // surfaceTensionN is 4; a multiply chain beats Math.pow, which was
            // costing more than the rest of the inner loop combined.
            const r2p = ratio * ratio
            corr = -stK * r2p * r2p
          }

          const w = spikyCoeff * (h - r) * (h - r)
          const s = (((li + lj + corr) * w) / r) * invRho0
          dx0 += s * dx
          dy0 += s * dy
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
