import { KIND_FLUID, type ParticleStore } from './particles'

/**
 * A coarse column height field sampled from the fluid particles each frame.
 *
 * This is the MEMBER regime's water model: member buoyancy, hydrostatic wall
 * load, quadratic drag and wind shielding all read it, and all inherit its
 * resolution-independence - a member must not float or load differently at
 * different settings of a slider the player controls. Objects are the other
 * regime: they live in the particle pressure field directly (see fluid.ts),
 * which costs some resolution sensitivity in exchange for water genuinely
 * displacing around them. The particles still do the shoving and the
 * containment for both, through direct collision.
 *
 * Known limit shared by everything built on columns: one column holds one
 * surface, so a sealed vessel with different levels inside and outside at the
 * same x cannot be represented. The dome archetype is deferred partly for
 * this reason.
 */
export class WaterField {
  /** Column width in metres. */
  resolution = 1
  private x0 = 0
  private columns = new Float32Array(0)
  private counts = new Int32Array(0)
  private count = 0
  private floors = new Float32Array(0)
  private velX = new Float32Array(0)
  private velY = new Float32Array(0)

  /** Rebuild from the current fluid particles. Once per frame. */
  build(p: ParticleStore, x0: number, x1: number, particleArea: number): void {
    const n = Math.max(1, Math.ceil((x1 - x0) / this.resolution) + 1)
    if (this.columns.length < n) {
      this.columns = new Float32Array(n)
      this.counts = new Int32Array(n)
      this.velX = new Float32Array(n)
      this.velY = new Float32Array(n)
    }
    this.velX.fill(0, 0, n)
    this.velY.fill(0, 0, n)
    this.x0 = x0
    this.count = n
    this.columns.fill(-Infinity, 0, n)
    this.counts.fill(0, 0, n)

    const alive = p.slots.alive
    // Track the lowest particle per column as the column floor, and count
    // particles for the volume calculation below.
    const floors = this.floors
    if (floors.length < n) this.floors = new Float32Array(n)
    this.floors.fill(Infinity, 0, n)

    for (let i = 0; i < p.highWater; i++) {
      if (alive[i] !== 1 || p.kind[i] !== KIND_FLUID) continue
      const c = Math.floor((p.posX[i]! - x0) / this.resolution)
      if (c < 0 || c >= n) continue
      this.counts[c]!++
      this.velX[c]! += p.velX[i]!
      this.velY[c]! += p.velY[i]!
      if (p.posY[i]! < this.floors[c]!) this.floors[c] = p.posY[i]!
    }

    // Surface from the VOLUME of water in the column, not the topmost particle.
    // Taking the highest particle lets spray define the surface: a heavy object
    // slamming in threw droplets 18 m up and buoyancy then acted as though the
    // water were 18 m deep. Volume ignores spray almost entirely.
    const area = particleArea
    for (let c = 0; c < n; c++) {
      const k = this.counts[c]!
      if (k < 3) {
        this.columns[c] = -Infinity
        continue
      }
      const depth = (k * area) / this.resolution
      this.columns[c] = this.floors[c]! + depth
      this.velX[c]! /= k
      this.velY[c]! /= k
    }

    // One smoothing pass, so a choppy surface does not make lift chatter.
    for (let c = 1; c < n - 1; c++) {
      const a = this.columns[c - 1]!
      const b = this.columns[c]!
      const d = this.columns[c + 1]!
      if (b === -Infinity) continue
      let sum = b
      let k = 1
      if (a !== -Infinity) {
        sum += a
        k++
      }
      if (d !== -Infinity) {
        sum += d
        k++
      }
      this.columns[c] = sum / k
    }
  }

  /** Water surface height at x, or -Infinity if dry. */
  surfaceAt(x: number): number {
    if (this.count === 0) return -Infinity
    const c = Math.floor((x - this.x0) / this.resolution)
    if (c < 0 || c >= this.count) return -Infinity
    return this.columns[c]!
  }

  /** Mean fluid velocity in the column at x. Zero where dry. */
  velocityAt(x: number, out: { x: number; y: number }): void {
    out.x = 0
    out.y = 0
    if (this.count === 0) return
    const c = Math.floor((x - this.x0) / this.resolution)
    if (c < 0 || c >= this.count) return
    out.x = this.velX[c]!
    out.y = this.velY[c]!
  }

  /** How much of a particle of radius r at (x, y) is under the surface, 0..1. */
  submergedFraction(x: number, y: number, r: number): number {
    const surface = this.surfaceAt(x)
    if (surface === -Infinity) return 0
    const depth = surface - (y - r)
    if (depth <= 0) return 0
    const span = 2 * r
    return depth >= span ? 1 : depth / span
  }
}
