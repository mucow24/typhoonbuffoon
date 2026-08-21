import type { ParticleStore } from './particles'

/**
 * Uniform-grid neighbour search, built fresh each substep with a counting sort.
 *
 * Rebuilding beats incremental maintenance here: fluid particles move far
 * enough every substep that most would change cell anyway, and a counting sort
 * over a flat Int32Array has no allocation and perfect locality.
 */
export class SpatialHash {
  private cellSize: number
  private invCell: number
  private readonly tableSize: number
  private counts: Int32Array
  /** Prefix sums per bucket. Public so hot loops can avoid a callback. */
  starts: Int32Array
  entries: Int32Array
  private capacity = 0
  readonly scratch = new Int32Array(9)

  constructor(cellSize = 0.5, tableSize = 1 << 16) {
    this.cellSize = cellSize
    this.invCell = 1 / cellSize
    this.tableSize = tableSize
    this.counts = new Int32Array(tableSize + 1)
    this.starts = new Int32Array(tableSize + 1)
    this.entries = new Int32Array(0)
  }

  setCellSize(size: number): void {
    this.cellSize = size
    this.invCell = 1 / size
  }

  private hash(cx: number, cy: number): number {
    // Large primes; the sign masking keeps negative coordinates well distributed.
    const h = (cx * 92837111) ^ (cy * 689287499)
    return (h & 0x7fffffff) % this.tableSize
  }

  /** Rebuild over every live particle matching the kind mask (bit per kind). */
  build(p: ParticleStore, kindMask: number): void {
    const n = p.highWater
    if (this.capacity < n) {
      this.entries = new Int32Array(Math.max(n, 1024))
      this.capacity = this.entries.length
    }

    this.counts.fill(0)
    const alive = p.slots.alive

    for (let i = 0; i < n; i++) {
      if (alive[i] !== 1 || (kindMask & (1 << p.kind[i]!)) === 0) continue
      const cx = Math.floor(p.posX[i]! * this.invCell)
      const cy = Math.floor(p.posY[i]! * this.invCell)
      this.counts[this.hash(cx, cy)]!++
    }

    let sum = 0
    for (let b = 0; b < this.tableSize; b++) {
      this.starts[b] = sum
      sum += this.counts[b]!
    }
    this.starts[this.tableSize] = sum

    const cursor = this.counts
    for (let b = 0; b < this.tableSize; b++) cursor[b] = this.starts[b]!

    for (let i = 0; i < n; i++) {
      if (alive[i] !== 1 || (kindMask & (1 << p.kind[i]!)) === 0) continue
      const cx = Math.floor(p.posX[i]! * this.invCell)
      const cy = Math.floor(p.posY[i]! * this.invCell)
      const b = this.hash(cx, cy)
      this.entries[cursor[b]!++] = i
    }
  }

  /**
   * Visit candidates within one cell ring of (x, y). Hash collisions mean the
   * callback can see particles outside the radius; callers already range-check.
   */
  /**
   * Fill `scratch` with the unique buckets covering the ring around (x, y) and
   * return how many. Callers iterate starts/entries directly - a callback per
   * neighbour allocates a closure per particle and blocks inlining.
   */
  collectBuckets(x: number, y: number): number {
    const cx = Math.floor(x * this.invCell)
    const cy = Math.floor(y * this.invCell)
    const buckets = this.scratch
    let count = 0
    for (let oy = -1; oy <= 1; oy++) {
      for (let ox = -1; ox <= 1; ox++) {
        const b = this.hash(cx + ox, cy + oy)
        let seen = false
        for (let s = 0; s < count; s++) {
          if (buckets[s] === b) {
            seen = true
            break
          }
        }
        if (!seen) buckets[count++] = b
      }
    }
    return count
  }

  forEachNeighbour(x: number, y: number, visit: (j: number) => void): void {
    const cx = Math.floor(x * this.invCell)
    const cy = Math.floor(y * this.invCell)

    // Two of the nine cells can hash to the same bucket. Visiting it twice
    // would double-count that neighbour in the density sum, so dedupe first.
    const buckets = this.scratch
    let count = 0
    for (let oy = -1; oy <= 1; oy++) {
      for (let ox = -1; ox <= 1; ox++) {
        const b = this.hash(cx + ox, cy + oy)
        let seen = false
        for (let s = 0; s < count; s++) {
          if (buckets[s] === b) {
            seen = true
            break
          }
        }
        if (!seen) buckets[count++] = b
      }
    }

    for (let s = 0; s < count; s++) {
      const b = buckets[s]!
      const end = this.starts[b + 1]!
      for (let k = this.starts[b]!; k < end; k++) visit(this.entries[k]!)
    }
  }
}
