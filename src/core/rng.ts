/**
 * Seeded RNG. Everything stochastic in the sim goes through one of these so runs
 * are reproducible - which is what keeps deterministic replay cheap to add later
 * (see docs/PLAN.md, stability rule 5).
 */
export class Rng {
  private state: number

  constructor(seed = 0x9e3779b9) {
    // Avoid the degenerate all-zero state.
    this.state = seed >>> 0 || 0x9e3779b9
  }

  /** mulberry32 - small, fast, good enough for gameplay noise. */
  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0
    let t = this.state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }

  /** Uniform in [min, max). */
  range(min: number, max: number): number {
    return min + this.next() * (max - min)
  }

  /** Uniform in [-spread, spread). */
  signed(spread = 1): number {
    return this.range(-spread, spread)
  }

  int(minInclusive: number, maxExclusive: number): number {
    return Math.floor(this.range(minInclusive, maxExclusive))
  }

  fork(): Rng {
    return new Rng((this.next() * 4294967296) >>> 0)
  }
}

/**
 * Value noise over a 1D axis (time), smoothed with smoothstep. Used for wind
 * gusts, where the important property is that it is BAND LIMITED - see the wind
 * notes in docs/PLAN.md. Per-frame randomness would excite exactly the
 * structural modes the damping exists to suppress.
 */
export class ValueNoise1D {
  private readonly table: Float32Array

  constructor(rng: Rng, size = 256) {
    this.table = new Float32Array(size)
    for (let i = 0; i < size; i++) this.table[i] = rng.signed(1)
  }

  /** Sample at position t (in table units). Result in roughly -1..1. */
  sample(t: number): number {
    const n = this.table.length
    const i = Math.floor(t)
    const f = t - i
    const a = this.table[((i % n) + n) % n]!
    const b = this.table[(((i + 1) % n) + n) % n]!
    const s = f * f * (3 - 2 * f)
    return a + (b - a) * s
  }

  /**
   * Octave sum. `periods` are in seconds; the shortest one should stay well
   * above the sim timestep or the forcing stops being band limited.
   */
  octaves(timeSec: number, periods: readonly number[], weights: readonly number[]): number {
    let sum = 0
    let totalWeight = 0
    for (let i = 0; i < periods.length; i++) {
      const w = weights[i] ?? 1
      sum += this.sample(timeSec / periods[i]! + i * 37.17) * w
      totalWeight += w
    }
    return totalWeight > 0 ? sum / totalWeight : 0
  }
}
