import { Rng, ValueNoise1D } from '../core/rng'

/**
 * Wind as a band-limited gust field.
 *
 * The band limit is the whole point. Per-frame noise is high-frequency forcing
 * and would excite exactly the structural modes the constraint damping exists
 * to suppress - the solver can be perfectly damped and the result still reads
 * as buzzing, because the *forcing* is jittery. Gusts evolve over seconds.
 */
export class WindField {
  /** Sustained wind speed, m/s. */
  baseSpeed = 0
  /** +1 blows to the right, -1 to the left. */
  direction = -1
  /** Gust amplitude as a fraction of base speed. */
  gustiness = 0.4

  /** Gust periods in seconds. The shortest is deliberately kept near 0.5s. */
  private readonly periods = [8.5, 3.1, 1.2]
  private readonly weights = [1, 0.5, 0.22]
  private readonly noise: ValueNoise1D
  private time = 0

  constructor(seed = 0x5eed) {
    this.noise = new ValueNoise1D(new Rng(seed), 512)
  }

  advance(dt: number): void {
    this.time += dt
  }

  reset(): void {
    this.time = 0
  }

  /**
   * Horizontal wind velocity at a world position, m/s. The x term gives gusts
   * some spatial extent so a wide structure is not loaded perfectly uniformly.
   */
  velocityAt(x: number): number {
    if (this.baseSpeed <= 0) return 0
    const gust = this.noise.octaves(this.time + x * 0.02, this.periods, this.weights)
    return this.direction * this.baseSpeed * (1 + this.gustiness * gust)
  }

  /** Current gust factor, for HUD and for driving visual effects. */
  gustFactor(): number {
    return 1 + this.gustiness * this.noise.octaves(this.time, this.periods, this.weights)
  }
}

/** Air density, kg/m^3. */
export const AIR_DENSITY = 1.225
