import { Rng, ValueNoise1D } from '../core/rng'
import { clamp, lerp, smoothstep } from '../core/math'

/**
 * The ground, as an evenly-sampled polyline in world metres (Y up, sea level
 * at y = 0). A polyline rather than a bitmap, which is what leaves room for
 * erosion and destruction later without changing the representation.
 */
export class Terrain {
  /** Height samples, left to right across the field. */
  readonly heights: Float32Array
  readonly x0: number
  readonly x1: number
  readonly spacing: number

  constructor(heights: Float32Array, x0: number, x1: number) {
    this.heights = heights
    this.x0 = x0
    this.x1 = x1
    this.spacing = heights.length > 1 ? (x1 - x0) / (heights.length - 1) : 1
  }

  /** Ground height at a world x, linearly interpolated, clamped at the edges. */
  heightAt(x: number): number {
    const n = this.heights.length
    if (n === 0) return 0
    const t = (x - this.x0) / this.spacing
    if (t <= 0) return this.heights[0]!
    if (t >= n - 1) return this.heights[n - 1]!
    const i = Math.floor(t)
    return lerp(this.heights[i]!, this.heights[i + 1]!, t - i)
  }

  /** Outward-ish surface normal at x, unit length. */
  normalAt(x: number): { nx: number; ny: number } {
    const h = this.spacing
    const dy = this.heightAt(x + h) - this.heightAt(x - h)
    const len = Math.hypot(dy, 2 * h)
    return { nx: -dy / len, ny: (2 * h) / len }
  }

  /** Signed distance above the surface. Negative means buried. */
  depthBelowSurface(x: number, y: number): number {
    return this.heightAt(x) - y
  }

  get minHeight(): number {
    let m = Infinity
    for (const h of this.heights) if (h < m) m = h
    return m === Infinity ? 0 : m
  }

  get maxHeight(): number {
    let m = -Infinity
    for (const h of this.heights) if (h > m) m = h
    return m === -Infinity ? 0 : m
  }
}

export interface BeachOptions {
  widthM: number
  /** Height of the land at the left edge. */
  landHeight?: number
  /** Depth of the sea floor at the right edge. */
  seaDepth?: number
  /** Where the waterline crosses, as a fraction of the width. */
  shoreAt?: number
  sampleSpacing?: number
  seed?: number
}

/**
 * Default scene: land on the left, sloping down past the waterline into the sea
 * on the right. Deterministic from the seed, so the same width always gives the
 * same beach.
 */
export function generateBeach(opts: BeachOptions): Terrain {
  const {
    widthM,
    landHeight = 7,
    seaDepth = -11,
    shoreAt = 0.58,
    sampleSpacing = 1.5,
    seed = 0xbeac4,
  } = opts

  const x0 = -widthM * 0.5
  const x1 = widthM * 0.5
  const count = Math.max(2, Math.round(widthM / sampleSpacing) + 1)

  const rng = new Rng(seed)
  const dunes = new ValueNoise1D(rng, 128)

  const heights = new Float32Array(count)
  for (let i = 0; i < count; i++) {
    const t = i / (count - 1)

    // Main profile: a shelf of land, a shoulder at the shore, then the sea floor.
    const base = lerp(landHeight, seaDepth, smoothstep(shoreAt - 0.34, shoreAt + 0.38, t))

    // Dunes only on the dry side, fading out as we approach the water.
    const dryness = clamp(1 - smoothstep(shoreAt - 0.22, shoreAt + 0.05, t), 0, 1)
    const dune = dunes.octaves(t * widthM, [34, 13], [1, 0.4]) * 1.5 * dryness

    // A gentle sandbar just offshore, so flooding has something to break over.
    const bar = Math.exp(-Math.pow((t - (shoreAt + 0.17)) / 0.06, 2)) * 1.6

    heights[i] = base + dune + bar
  }

  return new Terrain(heights, x0, x1)
}
