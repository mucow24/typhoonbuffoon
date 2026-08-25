import { describe, it, expect } from 'vitest'
import { basinTerrain, fillWater, makeWorld } from '../harness'

/**
 * Throughput regression guards.
 *
 * Wall-clock assertions are usually flaky poison in CI, so these are shaped
 * to be machine-independent: a RATIO that catches accidental superlinear
 * scaling (the class of bug that turns "one more dump click" into a cliff),
 * and one absolute ceiling so generous that only an order-of-magnitude
 * regression can reach it. Measured on the dev machine: per-particle cost is
 * FLAT across 1k -> 4k fluid (ratio 0.99), ~13 ms per step at 4k.
 *
 * Method: the ratio bound was verified red against a degraded neighbour
 * search (hash cell size x100, which turns the per-frame pair build into
 * near-full scans with distance rejections - measured ratio ~1.9), and the
 * per-particle cap in the pair build is what defeats radius-inflation
 * mutations before they reach this test.
 */

function stepCost(fluidTarget: number): { perParticle: number; mean: number; fluid: number } {
  const sim = makeWorld({ widthM: 60, spacing: 0.25, terrain: basinTerrain(60, 0, 15) })
  // Grow DEPTH with the target, not width - the basin clips a too-wide pool
  // and quietly measures fewer particles than the label claims.
  const depth = (fluidTarget * 0.25 * 0.25) / 40
  fillWater(sim, { x0: -20, x1: 20, yTop: depth })
  const fluid = sim.fluidCount

  // Warm up the JIT and let the pool compact before timing.
  for (let i = 0; i < 120; i++) sim.step(1 / 60)

  const frames = 120
  const t0 = performance.now()
  for (let i = 0; i < frames; i++) sim.step(1 / 60)
  const mean = (performance.now() - t0) / frames
  return { perParticle: mean / fluid, mean, fluid }
}

describe('fluid throughput', () => {
  it('scales linearly with particle count and stays inside the ceiling', () => {
    const small = stepCost(1000)
    const large = stepCost(4000)
    // The scenario must actually produce the counts it claims to compare.
    expect(small.fluid).toBeGreaterThan(800)
    expect(large.fluid).toBeGreaterThan(3400)

    // Per-particle cost must not grow materially with the particle count:
    // measured flat (0.99); 1.6x allows cache effects and machine noise, and
    // catches the degraded-hash mutation at ~1.9.
    expect(large.perParticle).toBeLessThan(small.perParticle * 1.6)

    // And the absolute ceiling: ~13 ms measured, 100 ms allowed. If this
    // fires, the sim got an order of magnitude slower and the dump tool is a
    // freeze button again.
    expect(large.mean).toBeLessThan(100)
  }, 300000)
})
