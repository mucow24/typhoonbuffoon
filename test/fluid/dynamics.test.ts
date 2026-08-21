import { describe, it, expect } from 'vitest'
import { Terrain } from '../../src/world/terrain'
import {
  basinTerrain,
  densityStats,
  expectFinite,
  expectMonotonic,
  expectNear,
  expectNoEnergyGain,
  expectNoEscapes,
  expectSpeedBelow,
  fillWater,
  makeWorld,
  rampTerrain,
  run,
  settle,
  surfaceProfile,
  waterFront,
} from '../harness'

/**
 * Water in motion, and water under its own weight.
 *
 * A fluid that merely sits still is a pile of sand. These check the properties
 * that make it a fluid: pressure supports the column above, it finds its own
 * level across a barrier, and a released column collapses at roughly the speed
 * gravity allows rather than at whatever speed the solver invents.
 */

describe('hydrostatics', () => {
  it('supports a deep column without crushing the bottom', () => {
    const sim = makeWorld({ widthM: 24, spacing: 0.35, terrain: basinTerrain(24, 0, 20) })
    fillWater(sim, { x0: -8, x1: 8, yTop: 8 })
    settle(sim, 15)

    const d = densityStats(sim)

    // Real water is near enough incompressible: 8 m of head is under one extra
    // atmosphere and compresses it by well under a tenth of a percent. The
    // solver will not be that good, but the bottom of the pool being tens of
    // percent denser than the top means pressure is not being resolved.
    expectNear(d.max, d.restDensity, {
      rel: 0.15,
      label: 'peak density in an 8 m column vs rest density',
    })
    expect(d.overCompressed).toBeLessThan(0.05)
  })

  it('does not compress more as the column gets deeper', () => {
    const peakDensity = (depth: number) => {
      const sim = makeWorld({ widthM: 24, spacing: 0.35, terrain: basinTerrain(24, 0, 30) })
      fillWater(sim, { x0: -8, x1: 8, yTop: depth })
      settle(sim, 15)
      return densityStats(sim).max / densityStats(sim).restDensity
    }

    // If compression scales with depth, the fluid is acting like a spring
    // mattress rather than a liquid, and deep water will behave differently
    // from shallow water for no physical reason.
    const shallow = peakDensity(2)
    const deep = peakDensity(8)
    expectNear(deep, shallow, { rel: 0.1, abs: 0.05, label: 'compression at 8 m vs 2 m depth' })
  })

  it('finds its own level across a dividing barrier', () => {
    // A terrain sill lower than the water: both sides must equalise.
    const width = 30
    const sim = makeWorld({ widthM: width, spacing: 0.35 })
    const sill = 1.5
    const heights = new Float32Array(Math.round(width / 0.5) + 1)
    for (let i = 0; i < heights.length; i++) {
      const x = -width / 2 + i * 0.5
      heights[i] = Math.abs(x) < 1 ? sill : 0
      if (Math.abs(x) > width / 2 - 2) heights[i] = 20
    }
    sim.terrain = new Terrain(heights, -width / 2, width / 2)

    // All the water starts on the left.
    fillWater(sim, { x0: -13, x1: -1, yTop: 5 })
    settle(sim, 30)

    const left = surfaceProfile(sim, { x0: -12, x1: -2, columnWidth: 1 })
    const right = surfaceProfile(sim, { x0: 2, x1: 12, columnWidth: 1 })

    expect(right.wetColumns).toBeGreaterThan(3)
    expectNear(right.mean, left.mean, {
      rel: 0.2,
      abs: 0.5,
      label: 'right-hand level vs left-hand level across a sill',
    })
  })

  it('runs downhill and pools at the low end', () => {
    const sim = makeWorld({ widthM: 30, spacing: 0.35, terrain: rampTerrain(30, 6, 0) })
    fillWater(sim, { x0: -14, x1: -8, yTop: 8 })
    settle(sim, 20)

    const uphill = surfaceProfile(sim, { x0: -14, x1: -8, columnWidth: 1 })
    const downhill = surfaceProfile(sim, { x0: 6, x1: 13, columnWidth: 1 })

    // Water released high on a ramp ends up low on the ramp. Anything else is
    // not obeying gravity.
    expect(downhill.wetColumns).toBeGreaterThan(uphill.wetColumns)
  })
})

describe('dam break', () => {
  it('collapses at roughly the speed gravity allows', () => {
    const depth = 4
    const sim = makeWorld({ widthM: 40, spacing: 0.35, terrain: basinTerrain(40, 0, 15) })
    fillWater(sim, { x0: -16, x1: -8, yTop: depth })

    const fronts: { t: number; x: number }[] = []
    const trace = run(sim, {
      seconds: 6,
      sampleEvery: 0.5,
      box: { x0: -25, x1: 25, y0: -10, y1: 60 },
      each: (t, step) => {
        if (step % 30 === 0) fronts.push({ t, x: waterFront(sim) })
      },
    })

    expectFinite(trace)
    expectNoEscapes(trace, 'dam-break water')

    // Shallow-water theory puts the front of a collapsing column at about
    // 2*sqrt(g*h). Faster than that is energy the solver invented.
    const theoreticalFront = 2 * Math.sqrt(9.81 * depth)
    expectSpeedBelow(trace, theoreticalFront * 1.5, 'dam-break water')

    // And it must actually advance.
    const advancing = fronts.filter((f) => Number.isFinite(f.x)).map((f) => f.x)
    expect(advancing.length).toBeGreaterThan(4)
    expect(advancing[advancing.length - 1]!).toBeGreaterThan(advancing[0]! + 2)
  })

  it('does not gain energy while collapsing', () => {
    const sim = makeWorld({ widthM: 40, spacing: 0.4, terrain: basinTerrain(40, 0, 15) })
    fillWater(sim, { x0: -16, x1: -8, yTop: 4 })
    const trace = run(sim, { seconds: 8, box: { x0: -25, x1: 25, y0: -10, y1: 60 } })
    expectNoEnergyGain(trace, { tolerance: 0.05, label: 'dam break' })
  })
})

describe('splash response', () => {
  it('settles again after being disturbed', () => {
    const sim = makeWorld({ widthM: 30, spacing: 0.4, terrain: basinTerrain(30, 0, 12) })
    fillWater(sim, { x0: -11, x1: 11, yTop: 3 })
    settle(sim, 10)

    // Kick the middle of the pool upward, then let it recover.
    const p = sim.particles
    for (let i = 0; i < p.highWater; i++) {
      if (p.slots.alive[i] !== 1 || p.kind[i] !== 1) continue
      if (Math.abs(p.posX[i]!) < 2) p.velY[i]! += 4
    }

    const trace = run(sim, { seconds: 20, box: { x0: -20, x1: 20, y0: -10, y1: 60 } })

    // The disturbance must decay, not persist or grow.
    const early = Math.max(...trace.samples.filter((s) => s.t < 3).map((s) => s.kinetic))
    const late = Math.max(...trace.samples.filter((s) => s.t > 15).map((s) => s.kinetic))
    expect(late).toBeLessThan(early * 0.35)
  })

  it('responds more to a bigger disturbance', () => {
    const peakFor = (kick: number) => {
      const sim = makeWorld({ widthM: 30, spacing: 0.45, terrain: basinTerrain(30, 0, 12) })
      fillWater(sim, { x0: -11, x1: 11, yTop: 3 })
      settle(sim, 8)
      const p = sim.particles
      for (let i = 0; i < p.highWater; i++) {
        if (p.slots.alive[i] !== 1 || p.kind[i] !== 1) continue
        if (Math.abs(p.posX[i]!) < 2) p.velY[i]! += kick
      }
      const trace = run(sim, { seconds: 6, box: { x0: -20, x1: 20, y0: -10, y1: 80 } })
      return trace.max('kinetic')
    }

    expectMonotonic([peakFor(1), peakFor(3), peakFor(6)], 'increasing', 'splash energy vs kick size')
  })
})
