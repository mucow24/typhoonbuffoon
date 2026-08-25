import { describe, it, expect } from 'vitest'
import { Conditions } from '../../src/game/conditions'
import { Field } from '../../src/world/field'
import { SimWorld } from '../../src/sim/world'
import { KIND_FLUID } from '../../src/sim/particles'
import {
  expectFinite,
  expectSettles,
  expectNoEscapes,
  expectSpeedBelow,
  fillWater,
  makeWorld,
  basinTerrain,
  positionFingerprint,
  run,
  settle,
  surfaceProfile,
} from '../harness'

/**
 * The paths the game actually takes.
 *
 * The fluid was validated on a block spawned onto a clean lattice. The game
 * never does that: it admits water at the field edges from the flood slider,
 * frame after frame, into space that may already be occupied. That path was
 * never exercised and it is the one that detonates.
 */

function gameLikeWorld(spacing = 0.4) {
  const field = new Field(80)
  const sim = new SimWorld()
  sim.terrain = field.terrain
  sim.boundsX0 = field.left
  sim.boundsX1 = field.right
  sim.fluid.spacing = spacing
  return { sim, field, conditions: new Conditions(sim, field) }
}

describe('flood inflow', () => {
  it('fills to the requested level without exploding', () => {
    const { sim, conditions } = gameLikeWorld()
    conditions.floodLevelM = 3

    const trace = run(sim, {
      seconds: 30,
      box: { x0: -60, x1: 60, y0: -40, y1: 120 },
      each: () => conditions.update(1 / 60),
    })

    expectFinite(trace)
    expectNoEscapes(trace, 'flood water')
    // Water is admitted at the field edges and falls to the seabed, which on
    // the generated beach reaches about -11 m. Free fall over that is
    // sqrt(2*9.81*11) = 14.7 m/s, so anything at or under roughly that is the
    // water doing what gravity says. The previous limit of 12 m/s was below
    // physics and was failing the sim for being correct.
    const seabed = Math.abs(sim.terrain!.minHeight)
    const freeFall = Math.sqrt(2 * 9.81 * seabed)
    expectSpeedBelow(trace, freeFall * 1.3, 'flood inflow')
  })

  it('settles near the level the slider asked for', () => {
    const { sim, conditions } = gameLikeWorld()
    conditions.floodLevelM = 2

    for (let i = 0; i < 60 * 40; i++) {
      conditions.update(1 / 60)
      sim.step(1 / 60)
    }

    // Measure over the SEA side of the shoreline (the Field(80) beach crosses
    // y=0 near x=6), where the flood actually pools. The old window straddled
    // the dry dunes, so thin runoff films on the slope - columns whose
    // "surface" is just the local terrain height - dragged the mean metres
    // above the true water level.
    const surface = surfaceProfile(sim, { x0: 10, x1: 35, columnWidth: 2 })
    expect(surface.wetColumns).toBeGreaterThan(4)
    // The slider is in metres. If it does not mean metres, it is a dial with a
    // number printed on it.
    expect(Math.abs(surface.mean - 2)).toBeLessThan(0.8)
  })

  it('stops admitting water once it reaches the target', () => {
    const { sim, conditions } = gameLikeWorld()
    conditions.floodLevelM = 2
    for (let i = 0; i < 60 * 30; i++) {
      conditions.update(1 / 60)
      sim.step(1 / 60)
    }
    const settled = sim.particles.countOfKind(KIND_FLUID)
    for (let i = 0; i < 60 * 10; i++) {
      conditions.update(1 / 60)
      sim.step(1 / 60)
    }
    const later = sim.particles.countOfKind(KIND_FLUID)
    expect(Math.abs(later - settled) / Math.max(settled, 1)).toBeLessThan(0.15)
  })
})

describe('spawning into occupied space', () => {
  it('the dump tool skips occupied space instead of double-filling it', () => {
    // The production path: every spawn route (edge inflow, dump tool) now
    // refuses to create water where water already is - an overlapped pair is
    // a density error the solver can only answer violently, so the right fix
    // is to never manufacture one.
    const sim = makeWorld({ widthM: 30, spacing: 0.4, terrain: basinTerrain(30, 0, 12) })
    fillWater(sim, { x0: -10, x1: 10, yTop: 3 })
    settle(sim, 10)

    const before = sim.particles.countOfKind(KIND_FLUID)
    sim.spawnBlock(0, 2, 6, 3)
    const added = sim.particles.countOfKind(KIND_FLUID) - before
    // Most of that block is already water; only the gaps and the headroom
    // above the surface admit anything.
    expect(added).toBeLessThan(before * 0.3)

    const trace = run(sim, { seconds: 10, box: { x0: -20, x1: 20, y0: -10, y1: 80 } })
    expectFinite(trace)
    expectNoEscapes(trace, 'dump-over-water')
    expectSpeedBelow(trace, 12, 'dump-over-water')
    expectSettles(trace, { below: 0.6, byFraction: 0.7, label: 'dump-over-water' })
  })

  it('raw overlapped spawns relax without a sustained fountain', () => {
    // The impossible state itself, forced through the harness: particles
    // materialised inside the pool. No production path can reach this any
    // more, so the bar is robustness, not beauty: bounded burst, no NaNs,
    // calm afterwards. Pre-fix (high correction cap, no relaunch damping)
    // this churned at 30+ m/s permanently.
    const sim = makeWorld({ widthM: 30, spacing: 0.4, terrain: basinTerrain(30, 0, 12) })
    fillWater(sim, { x0: -10, x1: 10, yTop: 3 })
    settle(sim, 10)

    const before = sim.particles.countOfKind(KIND_FLUID)
    fillWater(sim, { x0: -2, x1: 2, yBottom: 1, yTop: 2.5, seed: 999, jitter: 0.3 })
    const added = sim.particles.countOfKind(KIND_FLUID) - before
    expect(added).toBeGreaterThan(10)

    const trace = run(sim, { seconds: 15, box: { x0: -20, x1: 20, y0: -10, y1: 80 } })
    expectFinite(trace)
    // A few droplets of the initial burst may leave the tall box; a stream is
    // a detonation. Measured burst: ~6% of the pool. Bounded by the global
    // speed cap either way.
    expect(trace.max('escaped')).toBeLessThan((before + added) * 0.08)
    expectSpeedBelow(trace, 46, 'overlapped water')
    const tail = trace.samples.filter((s) => s.t >= 10.5)
    expect(Math.max(...tail.map((s) => s.p99Speed))).toBeLessThan(3)
  })
})

describe('waves', () => {
  it('drives water without unbounded acceleration', () => {
    const { sim, conditions } = gameLikeWorld(0.45)
    conditions.floodLevelM = 3
    for (let i = 0; i < 60 * 25; i++) {
      conditions.update(1 / 60)
      sim.step(1 / 60)
    }

    conditions.waveStrength = 'extreme'
    const trace = run(sim, {
      seconds: 20,
      box: { x0: -60, x1: 60, y0: -40, y1: 120 },
      each: () => conditions.update(1 / 60),
    })

    expectFinite(trace)
    expectNoEscapes(trace, 'wave-driven water')
    // The paddle drives at up to 9 m/s. Water reaching several times that is
    // the solver amplifying, not the wave.
    expectSpeedBelow(trace, 25, 'wave-driven water')
  })
})

describe('determinism', () => {
  it('produces identical results from identical inputs', () => {
    const once = () => {
      const sim = makeWorld({ widthM: 30, spacing: 0.45, terrain: basinTerrain(30, 0, 12) })
      fillWater(sim, { x0: -10, x1: 10, yTop: 3 })
      settle(sim, 5)
      return positionFingerprint(sim)
    }
    // Replay, fair comparison between runs, and reproducible bug reports all
    // depend on this.
    expect(once()).toBe(once())
  })

  it('produces identical results for the game inflow path', () => {
    const once = () => {
      const { sim, conditions } = gameLikeWorld(0.5)
      conditions.floodLevelM = 2
      for (let i = 0; i < 600; i++) {
        conditions.update(1 / 60)
        sim.step(1 / 60)
      }
      return positionFingerprint(sim)
    }
    expect(once()).toBe(once())
  })
})
