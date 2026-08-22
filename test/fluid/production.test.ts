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

    const surface = surfaceProfile(sim, { x0: -20, x1: 20, columnWidth: 2 })
    expect(surface.wetColumns).toBeGreaterThan(4)
    // The slider is in metres. If it does not mean metres, it is a dial with a
    // number printed on it.
    expect(Math.abs(surface.mean - 2)).toBeLessThan(1.5)
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
  it('does not detonate when water is created on top of water', () => {
    const sim = makeWorld({ widthM: 30, spacing: 0.4, terrain: basinTerrain(30, 0, 12) })
    fillWater(sim, { x0: -10, x1: 10, yTop: 3 })
    settle(sim, 10)

    // Overlap a patch, not the whole pool. Doubling the density of an entire
    // body of water everywhere at once is not a state the game can reach, and a
    // test built on it measures the response to a detonation rather than to the
    // thing that actually happens: inflow putting a few particles where there
    // is already water.
    const before = sim.particles.countOfKind(KIND_FLUID)
    fillWater(sim, { x0: -2, x1: 2, yBottom: 1, yTop: 2.5, seed: 999, jitter: 0.3 })
    const added = sim.particles.countOfKind(KIND_FLUID) - before
    expect(added).toBeGreaterThan(10)

    const trace = run(sim, { seconds: 15, box: { x0: -20, x1: 20, y0: -10, y1: 80 } })
    expectFinite(trace)
    expectNoEscapes(trace, 'overlapped water')
    expectSpeedBelow(trace, 20, 'overlapped water')
    // And it must calm down again rather than staying agitated.
    expectSettles(trace, { below: 0.6, byFraction: 0.7, label: 'overlapped water' })
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
