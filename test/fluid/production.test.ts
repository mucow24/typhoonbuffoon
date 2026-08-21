import { describe, it, expect } from 'vitest'
import { Conditions } from '../../src/game/conditions'
import { Field } from '../../src/world/field'
import { SimWorld } from '../../src/sim/world'
import { KIND_FLUID } from '../../src/sim/particles'
import {
  expectFinite,
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
    // Water arriving under gravity from a 3 m head has no business exceeding
    // about 8 m/s. The inflow itself is injected at 2.5 m/s.
    expectSpeedBelow(trace, 12, 'flood inflow')
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

    // Deliberately spawn a second body straight through the first. The inflow
    // path does exactly this, every frame, at the field edges.
    fillWater(sim, { x0: -10, x1: 10, yTop: 3, seed: 999, jitter: 0.3 })

    const trace = run(sim, { seconds: 15, box: { x0: -20, x1: 20, y0: -10, y1: 80 } })
    expectFinite(trace)
    expectNoEscapes(trace, 'overlapped water')
    expectSpeedBelow(trace, 20, 'overlapped water')
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
