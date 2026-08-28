import { describe, it, expect } from 'vitest'
import { WaterEmitter } from '../../src/game/waterEmitter'
import { KIND_FLUID } from '../../src/sim/particles'
import type { SimWorld } from '../../src/sim/world'
import {
  basinTerrain,
  expectFinite,
  expectNoEscapes,
  expectSpeedBelow,
  fillWater,
  makeWorld,
  maxSpeed,
  run,
  settle,
} from '../harness'

/**
 * The water tool: spawnDisc (the sim-side spawn) and WaterEmitter (the
 * game-side rate accounting). The slider is a flow in m²/s; these tests hold
 * it to meaning that, the same way the flood slider is held to metres.
 */

const fluid = (sim: SimWorld) => sim.particles.countOfKind(KIND_FLUID)

/** Closest fluid pair, brute force - the spacing the guards actually produced. */
function minPairDistance(sim: SimWorld): number {
  const p = sim.particles
  const idx: number[] = []
  for (let i = 0; i < p.highWater; i++) {
    if (p.slots.alive[i] === 1 && p.kind[i] === KIND_FLUID) idx.push(i)
  }
  let min = Infinity
  for (let a = 0; a < idx.length; a++) {
    for (let b = a + 1; b < idx.length; b++) {
      const i = idx[a]!
      const j = idx[b]!
      const d = Math.hypot(p.posX[i]! - p.posX[j]!, p.posY[i]! - p.posY[j]!)
      if (d < min) min = d
    }
  }
  return min
}

describe('spawnDisc', () => {
  it('spawns inside the disc and never below the terrain', () => {
    const sim = makeWorld({ widthM: 30, spacing: 0.4, floor: 0 })
    const added = sim.spawnDisc(0, 1.5, 2)
    expect(added).toBeGreaterThan(10)

    const p = sim.particles
    for (let i = 0; i < p.highWater; i++) {
      if (p.slots.alive[i] !== 1 || p.kind[i] !== KIND_FLUID) continue
      const d = Math.hypot(p.posX[i]! - 0, p.posY[i]! - 1.5)
      // Grid point inside the disc, plus jitter of up to spacing * 0.125.
      expect(d).toBeLessThan(2 + 0.4 * 0.13)
      // Ground guard: grid y >= ground + spacing, minus jitter.
      expect(p.posY[i]!).toBeGreaterThan(0.4 - 0.4 * 0.13)
    }
  })

  it('skips space that is already water', () => {
    const sim = makeWorld({ widthM: 30, spacing: 0.4, terrain: basinTerrain(30, 0, 12) })
    fillWater(sim, { x0: -10, x1: 10, yTop: 3 })
    settle(sim, 10)

    // A disc fully inside the pool: only the gaps admit anything.
    const openCapacity = (Math.PI * 1.5 ** 2) / (0.4 * 0.4)
    const added = sim.spawnDisc(0, 1, 1.5)
    expect(added).toBeLessThan(openCapacity * 0.2)

    const trace = run(sim, { seconds: 8, box: { x0: -20, x1: 20, y0: -10, y1: 80 } })
    expectFinite(trace)
    expectNoEscapes(trace, 'disc-over-water')
    expectSpeedBelow(trace, 12, 'disc-over-water')
  })

  it('repeated calls between steps stack nothing (splashing while paused)', () => {
    // hasFluidNear reads the hash built during step(). Without the
    // recent-spawn guard, every splash made while paused lands exactly on the
    // previous one, and the density error detonates on unpause. A repeat
    // call may still admit a few particles into genuine gaps the first
    // call's jitter left open - what it must never do is put anything
    // inside the clearance of an earlier spawn.
    const sim = makeWorld({ widthM: 30, spacing: 0.4, floor: 0 })
    const first = sim.spawnDisc(0, 2, 1.5)
    expect(first).toBeGreaterThan(10)

    const again = sim.spawnDisc(0, 2, 1.5)
    expect(again).toBeLessThan(first * 0.15)
    expect(minPairDistance(sim)).toBeGreaterThanOrEqual(0.4 * 0.95 * 0.999)

    // After a step the hash sees the spawns and takes over the guard.
    sim.step(1 / 60)
    const afterStep = sim.spawnDisc(0, 2, 1.5)
    expect(afterStep).toBeLessThan(first * 0.3)
  })
})

describe('WaterEmitter', () => {
  it('delivers the flow the slider asks for', () => {
    const sim = makeWorld({ widthM: 40, spacing: 0.4, terrain: basinTerrain(40, 0, 14) })
    const em = new WaterEmitter()
    // 6 m²/s at spacing 0.4 is 0.625 particles/frame: exercises the
    // fractional carry on every single frame.
    em.flow = 6
    for (let i = 0; i < 60 * 8; i++) {
      em.update(sim, 1 / 60, 0, 10)
      sim.step(1 / 60)
    }

    // The slider is in m²/s. If it does not mean that, it is a dial with a
    // number printed on it: 8 s at 6 m²/s is 48 m² of water.
    const expected = (6 * 8) / (0.4 * 0.4)
    expect(fluid(sim)).toBeGreaterThan(expected * 0.8)
    expect(fluid(sim)).toBeLessThanOrEqual(expected + 1)

    // And the stream must be a stream, not a detonation.
    const trace = run(sim, { seconds: 8, box: { x0: -25, x1: 25, y0: -10, y1: 80 } })
    expectFinite(trace)
    expectNoEscapes(trace, 'water tool stream')
    expectSpeedBelow(trace, Math.sqrt(2 * 9.81 * 10) * 1.3, 'water tool stream')
  })

  it('sub-particle flow rates accumulate instead of rounding to zero', () => {
    const sim = makeWorld({ widthM: 30, spacing: 0.4, floor: 0 })
    const em = new WaterEmitter()
    em.flow = 1 // 0.104 particles/frame
    let spawned = 0
    for (let i = 0; i < 60; i++) {
      spawned += em.update(sim, 1 / 60, 0, 6)
      sim.step(1 / 60)
    }
    // 1 m²/s for 1 s at spacing 0.4 is 6.25 particles.
    expect(spawned).toBeGreaterThanOrEqual(5)
    expect(spawned).toBeLessThanOrEqual(7)
  })

  it('a blocked emitter drops flow instead of banking a burst', () => {
    const sim = makeWorld({ widthM: 30, spacing: 0.4, terrain: basinTerrain(30, 0, 12) })
    fillWater(sim, { x0: -10, x1: 10, yTop: 4 })
    settle(sim, 8)

    const em = new WaterEmitter()
    em.flow = 12
    let during = 0
    for (let i = 0; i < 60 * 3; i++) {
      during += em.update(sim, 1 / 60, 0, 1.5) // held deep inside the pool
      sim.step(1 / 60)
    }
    expect(during).toBeLessThan(30)

    // One frame in open air afterwards emits one frame's worth, not the
    // three seconds the pool refused.
    const burst = em.update(sim, 1 / 60, 0, 30)
    expect(burst).toBeLessThanOrEqual(Math.ceil(12 / 60 / (0.4 * 0.4)) + 1)
  })

  it('a paused splash respects the spawn clearance and does not fizz on unpause', () => {
    // A click while paused spawns the whole burst with no steps in between,
    // so nothing relaxes the lattice before it is complete. The clearance
    // must hold for the positions particles ACTUALLY get, and it must be
    // wide enough that a jittered burst spawns sub-rest-density: guarding
    // the unjittered grid point admitted pairs down to ~0.77 * spacing, and
    // even at 0.85 the worst particle reads ~7% OVER rest density - poly6 is
    // steep at close range - which the solve discharges at the correction
    // cap. Both versions fizzed upward at 8-9 m/s on unpause.
    const sim = makeWorld({ widthM: 30, spacing: 0.25, floor: 0 })
    const em = new WaterEmitter() // default flow: an 84-particle burst
    const n = em.splash(sim, 0, 6)
    expect(n).toBeGreaterThan(50)

    // The guard's contract, checked against actual positions: no pair
    // closer than the sub-rest-density clearance.
    expect(minPairDistance(sim)).toBeGreaterThanOrEqual(0.25 * 0.95 * 0.999)

    // Unpause. In free fall the blob reaches g * dt ~ 0.16 m/s after one
    // step, ~1 m/s after six. The over-density discharge saturates the
    // 8 m/s correction cap; 3 m/s separates the regimes cleanly.
    for (let step = 1; step <= 6; step++) {
      sim.step(1 / 60)
      expect(maxSpeed(sim, KIND_FLUID), `max fluid speed after step ${step}`).toBeLessThan(3)
    }
  })

  it('a click splash is one flow-sized burst', () => {
    const sim = makeWorld({ widthM: 40, spacing: 0.4, floor: 0 })
    const em = new WaterEmitter()
    em.flow = 12
    const n = em.splash(sim, 0, 6)
    // splashSeconds of flow, converted at spacing²  per particle - exact,
    // because the disc is sized to hold the whole burst.
    expect(n).toBe(Math.round((12 * em.splashSeconds) / (0.4 * 0.4)))

    const sim2 = makeWorld({ widthM: 40, spacing: 0.4, floor: 0 })
    em.flow = 48
    const n2 = em.splash(sim2, 0, 6)
    expect(n2).toBe(Math.round((48 * em.splashSeconds) / (0.4 * 0.4)))
  })
})
