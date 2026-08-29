import { describe, it, expect } from 'vitest'
import { KIND_FLUID } from '../../src/sim/particles'
import type { SimWorld } from '../../src/sim/world'
import type { SolverBackend } from '../../src/sim/solver'
import { flatTerrain, makeWorld } from '../harness'

/**
 * Water-tool admission across the solver seam. spawnDisc has two guards:
 * the neighbour hash (built when a solver frame's results are visible) and
 * the recent-spawn list (covering spawns the hash has not seen yet). The
 * hand-off between them must follow the HASH, not the step counter: the
 * pipelined GPU backend can run several steps before a frame's results
 * land, and clearing the list per step opens a window where a spawn is in
 * NEITHER guard - the next stream tick double-fills the same space and the
 * density error discharges at the speed cap, which on screen is hyper-
 * velocity spray off the water tool strong enough to break a support.
 */

function fluidCount(w: SimWorld): number {
  const p = w.particles
  let n = 0
  for (let i = 0; i < p.highWater; i++) {
    if (p.slots.alive[i] === 1 && p.kind[i] === KIND_FLUID) n++
  }
  return n
}

describe('spawn admission guards across the solver seam', () => {
  it('blocks a repeat spawn while no solver frame has landed', () => {
    const w = makeWorld({ widthM: 40, terrain: flatTerrain(40, 0) })
    // A backend whose frames never land inside the window - the pipelined
    // GPU under real fence latency. It never rebuilds the neighbour hash,
    // so the recent-spawn list is the ONLY thing standing between two
    // stream ticks aimed at the same spot.
    const inFlight: SolverBackend = {
      sync() {},
      step() {},
      readback: () => Promise.resolve(),
    }
    w.solver = inFlight

    const first = w.spawnDisc(0, 8, 1.2)
    expect(first).toBeGreaterThan(10)
    w.step(1 / 60)
    const again = w.spawnDisc(0, 8, 1.2)
    expect(again).toBe(0)
    expect(fluidCount(w)).toBe(first)
  })

  it('hands occupancy back to the hash once a frame with the spawns lands', () => {
    // CPU reference: every step() rebuilds the hash with the spawns in it,
    // so the same spot stays blocked (now by the hash) while genuinely
    // clear space still admits water - the guard must not become a latch.
    const w = makeWorld({ widthM: 40, terrain: flatTerrain(40, 0) })
    const first = w.spawnDisc(0, 8, 1.2)
    expect(first).toBeGreaterThan(10)
    w.step(1 / 60)
    expect(w.spawnDisc(0, 8, 1.2)).toBe(0)
    expect(w.spawnDisc(6, 8, 1.2)).toBeGreaterThan(10)
  })
})
