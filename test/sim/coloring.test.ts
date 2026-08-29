import { describe, it, expect } from 'vitest'
import { colorConstraints } from '../../src/sim/gpu/coloring'
import { buildBeam } from '../../src/scenes/demos'
import { flatTerrain, makeWorld } from '../harness'

/**
 * Graph colouring for the GPU joints solve: constraints inside one colour
 * must share NO particle, because they apply position corrections in
 * parallel. A colouring that violates that is a data race on the far side of
 * a driver, where it would surface as intermittent jitter - so the validity
 * property is proven here, in Node, against real beam topology.
 */

/** Every alive constraint's endpoint list, from a SimWorld's distance table. */
function distanceEndpoints(sim: ReturnType<typeof makeWorld>): number[][] {
  const d = sim.distance
  const out: number[][] = []
  for (let i = 0; i < d.highWater; i++) {
    out.push(d.slots.alive[i] === 1 ? [d.a[i]!, d.b[i]!] : [])
  }
  return out
}

describe('constraint colouring', () => {
  it('no two constraints in a colour share a particle (chain topology)', () => {
    const sim = makeWorld({ widthM: 40, terrain: flatTerrain(40, 0) })
    buildBeam(sim, { x0: -8, y0: 2, x1: 8, y1: 2, material: 'wood', segments: 12 })
    const endpoints = distanceEndpoints(sim)
    const groups = colorConstraints(endpoints)

    const seen = new Set<number>()
    let total = 0
    for (const group of groups) {
      const particlesInColour = new Set<number>()
      for (const ci of group) {
        expect(seen.has(ci)).toBe(false) // each constraint exactly once
        seen.add(ci)
        total++
        for (const pIdx of endpoints[ci]!) {
          expect(particlesInColour.has(pIdx)).toBe(false)
          particlesInColour.add(pIdx)
        }
      }
    }
    // Every alive constraint got a colour.
    const alive = endpoints.filter((e) => e.length > 0).length
    expect(total).toBe(alive)
    // A chain is 2-colourable; greedy may use a couple more at junctions, but
    // double digits would mean the adjacency is wrong.
    expect(groups.length).toBeGreaterThanOrEqual(2)
    expect(groups.length).toBeLessThanOrEqual(4)
  })

  it('handles shared junctions: a star of constraints ends up fully serialised', () => {
    // Five constraints all sharing particle 0 - no two may share a colour.
    const endpoints = [
      [0, 1],
      [0, 2],
      [0, 3],
      [0, 4],
      [0, 5],
    ]
    const groups = colorConstraints(endpoints)
    expect(groups.length).toBe(5)
    for (const g of groups) expect(g.length).toBe(1)
  })

  it('leaves dead constraints (empty endpoint lists) out entirely', () => {
    const endpoints = [[0, 1], [], [1, 2], []]
    const groups = colorConstraints(endpoints)
    const all = groups.flat()
    expect(all.sort()).toEqual([0, 2])
  })

  it('colours three-particle bend constraints', () => {
    // A bend chain: (0,1,2), (1,2,3), (2,3,4) - all overlap, need 3 colours;
    // (5,6,7) is independent and can share a colour with any of them.
    const endpoints = [
      [0, 1, 2],
      [1, 2, 3],
      [2, 3, 4],
      [5, 6, 7],
    ]
    const groups = colorConstraints(endpoints)
    for (const group of groups) {
      const used = new Set<number>()
      for (const ci of group) {
        for (const pIdx of endpoints[ci]!) {
          expect(used.has(pIdx)).toBe(false)
          used.add(pIdx)
        }
      }
    }
    expect(groups.length).toBe(3)
  })
})
