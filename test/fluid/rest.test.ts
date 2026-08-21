import { describe, it, expect } from 'vitest'
import {
  basinTerrain,
  expectFlatSurface,
  expectFinite,
  expectNear,
  expectNoEnergyGain,
  expectNoEscapes,
  expectSettles,
  expectSpeedBelow,
  expectSurfaceStable,
  expectVolumeConserved,
  fillWater,
  makeWorld,
  nearestNeighbourStats,
  run,
  settle,
  surfaceProfile,
} from '../harness'

/**
 * Water at rest.
 *
 * These are the cheapest possible checks on a fluid and the ones that catch the
 * most: a pool that is left alone must stay where it is, keep its level, keep
 * its volume, and never gain energy. Anything that fails here cannot possibly
 * behave under a wave or against a wall.
 */

const POOL_WIDTH = 30
const FLOOR = 0

function stillPool(spacing = 0.4, depth = 3) {
  const sim = makeWorld({
    widthM: 40,
    spacing,
    terrain: basinTerrain(40, FLOOR, 12),
  })
  const count = fillWater(sim, { x0: -POOL_WIDTH / 2, x1: POOL_WIDTH / 2, yTop: FLOOR + depth })
  return { sim, count }
}

describe('water at rest', () => {
  it('does not gain energy when left alone', () => {
    const { sim } = stillPool()
    // Deliberately NOT settled first. Settling before measuring lets the blow-up
    // happen off-camera and inflates the baseline, which is how this assertion
    // passed while the pool was reaching 44 m/s.
    const trace = run(sim, {
      seconds: 20,
      box: { x0: -25, x1: 25, y0: -20, y1: 60 },
      surface: { x0: -POOL_WIDTH / 2, x1: POOL_WIDTH / 2 },
    })

    expectFinite(trace)
    expectNoEnergyGain(trace, { tolerance: 0.05, label: 'still pool' })
  })

  it('comes to rest instead of churning forever', () => {
    const { sim } = stillPool()
    const trace = run(sim, {
      seconds: 25,
      box: { x0: -25, x1: 25, y0: -20, y1: 60 },
      surface: { x0: -POOL_WIDTH / 2, x1: POOL_WIDTH / 2 },
    })

    // Water poured in has to slosh a little, but by the back half of a
    // 25 second run it should be effectively still.
    expectSettles(trace, { below: 0.35, byFraction: 0.6, label: 'still pool' })
  })

  it('never launches a particle', () => {
    const { sim } = stillPool()
    const trace = run(sim, {
      seconds: 20,
      box: { x0: -25, x1: 25, y0: -20, y1: 60 },
    })

    // Nothing in a 3 m deep pool has any business moving faster than free-fall
    // from its own surface, which is about 7.7 m/s.
    expectSpeedBelow(trace, 8, 'still pool')
    expectNoEscapes(trace, 'water particles')
  })

  it('holds a level surface', () => {
    const { sim } = stillPool()
    settle(sim, 5)
    const trace = run(sim, {
      seconds: 20,
      surface: { x0: -POOL_WIDTH / 2 + 2, x1: POOL_WIDTH / 2 - 2, columnWidth: 1 },
      box: { x0: -25, x1: 25, y0: -20, y1: 60 },
    })

    expectFlatSurface(trace, { stdDevBelow: 0.25 })
    expectSurfaceStable(trace, { driftBelow: 0.2 })
  })

  it('conserves its volume', () => {
    const { sim } = stillPool()
    const trace = run(sim, { seconds: 15, box: { x0: -25, x1: 25, y0: -20, y1: 60 } })
    expectVolumeConserved(trace)
  })

  it('settles at the level its volume implies', () => {
    const { sim, count } = stillPool(0.4, 3)
    settle(sim, 12)

    // Conservation: the water has to go somewhere, and the basin floor is flat,
    // so the level follows from particle count alone.
    const area = count * sim.fluid.spacing * sim.fluid.spacing
    const wetted = POOL_WIDTH
    const predicted = FLOOR + area / wetted

    const surface = surfaceProfile(sim, {
      x0: -POOL_WIDTH / 2 + 2,
      x1: POOL_WIDTH / 2 - 2,
      columnWidth: 1,
    })

    expectNear(surface.mean, predicted, {
      rel: 0.15,
      abs: sim.fluid.spacing,
      label: 'settled water level vs volume/width',
    })
  })

  it('does not clump into strings', () => {
    const { sim } = stillPool()
    settle(sim, 12)

    const nn = nearestNeighbourStats(sim)
    // Particles should sit roughly a spacing apart. Collapsing to a fraction of
    // that is the tensile instability, and it looks like blue spaghetti.
    expect(nn.p05).toBeGreaterThan(sim.fluid.spacing * 0.45)
    expect(nn.mean).toBeGreaterThan(sim.fluid.spacing * 0.6)
  })

  it('reaches the same level at either resolution', () => {
    const level = (spacing: number) => {
      const sim = makeWorld({ widthM: 40, spacing, terrain: basinTerrain(40, FLOOR, 12) })
      fillWater(sim, { x0: -POOL_WIDTH / 2, x1: POOL_WIDTH / 2, yTop: FLOOR + 3 })
      settle(sim, 12)
      return surfaceProfile(sim, {
        x0: -POOL_WIDTH / 2 + 2,
        x1: POOL_WIDTH / 2 - 2,
        columnWidth: 1,
      }).mean
    }

    // Resolution is a player-facing slider. If the water sits at a different
    // height depending on it, every level is silently different per setting.
    const coarse = level(0.5)
    const fine = level(0.3)
    expectNear(fine, coarse, { rel: 0.12, abs: 0.35, label: 'level at 0.3 m vs 0.5 m spacing' })
  })
})
