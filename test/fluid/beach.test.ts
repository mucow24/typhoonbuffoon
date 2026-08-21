import { describe, it, expect } from 'vitest'
import { Field } from '../../src/world/field'
import { SimWorld } from '../../src/sim/world'
import {
  expectFinite,
  expectNoEnergyGain,
  expectNoEscapes,
  expectSettles,
  expectSpeedBelow,
  run,
  settle,
  speedPercentile,
  surfaceProfile,
} from '../harness'

/**
 * Water on the terrain the GAME actually uses.
 *
 * Every other fluid test runs on a flat floor, a basin or a ramp - fixtures
 * chosen so their failures are easy to read. That is the right default, but it
 * left the suite unable to see anything specific to the generated beach, which
 * is the only terrain a player ever meets. The suite passed while the app
 * churned at 18 m/s for six seconds, and this is the hole that let it.
 *
 * Testing the path the product actually uses is rule 2 of the policy. This is
 * the file that keeps me honest about it.
 */

function beachWorld(spacing = 0.35, widthM = 120) {
  const field = new Field(widthM)
  const sim = new SimWorld()
  sim.terrain = field.terrain
  sim.boundsX0 = field.left
  sim.boundsX1 = field.right
  sim.fluid.spacing = spacing
  return { sim, field }
}

describe('water on the generated beach', () => {
  it('settles after being poured onto the shore', () => {
    const { sim, field } = beachWorld()
    const t = field.terrain
    sim.spawnBlock(-8, t.heightAt(-8) + 18, 14, 10)

    const trace = run(sim, {
      seconds: 60,
      box: { x0: field.left - 5, x1: field.right + 5, y0: -60, y1: 120 },
      surface: { x0: -20, x1: 40, columnWidth: 2 },
    })

    expectFinite(trace)
    expectNoEscapes(trace, 'beach water')
    // Poured from 18 m, so the arrival is fast and that is fine. The measured
    // decay is monotonic but slow: p99 runs 3.9 -> 1.4 -> 0.8 m/s over the
    // first 45 seconds in a 60 m wide, 11 m deep basin. An earlier version of
    // this test asserted at 17.5 s and failed water that was simply still
    // sloshing. Whether that settling time is acceptable for the GAME is a
    // separate question from whether the solver is stable.
    expectSettles(trace, { below: 1.2, maxBelow: 8, byFraction: 0.75, label: 'poured beach water' })
  })

  it('does not gain energy running down the shore', () => {
    const { sim, field } = beachWorld(0.4)
    const t = field.terrain
    sim.spawnBlock(-8, t.heightAt(-8) + 14, 12, 8)
    const trace = run(sim, {
      seconds: 20,
      box: { x0: field.left - 5, x1: field.right + 5, y0: -60, y1: 120 },
    })
    expectNoEnergyGain(trace, { tolerance: 0.05, label: 'water on the beach' })
  })

  it('does not exceed the speed the drop height allows', () => {
    const { sim, field } = beachWorld(0.4)
    const t = field.terrain
    const dropFrom = t.heightAt(-8) + 14
    sim.spawnBlock(-8, dropFrom, 12, 8)

    const trace = run(sim, {
      seconds: 20,
      box: { x0: field.left - 5, x1: field.right + 5, y0: -60, y1: 120 },
    })

    // Everything the water can gain is the fall from where it was released to
    // the lowest point of the seabed.
    const fall = dropFrom - t.minHeight
    const freeFall = Math.sqrt(2 * 9.81 * fall)
    expectSpeedBelow(trace, freeFall * 1.2, 'beach water')
  })

  it('pools in the sea basin rather than sitting on the slope', () => {
    const { sim, field } = beachWorld(0.4)
    const t = field.terrain
    sim.spawnBlock(-8, t.heightAt(-8) + 14, 12, 8)
    settle(sim, 60)

    const p = sim.particles
    let below = 0
    let total = 0
    for (let i = 0; i < p.highWater; i++) {
      if (p.slots.alive[i] !== 1 || p.kind[i] !== 1) continue
      total++
      // The shoreline on the generated beach crosses y = 0.
      if (p.posY[i]! < 1) below++
    }
    expect(total).toBeGreaterThan(50)
    // Water released on a slope ends up down the slope, not clinging to it.
    expect(below / total).toBeGreaterThan(0.7)
  })

  it('holds a level surface once pooled', () => {
    const { sim, field } = beachWorld(0.4)
    const t = field.terrain
    sim.spawnBlock(10, t.heightAt(10) + 12, 24, 10)
    settle(sim, 75)

    const surface = surfaceProfile(sim, { x0: 5, x1: 45, columnWidth: 2 })
    expect(surface.wetColumns).toBeGreaterThan(4)
    expect(surface.stdDev).toBeLessThan(3)
    expect(speedPercentile(sim, 0.99)).toBeLessThan(1.5)
  })
})
