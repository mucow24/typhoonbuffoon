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
  it('settles with a structure standing in it, as the seeded level has', () => {
    // The app always has the house and its anchors present. Every other test in
    // this file pours water into an empty world, and the empty world settles
    // while the app plateaus around 3 m/s - so the object was the difference,
    // not the resolution or the volume. This is the configuration a player is
    // actually looking at.
    const { sim, field } = beachWorld(0.35)
    const t = field.terrain
    const ground = t.heightAt(-8)
    sim.addObject({ cx: -8, cy: ground + 9, width: 8, height: 4.5, density: 150 })
    sim.spawnBlock(-8, ground + 18, 14, 10)

    const trace = run(sim, {
      seconds: 60,
      box: { x0: field.left - 5, x1: field.right + 5, y0: -60, y1: 140 },
    })

    expectFinite(trace)
    expectNoEscapes(trace, 'water around a structure')
    expectSettles(trace, { below: 1.2, maxBelow: 8, byFraction: 0.75, label: 'water around a structure' })
  })

  it('settles after a large dump from height, as the sandbox tool does', () => {
    // Matches what the dump-water tool produces in the app: a bigger volume
    // released from higher up than the other cases here. Measured in the app,
    // this plateaus around 3 m/s and stops decaying, where the smaller pours
    // reach 0.5. Sized to the real thing so the suite sees it.
    const { sim, field } = beachWorld(0.35)
    const t = field.terrain
    sim.spawnBlock(-8, t.heightAt(-8) + 18, 14, 10)

    const trace = run(sim, {
      seconds: 60,
      box: { x0: field.left - 5, x1: field.right + 5, y0: -60, y1: 140 },
    })

    expectFinite(trace)
    expectNoEscapes(trace, 'dumped beach water')
    expectSettles(trace, { below: 1.2, maxBelow: 8, byFraction: 0.75, label: 'dumped beach water' })
  })

  it('settles at the resolution the game actually ships with', () => {
    // The default in the app is 0.25 m. Every other test here uses 0.4-0.45,
    // which is coarser than anything a player will run, and coarser water is
    // calmer - so the suite was passing at a setting the product never uses.
    const { sim, field } = beachWorld(0.25)
    const t = field.terrain
    sim.spawnBlock(-8, t.heightAt(-8) + 14, 12, 8)

    const trace = run(sim, {
      seconds: 60,
      box: { x0: field.left - 5, x1: field.right + 5, y0: -60, y1: 120 },
    })

    expectFinite(trace)
    expectNoEscapes(trace, 'beach water at game resolution')
    expectSettles(trace, { below: 1.2, maxBelow: 8, byFraction: 0.75, label: 'beach water at 0.25 m' })
  })

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

  it('dissipates running down the shore - bounded transient, no sustained gain', () => {
    const { sim, field } = beachWorld(0.4)
    const t = field.terrain
    sim.spawnBlock(-8, t.heightAt(-8) + 14, 12, 8)
    const trace = run(sim, {
      seconds: 20,
      box: { x0: field.left - 5, x1: field.right + 5, y0: -60, y1: 120 },
    })
    // The impact of a 14 m drop pops transiently (unilateral PBF - see the
    // dam-break test); it must be bounded and be GONE: from t=5 the total sits
    // below the start and keeps falling.
    expectNoEnergyGain(trace, { tolerance: 0.25, label: 'water on the beach (transient)' })
    const start = trace.first.total
    for (const s of trace.samples.filter((x) => x.t >= 5)) {
      expect(s.total).toBeLessThan(start * 1.0)
    }
    expect(trace.last.total).toBeLessThan(start * 0.9)
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
    // the lowest point of the seabed, plus a little splash margin - impacts
    // redirect momentum and a droplet briefly beats the bulk figure.
    const fall = dropFrom - t.minHeight
    const freeFall = Math.sqrt(2 * 9.81 * fall)
    expectSpeedBelow(trace, freeFall * 1.25, 'beach water')
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

describe('water at rest is at rest', () => {
  /**
   * The bar: settled water has no current in it, and a structure standing in it
   * is not a propulsion device. Net velocity is the sharp test - a body of water
   * can have small local jitter and still be going nowhere, but a persistent
   * drift means momentum is being manufactured.
   */
  const netVelocity = (sim: SimWorld) => {
    const p = sim.particles
    let mx = 0
    let my = 0
    let mass = 0
    for (let i = 0; i < p.highWater; i++) {
      if (p.slots.alive[i] !== 1 || p.kind[i] !== 1) continue
      const m = 1 / p.invMass[i]!
      mx += m * p.velX[i]!
      my += m * p.velY[i]!
      mass += m
    }
    return mass > 0 ? Math.hypot(mx, my) / mass : 0
  }

  const meanSpeedOf = (sim: SimWorld) => {
    const p = sim.particles
    let sum = 0
    let n = 0
    for (let i = 0; i < p.highWater; i++) {
      if (p.slots.alive[i] !== 1 || p.kind[i] !== 1) continue
      sum += Math.hypot(p.velX[i]!, p.velY[i]!)
      n++
    }
    return n > 0 ? sum / n : 0
  }

  it('has no current once settled, with nothing in it', () => {
    const { sim, field } = beachWorld(0.4)
    sim.spawnBlock(-8, field.terrain.heightAt(-8) + 14, 12, 8)
    settle(sim, 120)
    expect(meanSpeedOf(sim)).toBeLessThan(0.05)
    // Recalibrated after the pair-based solver rewrite: measured 0.020 under
    // the per-particle walk, 0.026 under pair iteration (deterministic -
    // different summation order, slightly different film drainage). The
    // regime this exists to catch is the unsupported-bed creep the audit
    // measured at 0.087 m/s, three times above this bar.
    expect(netVelocity(sim)).toBeLessThan(0.035)
  })

  it('has no current once settled, with a structure standing in it', () => {
    const { sim, field } = beachWorld(0.4)
    const ground = field.terrain.heightAt(-8)
    sim.addObject({ cx: -8, cy: ground + 9, width: 8, height: 4.5, density: 150 })
    sim.spawnBlock(-8, ground + 18, 14, 10)
    settle(sim, 120)
    // A structure must not act as a pump. Net velocity is the sharp pump
    // detector and stays strict. Mean speed is looser here than in the empty
    // case for two real reasons: water cascading off the house spreads into
    // thin films that genuinely keep draining at this timescale, and the
    // house itself floats off as a raft whose wake is still relaxing
    // (measured 0.135 m/s at t=120 s and falling). What this must never show
    // again is the churn regime the audit measured - sustained 0.35 m/s "at
    // rest" with the water being actively stirred.
    expect(meanSpeedOf(sim)).toBeLessThan(0.18)
    // Net drift is looser than the empty case for the same reason as the mean:
    // the films still draining downslope are a genuinely directional flow
    // (measured 0.025 m/s at t=120 s and falling). A pump reads far above this
    // and does not decay.
    expect(netVelocity(sim)).toBeLessThan(0.04)
  })
})
