import { describe, it, expect } from 'vitest'
import { Conditions, WIND_MAX_KPH } from '../../src/game/conditions'
import { Field } from '../../src/world/field'
import { kphToMs } from '../../src/core/math'
import { basinTerrain, makeWorld } from '../harness'

/**
 * The wind slider is signed: -250..250 kph, sign meaning direction.
 *
 * The trap this guards against is silent: WindField carries speed and heading
 * separately, and both `velocityAt` and `applyWind` bail on `baseSpeed <= 0`.
 * Feeding a signed kph straight into baseSpeed therefore produces a slider
 * whose entire left half is DEAD AIR - no error, no NaN, just a storm that
 * does nothing. Conditions has to split the sign off into `direction`.
 * Method: red-first - the pre-change update() left direction untouched and
 * baseSpeed negative.
 */

function makeConditions() {
  const sim = makeWorld({ widthM: 40, spacing: 0.5, terrain: basinTerrain(40, 0, 2) })
  return { sim, conditions: new Conditions(sim, new Field(40)) }
}

describe('signed wind', () => {
  it('blows to the right at full positive throttle', () => {
    const { sim, conditions } = makeConditions()
    conditions.windKph = WIND_MAX_KPH
    conditions.update(1 / 60)

    expect(sim.wind.baseSpeed).toBeCloseTo(kphToMs(250), 6)
    expect(sim.wind.direction).toBe(1)
    expect(sim.wind.velocityAt(0)).toBeGreaterThan(0)
  })

  it('blows to the left at full negative throttle', () => {
    const { sim, conditions } = makeConditions()
    conditions.windKph = -WIND_MAX_KPH
    conditions.update(1 / 60)

    // Speed is a magnitude: -250 kph is a category 5 pointing the other way,
    // not a negative wind speed the solver will discard.
    expect(sim.wind.baseSpeed).toBeCloseTo(kphToMs(250), 6)
    expect(sim.wind.direction).toBe(-1)
    expect(sim.wind.velocityAt(0)).toBeLessThan(0)
  })

  it('is mirror-symmetric about zero', () => {
    const right = makeConditions()
    right.conditions.windKph = 175
    right.conditions.update(1 / 60)
    const vRight = right.sim.wind.velocityAt(3)

    const left = makeConditions()
    left.conditions.windKph = -175
    left.conditions.update(1 / 60)
    const vLeft = left.sim.wind.velocityAt(3)

    expect(vLeft).toBeCloseTo(-vRight, 9)
  })

  it('is calm at the detent', () => {
    const { sim, conditions } = makeConditions()
    conditions.windKph = 0
    conditions.update(1 / 60)

    expect(sim.wind.baseSpeed).toBe(0)
    expect(sim.wind.velocityAt(0)).toBe(0)
  })

  it('rates a leftward storm as severe as the same storm rightward', () => {
    const a = makeConditions().conditions
    const b = makeConditions().conditions
    a.windKph = 200
    b.windKph = -200

    expect(b.severity()).toBeCloseTo(a.severity(), 9)
    // A storm blowing the other way is not LESS severe than calm air.
    expect(b.severity()).toBeGreaterThan(0)
  })

  it('resets a leftward storm to calm', () => {
    const { sim, conditions } = makeConditions()
    conditions.windKph = -120
    conditions.update(1 / 60)
    conditions.reset()

    expect(conditions.windKph).toBe(0)
    expect(sim.wind.baseSpeed).toBe(0)
    expect(sim.wind.velocityAt(0)).toBe(0)
  })
})
