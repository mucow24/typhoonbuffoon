import { describe, it, expect } from 'vitest'
import { GameLoop } from '../../src/core/loop'

/**
 * The fixed-timestep accumulator, at its worst moment.
 *
 * The failure this guards against was measured in the app: once one sim step
 * costs more than the 16.7 ms budget, the accumulator queues catch-up steps,
 * the frame gets slower, more steps queue - and the loop pins itself at its
 * wall-clock clamp running 15 steps a frame. A sim 60% over budget became a
 * sub-1-fps freeze. Overload must mean SLOW MOTION (drop sim time, stay
 * responsive), never a spiral. Method: red-first - the pre-fix loop ran 15
 * steps in this scenario.
 */

function makeLoop(opts: { maxCatchUpSteps?: number } = {}) {
  let steps = 0
  const loop = new GameLoop({
    fixedHz: 60,
    ...opts,
    fixedUpdate: () => {
      steps++
    },
    render: () => {},
  })
  return { loop, stepCount: () => steps }
}

describe('catch-up bounding', () => {
  it('never runs more than maxCatchUpSteps fixed steps in one frame', () => {
    const { loop, stepCount } = makeLoop()
    loop.beginFrames(0)
    // One second of wall clock arrives at once - a 60-step debt.
    loop.advance(1000)
    expect(stepCount()).toBeLessThanOrEqual(3)
    expect(loop.stats.stepsLastFrame).toBeLessThanOrEqual(3)
    expect(loop.stats.starved).toBe(true)
  })

  it('drops the un-runnable debt instead of carrying it forward', () => {
    const { loop, stepCount } = makeLoop()
    loop.beginFrames(0)
    loop.advance(500)
    const afterSpike = stepCount()
    // Back to healthy 60 Hz frames: each one runs AT MOST one catch-up step
    // beyond its own - the spike's debt must be gone, not amortised over the
    // next minute of gameplay.
    loop.advance(516)
    loop.advance(533)
    expect(stepCount() - afterSpike).toBeLessThanOrEqual(4)
  })

  it('runs exactly one step per frame at a healthy 60 Hz', () => {
    const { loop, stepCount } = makeLoop()
    loop.beginFrames(0)
    let t = 0
    for (let i = 0; i < 60; i++) {
      t += 1000 / 60
      loop.advance(t)
    }
    expect(stepCount()).toBeGreaterThanOrEqual(59)
    expect(stepCount()).toBeLessThanOrEqual(61)
    expect(loop.stats.starved).toBe(false)
  })

  it('catches up small hiccups without dropping time', () => {
    const { loop, stepCount } = makeLoop()
    loop.beginFrames(0)
    // A 50 ms frame (three steps of debt, within the catch-up allowance),
    // then a normal one: no sim time may be lost.
    loop.advance(50)
    loop.advance(50 + 1000 / 60)
    expect(stepCount()).toBe(4)
    expect(loop.stats.starved).toBe(false)
  })

  it('honours a configured maxCatchUpSteps', () => {
    const { loop, stepCount } = makeLoop({ maxCatchUpSteps: 1 })
    loop.beginFrames(0)
    loop.advance(1000)
    expect(stepCount()).toBe(1)
  })

  it('keeps simTime consistent with the steps actually run', () => {
    const { loop } = makeLoop()
    loop.beginFrames(0)
    loop.advance(1000)
    expect(loop.stats.simTime).toBeCloseTo(loop.stats.totalSteps / 60, 9)
  })
})
