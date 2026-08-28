import { describe, it, expect } from 'vitest'
import { snapToDetent } from '../../src/core/math'

/**
 * The detent under a slider thumb.
 *
 * A wind slider that spans -250..250 has to be able to return to exactly zero.
 * Without a detent, "calm" is a single pixel on a 500-wide range and the storm
 * never quite stops - the mast keeps leaning at 5 kph. The rule is a magnetic
 * zone around the detent value and NOTHING anywhere else: the 50 kph tick marks
 * are scenery, and a slider that snapped to every one of them could not be set
 * to 175. Method: red-first - snapToDetent did not exist.
 */
describe('snapToDetent', () => {
  const zero = [0]

  it('pulls a value inside the radius onto the detent', () => {
    expect(snapToDetent(7, zero, 12)).toBe(0)
    expect(snapToDetent(-7, zero, 12)).toBe(0)
  })

  it('leaves a value outside the radius exactly where it was', () => {
    expect(snapToDetent(40, zero, 12)).toBe(40)
    expect(snapToDetent(-175, zero, 12)).toBe(-175)
  })

  it('holds the boundary of the radius inclusively', () => {
    expect(snapToDetent(12, zero, 12)).toBe(0)
    expect(snapToDetent(12.5, zero, 12)).toBe(12.5)
  })

  it('does not snap to values that are merely tick marks', () => {
    // Ticks every 50 kph, detent only at zero: 48 stays 48.
    expect(snapToDetent(48, zero, 12)).toBe(48)
    expect(snapToDetent(152, zero, 12)).toBe(152)
  })

  it('picks the nearest detent when several are in range', () => {
    expect(snapToDetent(11, [0, 20], 12)).toBe(20)
    expect(snapToDetent(9, [0, 20], 12)).toBe(0)
  })

  it('is inert with no detents or a zero radius', () => {
    expect(snapToDetent(3, [], 12)).toBe(3)
    expect(snapToDetent(3, zero, 0)).toBe(3)
  })
})
