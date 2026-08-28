import { describe, it, expect } from 'vitest'
import { THUMB_PX, atFraction, fillRange, fillOrigin, fraction, offsetPx } from '../../src/ui/sliderGeometry'

/**
 * Where a slider's thumb, ticks and fill actually sit.
 *
 * A native range paints its fill from `min` to the thumb, which on a signed
 * range means the wind slider reads HALF FULL at dead calm - the bar says
 * "125 kph of something" when the storm is off. Drawing the track ourselves
 * fixes that, but only if the fill and the tick marks agree with where the
 * browser puts the thumb: the thumb CENTRE travels between `thumb/2` and
 * `width - thumb/2`, not across the whole track, so anything positioned at a
 * flat percentage drifts by half a thumb at the ends and the fill stops short
 * of (or runs past) the thumb it is supposed to reach.
 *
 * Method: red-first - none of this existed.
 */
describe('slider geometry', () => {
  describe('fraction', () => {
    it('maps a value onto its position in the range', () => {
      expect(fraction(0, -250, 250)).toBe(0.5)
      expect(fraction(125, -250, 250)).toBe(0.75)
      expect(fraction(-250, -250, 250)).toBe(0)
      expect(fraction(250, -250, 250)).toBe(1)
    })

    it('clamps out-of-range values and survives a degenerate range', () => {
      expect(fraction(400, -250, 250)).toBe(1)
      expect(fraction(-400, -250, 250)).toBe(0)
      expect(fraction(7, 3, 3)).toBe(0)
    })
  })

  describe('offsetPx', () => {
    it('insets the ends by half a thumb and leaves the centre alone', () => {
      // At the far left the thumb centre is half a thumb IN from the edge.
      expect(offsetPx(0)).toBe(THUMB_PX / 2)
      expect(offsetPx(1)).toBe(-THUMB_PX / 2)
      expect(offsetPx(0.5)).toBe(0)
    })

    it('moves the correction linearly across the track', () => {
      expect(offsetPx(0.25)).toBeCloseTo(THUMB_PX / 4, 9)
      expect(offsetPx(0.75)).toBeCloseTo(-THUMB_PX / 4, 9)
    })
  })

  describe('atFraction', () => {
    it('writes a CSS length that combines the percentage and the inset', () => {
      expect(atFraction(0)).toBe('calc(0% + 7px)')
      expect(atFraction(1)).toBe('calc(100% - 7px)')
      expect(atFraction(0.5)).toBe('calc(50% + 0px)')
    })
  })

  describe('fillOrigin', () => {
    it('is zero when zero is inside the range', () => {
      expect(fillOrigin(-250, 250)).toBe(0)
    })

    it('falls back to the low end when zero is outside it', () => {
      // A 1..32 substep slider has no meaningful "zero" to fill from.
      expect(fillOrigin(1, 32)).toBe(1)
      expect(fillOrigin(-6, -1)).toBe(-1)
    })
  })

  describe('fillRange', () => {
    it('runs from the centre outward on a signed range', () => {
      expect(fillRange(100, 0, -250, 250)).toEqual([0.5, 0.7])
      expect(fillRange(-100, 0, -250, 250)).toEqual([0.3, 0.5])
    })

    it('is empty at the origin', () => {
      const [lo, hi] = fillRange(0, 0, -250, 250)
      expect(lo).toBe(hi)
      expect(lo).toBe(0.5)
    })

    it('reaches the ends', () => {
      expect(fillRange(250, 0, -250, 250)).toEqual([0.5, 1])
      expect(fillRange(-250, 0, -250, 250)).toEqual([0, 0.5])
    })

    it('runs from the left on an unsigned range, as a native track does', () => {
      expect(fillRange(100, 0, 0, 250)).toEqual([0, 0.4])
    })

    it('always returns low end first', () => {
      const [lo, hi] = fillRange(-200, 0, -250, 250)
      expect(lo).toBeLessThan(hi)
    })
  })
})
