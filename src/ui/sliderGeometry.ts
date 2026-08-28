/**
 * Slider metrics: where the thumb, the tick marks and the track fill go.
 *
 * Pure geometry, kept out of controls.ts so it can be tested - the CSS below
 * it is drawing, but this is arithmetic, and it is the arithmetic that goes
 * wrong. Everything here is expressed in the browser's own coordinates: a
 * range input's thumb CENTRE travels from `thumb/2` to `width - thumb/2`,
 * never across the full track, so positions are a percentage plus a
 * half-thumb correction rather than a bare percentage.
 */

import { clamp } from '../core/math'

/** Thumb diameter in px. The kit draws its own thumb, so this is exact. */
export const THUMB_PX = 14
/** Track height in px. */
export const TRACK_PX = 4

/** Position of a value within its range, 0..1, clamped. */
export function fraction(value: number, min: number, max: number): number {
  if (max === min) return 0
  return clamp((value - min) / (max - min), 0, 1)
}

/** The half-thumb correction, in px, to add to a flat percentage. */
export function offsetPx(frac: number): number {
  return (0.5 - frac) * THUMB_PX
}

/** A CSS length landing exactly on where the thumb centre can reach. */
export function atFraction(frac: number): string {
  const pct = Number((frac * 100).toFixed(4))
  const off = Number(offsetPx(frac).toFixed(3))
  return `calc(${pct}% ${off < 0 ? '-' : '+'} ${Math.abs(off)}px)`
}

/**
 * Where the fill starts. A filled bar means "this much of something", so it
 * grows from zero - which on a signed range is the middle, and on every
 * range that never reaches zero is simply the low end, exactly as a native
 * track already behaves.
 */
export function fillOrigin(min: number, max: number): number {
  return clamp(0, min, max)
}

/** The span the fill covers, as 0..1 fractions, low end first. */
export function fillRange(
  value: number,
  origin: number,
  min: number,
  max: number,
): [number, number] {
  const v = fraction(value, min, max)
  const o = fraction(origin, min, max)
  return v < o ? [v, o] : [o, v]
}
