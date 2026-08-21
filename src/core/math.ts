export interface Vec2 {
  x: number
  y: number
}

export const vec2 = (x = 0, y = 0): Vec2 => ({ x, y })

export const TAU = Math.PI * 2

export function clamp(v: number, min: number, max: number): number {
  return v < min ? min : v > max ? max : v
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/** Inverse lerp, clamped to 0..1. Returns 0 when a === b. */
export function invLerp(a: number, b: number, v: number): number {
  if (a === b) return 0
  return clamp((v - a) / (b - a), 0, 1)
}

export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = invLerp(edge0, edge1, x)
  return t * t * (3 - 2 * t)
}

/** Exponential smoothing that is stable across variable frame times. */
export function damp(current: number, target: number, halfLifeSec: number, dt: number): number {
  if (halfLifeSec <= 0) return target
  return lerp(target, current, Math.pow(2, -dt / halfLifeSec))
}

export function dist(ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax
  const dy = by - ay
  return Math.hypot(dx, dy)
}

export function distSq(ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax
  const dy = by - ay
  return dx * dx + dy * dy
}

/** km/h -> m/s */
export const kphToMs = (kph: number): number => kph * (1000 / 3600)

/** m/s -> km/h */
export const msToKph = (ms: number): number => ms * (3600 / 1000)
