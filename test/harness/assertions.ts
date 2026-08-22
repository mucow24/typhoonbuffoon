import { expect } from 'vitest'
import type { Trace, TraceSample } from './trace'

/**
 * Domain assertions.
 *
 * Every one of these fails with the shape of the run attached, not just a
 * boolean. "expected true to be false" on a physics failure tells you nothing;
 * "energy grew 340x, peaking at t=3.2s" tells you where to look.
 */

const fmt = (v: number) => (Number.isFinite(v) ? v.toPrecision(4) : String(v))

/**
 * A closed system must not gain energy.
 *
 * PBD dissipates, so the expected direction is down. Growth means the solver is
 * doing work on the system, which is the signature of the position-correction
 * feedback that launches particles.
 */
export function expectNoEnergyGain(trace: Trace, opts: { tolerance?: number; label?: string } = {}): void {
  const tolerance = opts.tolerance ?? 0.05
  const start = trace.first.total
  const peak = trace.max('total')
  const scale = Math.max(Math.abs(start), 1e-6)
  const growth = (peak - start) / scale

  if (growth > tolerance) {
    throw new Error(
      `${opts.label ?? 'system'} gained energy: ${fmt(start)} J -> peak ${fmt(peak)} J ` +
        `(+${(growth * 100).toFixed(1)}%, tolerance ${(tolerance * 100).toFixed(0)}%)\n` +
        `  ${trace.describe('total', ' J')}\n` +
        `  ${trace.describe('kinetic', ' J')}\n` +
        `  ${trace.describe('maxSpeed', ' m/s')}`,
    )
  }
}

/** Nothing may reach an implausible speed. Catches launches the energy test can average away. */
export function expectSpeedBelow(trace: Trace, limit: number, label = 'particles'): void {
  const peak = trace.max('maxSpeed')
  if (peak > limit) {
    const at = trace.argmax('maxSpeed')
    throw new Error(
      `${label} exceeded ${limit} m/s: peak ${fmt(peak)} m/s at t=${at.t.toFixed(2)}s\n` +
        `  ${trace.describe('maxSpeed', ' m/s')}`,
    )
  }
}

/**
 * The system must come to rest.
 *
 * Judged on the 99th percentile, with a separate looser bound on the outright
 * maximum. Asserting only on the max means one twitching particle out of six
 * hundred fails a pool whose mean speed is 0.03 m/s, and it gets stricter as
 * the particle count rises even though the water is calmer.
 */
export function expectSettles(
  trace: Trace,
  opts: { below: number; maxBelow?: number; byFraction?: number; label?: string },
): void {
  const byFraction = opts.byFraction ?? 0.6
  const cutoff = trace.last.t * byFraction
  const tail = trace.samples.filter((s) => s.t >= cutoff)
  const worstTypical = Math.max(...tail.map((s) => s.p99Speed))
  if (worstTypical > opts.below) {
    throw new Error(
      `${opts.label ?? 'system'} never settled: 99th-percentile speed after ` +
        `t=${cutoff.toFixed(1)}s was ${fmt(worstTypical)} m/s, expected below ${opts.below}\n` +
        `  ${trace.describe('p99Speed', ' m/s')}\n  ${trace.describe('maxSpeed', ' m/s')}`,
    )
  }
  const maxBelow = opts.maxBelow ?? opts.below * 8
  const worstAny = Math.max(...tail.map((s) => s.maxSpeed))
  if (worstAny > maxBelow) {
    throw new Error(
      `${opts.label ?? 'system'} had a particle at ${fmt(worstAny)} m/s after ` +
        `t=${cutoff.toFixed(1)}s, well past the ${maxBelow} m/s a settled system allows\n` +
        `  ${trace.describe('maxSpeed', ' m/s')}`,
    )
  }
}

/** No particle may leave the plausible region. */
export function expectNoEscapes(trace: Trace, label = 'particles'): void {
  const peak = trace.max('escaped')
  if (peak > 0) {
    const at = trace.argmax('escaped')
    throw new Error(
      `${peak} ${label} left the world (first seen by t=${at.t.toFixed(2)}s). ` +
        `This is a blow-up, not a tuning problem.\n  ${trace.describe('maxSpeed', ' m/s')}`,
    )
  }
}

/** Water is neither created nor destroyed in a closed scenario. */
export function expectVolumeConserved(trace: Trace, tolerance = 0.001): void {
  const start = trace.first.waterVolume
  for (const s of trace.samples) {
    const drift = Math.abs(s.waterVolume - start) / Math.max(start, 1e-9)
    if (drift > tolerance) {
      throw new Error(
        `water volume changed: ${fmt(start)} m^2 -> ${fmt(s.waterVolume)} m^2 at t=${s.t.toFixed(2)}s ` +
          `(${(drift * 100).toFixed(2)}%, tolerance ${(tolerance * 100).toFixed(2)}%)`,
      )
    }
  }
}

/** A body of water at rest has a level surface. */
export function expectFlatSurface(trace: Trace, opts: { stdDevBelow: number; byFraction?: number }): void {
  const byFraction = opts.byFraction ?? 0.6
  const cutoff = trace.last.t * byFraction
  const tail = trace.samples.filter((s) => s.t >= cutoff && Number.isFinite(s.surfaceStdDev))
  if (tail.length === 0) throw new Error('no wet columns to measure a surface from')
  const worst = Math.max(...tail.map((s) => s.surfaceStdDev))
  if (worst > opts.stdDevBelow) {
    throw new Error(
      `water surface was not level: std dev ${fmt(worst)} m after t=${cutoff.toFixed(1)}s, ` +
        `expected below ${opts.stdDevBelow} m\n  ${trace.describe('surfaceStdDev', ' m')}\n` +
        `  ${trace.describe('surfaceMean', ' m')}`,
    )
  }
}

/** The surface must not drift up or down once settled - a slow creep is still a bug. */
export function expectSurfaceStable(trace: Trace, opts: { driftBelow: number; byFraction?: number }): void {
  const byFraction = opts.byFraction ?? 0.5
  const cutoff = trace.last.t * byFraction
  const tail = trace.samples.filter((s) => s.t >= cutoff && Number.isFinite(s.surfaceMean))
  if (tail.length < 2) throw new Error('not enough samples to measure surface drift')
  const first = tail[0]!.surfaceMean
  const last = tail[tail.length - 1]!.surfaceMean
  const drift = Math.abs(last - first)
  if (drift > opts.driftBelow) {
    throw new Error(
      `water surface drifted ${fmt(drift)} m between t=${tail[0]!.t.toFixed(1)}s and ` +
        `t=${tail[tail.length - 1]!.t.toFixed(1)}s, expected below ${opts.driftBelow} m\n` +
        `  ${trace.describe('surfaceMean', ' m')}`,
    )
  }
}

export function expectFinite(trace: Trace, keys: (keyof TraceSample)[] = ['total', 'maxSpeed', 'surfaceMean']): void {
  for (const s of trace.samples) {
    for (const k of keys) {
      const v = s[k]
      if (typeof v === 'number' && !Number.isFinite(v) && !(k === 'surfaceMean' && s.waterVolume === 0)) {
        throw new Error(`${String(k)} became ${v} at t=${s.t.toFixed(2)}s`)
      }
    }
  }
}

/** Value within a relative tolerance of a reference, with a readable message. */
export function expectNear(
  actual: number,
  expected: number,
  opts: { rel?: number; abs?: number; label: string },
): void {
  const rel = opts.rel ?? 0.1
  const abs = opts.abs ?? 0
  const allowed = Math.max(abs, Math.abs(expected) * rel)
  const delta = Math.abs(actual - expected)
  if (!(delta <= allowed)) {
    throw new Error(
      `${opts.label}: got ${fmt(actual)}, expected ${fmt(expected)} ` +
        `(off by ${fmt(delta)}, allowed ${fmt(allowed)} = ${(rel * 100).toFixed(0)}%)`,
    )
  }
  expect(delta).toBeLessThanOrEqual(allowed)
}

/** Ordering assertion, for "more load must mean more deflection" style checks. */
export function expectMonotonic(
  values: number[],
  direction: 'increasing' | 'decreasing',
  label: string,
): void {
  for (let i = 1; i < values.length; i++) {
    const ok = direction === 'increasing' ? values[i]! >= values[i - 1]! : values[i]! <= values[i - 1]!
    if (!ok) {
      throw new Error(
        `${label} was not ${direction}: [${values.map(fmt).join(', ')}] breaks at index ${i}`,
      )
    }
  }
}
