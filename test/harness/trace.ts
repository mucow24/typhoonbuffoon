import type { SimWorld } from '../../src/sim/world'
import { energy, escapedCount, maxSpeed, speedPercentile, surfaceProfile, waterVolume } from './probes'

export interface TraceSample {
  t: number
  kinetic: number
  potential: number
  total: number
  maxSpeed: number
  p99Speed: number
  waterVolume: number
  surfaceMean: number
  surfaceStdDev: number
  escaped: number
}

export interface RunOptions {
  seconds: number
  /** Simulated seconds between samples. */
  sampleEvery?: number
  /** Bounds outside which a particle counts as escaped. */
  box?: { x0: number; x1: number; y0: number; y1: number }
  /** Column range for the surface probe. */
  surface?: { x0: number; x1: number; columnWidth?: number }
  /** Called every step, before stepping. For driving wind, waves, inflow. */
  each?: (t: number, step: number) => void
  hz?: number
}

/**
 * A time series of a run.
 *
 * Physics failures here are almost always about EVOLUTION - energy creeping up,
 * a surface drifting, a structure never settling. A final-value assertion
 * cannot see any of those, which is precisely how a fluid that detonates on
 * contact got reported as working.
 */
export class Trace {
  readonly samples: TraceSample[] = []

  get first(): TraceSample {
    const s = this.samples[0]
    if (!s) throw new Error('trace is empty')
    return s
  }

  get last(): TraceSample {
    const s = this.samples[this.samples.length - 1]
    if (!s) throw new Error('trace is empty')
    return s
  }

  max(key: keyof TraceSample): number {
    return Math.max(...this.samples.map((s) => s[key]))
  }

  min(key: keyof TraceSample): number {
    return Math.min(...this.samples.map((s) => s[key]))
  }

  /** Sample at which `key` peaks, for diagnostics. */
  argmax(key: keyof TraceSample): TraceSample {
    return this.samples.reduce((a, b) => (b[key] > a[key] ? b : a))
  }

  series(key: keyof TraceSample): number[] {
    return this.samples.map((s) => s[key])
  }

  /** Human-readable summary, used in assertion messages so failures are diagnosable. */
  describe(key: keyof TraceSample, unit = ''): string {
    const vals = this.series(key)
    const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : String(v))
    const peak = this.argmax(key)
    return (
      `${String(key)}: start=${fmt(vals[0]!)}${unit} end=${fmt(vals[vals.length - 1]!)}${unit} ` +
      `peak=${fmt(peak[key])}${unit} at t=${peak.t.toFixed(2)}s`
    )
  }

  /** Compact table, for when a failure message needs the whole shape. */
  table(keys: (keyof TraceSample)[]): string {
    const head = ['t', ...keys.map(String)].join('\t')
    const rows = this.samples.map((s) =>
      [s.t.toFixed(2), ...keys.map((k) => (Number.isFinite(s[k]) ? s[k].toFixed(3) : String(s[k])))].join('\t'),
    )
    return [head, ...rows].join('\n')
  }
}

/**
 * Step a world forward, recording a trace.
 *
 * Sampling is by simulated time, not by frame, so a test reads the same whether
 * the step rate changes.
 */
export function run(sim: SimWorld, opts: RunOptions): Trace {
  const hz = opts.hz ?? 60
  const dt = 1 / hz
  const steps = Math.round(opts.seconds * hz)
  const sampleEvery = opts.sampleEvery ?? 0.25
  const sampleStride = Math.max(1, Math.round(sampleEvery * hz))
  const box = opts.box ?? { x0: -1e4, x1: 1e4, y0: -1e4, y1: 1e4 }
  const surf = opts.surface ?? { x0: sim.boundsX0, x1: sim.boundsX1 }

  const trace = new Trace()

  const sample = (t: number) => {
    const e = energy(sim)
    const s = surfaceProfile(sim, { ...surf })
    trace.samples.push({
      t,
      kinetic: e.kinetic,
      potential: e.potential,
      total: e.total,
      maxSpeed: maxSpeed(sim),
      p99Speed: speedPercentile(sim, 0.99),
      waterVolume: waterVolume(sim),
      surfaceMean: s.mean,
      surfaceStdDev: s.stdDev,
      escaped: escapedCount(sim, box),
    })
  }

  sample(0)
  for (let i = 0; i < steps; i++) {
    opts.each?.(i * dt, i)
    sim.step(dt)
    if ((i + 1) % sampleStride === 0) sample((i + 1) * dt)
  }
  if (trace.samples[trace.samples.length - 1]!.t < steps * dt) sample(steps * dt)

  return trace
}

/** Step without recording, for settling a scene before the interesting part. */
export function settle(sim: SimWorld, seconds: number, hz = 60): void {
  const dt = 1 / hz
  const steps = Math.round(seconds * hz)
  for (let i = 0; i < steps; i++) sim.step(dt)
}
