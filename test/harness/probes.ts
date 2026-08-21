import { KIND_FLUID, KIND_NODE, KIND_OBJECT } from '../../src/sim/particles'
import type { SimWorld } from '../../src/sim/world'

/**
 * Measurements taken from a SimWorld from the OUTSIDE.
 *
 * These deliberately avoid the sim's own derived state wherever an independent
 * calculation is possible - the water surface is recomputed here rather than
 * read from WaterField, because asserting that WaterField agrees with itself
 * would prove nothing. The one exception is density, which has no external
 * definition and is read from the solver.
 */

export interface EnergyBreakdown {
  kinetic: number
  potential: number
  total: number
}

const isAlive = (sim: SimWorld, i: number): boolean => sim.particles.slots.alive[i] === 1

/** Kinetic energy of every dynamic particle, joules (per metre of depth). */
export function kineticEnergy(sim: SimWorld, kind?: number): number {
  const p = sim.particles
  let e = 0
  for (let i = 0; i < p.highWater; i++) {
    if (!isAlive(sim, i) || p.invMass[i] === 0) continue
    if (kind !== undefined && p.kind[i] !== kind) continue
    const m = 1 / p.invMass[i]!
    e += 0.5 * m * (p.velX[i]! ** 2 + p.velY[i]! ** 2)
  }
  return e
}

/** Gravitational potential relative to `datum`. */
export function potentialEnergy(sim: SimWorld, datum = 0, kind?: number): number {
  const p = sim.particles
  const g = Math.abs(sim.gravity)
  let e = 0
  for (let i = 0; i < p.highWater; i++) {
    if (!isAlive(sim, i) || p.invMass[i] === 0) continue
    if (kind !== undefined && p.kind[i] !== kind) continue
    const m = 1 / p.invMass[i]!
    e += m * g * (p.posY[i]! - datum)
  }
  return e
}

export function energy(sim: SimWorld, datum = 0, kind?: number): EnergyBreakdown {
  const kinetic = kineticEnergy(sim, kind)
  const potential = potentialEnergy(sim, datum, kind)
  return { kinetic, potential, total: kinetic + potential }
}

export function maxSpeed(sim: SimWorld, kind?: number): number {
  const p = sim.particles
  let m = 0
  for (let i = 0; i < p.highWater; i++) {
    if (!isAlive(sim, i)) continue
    if (kind !== undefined && p.kind[i] !== kind) continue
    const s = Math.hypot(p.velX[i]!, p.velY[i]!)
    if (s > m) m = s
  }
  return m
}

export function meanSpeed(sim: SimWorld, kind = KIND_FLUID): number {
  const p = sim.particles
  let sum = 0
  let n = 0
  for (let i = 0; i < p.highWater; i++) {
    if (!isAlive(sim, i) || p.kind[i] !== kind) continue
    sum += Math.hypot(p.velX[i]!, p.velY[i]!)
    n++
  }
  return n > 0 ? sum / n : 0
}

/** Particles that have left any plausible region. A non-zero count is a blow-up. */
export function escapedCount(sim: SimWorld, box: { x0: number; x1: number; y0: number; y1: number }): number {
  const p = sim.particles
  let n = 0
  for (let i = 0; i < p.highWater; i++) {
    if (!isAlive(sim, i)) continue
    const x = p.posX[i]!
    const y = p.posY[i]!
    if (!Number.isFinite(x) || !Number.isFinite(y)) n++
    else if (x < box.x0 || x > box.x1 || y < box.y0 || y > box.y1) n++
  }
  return n
}

export interface SurfaceProfile {
  /** Column centres, world x. */
  x: number[]
  /** Top of the water in each column, or NaN where dry. */
  top: number[]
  /** Columns holding at least `minParticles`. */
  wetColumns: number
  mean: number
  stdDev: number
  min: number
  max: number
}

/**
 * Water surface, computed independently of the sim's own WaterField.
 *
 * Uses a high percentile rather than the maximum so that a few droplets of
 * spray do not define the surface - the same mistake that made buoyancy behave
 * as though the water were eighteen metres deep.
 */
export function surfaceProfile(
  sim: SimWorld,
  opts: { x0: number; x1: number; columnWidth?: number; minParticles?: number; percentile?: number } = {
    x0: -60,
    x1: 60,
  },
): SurfaceProfile {
  const columnWidth = opts.columnWidth ?? 1
  const minParticles = opts.minParticles ?? 3
  const percentile = opts.percentile ?? 0.95
  const n = Math.max(1, Math.ceil((opts.x1 - opts.x0) / columnWidth))

  const columns: number[][] = Array.from({ length: n }, () => [])
  const p = sim.particles
  for (let i = 0; i < p.highWater; i++) {
    if (!isAlive(sim, i) || p.kind[i] !== KIND_FLUID) continue
    const c = Math.floor((p.posX[i]! - opts.x0) / columnWidth)
    if (c < 0 || c >= n) continue
    columns[c]!.push(p.posY[i]!)
  }

  const x: number[] = []
  const top: number[] = []
  const tops: number[] = []
  for (let c = 0; c < n; c++) {
    const col = columns[c]!
    x.push(opts.x0 + (c + 0.5) * columnWidth)
    if (col.length < minParticles) {
      top.push(NaN)
      continue
    }
    col.sort((a, b) => a - b)
    const idx = Math.min(col.length - 1, Math.floor(col.length * percentile))
    top.push(col[idx]!)
    tops.push(col[idx]!)
  }

  if (tops.length === 0) {
    return { x, top, wetColumns: 0, mean: NaN, stdDev: NaN, min: NaN, max: NaN }
  }
  const mean = tops.reduce((a, b) => a + b, 0) / tops.length
  const variance = tops.reduce((a, b) => a + (b - mean) ** 2, 0) / tops.length
  return {
    x,
    top,
    wetColumns: tops.length,
    mean,
    stdDev: Math.sqrt(variance),
    min: Math.min(...tops),
    max: Math.max(...tops),
  }
}

/** Volume of simulated water, m^2 (per metre of depth). Mass is conserved, so this should be too. */
export function waterVolume(sim: SimWorld): number {
  return sim.particles.countOfKind(KIND_FLUID) * sim.fluid.spacing * sim.fluid.spacing
}

export interface DensityStats {
  mean: number
  max: number
  min: number
  /** Fraction of particles more than 10% over rest density. */
  overCompressed: number
  restDensity: number
}

/** Reads the solver's own density field. There is no external definition to check against. */
export function densityStats(sim: SimWorld): DensityStats {
  const f = sim.fluid
  const d = f.density
  const n = f.liveCount
  if (n === 0) {
    return { mean: NaN, max: NaN, min: NaN, overCompressed: 0, restDensity: f.restDensity }
  }
  let sum = 0
  let max = -Infinity
  let min = Infinity
  let over = 0
  for (let a = 0; a < n; a++) {
    const v = d[a]!
    sum += v
    if (v > max) max = v
    if (v < min) min = v
    if (v > f.restDensity * 1.1) over++
  }
  return { mean: sum / n, max, min, overCompressed: over / n, restDensity: f.restDensity }
}

/**
 * Distance to the nearest other fluid particle, per particle. Detects the
 * tensile-instability clumping where particles collapse into strings.
 */
export function nearestNeighbourStats(sim: SimWorld): { mean: number; min: number; p05: number } {
  const p = sim.particles
  const idx: number[] = []
  for (let i = 0; i < p.highWater; i++) {
    if (isAlive(sim, i) && p.kind[i] === KIND_FLUID) idx.push(i)
  }
  if (idx.length < 2) return { mean: NaN, min: NaN, p05: NaN }

  const dists: number[] = []
  // O(n^2) is fine: harness scenarios are small by design.
  for (let a = 0; a < idx.length; a++) {
    let best = Infinity
    for (let b = 0; b < idx.length; b++) {
      if (a === b) continue
      const d = (p.posX[idx[a]!]! - p.posX[idx[b]!]!) ** 2 + (p.posY[idx[a]!]! - p.posY[idx[b]!]!) ** 2
      if (d < best) best = d
    }
    dists.push(Math.sqrt(best))
  }
  dists.sort((a, b) => a - b)
  return {
    mean: dists.reduce((a, b) => a + b, 0) / dists.length,
    min: dists[0]!,
    p05: dists[Math.floor(dists.length * 0.05)]!,
  }
}

/** Peak |strain| / breakStrain across live members, as a fraction. */
export function peakMemberLoad(sim: SimWorld): number {
  const d = sim.distance
  let peak = 0
  for (let i = 0; i < d.highWater; i++) {
    if (d.slots.alive[i] !== 1) continue
    const rest = d.rest[i]!
    if (rest <= 0) continue // welds carry no meaningful strain
    peak = Math.max(peak, Math.abs(d.strain[i]!))
  }
  return peak
}

export function liveMemberCount(sim: SimWorld): number {
  return sim.distance.count
}

export function particleCounts(sim: SimWorld): { nodes: number; fluid: number; objects: number } {
  return {
    nodes: sim.particles.countOfKind(KIND_NODE),
    fluid: sim.particles.countOfKind(KIND_FLUID),
    objects: sim.particles.countOfKind(KIND_OBJECT),
  }
}

/** Snapshot of every particle position, for determinism and reset-fidelity checks. */
export function positionFingerprint(sim: SimWorld): string {
  const p = sim.particles
  const parts: string[] = []
  for (let i = 0; i < p.highWater; i++) {
    if (!isAlive(sim, i)) continue
    parts.push(`${p.posX[i]!.toFixed(6)},${p.posY[i]!.toFixed(6)}`)
  }
  return parts.join(';')
}
