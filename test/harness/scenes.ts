import { KIND_FLUID } from '../../src/sim/particles'
import { MATERIALS, type MaterialId } from '../../src/sim/materials'
import { SimWorld } from '../../src/sim/world'
import { Terrain } from '../../src/world/terrain'
import { buildBeam } from '../../src/scenes/demos'

/**
 * Scenario builders.
 *
 * Terrain here is FLAT or a simple analytic shape, never the game's generated
 * beach. A test that has to reason about a procedural dune profile is a test
 * whose failures are hard to interpret, and hydrostatics needs a known floor.
 */

export interface WorldOptions {
  widthM?: number
  /** Flat floor height. */
  floor?: number
  spacing?: number
  substeps?: number
  /** Terrain override, for slopes and vessels. */
  terrain?: Terrain
}

export function flatTerrain(widthM: number, height: number, sampleSpacing = 1): Terrain {
  const count = Math.max(2, Math.round(widthM / sampleSpacing) + 1)
  const heights = new Float32Array(count).fill(height)
  return new Terrain(heights, -widthM / 2, widthM / 2)
}

/**
 * A basin: flat floor with SLOPED walls at the ends.
 *
 * The walls ramp rather than step. A heightfield cannot represent a vertical
 * face - the contact projection is vertical, so a particle a hair inside a
 * near-vertical wall is teleported the full wall height in one substep, which
 * manufactures potential energy out of nothing. That is a defect of the test
 * fixture, not of the fluid, and it hid the fact that the solver is stable.
 */
export function basinTerrain(
  widthM: number,
  floor: number,
  rimHeight: number,
  rimFraction = 0.16,
): Terrain {
  const sampleSpacing = 0.5
  const count = Math.max(4, Math.round(widthM / sampleSpacing) + 1)
  const heights = new Float32Array(count)
  const rim = Math.max(2, Math.floor(count * rimFraction))
  for (let i = 0; i < count; i++) {
    const fromLeft = i / rim
    const fromRight = (count - 1 - i) / rim
    const t = Math.min(fromLeft, fromRight, 1)
    heights[i] = floor + (rimHeight - floor) * (1 - t)
  }
  return new Terrain(heights, -widthM / 2, widthM / 2)
}

/** Constant slope from `leftHeight` to `rightHeight`. */
export function rampTerrain(widthM: number, leftHeight: number, rightHeight: number): Terrain {
  const sampleSpacing = 0.5
  const count = Math.max(2, Math.round(widthM / sampleSpacing) + 1)
  const heights = new Float32Array(count)
  for (let i = 0; i < count; i++) {
    heights[i] = leftHeight + (rightHeight - leftHeight) * (i / (count - 1))
  }
  return new Terrain(heights, -widthM / 2, widthM / 2)
}

export function makeWorld(opts: WorldOptions = {}): SimWorld {
  const widthM = opts.widthM ?? 40
  const sim = new SimWorld()
  sim.terrain = opts.terrain ?? flatTerrain(widthM, opts.floor ?? 0)
  sim.boundsX0 = -widthM / 2
  sim.boundsX1 = widthM / 2
  sim.fluid.spacing = opts.spacing ?? 0.25
  if (opts.substeps !== undefined) sim.substeps = opts.substeps
  return sim
}

export interface FillOptions {
  x0: number
  x1: number
  /** Bottom of the water body. Defaults to sitting on the terrain. */
  yBottom?: number
  yTop: number
  /** Jitter as a fraction of spacing. Zero gives a perfect lattice. */
  jitter?: number
  seed?: number
}

/**
 * Fill a region with water on a lattice, never below the terrain.
 *
 * Deliberately separate from SimWorld.fillTo: the harness must be able to
 * create a known, clean initial condition without depending on the production
 * spawn path, so that spawn bugs and solver bugs can be told apart.
 */
export function fillWater(sim: SimWorld, opts: FillOptions): number {
  const spacing = sim.fluid.spacing
  const mass = sim.fluid.particleMass
  const p = sim.particles
  const t = sim.terrain
  const jitter = opts.jitter ?? 0
  let seed = opts.seed ?? 12345
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0
    return seed / 4294967296 - 0.5
  }

  let n = 0
  for (let x = opts.x0 + spacing * 0.5; x < opts.x1; x += spacing) {
    const ground = t ? t.heightAt(x) : opts.yBottom
    const floor = ground ?? opts.yBottom
    if (floor === undefined || !Number.isFinite(floor)) {
      throw new Error('fillWater needs terrain or an explicit finite yBottom')
    }
    const bottom = Math.max(opts.yBottom ?? floor + spacing * 0.5, floor + spacing * 0.5)
    for (let y = bottom; y < opts.yTop; y += spacing) {
      p.create({
        x: x + rand() * spacing * jitter,
        y: y + rand() * spacing * jitter,
        invMass: 1 / mass,
        radius: spacing * 0.5,
        kind: KIND_FLUID,
      })
      n++
    }
  }
  return n
}

/** Water level a given particle count implies over a flat floor of known width. */
export function expectedLevel(sim: SimWorld, count: number, widthM: number, floor: number): number {
  const area = count * sim.fluid.spacing * sim.fluid.spacing
  return floor + area / widthM
}

export interface WallOptions {
  x: number
  yBottom: number
  yTop: number
  material?: MaterialId
  segments?: number
  /** Pin the top as well, making the wall immovable - isolates containment from strength. */
  rigid?: boolean
}

/** A vertical wall of members, for containment tests. */
export function buildWall(sim: SimWorld, opts: WallOptions): ReturnType<typeof buildBeam> {
  return buildBeam(sim, {
    x0: opts.x,
    y0: opts.yBottom,
    x1: opts.x,
    y1: opts.yTop,
    material: opts.material ?? 'wood',
    segments: opts.segments ?? Math.max(2, Math.round((opts.yTop - opts.yBottom) / 1.5)),
    pinStart: true,
    pinEnd: opts.rigid ?? false,
    clampStart: !(opts.rigid ?? false),
  })
}

export interface SimpleBeamOptions {
  length: number
  y: number
  x?: number
  material?: MaterialId
  segments?: number
  /** Extra mass hung on the mid-span node, kg. */
  loadKg?: number
}

/**
 * A beam pinned at both ends with a mid-span point load - the case with a
 * textbook deflection formula, which is what makes it worth testing.
 */
export function buildSimplySupported(sim: SimWorld, opts: SimpleBeamOptions) {
  const x = opts.x ?? 0
  const segments = opts.segments ?? 8
  const beam = buildBeam(sim, {
    x0: x - opts.length / 2,
    y0: opts.y,
    x1: x + opts.length / 2,
    y1: opts.y,
    material: opts.material ?? 'wood',
    segments,
    pinStart: true,
    pinEnd: true,
  })
  const mid = beam.nodes[Math.floor(beam.nodes.length / 2)]!
  if (opts.loadKg && opts.loadKg > 0) {
    const existing = sim.particles.massOf(mid)
    sim.particles.invMass[mid] = 1 / (existing + opts.loadKg)
  }
  return { ...beam, mid }
}

/** Analytic mid-span deflection of a simply supported beam under a central point load. */
export function analyticMidspanDeflection(loadN: number, length: number, EI: number): number {
  return (loadN * length ** 3) / (48 * EI)
}

/** Analytic tip deflection of a cantilever under a tip point load. */
export function analyticCantileverTip(loadN: number, length: number, EI: number): number {
  return (loadN * length ** 3) / (3 * EI)
}

export const materialEI = (id: MaterialId): number => MATERIALS[id].flexuralRigidity

/** Draft of a floating box by Archimedes: submerged fraction equals density ratio. */
export function archimedesSubmergedFraction(bodyDensity: number, waterDensity = 1000): number {
  return Math.min(1, bodyDensity / waterDensity)
}

/** Highest live fluid particle. */
export function topOfWater(sim: SimWorld): number {
  const p = sim.particles
  let top = -Infinity
  for (let i = 0; i < p.highWater; i++) {
    if (p.slots.alive[i] !== 1 || p.kind[i] !== KIND_FLUID) continue
    if (p.posY[i]! > top) top = p.posY[i]!
  }
  return top
}

/** Furthest-right live fluid particle, for dam-break front tracking. */
export function waterFront(sim: SimWorld, minParticlesAhead = 3): number {
  const p = sim.particles
  const xs: number[] = []
  for (let i = 0; i < p.highWater; i++) {
    if (p.slots.alive[i] !== 1 || p.kind[i] !== KIND_FLUID) continue
    xs.push(p.posX[i]!)
  }
  if (xs.length === 0) return NaN
  xs.sort((a, b) => b - a)
  // Ignore the leading few, so one stray droplet does not define the front.
  return xs[Math.min(minParticlesAhead - 1, xs.length - 1)]!
}

/** How much water lies beyond `x` - the leak measure for containment tests. */
export function waterBeyond(sim: SimWorld, x: number, side: 'left' | 'right'): number {
  const p = sim.particles
  let n = 0
  for (let i = 0; i < p.highWater; i++) {
    if (p.slots.alive[i] !== 1 || p.kind[i] !== KIND_FLUID) continue
    const px = p.posX[i]!
    if (side === 'right' ? px > x : px < x) n++
  }
  return n
}
