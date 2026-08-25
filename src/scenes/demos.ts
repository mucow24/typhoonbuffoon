import {
  MATERIALS,
  axialCompliance,
  bendCompliance,
  areaOf,
  massPerMetre,
  materialIndex,
  segmentsFor,
  type MaterialId,
} from '../sim/materials'
import type { SimWorld } from '../sim/world'

export interface ChainOptions {
  x: number
  y: number
  links?: number
  spacing?: number
  compliance?: number
  zeta?: number
  nodeMass?: number
}

/** Step 3 probe: a chain hanging from a pinned node. */
export function buildChain(sim: SimWorld, opts: ChainOptions): number[] {
  const { x, y, links = 14, spacing = 0.9, compliance = 1e-7, zeta = 0.9, nodeMass = 8 } = opts
  const p = sim.particles
  const ids: number[] = []

  for (let i = 0; i <= links; i++) {
    ids.push(
      p.create({ x: x + i * spacing, y, invMass: i === 0 ? 0 : 1 / nodeMass, radius: 0.14 }),
    )
  }
  for (let i = 0; i < links; i++) {
    sim.distance.create({ a: ids[i]!, b: ids[i + 1]!, rest: spacing, compliance, zeta })
  }
  return ids
}

export interface BeamOptions {
  x0: number
  y0: number
  x1: number
  y1: number
  material?: MaterialId
  /** 1 is a rigid link. Defaults from the material's segmentsPerMetre. */
  segments?: number
  /** Overrides, for probing. Normally everything comes from the material. */
  axialCompliance?: number
  flexuralRigidity?: number
  bendCompliance?: number
  zetaAxial?: number
  zetaBend?: number
  massPerMetre?: number
  radius?: number
  /**
   * Clamp the start against rotation, via a pinned ghost particle behind it.
   * Pinning one particle fixes a point, not a direction, so without this a
   * cantilever just swings bodily about its anchor.
   */
  clampStart?: boolean
  pinStart?: boolean
  pinEnd?: boolean
}

export interface Beam {
  nodes: number[]
  ghost: number
  distances: number[]
  bends: number[]
  material: MaterialId
}

/**
 * A segmented member. Bending needs three particles to measure an angle, so a
 * member that should visibly bow is a short chain; a stubby brace stays at
 * segments: 1 and behaves as a rigid link.
 *
 * Everything physical - compliance, mass, thickness, damping - is derived from
 * the material and the segment length, so a member means the same thing however
 * it is discretised.
 */
export function buildBeam(sim: SimWorld, opts: BeamOptions): Beam {
  const matId = opts.material ?? 'wood'
  const mat = MATERIALS[matId]

  const dx = opts.x1 - opts.x0
  const dy = opts.y1 - opts.y0
  const length = Math.hypot(dx, dy)
  const n = Math.max(1, Math.round(opts.segments ?? segmentsFor(mat, length)))
  const spacing = length / n
  const ux = dx / length
  const uy = dy / length

  const axialAlpha = opts.axialCompliance ?? axialCompliance(mat, spacing)
  const bendAlpha =
    opts.bendCompliance ??
    (opts.flexuralRigidity ? spacing / opts.flexuralRigidity : bendCompliance(mat, spacing))
  const mpm = opts.massPerMetre ?? massPerMetre(mat)
  const radius = opts.radius ?? mat.section * 0.5
  const zetaAxial = opts.zetaAxial ?? mat.zetaAxial
  const zetaBend = opts.zetaBend ?? mat.zetaBend
  const matIdx = materialIndex(matId)
  const nodeMass = Math.max(0.01, mpm * spacing)

  const p = sim.particles
  const nodes: number[] = []
  for (let i = 0; i <= n; i++) {
    const pinned = (i === 0 && opts.pinStart) || (i === n && opts.pinEnd)
    nodes.push(
      p.create({
        x: opts.x0 + ux * spacing * i,
        y: opts.y0 + uy * spacing * i,
        invMass: pinned ? 0 : 1 / nodeMass,
        radius,
        // Rest volume: the member's cross-section times the length this node
        // is responsible for. Never recomputed from deformed geometry.
        volume: areaOf(mat) * spacing,
      }),
    )
  }

  const distances: number[] = []
  for (let i = 0; i < n; i++) {
    distances.push(
      sim.distance.create({
        a: nodes[i]!,
        b: nodes[i + 1]!,
        rest: spacing,
        compliance: axialAlpha,
        zeta: zetaAxial,
        material: matIdx,
      }),
    )
  }

  const bends: number[] = []
  for (let i = 0; i + 2 <= n; i++) {
    bends.push(
      sim.bend.create({
        a: nodes[i]!,
        b: nodes[i + 1]!,
        c: nodes[i + 2]!,
        restAngle: 0,
        compliance: bendAlpha,
        zeta: zetaBend,
        material: matIdx,
      }),
    )
  }

  let ghost = -1
  if (opts.clampStart && n >= 1) {
    ghost = p.create({
      x: opts.x0 - ux * spacing,
      y: opts.y0 - uy * spacing,
      invMass: 0,
      radius,
    })
    p.pin(nodes[0]!)
    bends.push(
      sim.bend.create({
        a: ghost,
        b: nodes[0]!,
        c: nodes[1]!,
        restAngle: 0,
        compliance: bendAlpha,
        zeta: zetaBend,
        material: matIdx,
      }),
    )
  }

  return { nodes, ghost, distances, bends, material: matId }
}

export interface LoadTestOptions {
  x: number
  y: number
  material?: MaterialId
  length?: number
  segments?: number
  /** Point mass hung off the tip, kg. Crank it until the member fails. */
  tipMassKg?: number
}

/**
 * Step 5 probe: a cantilever with a weight on the end. Wind the load up and
 * watch the difference in how the two materials give up - wood stretches then
 * snaps with no warning, steel yields, sags, holds the sag, then goes.
 */
export function buildLoadTest(sim: SimWorld, opts: LoadTestOptions): Beam & { weight: number } {
  const { x, y, material = 'wood', length = 10, segments, tipMassKg = 0 } = opts
  const beam = buildBeam(sim, {
    x0: x,
    y0: y,
    x1: x + length,
    y1: y,
    material,
    segments,
    clampStart: true,
  })

  const tip = beam.nodes[beam.nodes.length - 1]!
  const p = sim.particles
  if (tipMassKg > 0) {
    const existing = p.massOf(tip)
    p.invMass[tip] = 1 / (existing + tipMassKg)
  }
  return { ...beam, weight: tipMassKg }
}
