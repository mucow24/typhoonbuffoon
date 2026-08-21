import type { SimWorld } from '../sim/world'

export interface ChainOptions {
  x: number
  y: number
  links?: number
  spacing?: number
  compliance?: number
  zeta?: number
  /** Mass per node, kg. Heavier nodes give slower, lower-frequency motion. */
  nodeMass?: number
}

/**
 * Step 3 probe: a chain hanging from a pinned node. Makes compliance and
 * damping visible and tunable before anything depends on them.
 */
export function buildChain(sim: SimWorld, opts: ChainOptions): number[] {
  const { x, y, links = 14, spacing = 0.9, compliance = 1e-7, zeta = 0.9, nodeMass = 8 } = opts

  const p = sim.particles
  const ids: number[] = []

  for (let i = 0; i <= links; i++) {
    ids.push(
      p.create({
        x: x + i * spacing,
        y,
        invMass: i === 0 ? 0 : 1 / nodeMass,
        radius: 0.14,
      }),
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
  /** 1 is a rigid link. Trunks and long spans want 4-8. */
  segments?: number
  axialCompliance?: number
  /**
   * Flexural rigidity E*I in N*m^2. Preferred over a raw bendCompliance: the
   * joint compliance a segmented beam needs is segmentLength / EI, so quoting
   * EI keeps the material meaning the same when the segment count changes.
   * With a fixed compliance instead, droop varied 2.1x across 3-12 segments;
   * derived from EI it holds to about +/-10% from 4 segments up.
   *
   * For scale: a 0.3 m square timber post is around 7e6 N*m^2.
   */
  flexuralRigidity?: number
  bendCompliance?: number
  zetaAxial?: number
  zetaBend?: number
  /** kg per metre of member. */
  massPerMetre?: number
  radius?: number
  /**
   * Clamp the start against rotation, via a pinned ghost particle behind it.
   * Without this a cantilever just swings bodily about its anchor: pinning one
   * particle fixes a point, not a direction.
   */
  clampStart?: boolean
  pinStart?: boolean
  pinEnd?: boolean
}

export interface Beam {
  /** Real nodes from start to end. */
  nodes: number[]
  /** Pinned ghost behind the start, or -1. */
  ghost: number
  distances: number[]
  bends: number[]
}

/**
 * A segmented member. Bending needs three particles to measure an angle, so a
 * member that should visibly bow is a short chain; a stubby brace stays at
 * segments: 1 and behaves as a rigid link.
 */
export function buildBeam(sim: SimWorld, opts: BeamOptions): Beam {
  const {
    x0,
    y0,
    x1,
    y1,
    segments = 6,
    axialCompliance = 1e-7,
    flexuralRigidity,
    bendCompliance,
    zetaAxial = 0.9,
    zetaBend = 0.9,
    massPerMetre = 45,
    radius = 0.2,
    clampStart = false,
    pinStart = false,
    pinEnd = false,
  } = opts

  const p = sim.particles
  const dx = x1 - x0
  const dy = y1 - y0
  const length = Math.hypot(dx, dy)
  const n = Math.max(1, Math.round(segments))
  const spacing = length / n
  const ux = dx / length
  const uy = dy / length
  const nodeMass = Math.max(0.01, massPerMetre * spacing)
  const bendAlpha =
    bendCompliance ?? (flexuralRigidity ? spacing / flexuralRigidity : spacing / 4e6)

  const nodes: number[] = []
  for (let i = 0; i <= n; i++) {
    const pinned = (i === 0 && pinStart) || (i === n && pinEnd)
    nodes.push(
      p.create({
        x: x0 + ux * spacing * i,
        y: y0 + uy * spacing * i,
        invMass: pinned ? 0 : 1 / nodeMass,
        radius,
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
        compliance: axialCompliance,
        zeta: zetaAxial,
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
      }),
    )
  }

  let ghost = -1
  if (clampStart && n >= 1) {
    ghost = p.create({
      x: x0 - ux * spacing,
      y: y0 - uy * spacing,
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
      }),
    )
  }

  return { nodes, ghost, distances, bends }
}
