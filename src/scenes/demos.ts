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
 * Step 3 probe: a chain hanging from a pinned node. The point of it is to make
 * compliance and damping visible and tunable before anything depends on them.
 */
export function buildChain(sim: SimWorld, opts: ChainOptions): number[] {
  const {
    x,
    y,
    links = 14,
    spacing = 0.9,
    compliance = 1e-7,
    zeta = 0.9,
    nodeMass = 8,
  } = opts

  const p = sim.particles
  const ids: number[] = []

  for (let i = 0; i <= links; i++) {
    const id = p.create({
      x: x + i * spacing,
      y,
      invMass: i === 0 ? 0 : 1 / nodeMass,
      radius: 0.14,
    })
    ids.push(id)
  }

  for (let i = 0; i < links; i++) {
    sim.distance.create({
      a: ids[i]!,
      b: ids[i + 1]!,
      rest: spacing,
      compliance,
      zeta,
    })
  }

  return ids
}
