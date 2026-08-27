import { describe, it, expect } from 'vitest'
import { flatTerrain, makeWorld } from '../harness'

/**
 * Joinery (welds, anchor-mount links) must be invisible to every
 * environmental force pass. It carries a material index (defaulting to
 * wood) purely for bookkeeping, and before the guards each weld caught
 * `frontal = mat.section` of wood-section WIND (~1.1 kN at 250 kph, two per
 * member, concentrated at the joints, applied to steel structures as wood),
 * and mount links (rest > 0) additionally took wood-material water drag and
 * hydrostatic wall load inside the object they join.
 *
 * Each rig contains a pinned node, a free node, and ONE unbreakable
 * constraint, ARRANGED so the phantom force would act PERPENDICULAR to the
 * constraint - a phantom along the constraint is silently swallowed by the
 * constraint damper, which is exactly how a hidden static load hides from a
 * velocity assertion. Nothing else in a rig can push its free node
 * sideways, so the lateral state after stepping is exactly the phantom.
 *
 * Method: mutation-verified red per pass by deleting its `unbreakable`
 * guard in src/sim/world.ts - wind: lateral drift appears from still air;
 * hydrostatic: the hanging node swings out; drag: a 5 m/s node loses over
 * 1 m/s in one frame.
 */

function joineryRig(opts: { rest: number; ay: number; by: number; gravity?: number }) {
  const sim = makeWorld({ widthM: 60, spacing: 0.25, terrain: flatTerrain(60, -40) })
  if (opts.gravity !== undefined) sim.gravity = opts.gravity
  const p = sim.particles
  const a = p.create({ x: 0, y: opts.ay, invMass: 0, radius: 0.16 })
  const b = p.create({ x: 0, y: opts.by, invMass: 1 / 40, radius: 0.16 })
  sim.distance.create({
    a,
    b,
    rest: opts.rest,
    compliance: 1e-9,
    zeta: 0.95,
    unbreakable: true,
  })
  return { sim, b }
}

describe('joinery is invisible to environmental forces', () => {
  it('a weld catches no wind', () => {
    // A weld stretched open by a millimetre (any loaded weld is), gap
    // vertical so the horizontal wind phantom is perpendicular to it.
    const { sim, b } = joineryRig({ rest: 0, ay: 0, by: 0.001, gravity: 0 })
    sim.wind.baseSpeed = 70 // ~250 kph
    for (let f = 0; f < 10; f++) sim.step(1 / 60)
    // No lateral force exists in this rig: exactly zero, not merely small.
    expect(Math.abs(sim.particles.posX[b]!)).toBeLessThan(1e-9)
    expect(Math.abs(sim.particles.velX[b]!)).toBeLessThan(1e-9)
  })

  it('a mount link takes no hydrostatic wall load', () => {
    // The node HANGS below the pin (hydrostatics needs gravity on - a zero-g
    // rig reads zero pressure and proves nothing). Deep water to the left,
    // dry to the right: the exact reading that loads a real wall, made exact
    // by overriding the surface probe instead of pouring particles that
    // would also touch the rig through contacts.
    const { sim, b } = joineryRig({ rest: 1, ay: 1, by: 0 })
    sim.water.surfaceAt = (x: number) => (x < 0 ? 10 : -Infinity)
    for (let f = 0; f < 30; f++) sim.step(1 / 60)
    // Gravity and the link are both vertical; only a phantom can move it
    // sideways.
    expect(Math.abs(sim.particles.posX[b]!)).toBeLessThan(1e-9)
    expect(Math.abs(sim.particles.velX[b]!)).toBeLessThan(1e-9)
  })

  it('a mount link feels no water drag', () => {
    // Fully submerged everywhere: equal surfaces mean zero hydrostatic net,
    // isolating the drag pass; motion perpendicular to the link.
    const { sim, b } = joineryRig({ rest: 1, ay: 0, by: 1, gravity: 0 })
    sim.water.surfaceAt = () => 10
    sim.particles.velX[b] = 5
    sim.step(1 / 60)
    // The node keeps its speed apart from the structural linear damping
    // (0.35/s ~ 0.6% per frame) and the pendulum constraint redirecting it.
    // The un-guarded drag pass stripped over 1 m/s in this single frame.
    expect(sim.particles.velX[b]!).toBeGreaterThan(4.8)
  })
})
