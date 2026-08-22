import { describe, it, expect } from 'vitest'
import { buildBeam } from '../../src/scenes/demos'
import { MATERIALS } from '../../src/sim/materials'
import {
  analyticCantileverTip,
  analyticMidspanDeflection,
  buildSimplySupported,
  expectMonotonic,
  expectNear,
  expectNoEnergyGain,
  expectSettles,
  flatTerrain,
  makeWorld,
  peakMemberLoad,
  run,
  settle,
} from '../harness'

/**
 * Beams against textbook formulas.
 *
 * Every previous check on the structure solver compared it to itself - stiffer
 * EI droops less, more load droops more. Those cannot tell a beam that is 10x
 * too soft from one that is right. Euler-Bernoulli deflection is an external
 * reference, and it is the one that says whether wood is brittle by a factor of
 * two or a factor of two hundred.
 */

const G = 9.81

function cantilever(opts: { length: number; loadKg: number; material?: 'wood' | 'steel'; segments?: number }) {
  const sim = makeWorld({ widthM: 60, spacing: 0.5, terrain: flatTerrain(60, -50) })
  const y = 0
  const beam = buildBeam(sim, {
    x0: -opts.length / 2,
    y0: y,
    x1: opts.length / 2,
    y1: y,
    material: opts.material ?? 'wood',
    segments: opts.segments ?? 8,
    clampStart: true,
  })
  const tip = beam.nodes[beam.nodes.length - 1]!
  if (opts.loadKg > 0) {
    sim.particles.invMass[tip] = 1 / (sim.particles.massOf(tip) + opts.loadKg)
  }
  return { sim, beam, tip, startY: sim.particles.posY[tip]! }
}

describe('cantilever deflection', () => {
  it('matches the Euler-Bernoulli tip deflection for a wooden beam', () => {
    const length = 6
    const loadKg = 400
    const { sim, tip, startY } = cantilever({ length, loadKg })
    settle(sim, 20)

    const measured = startY - sim.particles.posY[tip]!
    const analytic = analyticCantileverTip(loadKg * G, length, MATERIALS.wood.flexuralRigidity)

    // A discretised beam is softer than the continuum, so a factor of a few is
    // expected. Orders of magnitude are not.
    expect(measured).toBeGreaterThan(analytic * 0.3)
    expect(measured).toBeLessThan(analytic * 6)
  })

  it('deflects proportionally more under proportionally more load', () => {
    const deflect = (loadKg: number) => {
      const { sim, tip, startY } = cantilever({ length: 6, loadKg })
      settle(sim, 20)
      return startY - sim.particles.posY[tip]!
    }
    // Linear elasticity: doubling the load doubles the deflection, until it
    // yields or breaks. Anything wildly non-linear at modest load is a bug.
    const a = deflect(100)
    const b = deflect(200)
    expectMonotonic([a, b, deflect(400)], 'increasing', 'tip deflection vs load')
    expectNear(b / Math.max(a, 1e-9), 2, { rel: 0.6, label: 'deflection ratio for 2x load' })
  })

  it('is stiffer in steel than in wood', () => {
    const woodBeam = cantilever({ length: 6, loadKg: 300, material: 'wood' })
    settle(woodBeam.sim, 20)
    const wood = woodBeam.startY - woodBeam.sim.particles.posY[woodBeam.tip]!

    const steelBeam = cantilever({ length: 6, loadKg: 300, material: 'steel' })
    settle(steelBeam.sim, 20)
    const steel = steelBeam.startY - steelBeam.sim.particles.posY[steelBeam.tip]!

    expect(steel).toBeLessThan(wood)
  })
})

describe('load capacity', () => {
  it('a 4 m timber beam carries half a tonne without shattering', () => {
    // A 0.3 m square timber section carrying 500 kg over 4 m is an entirely
    // ordinary joist. If this snaps, the material constants are wrong, not the
    // player's design.
    const { sim, beam } = cantilever({ length: 4, loadKg: 500 })
    const before = beam.distances.length
    settle(sim, 15)

    expect(sim.distance.count).toBeGreaterThanOrEqual(before)
    expect(peakMemberLoad(sim)).toBeLessThan(MATERIALS.wood.breakStrain)
  })

  it('a 4 m timber beam does break under a load far past its rating', () => {
    // The counterpart: strength has to mean something. 60 kN of axial capacity
    // should not survive 50 tonnes.
    const { sim } = cantilever({ length: 4, loadKg: 50_000 })
    const before = sim.distance.count
    settle(sim, 15)
    expect(sim.distance.count).toBeLessThan(before)
  })

  it('steel carries what wood cannot', () => {
    const survives = (material: 'wood' | 'steel') => {
      const { sim } = cantilever({ length: 4, loadKg: 6000, material })
      const before = sim.distance.count
      settle(sim, 15)
      return sim.distance.count >= before
    }
    expect(survives('steel')).toBe(true)
  })
})

describe('simply supported beam', () => {
  it('matches the textbook mid-span deflection', () => {
    const length = 6
    const loadKg = 400
    const sim = makeWorld({ widthM: 60, spacing: 0.5, terrain: flatTerrain(60, -50) })
    const beam = buildSimplySupported(sim, { length, y: 0, loadKg, segments: 8 })
    const startY = sim.particles.posY[beam.mid]!
    settle(sim, 20)

    const measured = startY - sim.particles.posY[beam.mid]!
    const analytic = analyticMidspanDeflection(loadKg * G, length, MATERIALS.wood.flexuralRigidity)

    expect(measured).toBeGreaterThan(analytic * 0.3)
    expect(measured).toBeLessThan(analytic * 6)
  })

  it('does not develop cable tension in place of bending resistance', () => {
    // A beam pinned at both ends resists a transverse load by bending. If the
    // model instead lets it sag into a catenary, the axial members carry the
    // whole load as tension and a perfectly reasonable beam snaps.
    const sim = makeWorld({ widthM: 60, spacing: 0.5, terrain: flatTerrain(60, -50) })
    buildSimplySupported(sim, { length: 6, y: 0, loadKg: 400, segments: 8 })
    settle(sim, 20)

    expect(peakMemberLoad(sim)).toBeLessThan(MATERIALS.wood.breakStrain * 0.75)
  })
})

describe('structural stability', () => {
  it('an unloaded structure does not gain energy', () => {
    const sim = makeWorld({ widthM: 60, spacing: 0.5, terrain: flatTerrain(60, -50) })
    buildBeam(sim, {
      x0: -3,
      y0: 0,
      x1: 3,
      y1: 0,
      material: 'wood',
      segments: 8,
      clampStart: true,
    })
    const trace = run(sim, { seconds: 20, box: { x0: -30, x1: 30, y0: -60, y1: 60 } })
    expectNoEnergyGain(trace, { tolerance: 0.05, label: 'unloaded cantilever' })
  })

  it('a loaded structure comes to rest instead of oscillating forever', () => {
    const { sim } = cantilever({ length: 6, loadKg: 200 })
    const trace = run(sim, { seconds: 25, box: { x0: -30, x1: 30, y0: -60, y1: 60 } })
    expectSettles(trace, { below: 0.15, byFraction: 0.6, label: 'loaded cantilever' })
  })

  it('responds symmetrically to a symmetric load', () => {
    const sim = makeWorld({ widthM: 60, spacing: 0.5, terrain: flatTerrain(60, -50) })
    const beam = buildSimplySupported(sim, { length: 8, y: 0, loadKg: 300, segments: 8 })
    settle(sim, 20)

    const p = sim.particles
    const n = beam.nodes.length
    for (let k = 1; k < Math.floor(n / 2); k++) {
      const left = p.posY[beam.nodes[k]!]!
      const right = p.posY[beam.nodes[n - 1 - k]!]!
      expectNear(left, right, { abs: 0.02, rel: 0.05, label: `symmetry of node ${k}` })
    }
  })
})
