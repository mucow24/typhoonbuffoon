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
  /**
   * Baseline-subtracted: the beam sags under its own 270 kg before any load
   * is hung on it, and Euler-Bernoulli's point-load formula knows nothing of
   * self-weight. Measuring the INCREMENT isolates the response the formula
   * predicts, which is what let the acceptance band tighten from the old
   * 0.3-6x (a 20x-wide window that certified almost anything) to 2-6x.
   */
  const incrementalDeflection = (loadKg: number, length = 6) => {
    const { sim, tip } = cantilever({ length, loadKg: 0 })
    settle(sim, 20)
    const baseline = sim.particles.posY[tip]!
    sim.particles.invMass[tip] = 1 / (sim.particles.massOf(tip) + loadKg)
    settle(sim, 20)
    return baseline - sim.particles.posY[tip]!
  }

  it('matches the Euler-Bernoulli tip deflection for a wooden beam', () => {
    const length = 6
    const loadKg = 400
    const measured = incrementalDeflection(loadKg, length)
    const analytic = analyticCantileverTip(loadKg * G, length, MATERIALS.wood.flexuralRigidity)

    // A discretised, one-iteration XPBD beam is softer than the continuum -
    // measured factor ~3.4x. The band is calibrated around that with room for
    // retuning, and verified red against EI halved and EI x10 (mutation).
    expect(measured).toBeGreaterThan(analytic * 2)
    expect(measured).toBeLessThan(analytic * 6)
  })

  it('deflects proportionally more under proportionally more load', () => {
    // Linear elasticity: doubling the load doubles the incremental
    // deflection, until it yields or breaks.
    const a = incrementalDeflection(100)
    const b = incrementalDeflection(200)
    const c = incrementalDeflection(400)
    expectMonotonic([a, b, c], 'increasing', 'tip deflection vs load')
    expectNear(b / Math.max(a, 1e-9), 2, { rel: 0.3, label: 'deflection ratio for 2x load' })
    expectNear(c / Math.max(b, 1e-9), 2, { rel: 0.3, label: 'deflection ratio for 4x load' })
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
    // Baseline-subtracted like the cantilever: settle unloaded, then hang the
    // load, so self-weight sag does not pollute the point-load comparison.
    const length = 6
    const loadKg = 400
    const sim = makeWorld({ widthM: 60, spacing: 0.5, terrain: flatTerrain(60, -50) })
    const beam = buildSimplySupported(sim, { length, y: 0, segments: 8 })
    settle(sim, 20)
    const baseline = sim.particles.posY[beam.mid]!
    sim.particles.invMass[beam.mid] = 1 / (sim.particles.massOf(beam.mid) + loadKg)
    settle(sim, 20)

    const measured = baseline - sim.particles.posY[beam.mid]!
    const analytic = analyticMidspanDeflection(loadKg * G, length, MATERIALS.wood.flexuralRigidity)

    expect(measured).toBeGreaterThan(analytic * 2)
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

describe('bending failure', () => {
  /**
   * Hand-built joint: outer particles pinned, middle one massive enough that
   * the solve barely moves it, so the recorded angle IS the imposed angle and
   * the break logic is tested at a known input. Red-first: no bend could fail
   * at all before this - a member could fold double and never snap.
   */
  function foldedJoint(sim: ReturnType<typeof makeWorld>, angle: number, material: 'wood' | 'steel') {
    const matIdx = material === 'wood' ? 0 : 1
    // Gravity off: the fixture holds a joint at a KNOWN angle, and the middle
    // particle would otherwise free-fall (gravitational acceleration does not
    // care how heavy it is).
    sim.gravity = 0
    const p = sim.particles
    const a = p.create({ x: 0, y: 5, invMass: 0 })
    const b = p.create({ x: 2, y: 5, invMass: 1e-9 })
    const c = p.create({ x: 2 + 2 * Math.cos(angle), y: 5 + 2 * Math.sin(angle), invMass: 0 })
    sim.distance.create({ a, b, rest: 2, compliance: 1e-7, zeta: 0.9, material: matIdx })
    sim.distance.create({ a: b, b: c, rest: 2, compliance: 1e-7, zeta: 0.9, material: matIdx })
    sim.bend.create({ a, b, c, restAngle: 0, compliance: 1e-6, zeta: 0.9, material: matIdx })
    return { a, b, c }
  }

  it('a joint folded past its break angle snaps the member there', () => {
    const sim = makeWorld({ widthM: 30, spacing: 0.5, terrain: flatTerrain(30, -20) })
    foldedJoint(sim, 1.2, 'wood') // far past wood's 0.28 rad
    expect(sim.bend.count).toBe(1)
    expect(sim.distance.count).toBe(2)

    sim.step(1 / 60)

    // The bend breaks AND severs one adjacent segment - a fracture, not a
    // freed hinge - and reports it.
    expect(sim.bend.count).toBe(0)
    expect(sim.distance.count).toBe(1)
    expect(sim.breakEvents.length).toBeGreaterThan(0)
  })

  it('a joint within its break angle holds', () => {
    const sim = makeWorld({ widthM: 30, spacing: 0.5, terrain: flatTerrain(30, -20) })
    foldedJoint(sim, 0.15, 'wood') // within wood's 0.28 rad
    for (let i = 0; i < 60; i++) sim.step(1 / 60)
    expect(sim.bend.count).toBe(1)
    expect(sim.distance.count).toBe(2)
  })

  it('steel takes a permanent set past its yield angle; wood does not', () => {
    const steelSim = makeWorld({ widthM: 30, spacing: 0.5, terrain: flatTerrain(30, -20) })
    foldedJoint(steelSim, 0.3, 'steel') // past steel's 0.12 rad yield, under 0.6 break
    for (let i = 0; i < 60 * 3; i++) steelSim.step(1 / 60)
    expect(steelSim.bend.count).toBe(1) // held
    expect(Math.abs(steelSim.bend.restAngle[0]!)).toBeGreaterThan(0.05) // stayed bent

    const woodSim = makeWorld({ widthM: 30, spacing: 0.5, terrain: flatTerrain(30, -20) })
    foldedJoint(woodSim, 0.2, 'wood') // wood never yields, whatever the angle
    for (let i = 0; i < 60 * 3; i++) woodSim.step(1 / 60)
    expect(woodSim.bend.count).toBe(1)
    expect(woodSim.bend.restAngle[0]!).toBe(0)
  })
})

describe('wood strength', () => {
  // Wood at the audit's constants had 60 kN of axial capacity - one strut
  // could barely hold one light house, and anything working near half its
  // strength dissolved through damage in seconds. "Tissue paper."
  it('a 4 m wood column carries five tonnes indefinitely', () => {
    // 5 t is 49 kN: a serious load a real 0.3 m timber post shrugs off.
    // Method: red under the pre-fix constants - 49 kN was 82% of capacity,
    // over the damage onset, and the column dissolved and snapped inside 15 s.
    const sim = makeWorld({ widthM: 30, spacing: 0.5, terrain: flatTerrain(30, 0) })
    const column = buildBeam(sim, {
      x0: 0,
      y0: 0,
      x1: 0,
      y1: 4,
      material: 'wood',
      segments: 3,
      pinStart: true,
      clampStart: false,
    })
    const top = column.nodes[column.nodes.length - 1]!
    sim.particles.invMass[top] = 1 / (sim.particles.massOf(top) + 5000)
    const before = sim.distance.count

    settle(sim, 15)

    expect(sim.distance.count).toBe(before)
    // And with margin: holding is not "barely not breaking".
    expect(peakMemberLoad(sim) / MATERIALS.wood.breakStrain).toBeLessThan(0.6)
  })

  it('the same column snaps under fifty tonnes', () => {
    // 50 t is 490 kN against 420 kN of capacity. Strength must still mean
    // something - mutation check: x10 axialStrengthN makes this go red.
    const sim = makeWorld({ widthM: 30, spacing: 0.5, terrain: flatTerrain(30, 0) })
    const column = buildBeam(sim, {
      x0: 0,
      y0: 0,
      x1: 0,
      y1: 4,
      material: 'wood',
      segments: 3,
      pinStart: true,
      clampStart: false,
    })
    const top = column.nodes[column.nodes.length - 1]!
    sim.particles.invMass[top] = 1 / (sim.particles.massOf(top) + 50000)
    const before = sim.distance.count
    settle(sim, 15)
    expect(sim.distance.count).toBeLessThan(before)
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
