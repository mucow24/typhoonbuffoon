import { describe, it, expect } from 'vitest'
import { Session } from '../../src/game/session'
import { Field } from '../../src/world/field'
import { KIND_FLUID } from '../../src/sim/particles'
import { SimWorld } from '../../src/sim/world'
import { defaultLevel } from '../../src/model/level'

/**
 * The live-edit seams: building, deleting and undoing against a RUNNING sim.
 *
 * Everything here is a regression test for a verified bug in the audit. The
 * sim holds typed-array indices; the session holds maps of those indices; and
 * every path where the two disagree - leaked welds, freed slots recycled into
 * water, snapshots that miss half the state - corrupted a live world in a way
 * no unit test of either side could see. Method for each: red-first against
 * the pre-fix build (re-verified by reverting the specific fix).
 */

function makeSession() {
  const field = new Field(40)
  const sim = new SimWorld()
  sim.terrain = field.terrain
  sim.boundsX0 = field.left
  sim.boundsX1 = field.right
  sim.fluid.spacing = 0.35
  const session = new Session(defaultLevel(field.widthM), sim, field)
  return { sim, session, field }
}

/** Every live distance constraint must reference live, non-fluid particles. */
function constraintIntegrity(sim: SimWorld): string[] {
  const problems: string[] = []
  const d = sim.distance
  const p = sim.particles
  for (let i = 0; i < d.highWater; i++) {
    if (d.slots.alive[i] !== 1) continue
    for (const end of [d.a[i]!, d.b[i]!]) {
      if (p.slots.alive[end] !== 1) {
        problems.push(`constraint ${i} references dead particle ${end}`)
      } else if (p.kind[end] === KIND_FLUID) {
        problems.push(`constraint ${i} references FLUID particle ${end}`)
      }
    }
  }
  return problems
}

describe('member lifecycle', () => {
  it('deleting a member removes its welds - water recycled into its slots is not captured', () => {
    const { sim, session } = makeSession()

    const a = session.addAnchor(-2, 5)
    const b = session.addAnchor(2, 5)
    session.addMember(a, b, 'wood')
    const membersBefore = sim.distance.count
    expect(membersBefore).toBeGreaterThan(0)

    const memberId = session.solution.members[0]!.id
    session.removeMember(memberId)

    // EVERY constraint goes with the member - segments AND welds. The leak
    // left two live rest-0 welds pointing at freed particle slots.
    expect(sim.distance.count).toBe(0)

    // The exact disaster: flood water recycles the freed slots. No live
    // constraint may touch it.
    sim.fillTo(6, -10, 10, 500)
    sim.step(1 / 60)
    expect(constraintIntegrity(sim)).toEqual([])
  })

  it('a break does not corrupt a later member built into the recycled slots', () => {
    const { sim, session } = makeSession()

    // Member A between two anchors...
    const a1 = session.addAnchor(-2, 8)
    const a2 = session.addAnchor(2, 8)
    const memberA = session.addMember(a1, a2, 'wood')!

    // ...broken by brute force: teleport one pinned anchor far away, so the
    // segments stretch past breakStrain and the sim frees their slots.
    const anchorIdx = (session as unknown as { nodeSim: Map<string, number> }).nodeSim.get(a2)!
    sim.particles.posX[anchorIdx] = 30
    sim.particles.prevX[anchorIdx] = 30
    for (let i = 0; i < 30; i++) sim.step(1 / 60)
    session.syncBreaks()
    const afterBreak = sim.distance.count
    expect(afterBreak).toBeLessThan(4) // something broke

    // Member B reuses the freed constraint slots.
    const b1 = session.addAnchor(-2, 3)
    const b2 = session.addAnchor(2, 3)
    session.addMember(b1, b2, 'wood')
    const bConstraints = sim.distance.count - afterBreak
    expect(bConstraints).toBeGreaterThan(0)

    // Deleting broken member A must not destroy any of B's constraints
    // through stale records: exactly A's remnants go, all of B survives.
    session.removeMember(memberA)
    expect(sim.distance.count).toBe(bConstraints)
    expect(constraintIntegrity(sim)).toEqual([])
  })
})

describe('play / reset', () => {
  it('reset reverts anchors and objects placed while running', () => {
    const { session } = makeSession()
    session.addAnchor(-5, 5)
    const anchorsBefore = session.doc.anchors.length
    const objectsBefore = session.doc.objects.length

    session.play()
    session.addAnchor(3, 4)
    session.addWorldObject({ x: 0, y: 8, width: 3, height: 2, density: 300 })
    expect(session.doc.anchors.length).toBe(anchorsBefore + 1)

    session.reset()
    // "Reset is exact" covers the document too - the plan's decision 7.
    expect(session.doc.anchors.length).toBe(anchorsBefore)
    expect(session.doc.objects.length).toBe(objectsBefore)
  })

  it('reset restores members deleted while running, without billing ghosts', () => {
    const { session } = makeSession()
    const a = session.addAnchor(-2, 5)
    const b = session.addAnchor(2, 5)
    session.addMember(a, b, 'wood')
    const costBefore = session.cost()

    session.play()
    session.removeMember(session.solution.members[0]!.id)
    session.reset()

    expect(session.solution.members.length).toBe(1)
    // Every billed member exists; every existing member is billed.
    expect(session.cost()).toBeCloseTo(costBefore, 6)
  })

  it('reset reverts an anchor deletion made mid-run and respawns the members on it', () => {
    const { sim, session } = makeSession()
    const a = session.addAnchor(-2, 5)
    const b = session.addAnchor(2, 5)
    session.addMember(a, b, 'wood')

    session.play()
    session.removeAnchor(a)
    expect(session.solution.members.length).toBe(0)

    session.reset()
    expect(session.doc.anchors.length).toBe(2)
    expect(session.solution.members.length).toBe(1)
    // And the member is genuinely in the sim, not just in the document -
    // the "billed but invisible" state is the one this guards against.
    expect(sim.distance.count).toBeGreaterThan(0)
  })
})

describe('build edits do not destroy the flood', () => {
  it('undo keeps the water', () => {
    const { sim, session } = makeSession()
    session.addAnchor(-5, 5)
    sim.fillTo(2, -18, 18, 2000)
    const water = sim.fluidCount
    expect(water).toBeGreaterThan(50)

    session.addAnchor(5, 5)
    session.undo()

    // Ctrl+Z reverts the BUILD. Deleting several thousand water particles and
    // re-flooding over tens of seconds is not "undo".
    expect(sim.fluidCount).toBe(water)
    expect(session.doc.anchors.length).toBe(1)
  })

  it('redo and anchor deletion keep the water too', () => {
    const { sim, session } = makeSession()
    const a = session.addAnchor(-5, 5)
    sim.fillTo(2, -18, 18, 2000)
    const water = sim.fluidCount

    session.removeAnchor(a)
    expect(sim.fluidCount).toBe(water)
    session.undo()
    expect(sim.fluidCount).toBe(water)
    session.redo()
    expect(sim.fluidCount).toBe(water)
  })

  it('reset DOES clear the water - ending the run means ending the storm', () => {
    const { sim, session } = makeSession()
    session.play()
    sim.fillTo(2, -18, 18, 2000)
    expect(sim.fluidCount).toBeGreaterThan(50)
    session.reset()
    expect(sim.fluidCount).toBe(0)
  })
})

describe('object anchors', () => {
  it('a member welded to an object does not fight it - the supported house holds still', () => {
    const { sim, session } = makeSession()

    // The starter-level shape: a house on two stilts plus X-bracing, each from
    // a terrain anchor up to an anchor bound to the house. The bracing matters
    // mechanically, not cosmetically: anchors are pin joints, and two parallel
    // pin-ended columns are a four-bar linkage that sways freely - an
    // UNBRACED version of this build genuinely collapses sideways, as it
    // should. Pre-fix, even the braced version pumped energy (capsule vs
    // weld) until the house flipped.
    const t = (session as unknown as { field: Field }).field.terrain
    const x = -8
    const g = t.heightAt(x)
    const houseId = session.addWorldObject({ x, y: g + 6, width: 6, height: 3, density: 150 })
    const g1 = session.addAnchor(x - 2, t.heightAt(x - 2))
    const g2 = session.addAnchor(x + 2, t.heightAt(x + 2))
    const h1 = session.addAnchor(x - 2, g + 4.5, houseId)
    const h2 = session.addAnchor(x + 2, g + 4.5, houseId)
    session.addMember(g1, h1, 'wood')
    session.addMember(g2, h2, 'wood')
    session.addMember(g1, h2, 'wood')
    session.addMember(g2, h1, 'wood')

    const cluster = (session as unknown as { objectSim: Map<string, { cy: number }> }).objectSim.get(houseId)!
    const startCy = cluster.cy
    for (let i = 0; i < 60 * 10; i++) sim.step(1 / 60)

    // Supported by its stilts: it may settle a little, it must not fall, fly,
    // or flip.
    expect(Math.abs(cluster.cy - startCy)).toBeLessThan(0.8)
    expect(sim.distance.count).toBeGreaterThan(0)
  })
})
