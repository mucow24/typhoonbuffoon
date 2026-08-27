import { describe, it, expect } from 'vitest'
import fixture from '../fixtures/steel-tower.json'
import { Field } from '../../src/world/field'
import { Session } from '../../src/game/session'
import { SimWorld } from '../../src/sim/world'
import { defaultLevel, migrateLevel, migrateSolution, claimIds, allIds } from '../../src/model/level'
import { peakStructuralLoad } from '../harness'

/**
 * A real player build, saved from the app the day it collapsed.
 *
 * This X-braced steel tower on two beach anchors is statically sound, and on
 * "play" it snapped 13-18 members inside a quarter of a second under nothing
 * but gravity, leaving debris spinning at the speed cap and hovering in
 * mid-air. Three distinct defects conspired, and this fixture is the
 * regression net over all of them:
 *
 *  - the XPBD mass-ratio instability: a 40 kg default joint node welded
 *    between ~400 kg steel segment ends self-excited at ~40% per frame -
 *    in vacuum, with gravity off (joints now weigh what their members do);
 *  - members drawn through dune bumps spawned with buried nodes, and the
 *    terrain heave against near-rigid axial constraints pumped energy every
 *    substep (members now conform to terrain at build, unstrained);
 *  - fragments driven past yield crept their rest state without bound into
 *    mutually contradictory constraints that buzzed forever (plastic set is
 *    now budgeted by ductility - steel tears instead).
 *
 * Method: red-first - this exact test fails on the pre-fix build with
 * broken > 0 on the second assertion.
 */

describe('the reported steel tower', () => {
  it('stands, settles, and stays whole for 30 seconds', () => {
    const doc = migrateLevel(fixture.level)
    const solution = migrateSolution(fixture.solution)
    claimIds(allIds(doc, solution))

    const field = new Field(doc.widthM)
    const sim = new SimWorld()
    sim.terrain = field.terrain
    sim.boundsX0 = field.left
    sim.boundsX1 = field.right
    sim.fluid.spacing = 0.25
    const session = new Session(doc, sim, field)
    session.solution = solution
    session.rebuild()

    const constraintsBefore = sim.distance.count
    expect(constraintsBefore).toBeGreaterThan(30)

    let broken = 0
    let peakLoad = 0
    for (let f = 0; f < 60 * 30; f++) {
      sim.step(1 / 60)
      session.syncBreaks()
      broken += sim.breakEvents.length
      sim.breakEvents.length = 0
      if (f % 10 === 0) peakLoad = Math.max(peakLoad, peakStructuralLoad(sim))
    }

    // Nothing breaks - not at play, not ever, with no load applied.
    expect(broken).toBe(0)
    expect(sim.distance.count).toBe(constraintsBefore)

    // The settling transient is modest, not a near-miss: a structure that
    // "survives" at 99% of failure fails the next time a butterfly lands.
    expect(peakLoad).toBeLessThan(0.6)

    // And it comes to REST - no perpetual buzzing, spinning, or hovering
    // debris. Every node, not just typical ones.
    const p = sim.particles
    let maxSpeed = 0
    for (let i = 0; i < p.highWater; i++) {
      if (p.slots.alive[i] !== 1 || p.invMass[i] === 0) continue
      maxSpeed = Math.max(maxSpeed, Math.hypot(p.velX[i]!, p.velY[i]!))
    }
    expect(maxSpeed).toBeLessThan(0.05)
    expect(peakStructuralLoad(sim)).toBeLessThan(0.1)
  }, 120000)

  it('a steel-braced house on object anchors settles dead calm', () => {
    // The object-anchored variant of the same collapse: members bolted to a
    // house cluster. Two distinct regressions live here - the joint-mass fix
    // must not touch cluster particles (mutating invMass under the shape
    // matcher desynchronises its cached masses: 11 members snapped), and an
    // object anchor must be a frame MOUNT, not a borrowed cluster particle
    // (one particle under two rigid masters rang at 2-5 m/s indefinitely).
    // Method: red-first against both pre-fix states.
    const field = new Field(120)
    const sim = new SimWorld()
    sim.terrain = field.terrain
    sim.boundsX0 = field.left
    sim.boundsX1 = field.right
    sim.fluid.spacing = 0.25
    const session = new Session(defaultLevel(field.widthM), sim, field)
    const t = field.terrain
    const x = -8
    const g = t.heightAt(x)
    const houseId = session.addWorldObject({ x, y: g + 5.5, width: 8, height: 4.5, density: 150 })
    const g1 = session.addAnchor(x - 2, t.heightAt(x - 2))
    const g2 = session.addAnchor(x + 2, t.heightAt(x + 2))
    const h1 = session.addAnchor(x - 2, g + 3.4, houseId)
    const h2 = session.addAnchor(x + 2, g + 3.4, houseId)
    session.addMember(g1, h1, 'steel')
    session.addMember(g2, h2, 'steel')
    session.addMember(g1, h2, 'steel')
    session.addMember(g2, h1, 'steel')

    let broken = 0
    for (let f = 0; f < 60 * 15; f++) {
      sim.step(1 / 60)
      session.syncBreaks()
      broken += sim.breakEvents.length
      sim.breakEvents.length = 0
    }

    expect(broken).toBe(0)
    const p = sim.particles
    let maxSpeed = 0
    for (let i = 0; i < p.highWater; i++) {
      if (p.slots.alive[i] !== 1 || p.invMass[i] === 0) continue
      maxSpeed = Math.max(maxSpeed, Math.hypot(p.velX[i]!, p.velY[i]!))
    }
    // DEAD calm - the pre-mount version "survived" while ringing at 2-5 m/s.
    expect(maxSpeed).toBeLessThan(0.05)
  }, 120000)
})
