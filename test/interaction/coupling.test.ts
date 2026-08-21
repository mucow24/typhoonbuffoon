import { describe, it, expect } from 'vitest'
import { Conditions } from '../../src/game/conditions'
import { Field } from '../../src/world/field'
import { KIND_FLUID } from '../../src/sim/particles'
import {
  archimedesSubmergedFraction,
  basinTerrain,
  buildWall,
  expectFinite,
  expectMonotonic,
  expectNear,
  expectNoEnergyGain,
  expectNoEscapes,
  expectSpeedBelow,
  fillWater,
  makeWorld,
  peakMemberLoad,
  run,
  settle,
  surfaceProfile,
  waterBeyond,
} from '../harness'

/**
 * Where the three systems meet.
 *
 * Water, wind and structures were each checked alone and all three were
 * reported working. The game only ever runs them together, and that is the
 * configuration that was never tested once. These are the tests that would have
 * caught it.
 */

describe('buoyancy', () => {
  it('floats a light box at the depth Archimedes predicts', () => {
    const sim = makeWorld({ widthM: 30, spacing: 0.35, terrain: basinTerrain(30, -6, 12) })
    fillWater(sim, { x0: -12, x1: 12, yTop: 0 })
    settle(sim, 12)

    const surfaceBefore = surfaceProfile(sim, { x0: -8, x1: 8, columnWidth: 1 }).mean
    const height = 1.6
    const density = 500
    const box = sim.addObject({ cx: 0, cy: surfaceBefore + 0.4, width: 3, height, density })
    settle(sim, 25)

    const surface = surfaceProfile(sim, { x0: -10, x1: 10, columnWidth: 1 }).mean
    const submerged = archimedesSubmergedFraction(density)
    // Centre of a floating box sits (0.5 - submergedFraction) * height above the
    // waterline. At half the density of water that is exactly the waterline.
    const predictedCentre = surface + (0.5 - submerged) * height

    expectNear(box.cy, predictedCentre, {
      abs: height * 0.4,
      rel: 0.5,
      label: 'floating box centre vs Archimedes',
    })
  })

  it('sinks a dense box to the floor and keeps it there', () => {
    const floor = -6
    const sim = makeWorld({ widthM: 30, spacing: 0.35, terrain: basinTerrain(30, floor, 12) })
    fillWater(sim, { x0: -12, x1: 12, yTop: 0 })
    settle(sim, 12)

    const box = sim.addObject({ cx: 0, cy: 1, width: 2.4, height: 1.4, density: 7850 })
    settle(sim, 25)

    expect(box.cy).toBeLessThan(floor + 3)
    expect(box.cy).toBeGreaterThan(floor - 1)
  })

  it('floats higher the lighter the object', () => {
    const heightOf = (density: number) => {
      const sim = makeWorld({ widthM: 30, spacing: 0.4, terrain: basinTerrain(30, -6, 12) })
      fillWater(sim, { x0: -12, x1: 12, yTop: 0 })
      settle(sim, 10)
      const box = sim.addObject({ cx: 0, cy: 1, width: 3, height: 1.6, density })
      settle(sim, 20)
      return box.cy
    }
    expectMonotonic([heightOf(900), heightOf(500), heightOf(200)], 'increasing', 'float height vs density')
  })

  it('lifts a floating object as the water rises', () => {
    const sim = makeWorld({ widthM: 30, spacing: 0.4, terrain: basinTerrain(30, -6, 20) })
    fillWater(sim, { x0: -12, x1: 12, yTop: -3 })
    settle(sim, 10)
    const box = sim.addObject({ cx: 0, cy: 0, width: 3, height: 1.6, density: 400 })
    settle(sim, 20)
    const low = box.cy

    fillWater(sim, { x0: -12, x1: 12, yBottom: -3, yTop: 0 })
    settle(sim, 25)

    // Rising flood water has to carry a buoyant object up with it.
    expect(box.cy).toBeGreaterThan(low + 1)
  })

  it('does not let a floating object gain energy', () => {
    const sim = makeWorld({ widthM: 30, spacing: 0.4, terrain: basinTerrain(30, -6, 12) })
    fillWater(sim, { x0: -12, x1: 12, yTop: 0 })
    settle(sim, 10)
    sim.addObject({ cx: 0, cy: 0.5, width: 3, height: 1.6, density: 500 })

    const trace = run(sim, { seconds: 25, box: { x0: -20, x1: 20, y0: -20, y1: 60 } })
    expectFinite(trace)
    expectNoEscapes(trace, 'water or object particles')
    expectSpeedBelow(trace, 12, 'floating box scene')
    expectNoEnergyGain(trace, { tolerance: 0.08, label: 'floating box' })
  })
})

describe('containment', () => {
  it('a wall keeps water on one side', () => {
    const sim = makeWorld({ widthM: 40, spacing: 0.35, terrain: basinTerrain(40, 0, 15) })
    buildWall(sim, { x: 0, yBottom: 0, yTop: 6, material: 'steel', rigid: true })
    fillWater(sim, { x0: -16, x1: -1.5, yTop: 4 })

    const trace = run(sim, { seconds: 20, box: { x0: -25, x1: 25, y0: -10, y1: 60 } })
    expectFinite(trace)

    // The whole flood-wall archetype rests on this. A handful of particles
    // squeezing past is tolerable; a stream is not.
    const leaked = waterBeyond(sim, 1.5, 'right')
    const total = sim.particles.countOfKind(KIND_FLUID)
    expect(leaked / Math.max(total, 1)).toBeLessThan(0.05)
  })

  it('loads the wall it is holding back', () => {
    const loadFor = (depth: number) => {
      const sim = makeWorld({ widthM: 40, spacing: 0.4, terrain: basinTerrain(40, 0, 15) })
      buildWall(sim, { x: 0, yBottom: 0, yTop: 8, material: 'wood' })
      fillWater(sim, { x0: -16, x1: -1.5, yTop: depth })
      settle(sim, 12)
      return peakMemberLoad(sim)
    }
    // Deeper water pushes harder. If the wall feels nothing, the water is
    // passing through it or the coupling is one-way in the wrong direction.
    expectMonotonic([loadFor(1), loadFor(3), loadFor(6)], 'increasing', 'wall load vs water depth')
  })
})

describe('wind on structures', () => {
  const deflectionAt = (kph: number) => {
    const sim = makeWorld({ widthM: 40, spacing: 0.5, terrain: basinTerrain(40, 0, 2) })
    const field = new Field(40)
    const conditions = new Conditions(sim, field)
    conditions.windKph = kph
    const mast = buildWall(sim, { x: 0, yBottom: 0, yTop: 10, material: 'wood', segments: 6 })
    const tip = mast.nodes[mast.nodes.length - 1]!
    const startX = sim.particles.posX[tip]!

    const trace = run(sim, {
      seconds: 20,
      box: { x0: -30, x1: 30, y0: -10, y1: 60 },
      each: (_t) => conditions.update(1 / 60),
    })
    return { deflection: sim.particles.posX[tip]! - startX, trace }
  }

  it('does not move a structure in still air', () => {
    const { deflection } = deflectionAt(0)
    expect(Math.abs(deflection)).toBeLessThan(0.05)
  })

  it('bends a mast further the harder it blows', () => {
    const a = Math.abs(deflectionAt(60).deflection)
    const b = Math.abs(deflectionAt(120).deflection)
    const c = Math.abs(deflectionAt(200).deflection)
    expectMonotonic([a, b, c], 'increasing', 'mast deflection vs wind speed')

    // Drag goes as v^2, so doubling the wind should roughly quadruple a small
    // deflection. Being far off means the force law is wrong.
    expectNear(b / Math.max(a, 1e-6), 4, { rel: 0.75, label: 'deflection ratio for 2x wind' })
  })

  it('bends the mast downwind, not upwind', () => {
    // The wind field blows in -x by default.
    const { deflection } = deflectionAt(150)
    expect(deflection).toBeLessThan(0)
  })

  it('sways smoothly rather than buzzing', () => {
    const sim = makeWorld({ widthM: 40, spacing: 0.5, terrain: basinTerrain(40, 0, 2) })
    const conditions = new Conditions(sim, new Field(40))
    conditions.windKph = 180
    const mast = buildWall(sim, { x: 0, yBottom: 0, yTop: 10, material: 'wood', segments: 6 })
    const tip = mast.nodes[mast.nodes.length - 1]!

    const xs: number[] = []
    run(sim, {
      seconds: 20,
      box: { x0: -30, x1: 30, y0: -10, y1: 60 },
      each: (t) => {
        conditions.update(1 / 60)
        if (t > 5) xs.push(sim.particles.posX[tip]!)
      },
    })

    const span = Math.max(...xs) - Math.min(...xs)
    let jitter = 0
    for (let i = 1; i < xs.length; i++) jitter += Math.abs(xs[i]! - xs[i - 1]!)
    const jitterPerFrame = jitter / xs.length

    // Sway is low-frequency: the path length per frame should be a small
    // fraction of the total swing. High-frequency buzz inverts that.
    expect(jitterPerFrame).toBeLessThan(Math.max(span, 0.01) * 0.2)
  })
})

describe('wind, water and structure together', () => {
  it('stays bounded with all three running', () => {
    const sim = makeWorld({ widthM: 40, spacing: 0.4, terrain: basinTerrain(40, -4, 15) })
    const conditions = new Conditions(sim, new Field(40))
    conditions.windKph = 160

    buildWall(sim, { x: 0, yBottom: -4, yTop: 8, material: 'wood', segments: 8 })
    fillWater(sim, { x0: -16, x1: -1.5, yTop: 1 })
    sim.addObject({ cx: -8, cy: 2, width: 2, height: 1.2, density: 400 })

    const trace = run(sim, {
      seconds: 25,
      box: { x0: -30, x1: 30, y0: -20, y1: 80 },
      each: () => conditions.update(1 / 60),
    })

    expectFinite(trace)
    expectNoEscapes(trace, 'particles in the combined scene')
    // Wind does work on the system, so energy may rise - but not without bound.
    expectSpeedBelow(trace, 25, 'combined wind/water/structure scene')
  })

  it('shields submerged members from the wind', () => {
    const tipDrift = (waterTop: number) => {
      const sim = makeWorld({ widthM: 40, spacing: 0.4, terrain: basinTerrain(40, 0, 15) })
      const conditions = new Conditions(sim, new Field(40))
      conditions.windKph = 200
      const mast = buildWall(sim, { x: 0, yBottom: 0, yTop: 10, material: 'wood', segments: 6 })
      const tip = mast.nodes[mast.nodes.length - 1]!
      const startX = sim.particles.posX[tip]!
      if (waterTop > 0) fillWater(sim, { x0: -16, x1: 16, yTop: waterTop })
      run(sim, {
        seconds: 15,
        box: { x0: -30, x1: 30, y0: -10, y1: 60 },
        each: () => conditions.update(1 / 60),
      })
      return Math.abs(sim.particles.posX[tip]! - startX)
    }

    // Most of the mast underwater means most of it is out of the wind, and the
    // water damps what is left.
    expect(tipDrift(8)).toBeLessThan(tipDrift(0))
  })
})
