import { describe, it, expect } from 'vitest'
import { Conditions } from '../../src/game/conditions'
import { Field } from '../../src/world/field'
import { KIND_FLUID } from '../../src/sim/particles'
import { MATERIALS } from '../../src/sim/materials'
import { SimWorld } from '../../src/sim/world'
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
  flatTerrain,
  makeWorld,
  peakStructuralLoad,
  run,
  settle,
  speedPercentile,
  surfaceProfile,
  waterBeyond,
} from '../harness'
import { buildBeam } from '../../src/scenes/demos'

/**
 * Where the three systems meet.
 *
 * Water, wind and structures were each checked alone and all three were
 * reported working. The game only ever runs them together, and that is the
 * configuration that was never tested once. These are the tests that would have
 * caught it.
 *
 * Calibration notes, so the bounds stay honest rather than cosy:
 *  - The pressure-coupled float rides HIGH by ~0.55 m for a 1.6 m box (a known
 *    systematic bias, constant across resolutions - see fluid.ts
 *    hullPressureFactor). Draft bounds document it instead of hiding it.
 *  - Every threshold here was verified to go red against the defect it names:
 *    draft/consistency against the pre-fix over-relaxed coupling, wall load
 *    and deflection against applyHydrostaticLoad disabled, rest-on-platform
 *    and stacking against the one-way contact pass. Method noted per test.
 */

const surfaceMean = (sim: SimWorld, x0: number, x1: number) =>
  surfaceProfile(sim, { x0, x1, columnWidth: 1 }).mean

describe('buoyancy', () => {
  it('floats a light box near the depth Archimedes predicts, and stays there', () => {
    const sim = makeWorld({ widthM: 30, spacing: 0.35, terrain: basinTerrain(30, -6, 12) })
    fillWater(sim, { x0: -12, x1: 12, yTop: 0 })
    settle(sim, 12)

    const surfaceBefore = surfaceMean(sim, -8, 8)
    const height = 1.6
    const density = 500
    const box = sim.addObject({ cx: 0, cy: surfaceBefore + 0.4, width: 3, height, density })
    settle(sim, 20)
    const settledCy = box.cy
    settle(sim, 5)

    // Settled means settled: the last five seconds move the box centimetres,
    // not tenths. (The pre-fix coupling bobbed at cyStd ~0.35 m forever.)
    expect(Math.abs(box.cy - settledCy)).toBeLessThan(0.08)

    // Exclude the box's own columns: under the box the 95th-percentile probe
    // reads the moat under the hull, which drags the mean down ~0.1 m.
    const surface =
      (surfaceMean(sim, -10, -2.5) + surfaceMean(sim, 2.5, 10)) / 2
    const submerged = archimedesSubmergedFraction(density)
    const predictedCentre = surface + (0.5 - submerged) * height

    // abs 0.65 covers the known +0.55 high-riding bias with a little noise
    // room; it still fails a box that sinks, launches, or rides a full extra
    // half-height high. Verified red against the pre-fix solver (box launched
    // at 0.25 m spacing) and against buoyancy doubled.
    expectNear(box.cy, predictedCentre, {
      abs: 0.65,
      rel: 0,
      label: 'floating box centre vs Archimedes',
    })
  })

  it('floats at the same draft whatever the resolution slider says', () => {
    // The player owns the resolution slider. Pre-fix, the same box floated
    // quiet at 0.40 m spacing and was LAUNCHED at the shipping 0.25 m; its
    // equilibrium draft moved ~0.9 m between slider stops. Method: red-first
    // against that build.
    const draftAt = (spacing: number) => {
      const sim = makeWorld({ widthM: 30, spacing, terrain: basinTerrain(30, -6, 12) })
      fillWater(sim, { x0: -12, x1: 12, yTop: 0 })
      settle(sim, 12)
      const surfaceBefore = surfaceMean(sim, -8, 8)
      const box = sim.addObject({ cx: 0, cy: surfaceBefore + 0.4, width: 3, height: 1.6, density: 500 })
      settle(sim, 20)
      const surface = (surfaceMean(sim, -10, -2.5) + surfaceMean(sim, 2.5, 10)) / 2
      return box.cy - surface
    }

    const coarse = draftAt(0.35)
    const fine = draftAt(0.25)
    expectNear(fine, coarse, {
      abs: 0.35,
      rel: 0,
      label: 'draft at 0.25 m vs 0.35 m spacing',
    })
  })

  it('sinks a dense box to the floor and keeps it there', () => {
    const floor = -6
    const sim = makeWorld({ widthM: 30, spacing: 0.35, terrain: basinTerrain(30, floor, 12) })
    fillWater(sim, { x0: -12, x1: 12, yTop: 0 })
    settle(sim, 12)

    const box = sim.addObject({ cx: 0, cy: 1, width: 2.4, height: 1.4, density: 7850 })
    settle(sim, 20)
    const restingCy = box.cy

    // ON the floor, not merely below the surface - steel does not hover.
    expect(restingCy).toBeLessThan(floor + 1.4)
    expect(restingCy).toBeGreaterThan(floor - 1)
    settle(sim, 5)
    // Staying put allows a slow grind against the bed (measured 0.07 m over
    // these five seconds), not a bounce or a walk.
    expect(Math.abs(box.cy - restingCy)).toBeLessThan(0.12)
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
    const h900 = heightOf(900)
    const h500 = heightOf(500)
    const h200 = heightOf(200)
    expectMonotonic([h900, h500, h200], 'increasing', 'float height vs density')
    // And not merely by numerical noise: 700 kg/m^3 of density difference on a
    // 1.6 m box is 1.1 m of draft by Archimedes; a quarter of that at least.
    expect(h200 - h900).toBeGreaterThan(0.3)
  })

  it('lifts a floating object as the water rises', () => {
    // The water rises GRADUALLY, as the flood slider raises it - poured in
    // thin layers. Dumping 3 m of water around a floater in one call buries
    // it under the collapsing walls of its own exclusion hole, which is a
    // fixture artefact, not a game state.
    const sim = makeWorld({ widthM: 30, spacing: 0.4, terrain: basinTerrain(30, -6, 20) })
    fillWater(sim, { x0: -12, x1: 12, yTop: -3 })
    settle(sim, 10)
    const box = sim.addObject({ cx: 0, cy: -2, width: 3, height: 1.6, density: 400 })
    settle(sim, 15)
    const low = box.cy

    for (let level = -2.7; level <= 0.01; level += 0.3) {
      fillWater(sim, { x0: -12, x1: 12, yBottom: level - 0.3, yTop: level, jitter: 0.15 })
      settle(sim, 2)
    }
    settle(sim, 15)

    // Rising flood water has to carry a buoyant object up with it. The
    // surface rose ~2.8 m; the box must track most of it. Verified red against
    // the pre-fix build, where the box ended BELOW where it started.
    expect(box.cy).toBeGreaterThan(low + 1.8)
  })

  it('a buoyant object released at depth surfaces', () => {
    // The scenario that exposed the pressure-field buoyancy as surface-only:
    // PBF's per-substep lambda carries no depth gradient in the bulk, so a
    // wood crate released 5 m down hovered there forever at every interface
    // stiffness. The analytic rest-volume term is what fixes it. Method:
    // red-first against the pressure-only build (box stayed at -4.7).
    const sim = makeWorld({ widthM: 30, spacing: 0.35, terrain: basinTerrain(30, -8, 12) })
    fillWater(sim, { x0: -12, x1: 12, yTop: 0 })
    settle(sim, 10)
    const box = sim.addObject({ cx: 0, cy: -5, width: 2, height: 1.2, density: 300 })
    settle(sim, 40)

    const surface = (surfaceMean(sim, -10, -2) + surfaceMean(sim, 2, 10)) / 2
    // Up from -5 to the neighbourhood of the surface - not pinned at depth,
    // not launched into orbit. The rise is deliberately unhurried (hull
    // viscosity entrains a water plug, and the buoyant-acceleration cap
    // plays the role of drag-limited rise), hence the 40 s.
    expect(box.cy).toBeGreaterThan(-2)
    expect(Math.abs(box.cy - surface)).toBeLessThan(1.5)
  })

  it('survives an object dropped INTO the water - the splash-entry case', () => {
    // The reported disaster: throw something in and the water erupts. Run at
    // the shipping 0.25 m resolution. Method: red-first - the pre-fix build
    // peaked at 40 m/s with 74 escapes on this scene.
    const sim = makeWorld({ widthM: 30, spacing: 0.25, terrain: basinTerrain(30, -5, 12) })
    fillWater(sim, { x0: -12, x1: 12, yTop: 0 })
    settle(sim, 10)

    const box = sim.addObject({ cx: 0, cy: 3, width: 2, height: 1.2, density: 500 })
    const trace = run(sim, { seconds: 12, box: { x0: -20, x1: 20, y0: -15, y1: 60 } })

    expectFinite(trace)
    expectNoEscapes(trace, 'splash-entry water')
    // Impact speed from 3 m is ~7 m/s; the splash may be somewhat faster but
    // an order more is the solver, not the splash. Measured peak: 4.9 m/s.
    expectSpeedBelow(trace, 12, 'splash-entry scene')
    // It calms down again.
    expect(speedPercentile(sim, 0.99, KIND_FLUID)).toBeLessThan(0.4)
    // And the box ends up floating at the surface, not launched or sunk.
    const surface = (surfaceMean(sim, -10, -2) + surfaceMean(sim, 2, 10)) / 2
    expect(box.cy - surface).toBeGreaterThan(-0.5)
    expect(box.cy - surface).toBeLessThan(0.9)
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

  it('loads the wall it is holding back, harder the deeper the water', () => {
    // THE flood-wall assertion. Pre-fix, a wall holding 1, 3 and 6 m of water
    // reported byte-identical member load and exactly zero deflection - water
    // exerted no force on members at all - and the old version of this test
    // passed anyway because its monotonic assert accepted ties. Method:
    // red-first against that build, and re-verified red with
    // applyHydrostaticLoad disabled.
    const loadFor = (depth: number) => {
      const sim = makeWorld({ widthM: 40, spacing: 0.4, terrain: basinTerrain(40, 0, 15) })
      const wall = buildWall(sim, { x: 0, yBottom: 0, yTop: 8, material: 'wood' })
      const tip = wall.nodes[wall.nodes.length - 1]!
      fillWater(sim, { x0: -16, x1: -1.5, yTop: depth })
      settle(sim, 12)
      return { load: peakStructuralLoad(sim), deflection: sim.particles.posX[tip]! }
    }

    const d2 = loadFor(2)
    const d4 = loadFor(4)
    const d6 = loadFor(6)

    // Strictly increasing, by a real margin - hydrostatic load grows like H^2.
    expectMonotonic([d2.load, d4.load, d6.load], 'increasing', 'wall load vs water depth', {
      byFactor: 1.25,
    })
    // And the absolute magnitude is a structural event: 6 m of water on an
    // 8 m wood wall loads it to over half of failure (measured 0.59; the
    // hand-calculated hydrostatic root-joint figure alone is ~0.3). The bound
    // sits at 0.45 deliberately - capsule contacts alone measure 0.39, so
    // this line is what fails if the analytic hydrostatic term is dropped
    // (mutation-verified red with applyHydrostaticLoad disabled).
    expect(d6.load).toBeGreaterThan(0.45)
    // The wall visibly leans away from the water (water is on the left, so
    // the tip moves right).
    expect(d6.deflection).toBeGreaterThan(0.03)
    expect(d6.deflection).toBeGreaterThan(d2.deflection)
  })

  it('a wood wall BREAKS under a deep enough head', () => {
    // Strength has to mean something in the water game, not just under point
    // loads: the siege loop is "the wall holds until it does not". Method:
    // red-first - no water-caused member breakage was possible pre-fix.
    const sim = makeWorld({ widthM: 40, spacing: 0.4, terrain: basinTerrain(40, 0, 20) })
    buildWall(sim, { x: 0, yBottom: 0, yTop: 9, material: 'wood', segments: 6 })
    const before = sim.distance.count
    fillWater(sim, { x0: -16, x1: -1.5, yTop: 8.5 })
    settle(sim, 15)
    expect(sim.distance.count).toBeLessThan(before)
  })
})

describe('objects against structures', () => {
  it('a crate resting on a platform bends it', () => {
    // Pre-fix the capsule contact was one-way: the crate was held up but the
    // platform felt nothing, so a loaded span never sagged. Method: red-first
    // against that build.
    const sim = makeWorld({ widthM: 30, spacing: 0.35, terrain: flatTerrain(30, 0) })
    const beam = buildBeam(sim, {
      x0: -3,
      y0: 2,
      x1: 3,
      y1: 2,
      material: 'wood',
      segments: 8,
      pinStart: true,
      pinEnd: true,
    })
    const mid = beam.nodes[Math.floor(beam.nodes.length / 2)]!
    const startY = sim.particles.posY[mid]!

    const crate = sim.addObject({ cx: 0, cy: 3.4, width: 1.2, height: 1.2, density: 600 })
    settle(sim, 12)

    // ~860 kg on a 6 m wood span: a real, visible sag.
    const sag = startY - sim.particles.posY[mid]!
    expect(sag).toBeGreaterThan(0.02)
    // And the crate is ON the platform, not through it.
    expect(crate.cy).toBeGreaterThan(sim.particles.posY[mid]!)
    expect(crate.cy).toBeLessThan(3.4)
  })

  it('two crates stack instead of passing through each other', () => {
    // Pre-fix there was no object-object contact at all. Method: red-first.
    const sim = makeWorld({ widthM: 30, spacing: 0.35, terrain: flatTerrain(30, 0) })
    const bottom = sim.addObject({ cx: 0, cy: 0.55, width: 2, height: 1, density: 400 })
    const top = sim.addObject({ cx: 0.1, cy: 2.1, width: 2, height: 1, density: 400 })
    settle(sim, 10)

    // The top crate rests ON the bottom one: centres a crate-height apart,
    // within the dent shape-matching allows.
    expect(top.cy - bottom.cy).toBeGreaterThan(0.7)
    // Neither burrowed into the ground.
    expect(bottom.cy).toBeGreaterThan(0.3)
  })

  it('waves shove a floating crate about', () => {
    // Driven through the production wave path, on the production beach.
    const field = new Field(40)
    const sim = new SimWorld()
    sim.terrain = field.terrain
    sim.boundsX0 = field.left
    sim.boundsX1 = field.right
    sim.fluid.spacing = 0.35
    const conditions = new Conditions(sim, field)

    // Flood the sea side, float a crate in the wave path (paddle zone starts
    // at right - 8 = 12).
    conditions.floodLevelM = 2
    for (let i = 0; i < 60 * 20; i++) {
      conditions.update(1 / 60)
      sim.step(1 / 60)
    }
    const crate = sim.addObject({ cx: 8, cy: 2.3, width: 1.6, height: 0.9, density: 300 })
    for (let i = 0; i < 60 * 5; i++) {
      conditions.update(1 / 60)
      sim.step(1 / 60)
    }

    let minX = Infinity
    let maxX = -Infinity
    conditions.waveStrength = 'extreme'
    for (let i = 0; i < 60 * 15; i++) {
      conditions.update(1 / 60)
      sim.step(1 / 60)
      minX = Math.min(minX, crate.cx)
      maxX = Math.max(maxX, crate.cx)
    }

    // Extreme waves must actually move a light floater - the audit found the
    // pre-fix drag was two orders of magnitude below a real wave load.
    expect(maxX - minX).toBeGreaterThan(0.8)
    expect(Number.isFinite(crate.cx)).toBe(true)
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
