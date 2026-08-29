import { describe, it, expect } from 'vitest'
import {
  FLAG_ALIVE,
  FLAG_PINNED,
  decodeSnapshot,
  encodeSnapshot,
  kindOfFlags,
} from '../../src/runtime/snapshot'
import { KIND_FLUID, KIND_NODE } from '../../src/sim/particles'
import { buildBeam } from '../../src/scenes/demos'
import { flatTerrain, makeWorld } from '../harness'

/**
 * The snapshot codec is the render path's only source of truth once the sim
 * lives in a worker: a bit packed wrong here is a particle drawn in the wrong
 * place with no other test between it and the screen. Assertions compare the
 * decoded views against the sim's own arrays - the codec's contract is
 * faithful transport, so the sim IS the external reference.
 */

function populatedWorld() {
  const sim = makeWorld({ widthM: 40, terrain: flatTerrain(40, 0) })
  sim.spawnBlock(0, 3, 3, 2)
  sim.addObject({ cx: 6, cy: 4, width: 2, height: 1.5, density: 400 })
  buildBeam(sim, { x0: -8, y0: 0, x1: -8, y1: 6, material: 'wood', segments: 4 })
  // A pinned particle, for the pinned flag.
  sim.particles.create({ x: -12, y: 1, invMass: 0, radius: 0.3 })
  sim.step(1 / 60)
  return sim
}

describe('snapshot codec', () => {
  it('round-trips particles: position, kind, alive and pinned per live slot', () => {
    const sim = populatedWorld()
    const body = decodeSnapshot(encodeSnapshot(sim, null))
    const p = sim.particles

    expect(body.particleCount).toBe(p.highWater)
    let fluidSeen = 0
    let pinnedSeen = 0
    for (let i = 0; i < p.highWater; i++) {
      const flags = body.flags[i]!
      expect((flags & FLAG_ALIVE) !== 0).toBe(p.slots.alive[i] === 1)
      if (p.slots.alive[i] !== 1) continue
      expect(body.posX[i]).toBeCloseTo(p.posX[i]!, 5)
      expect(body.posY[i]).toBeCloseTo(p.posY[i]!, 5)
      expect(kindOfFlags(flags)).toBe(p.kind[i])
      expect((flags & FLAG_PINNED) !== 0).toBe(p.invMass[i] === 0)
      if (p.kind[i] === KIND_FLUID) fluidSeen++
      if (p.invMass[i] === 0 && p.kind[i] === KIND_NODE) pinnedSeen++
    }
    // The scene must actually exercise what the flags claim to carry.
    expect(fluidSeen).toBeGreaterThan(50)
    expect(pinnedSeen).toBeGreaterThan(0)
  })

  it('round-trips every live member segment with endpoints, stress and material', () => {
    const sim = populatedWorld()
    const body = decodeSnapshot(encodeSnapshot(sim, null))
    const d = sim.distance
    const p = sim.particles

    let live = 0
    for (let i = 0; i < d.highWater; i++) if (d.slots.alive[i] === 1) live++
    expect(body.segmentCount).toBe(live)
    expect(live).toBeGreaterThanOrEqual(4) // the beam's segments actually present

    // Segments are packed in table order over live constraints.
    let k = 0
    for (let i = 0; i < d.highWater; i++) {
      if (d.slots.alive[i] !== 1) continue
      expect(body.segAx[k]).toBeCloseTo(p.posX[d.a[i]!]!, 5)
      expect(body.segAy[k]).toBeCloseTo(p.posY[d.a[i]!]!, 5)
      expect(body.segBx[k]).toBeCloseTo(p.posX[d.b[i]!]!, 5)
      expect(body.segBy[k]).toBeCloseTo(p.posY[d.b[i]!]!, 5)
      expect(body.segStrain[k]).toBeCloseTo(d.strain[i]!, 6)
      expect(body.segDamage[k]).toBeCloseTo(d.damage[i]!, 6)
      expect(body.segMaterial[k]).toBe(d.material[i])
      k++
    }
  })

  it('reuses a recycled buffer when it is big enough, allocates when not', () => {
    const sim = populatedWorld()
    const first = encodeSnapshot(sim, null)
    const again = encodeSnapshot(sim, first)
    expect(again).toBe(first)

    const tiny = new ArrayBuffer(16)
    const grown = encodeSnapshot(sim, tiny)
    expect(grown).not.toBe(tiny)
    // And the reallocated one still decodes faithfully.
    const body = decodeSnapshot(grown)
    expect(body.particleCount).toBe(sim.particles.highWater)
  })

  it('rejects a buffer that is not a snapshot', () => {
    expect(() => decodeSnapshot(new ArrayBuffer(64))).toThrow()
  })
})
