import { describe, it, expect, beforeAll } from 'vitest'
import { GpuSolver } from '../../src/sim/gpu/gpuSolver'
import type { SimWorld } from '../../src/sim/world'
import { buildBeam } from '../../src/scenes/demos'
import { buildWall, fillWater, flatTerrain, makeWorld } from '../harness'

/**
 * ISOLATING tests for the in-kernel coupling forces. The broad parity
 * scenes float a crate and load a wall - but the crate also rides hull
 * pressure and the wall also rides capsule contacts, and mutation testing
 * proved those scenes pass with the force kernels DISABLED. Each scene
 * here is built so its force is the only thing that can produce the
 * asserted motion, and each was verified red with its kernel knocked out.
 *
 * WATER DRAG has no isolating scene, deliberately: it co-fires with
 * displacement contacts by construction (submerged means in contact), and
 * measurement showed every reachable regime is either contact-saturated
 * (a driven current bulldozes a post identically with drag disabled) or
 * bistable (gentler currents sit on a bend-or-stand knife edge where
 * backends diverge chaotically). Drag is covered by the aggregate parity
 * bands (wall load, crate draft, submerged-fall trajectory), which is the
 * same combined-sum footing its CPU calibration stands on.
 */

let device: GPUDevice
const errors: GPUError[] = []

beforeAll(async () => {
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('no WebGPU adapter')
  device = await adapter.requestDevice()
  device.addEventListener('uncapturederror', (e) => {
    errors.push((e as GPUUncapturedErrorEvent).error)
  })
})

function twin(build: (sim: SimWorld) => void): { cpu: SimWorld; gpu: SimWorld } {
  const cpu = makeWorld({ widthM: 40, terrain: flatTerrain(40, 0) })
  const gpu = makeWorld({ widthM: 40, terrain: flatTerrain(40, 0) })
  build(cpu)
  build(gpu)
  gpu.solver = new GpuSolver(gpu, device)
  return { cpu, gpu }
}

async function run(cpu: SimWorld, gpu: SimWorld, frames: number): Promise<void> {
  for (let i = 0; i < frames; i++) {
    cpu.step(1 / 60)
    await gpu.stepAsync(1 / 60)
  }
  expect([...new Set(errors.map((e) => e.message))]).toEqual([])
}

/** x of the highest FREE structure node with x >= xMin - skips pinned
 *  nodes and anything (like a containment wall) left of the probe region. */
function freeTipX(w: SimWorld, xMin = -Infinity): number {
  const p = w.particles
  let x = NaN
  let bestY = -Infinity
  for (let i = 0; i < p.highWater; i++) {
    if (p.slots.alive[i] !== 1 || p.kind[i] !== 0 || p.invMass[i] === 0) continue
    if (p.posX[i]! < xMin) continue
    if (p.posY[i]! > bestY) {
      bestY = p.posY[i]!
      x = p.posX[i]!
    }
  }
  return x
}

describe('gpu coupling forces (isolated)', () => {
  it('analytic buoyancy lifts a deep node - the force pressure cannot provide', async () => {
    // A lone wood-volume node 2 m under the pool: no constraints, no
    // contacts of consequence, and PBF pressure carries no depth gradient
    // in the bulk (world.ts:293-300) - if this node rises, the analytic
    // buoyancy term did it. Lift = 9.81 * (1000 * 0.02 / 8) * 1 ~ 24.5 >> g.
    // Track the node's SLOT per world: the terrain boundary sampler creates
    // particles after ours on the first step, so highWater-1 is not it.
    const slots = new Map<SimWorld, number>()
    const { cpu, gpu } = twin((sim) => {
      fillWater(sim, { x0: -6, x1: 6, yTop: 3 })
      slots.set(sim, sim.particles.create({ x: 0, y: 1, invMass: 1 / 8, radius: 0.1, volume: 0.02 }))
    })
    const node = (w: SimWorld) => w.particles.posY[slots.get(w)!]!
    const y0 = node(gpu)
    await run(cpu, gpu, 60)
    // It rose, decisively (this is the assert the buoyancy mutation trips:
    // without lift the node falls), and lands near the CPU height - the
    // rise integrates 60 frames of slightly-diverging settling water, so
    // the band is trajectory-grade, not formula-grade.
    expect(node(gpu)).toBeGreaterThan(y0 + 0.3)
    expect(Math.abs(node(gpu) - node(cpu))).toBeLessThan(0.12)
  })

  it('hydrostatic pressure pushes a DRY member standing beside the pool', async () => {
    // A pinned-base wood post half a metre outside the water's edge: no
    // fluid within contact range, but the +-hydroOff (1.5 m) pressure
    // samples straddle the edge - net pressure pushes the post AWAY from
    // the pool. Any tip deflection here is the hydrostatic term alone.
    // A rigid wall CONTAINS the pond (on flat ground it otherwise collapses
    // and spreads to both sides of the post within the run, cancelling the
    // pressure difference - measured; that null scene passed with the
    // kernel disabled). The post stands dry half a metre beyond the wall.
    const { cpu, gpu } = twin((sim) => {
      buildWall(sim, { x: -4.5, yBottom: 0, yTop: 4, rigid: true })
      fillWater(sim, { x0: -14, x1: -5, yTop: 2.5 })
      buildBeam(sim, {
        x0: -4,
        y0: 0,
        x1: -4,
        y1: 3,
        material: 'wood',
        segments: 3,
        pinStart: true,
      })
    })
    await run(cpu, gpu, 90)
    // Pushed away from the water (toward +x), same deflection as CPU. The
    // probe region excludes the containment wall (its top node is free).
    expect(freeTipX(gpu, -4.3)).toBeGreaterThan(-3.97)
    expect(Math.abs(freeTipX(gpu, -4.3) - freeTipX(cpu, -4.3))).toBeLessThan(0.05)
  })

  it('water drag shapes a submerged member fall to match the CPU', async () => {
    // A free horizontal wood member released deep inside a pool. Its fall
    // is fought by drag AND displacement contacts; the tight band on the
    // trajectory is what makes the test sensitive - with the drag term
    // knocked out the GPU member falls measurably faster than the CPU one.
    const { cpu, gpu } = twin((sim) => {
      fillWater(sim, { x0: -8, x1: 8, yTop: 5 })
      buildBeam(sim, {
        x0: -1,
        y0: 3.5,
        x1: 1,
        y1: 3.5,
        material: 'steel',
        segments: 2,
      })
    })
    const meanY = (w: SimWorld) => {
      const p = w.particles
      let sum = 0
      let n = 0
      for (let i = 0; i < p.highWater; i++) {
        if (p.slots.alive[i] !== 1 || p.kind[i] !== 0) continue
        sum += p.posY[i]!
        n++
      }
      return sum / n
    }
    const y0 = meanY(gpu)
    await run(cpu, gpu, 90)
    // Packed PBF water plus the drag cap holds a thin member near-static,
    // so it settles slowly rather than plunging - the assert that carries
    // the sensitivity is the TIGHT band against the CPU trajectory: with
    // the drag term knocked out the GPU member falls visibly faster.
    expect(meanY(gpu)).toBeLessThan(y0 - 0.03)
    expect(Math.abs(meanY(gpu) - meanY(cpu))).toBeLessThan(0.08)
  })

})