import { describe, it, expect, beforeAll } from 'vitest'
import { GpuSolver } from '../../src/sim/gpu/gpuSolver'
import { CpuSolver } from '../../src/sim/solver'
import { SimWorld } from '../../src/sim/world'
import { KIND_FLUID } from '../../src/sim/particles'
import { buildBeam } from '../../src/scenes/demos'
import {
  basinTerrain,
  buildWall,
  fillWater,
  flatTerrain,
  makeWorld,
  topOfWater,
  waterBeyond,
} from '../harness'

/**
 * CPU/GPU parity: the same scenes stepped by both backends, compared where
 * comparison is honest. Micro-scenes with few interactions compare positions
 * tightly (f32 vs f64 drift only); scenes exercising the stated divergences
 * (colour-order joints, per-substep neighbour sets, atomic sums) compare
 * behaviour - settle speeds, containment, drafts - inside explicit bands.
 *
 * Everything runs against a real device; the rig test guarantees one exists.
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

/** Two identical worlds from one scene builder: CPU reference and GPU twin. */
function twinWorlds(build: (sim: SimWorld) => void): { cpu: SimWorld; gpu: SimWorld } {
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
  // Compare MESSAGES so a failure shows the actual WGSL/validation text.
  const unique = [...new Set(errors.map((e) => e.message))]
  expect(unique).toEqual([])
}

/** Max position difference across live particles (slot-aligned worlds). */
function maxPosDiff(a: SimWorld, b: SimWorld): number {
  let worst = 0
  const pa = a.particles
  const pb = b.particles
  expect(pb.highWater).toBe(pa.highWater)
  for (let i = 0; i < pa.highWater; i++) {
    if (pa.slots.alive[i] !== 1) continue
    worst = Math.max(
      worst,
      Math.hypot(pa.posX[i]! - pb.posX[i]!, pa.posY[i]! - pb.posY[i]!),
    )
  }
  return worst
}

describe('gpu parity', () => {
  it('compiles every kernel and steps an empty world without validation errors', async () => {
    const { cpu, gpu } = twinWorlds(() => {})
    await run(cpu, gpu, 2)
  })

  it('free fall onto terrain matches the CPU trajectory', async () => {
    const { cpu, gpu } = twinWorlds((sim) => {
      // Sparse: no particle interactions, just predict + terrain + velocity.
      for (let k = 0; k < 6; k++) {
        sim.particles.create({
          x: -15 + k * 6,
          y: 5 + k * 0.5,
          invMass: 1 / sim.fluid.particleMass,
          radius: sim.fluid.spacing * 0.5,
          kind: KIND_FLUID,
        })
      }
    })
    await run(cpu, gpu, 90)
    // Fell ~4-5 m and rested: f32-vs-f64 drift only.
    expect(maxPosDiff(cpu, gpu)).toBeLessThan(2e-3)
    expect(topOfWater(gpu)).toBeLessThan(1)
  })

  it('an over-dense pair relaxes identically', async () => {
    const { cpu, gpu } = twinWorlds((sim) => {
      const spec = (x: number) => ({
        x,
        y: 4,
        invMass: 1 / sim.fluid.particleMass,
        radius: sim.fluid.spacing * 0.5,
        kind: KIND_FLUID,
      })
      sim.particles.create(spec(0))
      sim.particles.create(spec(0.15)) // inside the kernel: pressure must act
    })
    await run(cpu, gpu, 30)
    const sep = (w: SimWorld) =>
      Math.abs(w.particles.posX[1]! - w.particles.posX[0]!)
    expect(sep(gpu)).toBeGreaterThan(0.15) // it DID push apart
    expect(Math.abs(sep(gpu) - sep(cpu))).toBeLessThan(0.02)
  })

  it('a stiff distance constraint matches the CPU solve', async () => {
    const { cpu, gpu } = twinWorlds((sim) => {
      const a = sim.particles.create({ x: 0, y: 6, invMass: 0, radius: 0.15 })
      const b = sim.particles.create({ x: 1.6, y: 6, invMass: 1 / 40, radius: 0.15 })
      sim.distance.create({ a, b, rest: 1.2, compliance: 1e-7, zeta: 0.9 })
    })
    await run(cpu, gpu, 60)
    expect(maxPosDiff(cpu, gpu)).toBeLessThan(3e-3)
    // And the constraint actually holds near rest length under gravity.
    const len = Math.hypot(
      gpu.particles.posX[1]! - gpu.particles.posX[0]!,
      gpu.particles.posY[1]! - gpu.particles.posY[0]!,
    )
    expect(len).toBeGreaterThan(1.1)
    expect(len).toBeLessThan(1.45)
  })

  it('a clamped cantilever sags the same way (colour-order joints band)', async () => {
    const { cpu, gpu } = twinWorlds((sim) => {
      buildBeam(sim, {
        x0: -4,
        y0: 6,
        x1: 4,
        y1: 6,
        material: 'wood',
        segments: 8,
        clampStart: true,
      })
    })
    await run(cpu, gpu, 120)
    // Both must be sagging, and by the same amount within the colour band.
    const tip = (w: SimWorld) => {
      let minY = Infinity
      let idx = -1
      const p = w.particles
      for (let i = 0; i < p.highWater; i++) {
        if (p.slots.alive[i] !== 1) continue
        if (p.posX[i]! > 3.5 && p.posY[i]! < minY) {
          minY = p.posY[i]!
          idx = i
        }
      }
      expect(idx).toBeGreaterThanOrEqual(0)
      return minY
    }
    const tipCpu = tip(cpu)
    const tipGpu = tip(gpu)
    expect(tipCpu).toBeLessThan(5.9) // it sags at all
    expect(Math.abs(tipGpu - tipCpu)).toBeLessThan(0.08)
  })

  it('a dropped object lands and rests at the same pose', async () => {
    const { cpu, gpu } = twinWorlds((sim) => {
      sim.addObject({ cx: 0, cy: 3, width: 2, height: 1, density: 400 })
    })
    await run(cpu, gpu, 150)
    const cCpu = cpu.clusters[0]!
    const cGpu = gpu.clusters[0]!
    expect(Math.abs(cGpu.cy - cCpu.cy)).toBeLessThan(0.05)
    expect(Math.abs(cGpu.cx - cCpu.cx)).toBeLessThan(0.05)
    expect(Math.abs(cGpu.angle - cCpu.angle)).toBeLessThan(0.05)
    // And it genuinely rests ON the ground (half-height 0.5 plus particle
    // radius standoff), neither buried nor floating away.
    expect(cGpu.cy).toBeGreaterThan(0.4)
    expect(cGpu.cy).toBeLessThan(0.75)
  })

  it('a rigid wall contains water on the GPU exactly as on the CPU', async () => {
    const build = (sim: SimWorld) => {
      buildWall(sim, { x: 4, yBottom: 0, yTop: 5, rigid: true })
      fillWater(sim, { x0: -6, x1: 3.5, yTop: 2.2 })
    }
    const { cpu, gpu } = twinWorlds(build)
    await run(cpu, gpu, 240)
    expect(waterBeyond(cpu, 4.6, 'right')).toBe(0)
    expect(waterBeyond(gpu, 4.6, 'right')).toBe(0)
  })

  it('a pool settles on the beach basin with reference-grade calm', async () => {
    const cpu = makeWorld({ widthM: 30, spacing: 0.45, terrain: basinTerrain(30, 0, 12) })
    const gpu = makeWorld({ widthM: 30, spacing: 0.45, terrain: basinTerrain(30, 0, 12) })
    for (const sim of [cpu, gpu]) fillWater(sim, { x0: -10, x1: 10, yTop: 3 })
    gpu.solver = new GpuSolver(gpu, device)

    await run(cpu, gpu, 300) // five seconds
    // Volume conserved exactly on both (the solver never creates/destroys).
    expect(gpu.fluidCount).toBe(cpu.fluidCount)

    const calm = (w: SimWorld) => {
      const p = w.particles
      let sum = 0
      let count = 0
      for (let i = 0; i < p.highWater; i++) {
        if (p.slots.alive[i] !== 1 || p.kind[i] !== KIND_FLUID) continue
        sum += Math.hypot(p.velX[i]!, p.velY[i]!)
        count++
      }
      return sum / count
    }
    const calmCpu = calm(cpu)
    const calmGpu = calm(gpu)
    // The settled mean speed is the physics; GPU must land in the same calm
    // regime, not merely "bounded" - within 2x of the reference, both small.
    expect(calmCpu).toBeLessThan(0.2)
    expect(calmGpu).toBeLessThan(Math.max(calmCpu * 2, 0.2))
    // Same surface height within roughly half a particle spacing. 95th
    // PERCENTILE, not max: max-Y reads a single residual-slosh straggler,
    // and slosh phase is exactly where f32 ordering makes the runs diverge.
    const surface = (w: SimWorld) => {
      const ys: number[] = []
      const p = w.particles
      for (let i = 0; i < p.highWater; i++) {
        if (p.slots.alive[i] !== 1 || p.kind[i] !== KIND_FLUID) continue
        ys.push(p.posY[i]!)
      }
      ys.sort((a, b) => a - b)
      return ys[Math.floor(ys.length * 0.95)]!
    }
    expect(Math.abs(surface(gpu) - surface(cpu))).toBeLessThan(0.25)
  })

  it('a light crate floats at the same draft (buoyancy + hull coupling)', async () => {
    const build = (sim: SimWorld) => {
      fillWater(sim, { x0: -8, x1: 8, yTop: 3 })
      sim.addObject({ cx: 0, cy: 3.6, width: 1.2, height: 0.8, density: 400 })
    }
    const { cpu, gpu } = twinWorlds(build)
    await run(cpu, gpu, 300) // five seconds of bobbing and settling
    // Compare MEAN height over a further second - an instantaneous sample
    // lands at whatever bob phase the run's float noise put it in.
    let sumCpu = 0
    let sumGpu = 0
    for (let i = 0; i < 60; i++) {
      cpu.step(1 / 60)
      await gpu.stepAsync(1 / 60)
      sumCpu += cpu.clusters[0]!.cy
      sumGpu += gpu.clusters[0]!.cy
    }
    const meanCpu = sumCpu / 60
    const meanGpu = sumGpu / 60
    // Both float (above the bed, below flight), at the same mean draft.
    expect(meanGpu).toBeGreaterThan(2)
    expect(meanGpu).toBeLessThan(4)
    expect(Math.abs(meanGpu - meanCpu)).toBeLessThan(0.3)
  })
})
