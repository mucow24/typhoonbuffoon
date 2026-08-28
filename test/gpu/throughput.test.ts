import { describe, it, expect } from 'vitest'
import { GpuSolver } from '../../src/sim/gpu/gpuSolver'
import { fillWater, flatTerrain, makeWorld } from '../harness'

/**
 * The 10x claim, measured where it was made: ~40k fluid particles at 60 Hz on
 * integrated-GPU-class hardware (docs/GPU_PLAN.md). This browser runs on the
 * Intel UHD (gen-12lp) adapter - the hardware floor, not the dev machine's
 * discrete GPU.
 *
 * Timed via stepAsync, so the number includes the host passes, full upload,
 * dispatch encoding, submission AND the readback stall - the true wall cost
 * of a frame, not a flattering kernel-only figure.
 *
 * Measured on 2026-08-28 after the optimisation pass (dense sorted grid,
 * packed vec4 gathers, fused per-particle kernels): 16.4-17.4 ms/frame,
 * median ~16.7 - i.e. 60 Hz at the hardware floor, within thermal noise,
 * with ~13 ms of that on the GPU and ~3.5 ms in host passes + the readback
 * stall (recoverable later by pipelining the readback a frame behind).
 * The CPU reference needs ~156 ms for this scene on a fast desktop core.
 *
 * The assertion is a REGRESSION GUARD at 20 ms on the median of three
 * batches: it trips on any real slowdown while staying quiet across the
 * +-1 ms the shared iGPU adds run to run.
 */

describe('gpu throughput', () => {
  it('steps ~40k fluid particles at 60 Hz class rates on the iGPU', async () => {
    const sim = makeWorld({ widthM: 240, spacing: 0.25, terrain: flatTerrain(240, 0) })
    fillWater(sim, { x0: -110, x1: 110, yTop: 11.4 })
    const adapter = await navigator.gpu.requestAdapter()
    const device = await adapter!.requestDevice()
    sim.solver = new GpuSolver(sim, device)

    const fluid = sim.fluidCount
    expect(fluid).toBeGreaterThan(38000)

    for (let i = 0; i < 30; i++) await sim.stepAsync(1 / 60) // warm up
    const batchMeans: number[] = []
    for (let b = 0; b < 3; b++) {
      const frames = 40
      const t0 = performance.now()
      for (let i = 0; i < frames; i++) await sim.stepAsync(1 / 60)
      batchMeans.push((performance.now() - t0) / frames)
    }
    batchMeans.sort((a, b) => a - b)
    const median = batchMeans[1]!

    console.log(
      `[gpu-throughput] ${fluid} fluid particles: ` +
        `${batchMeans.map((m) => m.toFixed(2)).join(' / ')} ms/frame (median ${median.toFixed(2)})`,
    )
    expect(median).toBeLessThan(20)
  })
})
