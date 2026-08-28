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
 * packed vec4 gathers, fused per-particle kernels) PLUS the review-mandated
 * support margin (the CPU's rq slack for pairs closing mid-frame, scaled by
 * substep so it only pays for drift actually accrued): 21-24 ms/frame,
 * median ~23. Without the margin the same rig measured ~16.7 - that number
 * had a physics gap. So the absolute floor adapter runs 40k at ~45 fps and
 * holds 60 Hz to ~29k; the readback stall (~3 ms, recoverable by pipelining
 * it a frame behind) is the known next lever. The CPU reference needs
 * ~156 ms for this scene on a fast desktop core.
 *
 * The assertion is a REGRESSION GUARD at 28 ms on the median of three
 * batches: it trips on any real slowdown while staying quiet across the
 * +-1.5 ms the shared iGPU adds run to run.
 */

describe('gpu throughput', () => {
  it('steps ~40k fluid particles at 60 Hz class rates on the iGPU', async () => {
    const sim = makeWorld({ widthM: 240, spacing: 0.25, terrain: flatTerrain(240, 0) })
    fillWater(sim, { x0: -110, x1: 110, yTop: 11.4 })
    const adapter = await navigator.gpu.requestAdapter()
    const device = await adapter!.requestDevice()
    // A validation error invalidates the command buffer and submit becomes a
    // no-op - which is FASTER. Without this listener, a fully broken solver
    // would pass the timing assertion below.
    const errors: string[] = []
    device.addEventListener('uncapturederror', (e) => {
      errors.push((e as GPUUncapturedErrorEvent).error.message)
    })
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
    expect([...new Set(errors)]).toEqual([])
    // The solver did real work: 2.5 s in, the fill has collapsed into a
    // settling pool (volume conserved, everything on the field).
    expect(sim.fluidCount).toBe(fluid)
    const p = sim.particles
    for (let i = 0; i < p.highWater; i++) {
      if (p.slots.alive[i] !== 1) continue
      expect(Number.isFinite(p.posX[i]!) && Number.isFinite(p.posY[i]!)).toBe(true)
    }
    expect(median).toBeLessThan(28)
  })
})
