import { describe, it, expect, beforeAll } from 'vitest'
import { GpuSolver } from '../../src/sim/gpu/gpuSolver'
import { KIND_FLUID } from '../../src/sim/particles'
import type { SimWorld } from '../../src/sim/world'
import { flatTerrain, makeWorld } from '../harness'

/**
 * The pipelined host pattern - reap() before each step(), frames left in
 * flight - must advance physics exactly one step per step().
 *
 * The failure this pins down, found live: a backend that re-simulates from
 * HOST-visible state (which lags the GPU while frames are in flight) builds
 * frame k+1 from the same stale state as frame k whenever a fence has not
 * landed in between. The world forks into two interleaved timelines, each
 * advancing at half rate - on screen the whole fluid blinks between two
 * states (measured: 88% of moving particles reversing direction every
 * snapshot after 20 s of the water tool), spawn guards check one timeline
 * while spawning into the other, and the density errors discharge at
 * hundreds of m/s.
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

/**
 * A block in free fall high above the ground: no contacts, no members, no
 * frame-head forces - gravity integration only. Both stepping patterns are
 * then dynamically identical and the ONLY variable is the pipeline.
 */
function fallWorld(): SimWorld {
  const w = makeWorld({ widthM: 40, terrain: flatTerrain(40, 0) })
  w.spawnBlock(0, 30, 3, 3)
  w.solver = new GpuSolver(w, device)
  return w
}

function meanFluidY(w: SimWorld): number {
  const p = w.particles
  let sum = 0
  let n = 0
  for (let i = 0; i < p.highWater; i++) {
    if (p.slots.alive[i] !== 1 || p.kind[i] !== KIND_FLUID) continue
    sum += p.posY[i]!
    n++
  }
  expect(n).toBeGreaterThan(50)
  return sum / n
}

describe('gpu pipelined stepping', () => {
  it('advances physics one step per step(), not as half-rate forked timelines', async () => {
    const FRAMES = 48
    const dt = 1 / 60

    // Reference: flush after every frame (exact depth-0 semantics).
    const flushed = fallWorld()
    const y0 = meanFluidY(flushed)
    for (let f = 0; f < FRAMES; f++) await flushed.stepAsync(dt)
    const fallFlushed = y0 - meanFluidY(flushed)

    // Host pattern: reap (non-blocking consume) before every step. Fences
    // cannot resolve between two steps issued in the same task turn, so at
    // least every second sync happens with a frame still in flight - the
    // exact condition that forks a host-state-driven backend.
    const piped = fallWorld()
    const solver = piped.solver as GpuSolver
    for (let f = 0; f < FRAMES; f++) {
      await solver.reap()
      piped.step(dt)
    }
    await solver.readback()
    const fallPiped = y0 - meanFluidY(piped)

    // 48 frames of free fall is ~2.0 m; a forked pipeline shows roughly half.
    expect(fallFlushed).toBeGreaterThan(1.5)
    expect(Math.abs(fallPiped - fallFlushed)).toBeLessThan(fallFlushed * 0.05)
    expect([...new Set(errors.map((e) => e.message))]).toEqual([])
  })

  it('keeps host and device coherent under stream + drain slot churn', async () => {
    // The water-tool path that broke live: the flood drain destroys pooled
    // particles the same tick the stream recycles their slots at the nozzle,
    // while two frames are still in flight. The write-stamp guard must keep
    // landing frames from clobbering recycled slots back to the dead
    // particle's position - a clobber leaves the host SoA disagreeing with
    // the device by tens of metres, which the next landed frame then
    // "corrects" as a teleport.
    const w = makeWorld({ widthM: 40, terrain: flatTerrain(40, 0) })
    w.spawnBlock(0, 1.2, 6, 1.5)
    w.solver = new GpuSolver(w, device)
    const solver = w.solver as GpuSolver
    const dt = 1 / 60
    const p = w.particles

    // HOST order per tick: reap, then churn (conditions drain + stream
    // spawn), then step. The clobber this pins down is TRANSIENT: a frame
    // captured before a recycle lands one or two reaps later, and without
    // the stamp guard it yanks the fresh nozzle particle back to the dead
    // pool particle's position until a newer frame heals it - so the
    // assertion must run at reap time on recent spawns, not after a flush.
    const tracked: { slot: number; y0: number; iter: number }[] = []
    for (let f = 0; f < 80; f++) {
      await solver.reap()
      for (const t of tracked) {
        if (p.slots.alive[t.slot] !== 1 || p.kind[t.slot] !== KIND_FLUID) continue
        expect(p.posY[t.slot]!).toBeGreaterThan(t.y0 - 1.5)
      }
      while (tracked.length > 0 && f - tracked[0]!.iter >= 3) tracked.shift()

      // Drain: kill every 10th live fluid particle (pool churn)...
      let k = 0
      for (let i = 0; i < p.highWater; i++) {
        if (p.slots.alive[i] !== 1 || p.kind[i] !== KIND_FLUID) continue
        if (++k % 10 === 0) p.destroy(i)
      }
      // ...and the stream refills through the recycled slots, tracked by
      // their creation stamp so the reap-time assertion knows them.
      const s0 = p.stampValue
      w.spawnDisc(0, 12, 0.5)
      for (let i = 0; i < p.highWater; i++) {
        if (p.slots.alive[i] === 1 && p.kind[i] === KIND_FLUID && p.writeStamp[i]! >= s0) {
          tracked.push({ slot: i, y0: p.posY[i]!, iter: f })
        }
      }
      w.step(dt)
    }
    await solver.readback()

    // Snapshot the settled host view, advance ONE quiet frame, flush, and
    // compare: with no host writes in between, every slot's motion must be
    // one physical frame's worth. A stamp-guard failure shows up here as a
    // slot yanked tens of metres between two consecutive flushed states.
    const beforeX = p.posX.slice(0, p.highWater)
    const beforeY = p.posY.slice(0, p.highWater)
    await w.stepAsync(dt)
    let maxMove = 0
    let maxSpeed = 0
    for (let i = 0; i < p.highWater; i++) {
      if (p.slots.alive[i] !== 1 || p.kind[i] !== KIND_FLUID) continue
      maxMove = Math.max(
        maxMove,
        Math.hypot(p.posX[i]! - beforeX[i]!, p.posY[i]! - beforeY[i]!),
      )
      maxSpeed = Math.max(maxSpeed, Math.hypot(p.velX[i]!, p.velY[i]!))
    }
    // One frame at the world speed cap, plus contact-correction headroom.
    expect(maxMove).toBeLessThan(w.maxSpeed * dt + 0.5)
    expect(maxSpeed).toBeLessThanOrEqual(w.maxSpeed * 1.05)
    expect([...new Set(errors.map((e) => e.message))]).toEqual([])
  })
})
