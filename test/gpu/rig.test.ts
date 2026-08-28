import { describe, it, expect } from 'vitest'

/**
 * The GPU test rig itself, proven before any kernel sits on it. If WebGPU is
 * unavailable in this environment the failure must be LOUD - a skipped GPU
 * suite reported as green is exactly the false confidence the testing policy
 * exists to prevent.
 */

describe('webgpu rig', () => {
  it('exposes a WebGPU adapter and creates a device', async () => {
    expect(navigator.gpu, 'navigator.gpu missing - browser launched without WebGPU').toBeDefined()
    const adapter = await navigator.gpu.requestAdapter()
    expect(adapter, 'no WebGPU adapter - check launch flags / GPU drivers').not.toBeNull()
    const device = await adapter!.requestDevice()
    expect(device).toBeDefined()
    expect(device.limits.maxComputeWorkgroupSizeX).toBeGreaterThanOrEqual(256)
    expect(device.limits.maxStorageBufferBindingSize).toBeGreaterThanOrEqual(128 * 1024 * 1024)
    device.destroy()
  })

  it('runs a trivial compute shader and reads the result back', async () => {
    const adapter = await navigator.gpu.requestAdapter()
    const device = await adapter!.requestDevice()
    const n = 256
    const buf = device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })
    const staging = device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    const module = device.createShaderModule({
      code: /* wgsl */ `
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          if (id.x >= ${n}u) { return; }
          data[id.x] = f32(id.x) * 3.0 + 1.0;
        }
      `,
    })
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    })
    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: buf } }],
    })
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bind)
    pass.dispatchWorkgroups(n / 64)
    pass.end()
    enc.copyBufferToBuffer(buf, 0, staging, 0, n * 4)
    device.queue.submit([enc.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    const out = new Float32Array(staging.getMappedRange().slice(0))
    staging.unmap()
    device.destroy()

    expect(out[0]).toBe(1)
    expect(out[100]).toBe(301)
    expect(out[255]).toBe(766)
  })
})
