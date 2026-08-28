import { MATERIALS, MATERIAL_IDS } from '../materials'
import { KIND_BOUNDARY, KIND_FLUID, KIND_OBJECT } from '../particles'
import type { SolverBackend } from '../solver'
import type { SimWorld } from '../world'
import { colorConstraints } from './coloring'
import {
  BF,
  CL_HEADER_F,
  DF,
  DU,
  BU,
  FC,
  FP_SCALE,
  FX,
  PF,
  PU,
  UNIFORM_BYTES,
  stagingLayout,
  writeUniforms,
  type StagingLayout,
} from './layout'
import { KERNELS, type KernelName } from './shaders'

const HASH_MASK = (1 << KIND_FLUID) | (1 << KIND_OBJECT) | (1 << KIND_BOUNDARY)

/** Workgroup width per kernel; gather-heavy kernels run wide for latency
 *  hiding on iGPUs. MUST match the @workgroup_size in shaders.ts. */
const WG: Partial<Record<KernelName, number>> = {
  gridClear: 256,
  gridScan1: 256,
  gridScan2: 1,
  gridScan3: 256,
  census: 256,
  fluidDensity: 256,
  integrate: 64,
  fluidCorrect: 256,
  contactsSolid: 256,
  contactsMember: 256,
  hullFluid: 256,
  hullSolid: 256,
  xsph: 256,
}

/**
 * The WebGPU backend: the same substep pipeline as CpuSolver, as compute
 * kernels (shaders.ts documents the reformulations kernel by kernel).
 *
 * The device is STATELESS ACROSS FRAMES by design: sync() re-uploads the full
 * particle/constraint/cluster state each frame (~1 MB at 40k particles,
 * fractions of a millisecond), step() encodes every substep into one compute
 * pass and submits, and readback() copies the results back into the world's
 * SoA arrays - which therefore stay canonical, so every host system (Session,
 * damage, water field, probes, snapshots) works identically on either
 * backend, and swapping backends mid-session needs no migration at all.
 */
export class GpuSolver implements SolverBackend {
  /** Probe for a device; null when WebGPU is unavailable or refuses one -
   *  the caller falls back to the CPU reference, never crashes. */
  static async create(w: SimWorld): Promise<GpuSolver | null> {
    try {
      const gpu = (globalThis.navigator as Navigator | undefined)?.gpu
      if (!gpu) return null
      // Ask for the DISCRETE GPU explicitly: on dual-GPU laptops the default
      // adapter is frequently the power-saver iGPU, which turns a 4090
      // machine into a gen-12lp machine and the flood into slow motion.
      const adapter =
        (await gpu.requestAdapter({ powerPreference: 'high-performance' })) ??
        (await gpu.requestAdapter())
      if (!adapter) return null
      const device = await adapter.requestDevice()
      return new GpuSolver(w, device)
    } catch {
      return null
    }
  }

  private readonly pipelines = new Map<KernelName, GPUComputePipeline>()
  private readonly bindGroups = new Map<KernelName, GPUBindGroup>()
  private readonly g0Layouts = new Map<KernelName, GPUBindGroupLayout>()
  private colorLayout!: GPUBindGroupLayout
  /** Per (colour-using kernel, colour slot) range bind groups. */
  private readonly colorGroups = new Map<string, GPUBindGroup>()
  private colorMetaBufs: GPUBuffer[] = []

  private cap = 0
  private distCap = 0
  private bendCap = 0
  private clusterCap = 0
  private terrCap = 4096
  private clusterStagingCap = 64
  private tableSize = 0
  private tableCap = 0
  private stag: StagingLayout | null = null

  private uni!: GPUBuffer
  private pf!: GPUBuffer
  private pu!: GPUBuffer
  private gridS!: GPUBuffer
  private gridA!: GPUBuffer
  private fx!: GPUBuffer
  private fc!: GPUBuffer
  private df!: GPUBuffer
  private du!: GPUBuffer
  private bf!: GPUBuffer
  private bu!: GPUBuffer
  private colorIdx!: GPUBuffer
  private clF!: GPUBuffer
  private clU!: GPUBuffer
  private terr!: GPUBuffer
  private matSec!: GPUBuffer
  private gath!: GPUBuffer
  private staging!: GPUBuffer

  /** Debug-only: kernels to skip while bisecting performance. */
  skipKernels: Set<KernelName> | null = null

  /** Scratch for u8/i32 -> u32 section conversion. */
  private scratchU32 = new Uint32Array(0)
  private uniData = new ArrayBuffer(UNIFORM_BYTES)

  /** Frame-varying dispatch data prepared in sync(), consumed by step(). */
  private distColors: number[][] = []
  private bendColors: number[][] = []
  /** Alive clusters in upload order, for pose write-back. */
  private liveClusters: { cx: number; cy: number; angle: number }[] = []
  private frameN = 0
  private frameDistHW = 0
  private frameBendHW = 0
  private pendingUniforms: Record<string, number> | null = null

  constructor(
    private readonly w: SimWorld,
    private readonly device: GPUDevice,
  ) {
    this.ensureCapacity()
    this.matSecUpload()
  }

  /** SolverBackend.dispose: release the device promptly on backend swap -
   *  the JS wrapper is tiny, the GPU allocations behind it are not. */
  dispose(): void {
    this.device.destroy()
  }

  // ------------------------------------------------------------------- sync

  sync(): void {
    const w = this.w
    const p = w.particles
    this.ensureCapacity()

    // BUFFER SIZING FIRST - grid geometry and the terrain buffer. Growth on
    // either path rebuilds every bind group, so both must precede anything
    // that creates per-frame bind groups (the colour groups below); a grow
    // that ran later silently cleared them, and joints stopped solving.
    // Grid cell = KERNEL radius h; gathers walk world-space reach windows
    // (see shaders.ts gatherLoop). Cells grow only if the field is so large
    // the table would blow its cap - that just adds candidates.
    const fGrid = w.fluid
    const tGrid = w.terrain
    const MARGIN = 4
    const TABLE_CAP = 1 << 22
    let cell = fGrid.spacing * 2
    const yLow = (tGrid ? tGrid.minHeight : 0) - 8
    const yHigh = (tGrid ? tGrid.maxHeight : 0) + 160
    for (;;) {
      const gw = Math.ceil((w.boundsX1 - w.boundsX0) / cell) + MARGIN * 2
      const gh = Math.ceil((yHigh - yLow) / cell) + MARGIN * 2
      if (gw * gh <= TABLE_CAP) break
      cell *= 1.5
    }
    const gridW = Math.ceil((w.boundsX1 - w.boundsX0) / cell) + MARGIN * 2
    const gridH = Math.ceil((yHigh - yLow) / cell) + MARGIN * 2
    this.tableSize = gridW * gridH
    this.ensureGrid(this.tableSize)
    const gridGeom = { gridW, gridH, gridX0: w.boundsX0 - MARGIN * cell, gridY0: yLow - MARGIN * cell, cell }

    if (tGrid && tGrid.heights.length > this.terrCap) {
      this.terrCap = Math.max(tGrid.heights.length, this.terrCap * 2)
      this.terr.destroy()
      this.terr = this.device.createBuffer({
        size: this.terrCap * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      })
      this.buildPipelines()
    }

    const q = this.device.queue
    const n = p.highWater
    this.frameN = n
    this.frameDistHW = w.distance.highWater
    this.frameBendHW = w.bend.highWater

    // Particle f32 sections straight from the SoA arrays.
    const wf = (sec: number, arr: Float32Array) => {
      if (n > 0) q.writeBuffer(this.pf, sec * this.cap * 4, arr.buffer, arr.byteOffset, n * 4)
    }
    wf(PF.posX, p.posX)
    wf(PF.posY, p.posY)
    wf(PF.velX, p.velX)
    wf(PF.velY, p.velY)
    wf(PF.accX, p.accX)
    wf(PF.accY, p.accY)
    wf(PF.invMass, p.invMass)
    wf(PF.radius, p.radius)
    wf(PF.volume, p.volume)

    // u32 sections via the conversion scratch.
    const wu = (buf: GPUBuffer, sec: number, cap: number, fill: (out: Uint32Array) => void) => {
      if (n === 0) return
      const out = this.scratchU32.subarray(0, Math.max(n, 1))
      fill(out)
      q.writeBuffer(buf, sec * cap * 4, out.buffer, out.byteOffset, out.length * 4)
    }
    wu(this.pu, PU.kind, this.cap, (o) => {
      for (let i = 0; i < n; i++) o[i] = p.kind[i]!
    })
    wu(this.pu, PU.cluster1, this.cap, (o) => {
      for (let i = 0; i < n; i++) o[i] = p.cluster[i]! + 1
    })
    wu(this.pu, PU.alive, this.cap, (o) => {
      for (let i = 0; i < n; i++) o[i] = p.slots.alive[i]!
    })

    // Distance constraints.
    const d = w.distance
    const dn = d.highWater
    const wdf = (sec: number, arr: Float32Array) => {
      if (dn > 0) q.writeBuffer(this.df, sec * this.distCap * 4, arr.buffer, arr.byteOffset, dn * 4)
    }
    wdf(DF.rest, d.rest)
    wdf(DF.compliance, d.compliance)
    wdf(DF.zeta, d.zeta)
    const wdi = (sec: number, arr: Int32Array) => {
      if (dn > 0) q.writeBuffer(this.du, sec * this.distCap * 4, arr.buffer, arr.byteOffset, dn * 4)
    }
    wdi(DU.a, d.a)
    wdi(DU.b, d.b)
    const wdu = (sec: number, fill: (out: Uint32Array) => void) => {
      if (dn === 0) return
      const out = this.scratchU32.subarray(0, dn)
      fill(out)
      q.writeBuffer(this.du, sec * this.distCap * 4, out.buffer, out.byteOffset, dn * 4)
    }
    wdu(DU.mat, (o) => {
      for (let i = 0; i < dn; i++) o[i] = d.material[i]!
    })
    wdu(DU.noCol1, (o) => {
      for (let i = 0; i < dn; i++) o[i] = d.noCollideCluster[i]! + 1
    })
    wdu(DU.alive, (o) => {
      for (let i = 0; i < dn; i++) o[i] = d.slots.alive[i]!
    })

    // Bends.
    const b = w.bend
    const bn = b.highWater
    const wbf = (sec: number, arr: Float32Array) => {
      if (bn > 0) q.writeBuffer(this.bf, sec * this.bendCap * 4, arr.buffer, arr.byteOffset, bn * 4)
    }
    wbf(BF.restAngle, b.restAngle)
    wbf(BF.compliance, b.compliance)
    wbf(BF.zeta, b.zeta)
    const wbi = (sec: number, arr: Int32Array) => {
      if (bn > 0) q.writeBuffer(this.bu, sec * this.bendCap * 4, arr.buffer, arr.byteOffset, bn * 4)
    }
    wbi(BU.a, b.a)
    wbi(BU.b, b.b)
    wbi(BU.c, b.c)
    if (bn > 0) {
      const out = this.scratchU32.subarray(0, bn)
      for (let i = 0; i < bn; i++) out[i] = b.slots.alive[i]!
      q.writeBuffer(this.bu, BU.alive * this.bendCap * 4, out.buffer, out.byteOffset, bn * 4)
    }

    // Colouring (cheap at structure scale; recomputed every frame so topology
    // changes never leave a stale race behind).
    const dEndpoints: number[][] = []
    for (let i = 0; i < dn; i++) {
      dEndpoints.push(d.slots.alive[i] === 1 ? [d.a[i]!, d.b[i]!] : [])
    }
    this.distColors = colorConstraints(dEndpoints)
    const bEndpoints: number[][] = []
    for (let i = 0; i < bn; i++) {
      bEndpoints.push(b.slots.alive[i] === 1 ? [b.a[i]!, b.b[i]!, b.c[i]!] : [])
    }
    this.bendColors = colorConstraints(bEndpoints)

    const flat: number[] = []
    const ranges: { start: number; count: number }[] = []
    for (const group of [...this.distColors, ...this.bendColors]) {
      ranges.push({ start: flat.length, count: group.length })
      for (const ci of group) flat.push(ci)
    }
    if (flat.length > 0) {
      q.writeBuffer(this.colorIdx, 0, new Uint32Array(flat))
    }
    this.ensureColorMeta(ranges)

    // Clusters.
    this.liveClusters = []
    let totalClusterParticles = 0
    const live: (typeof w.clusters)[number][] = []
    for (const c of w.clusters) {
      if (!c.alive) continue
      live.push(c)
      totalClusterParticles += c.particles.length
    }
    if (live.length > 0) {
      const headerEnd = live.length * CL_HEADER_F
      const clFData = new Float32Array(headerEnd + 3 * totalClusterParticles)
      const clUData = new Uint32Array(live.length * 2 + totalClusterParticles)
      let start = 0
      live.forEach((c, ci) => {
        this.liveClusters.push(c)
        const count = c.particles.length
        clFData[ci * CL_HEADER_F] = c.stiffness
        clFData[ci * CL_HEADER_F + 1] = c.maxCorrection
        clUData[ci * 2] = start
        clUData[ci * 2 + 1] = count
        const base = headerEnd + 3 * start
        clFData.set(c.restX.subarray(0, count), base)
        clFData.set(c.restY.subarray(0, count), base + count)
        clFData.set(c.mass.subarray(0, count), base + 2 * count)
        for (let k = 0; k < count; k++) clUData[live.length * 2 + start + k] = c.particles[k]!
        start += count
      })
      q.writeBuffer(this.clF, 0, clFData)
      q.writeBuffer(this.clU, 0, clUData)
    }

    // Terrain heights (small; sized in the buffer-sizing block above).
    const t = w.terrain
    if (t && t.heights.length > 0) {
      q.writeBuffer(this.terr, 0, t.heights.buffer, t.heights.byteOffset, t.heights.length * 4)
    }

    // Member union AABB for the contact early-out.
    let ax0 = Infinity
    let ay0 = Infinity
    let ax1 = -Infinity
    let ay1 = -Infinity
    let maxReach = 0
    for (let i = 0; i < dn; i++) {
      if (d.slots.alive[i] !== 1 || d.rest[i]! <= 1e-6) continue
      const ia = d.a[i]!
      const ib = d.b[i]!
      ax0 = Math.min(ax0, p.posX[ia]!, p.posX[ib]!)
      ax1 = Math.max(ax1, p.posX[ia]!, p.posX[ib]!)
      ay0 = Math.min(ay0, p.posY[ia]!, p.posY[ib]!)
      ay1 = Math.max(ay1, p.posY[ia]!, p.posY[ib]!)
      const mat = MATERIALS[MATERIAL_IDS[d.material[i]!] ?? 'wood']
      maxReach = Math.max(maxReach, Math.max(mat.section * 0.5, w.fluid.spacing * 0.75))
    }
    // Members move within the frame; pad generously (one full support radius).
    const pad = maxReach + w.fluid.spacing * 4
    const hasMembers = ax0 < Infinity

    // Uniforms (kernel constants exactly as fluid.ts computes them).
    const f = w.fluid
    f.calibrate()
    const spacing = f.spacing
    const hK = spacing * 2
    const h2 = hK * hK
    const rq = hK + spacing * f.supportMargin
    const poly6 = 4 / (Math.PI * Math.pow(hK, 8))
    const dq = h2 - 0.04 * h2
    const sCorrDenom = poly6 * dq * dq * dq

    this.pendingUniforms = {
      n,
      cap: this.cap,
      tableSize: this.tableSize,
      gridW: gridGeom.gridW,
      gridH: gridGeom.gridH,
      gridX0: gridGeom.gridX0,
      gridY0: gridGeom.gridY0,
      distHW: dn,
      bendHW: bn,
      clusterCount: live.length,
      fluidIters: f.iterations,
      waveOn: w.waveDrive ? 1 : 0,
      terrCount: t ? t.heights.length : 0,
      distCap: this.distCap,
      bendCap: this.bendCap,
      gravity: w.gravity,
      cellSize: gridGeom.cell,
      invCell: 1 / gridGeom.cell,
      spacing,
      hK,
      h2,
      rq,
      rq2: rq * rq,
      poly6,
      spiky: -30 / (Math.PI * Math.pow(hK, 5)),
      pmass: f.particleMass,
      restDensity: f.restDensity,
      invRho0: 1 / f.restDensity,
      selfRho: poly6 * h2 * h2 * h2 * f.particleMass,
      sCorrK: f.surfaceTensionK,
      invSCorrDenom: sCorrDenom > 1e-20 ? 1 / sCorrDenom : 0,
      eps: f.relaxation,
      maxC: f.maxDensityError,
      hullPressure: f.hullPressureFactor,
      xsphVisc: f.viscosity,
      hullC0: f.hullViscosity,
      hullC1: f.hullViscosityRate,
      hullCMax: f.hullViscosityMax,
      maxSpeed: w.maxSpeed,
      restitution: w.groundRestitution,
      contactRelax: w.contactRelaxation,
      normalDamping: w.contactNormalDamping,
      maxContactCorr: w.maxContactCorrection,
      maxTerrainPush: w.maxTerrainPush,
      boundsX0: w.boundsX0,
      boundsX1: w.boundsX1,
      terrX0: t ? t.x0 : 0,
      terrInvDx: t ? 1 / t.spacing : 1,
      terrDx: t ? t.spacing : 1,
      waveX0: w.waveDrive?.x0 ?? 0,
      wavePush: w.waveDrive?.push ?? 0,
      waveBlend: w.waveDrive?.blend ?? 0,
      fpScale: FP_SCALE,
      invFp: 1 / FP_SCALE,
      waterDensity: f.waterDensity,
      memAabbX0: hasMembers ? ax0 - pad : 1e9,
      memAabbY0: hasMembers ? ay0 - pad : 1e9,
      memAabbX1: hasMembers ? ax1 + pad : -1e9,
      memAabbY1: hasMembers ? ay1 + pad : -1e9,
      viscEverySub: f.viscosityEverySubstep ? 1 : 0,
      // dt/h/substep-dependent values are stamped in step().
      dt: 0,
      h: 0,
      maxCorrSub: 0,
      keepNode: 1,
      keepFluid: 1,
      fricSolid: 1,
      fricFluid: 1,
    }
  }

  // ------------------------------------------------------------------- step

  step(dt: number): void {
    const w = this.w
    const n = this.frameN
    const values = this.pendingUniforms
    if (!values) throw new Error('GpuSolver.step without sync')
    this.pendingUniforms = null

    const substeps = w.substeps
    const h = dt / substeps
    values.dt = dt
    values.h = h
    values.maxCorrSub = w.fluid.maxCorrectionSpeed * h
    values.reachSlope = (w.fluid.spacing * w.fluid.supportMargin) / substeps
    values.keepNode = w.linearDamping > 0 ? Math.exp(-w.linearDamping * h) : 1
    values.keepFluid = w.fluidDamping > 0 ? Math.exp(-w.fluidDamping * h) : 1
    values.fricSolid = Math.pow(w.groundFriction, 1 / substeps)
    values.fricFluid = Math.pow(w.fluidBedFriction, 1 / substeps)
    writeUniforms(this.uniData, values)
    this.device.queue.writeBuffer(this.uni, 0, this.uniData)

    const enc = this.device.createCommandEncoder()
    const pass = enc.beginComputePass()
    /** Dispatch `threads` invocations of kernel k at its workgroup width. */
    const run = (k: KernelName, threads: number) => {
      if (this.skipKernels?.has(k)) return
      pass.setPipeline(this.pipelines.get(k)!)
      pass.setBindGroup(0, this.bindGroups.get(k)!)
      pass.dispatchWorkgroups(Math.ceil(Math.max(threads, 1) / (WG[k] ?? 64)))
    }
    const runColors = (k: KernelName, colors: number[][], slotBase: number) => {
      for (let g = 0; g < colors.length; g++) {
        pass.setPipeline(this.pipelines.get(k)!)
        pass.setBindGroup(0, this.bindGroups.get(k)!)
        pass.setBindGroup(1, this.colorGroups.get(`${k}:${slotBase + g}`)!)
        pass.dispatchWorkgroups(Math.ceil(Math.max(colors[g]!.length, 1) / 64))
      }
    }

    if (n > 0) {
      // Frame setup: hash + census + wave.
      run('gridClear', this.tableSize)
      run('gridCount', n)
      run('gridScan1', this.tableSize)
      run('gridScan2', 1)
      run('gridScan3', this.tableSize)
      run('gridScatter', n)
      // Hull viscosity and its census only matter with dynamic solids in the
      // world; a pure flood pays nothing for them.
      const hasHulls = this.liveClusters.length > 0
      if (hasHulls) run('census', n)
      if (w.waveDrive) run('wave', n)

      const distSlotBase = 0
      const bendSlotBase = this.distColors.length

      for (let s = 0; s < substeps; s++) {
        run('predict', Math.max(n, this.frameDistHW, this.frameBendHW))
        if (this.liveClusters.length > 0) {
          pass.setPipeline(this.pipelines.get('cluster')!)
          pass.setBindGroup(0, this.bindGroups.get('cluster')!)
          pass.dispatchWorkgroups(this.liveClusters.length)
        }
        runColors('bendSolve', this.bendColors, bendSlotBase)
        runColors('distSolve', this.distColors, distSlotBase)

        for (let it = 0; it < w.fluid.iterations; it++) {
          run('packGather', n)
          run('fluidDensity', n)
          run('fluidCorrect', n)
          run('fluidApply', n)
        }

        run('contactsSolid', n)
        if (this.frameDistHW > 0) run('contactsMember', n)
        run('integrate', n)
        runColors('bendDamp', this.bendColors, bendSlotBase)
        runColors('distDamp', this.distColors, distSlotBase)
        if (hasHulls) {
          run('snapVel', n)
          run('hullFluid', n)
          run('hullSolid', n)
        }
        if (w.fluid.viscosityEverySubstep) {
          run('packXsph', n)
          run('xsph', n)
        }
      }

      if (!w.fluid.viscosityEverySubstep) {
        run('packXsph', n)
        run('xsph', n)
      }
    }
    pass.end()

    // Stage the frame's outputs for readback.
    const st = this.stag!
    const cp = (src: GPUBuffer, srcOff: number, dstOff: number, words: number) => {
      if (words > 0) enc.copyBufferToBuffer(src, srcOff, this.staging, dstOff, words * 4)
    }
    cp(this.pf, PF.posX * this.cap * 4, st.posX, n)
    cp(this.pf, PF.posY * this.cap * 4, st.posY, n)
    cp(this.pf, PF.velX * this.cap * 4, st.velX, n)
    cp(this.pf, PF.velY * this.cap * 4, st.velY, n)
    cp(this.pf, PF.density * this.cap * 4, st.density, n)
    cp(this.pf, PF.wetVX * this.cap * 4, st.wetVX, n)
    cp(this.pf, PF.wetVY * this.cap * 4, st.wetVY, n)
    cp(this.pu, PU.wet * this.cap * 4, st.wet, n)
    cp(this.df, DF.strain * this.distCap * 4, st.dStrain, this.frameDistHW)
    cp(this.bf, BF.angle * this.bendCap * 4, st.bAngle, this.frameBendHW)
    cp(this.clF, 0, st.clusterPose, this.liveClusters.length * CL_HEADER_F)

    this.device.queue.submit([enc.finish()])
  }

  // --------------------------------------------------------------- readback

  async readback(): Promise<void> {
    const w = this.w
    const p = w.particles
    const n = this.frameN
    const st = this.stag!
    await this.staging.mapAsync(GPUMapMode.READ)
    const buf = this.staging.getMappedRange()
    const f32 = (off: number, len: number) => new Float32Array(buf, off, len)
    const u32 = (off: number, len: number) => new Uint32Array(buf, off, len)

    if (n > 0) {
      p.posX.set(f32(st.posX, n))
      p.posY.set(f32(st.posY, n))
      p.velX.set(f32(st.velX, n))
      p.velY.set(f32(st.velY, n))

      // Wetness census into the fluid solver's public arrays (host buoyancy
      // gating reads them); replace wholesale so sizes always fit.
      if (w.fluid.solidWetCount.length < n) {
        w.fluid.solidWetCount = new Int32Array(this.cap)
        w.fluid.solidFluidVX = new Float32Array(this.cap)
        w.fluid.solidFluidVY = new Float32Array(this.cap)
      }
      w.fluid.solidWetCount.set(u32(st.wet, n))
      w.fluid.solidFluidVX.set(f32(st.wetVX, n))
      w.fluid.solidFluidVY.set(f32(st.wetVY, n))

      // Fluid density, compacted in ascending particle order - the same live
      // order the CPU solver's `indices` uses, so probes read it unchanged.
      if (w.fluid.density.length < n) w.fluid.density = new Float32Array(this.cap)
      const dens = f32(st.density, n)
      let live = 0
      for (let i = 0; i < n; i++) {
        if (p.slots.alive[i] === 1 && p.kind[i] === KIND_FLUID) {
          w.fluid.density[live++] = dens[i]!
        }
      }
      w.fluid.liveCount = live
    }

    if (this.frameDistHW > 0) w.distance.strain.set(f32(st.dStrain, this.frameDistHW))
    if (this.frameBendHW > 0) w.bend.angle.set(f32(st.bAngle, this.frameBendHW))

    const poses = f32(st.clusterPose, this.liveClusters.length * CL_HEADER_F)
    this.liveClusters.forEach((c, ci) => {
      c.cx = poses[ci * CL_HEADER_F + 2]!
      c.cy = poses[ci * CL_HEADER_F + 3]!
      c.angle = poses[ci * CL_HEADER_F + 4]!
    })

    this.staging.unmap()

    // Spawn-admission occupancy (hasFluidNear) reads the CPU hash, which the
    // CPU solver builds in beginFrame. Rebuild it here from the read-back
    // positions - same cell size, same kinds, same "last frame's hash"
    // semantics the admission guards were designed around.
    const f = w.fluid
    const rq = f.spacing * 2 + f.spacing * f.supportMargin
    f.hash.setCellSize(rq)
    f.hash.build(p, HASH_MASK)
  }

  // ------------------------------------------------------------- allocation

  private ensureCapacity(): void {
    const w = this.w
    const cap = w.particles.posX.length
    const distCap = w.distance.rest.length
    const bendCap = w.bend.restAngle.length
    let clusterParticles = 0
    let clusterCount = 0
    for (const c of w.clusters) {
      if (!c.alive) continue
      clusterCount++
      clusterParticles += c.particles.length
    }
    const clusterCap = Math.max(clusterCount * CL_HEADER_F + 3 * clusterParticles, 64)

    if (
      cap === this.cap &&
      distCap === this.distCap &&
      bendCap === this.bendCap &&
      clusterCap <= this.clusterCap &&
      clusterCount <= this.clusterStagingCap
    ) {
      return
    }

    this.cap = cap
    this.distCap = Math.max(distCap, 64)
    this.bendCap = Math.max(bendCap, 64)
    this.clusterCap = Math.max(clusterCap, this.clusterCap)
    // Staging cluster capacity GROWS (doubling) - a fixed cap here meant a
    // world with one cluster more than it re-allocated every buffer every
    // frame, forever.
    while (clusterCount > this.clusterStagingCap) this.clusterStagingCap *= 2
    this.scratchU32 = new Uint32Array(Math.max(cap, this.distCap, this.bendCap))

    const dev = this.device
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    const make = (words: number): GPUBuffer => dev.createBuffer({ size: words * 4, usage: S })

    this.uni?.destroy()
    this.uni = dev.createBuffer({
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.pf?.destroy()
    this.pf = make(PF.COUNT * cap)
    this.pu?.destroy()
    this.pu = make(PU.COUNT * cap)
    this.fx?.destroy()
    this.fx = make(FX.COUNT * cap)
    this.fc?.destroy()
    this.fc = make(FC.COUNT * cap)
    this.df?.destroy()
    this.df = make(DF.COUNT * this.distCap)
    this.du?.destroy()
    this.du = make(DU.COUNT * this.distCap)
    this.bf?.destroy()
    this.bf = make(BF.COUNT * this.bendCap)
    this.bu?.destroy()
    this.bu = make(BU.COUNT * this.bendCap)
    this.colorIdx?.destroy()
    this.colorIdx = make(this.distCap + this.bendCap)
    this.clF?.destroy()
    this.clF = make(this.clusterCap)
    this.clU?.destroy()
    this.clU = make(this.clusterCap)
    this.gath?.destroy()
    this.gath = make(cap * 4) // vec4f per particle
    this.terr?.destroy()
    this.terr = make(this.terrCap) // sized in sync's buffer-sizing block
    if (!this.matSec) {
      this.matSec = make(MATERIAL_IDS.length)
      this.matSecUpload()
    }

    this.stag = stagingLayout(cap, this.distCap, this.bendCap, this.clusterStagingCap)
    this.staging?.destroy()
    this.staging = dev.createBuffer({
      size: this.stag.totalBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })

    // Grid buffers embed a cap-sized entries region: particle-capacity growth
    // invalidates them too. ensureGrid rebuilds the bind groups.
    this.tableCap = 0
    this.ensureGrid(Math.max(this.tableSize, 4096))
  }

  /** Dense-grid buffers follow the field size and fluid resolution. */
  private ensureGrid(table: number): void {
    if (table <= this.tableCap) return
    this.tableCap = Math.max(table, Math.floor(this.tableCap * 1.5))
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    const blocks = Math.ceil(this.tableCap / 256)
    this.gridS?.destroy()
    this.gridS = this.device.createBuffer({
      // +1: the substep-counter word after the scan workspace (shaders.ts).
      size: (this.tableCap + 1 + this.cap + blocks + 1) * 4,
      usage: S,
    })
    this.gridA?.destroy()
    this.gridA = this.device.createBuffer({ size: this.tableCap * 2 * 4, usage: S })
    this.buildPipelines()
  }

  private matSecUpload(): void {
    const sections = new Float32Array(MATERIAL_IDS.length)
    MATERIAL_IDS.forEach((id, i) => (sections[i] = MATERIALS[id].section))
    this.device.queue.writeBuffer(this.matSec, 0, sections)
  }

  /** Buffers per kernel, in @binding order - must match shaders.ts. */
  private kernelBuffers(): Record<KernelName, GPUBuffer[]> {
    const {
      uni,
      pf,
      pu,
      gridS,
      gridA,
      fx,
      fc,
      df,
      du,
      bf,
      bu,
      colorIdx,
      clF,
      clU,
      terr,
      matSec,
      gath,
    } = this
    return {
      gridClear: [uni, gridA, gridS],
      packGather: [uni, pf, pu, gridS, gath],
      gridCount: [uni, pf, pu, gridA],
      gridScan1: [uni, gridS, gridA],
      gridScan2: [uni, gridS],
      gridScan3: [uni, gridS, gridA],
      gridScatter: [uni, pf, pu, gridS, gridA],
      census: [uni, pf, pu, gridS],
      wave: [uni, pf, pu],
      predict: [uni, pf, pu, df, bf, gridS],
      cluster: [uni, pf, pu, clF, clU],
      distSolve: [uni, pf, pu, df, du, colorIdx],
      distDamp: [uni, pf, pu, df, du, colorIdx],
      bendSolve: [uni, pf, pu, bf, bu, colorIdx],
      bendDamp: [uni, pf, pu, bf, bu, colorIdx],
      fluidDensity: [uni, pf, pu, gridS, gath],
      fluidCorrect: [uni, pf, pu, gridS, fx, gath],
      fluidApply: [uni, pf, pu, fx],
      contactsSolid: [uni, pf, pu, gridS],
      contactsMember: [uni, pf, pu, df, du, fx, fc, matSec],
      integrate: [uni, pf, pu, fx, fc, terr],
      snapVel: [uni, pf, pu],
      packXsph: [uni, pf, pu, gridS, gath],
      hullFluid: [uni, pf, pu, gridS],
      hullSolid: [uni, pf, pu, gridS],
      xsph: [uni, pf, pu, gridS, gath],
    }
  }

  private buildPipelines(): void {
    const dev = this.device
    const buffers = this.kernelBuffers()
    const colourKernels: KernelName[] = ['distSolve', 'distDamp', 'bendSolve', 'bendDamp']
    if (this.pipelines.size === 0) {
      // EXPLICIT layouts, not 'auto': auto drops bindings a kernel declares
      // but happens not to use, and then rejects the full bind group. The
      // binding table in kernelBuffers() is the single source of truth.
      this.colorLayout = dev.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
      })
      for (const [name, src] of Object.entries(KERNELS) as [KernelName, string][]) {
        const module = dev.createShaderModule({ code: src, label: name })
        const g0 = dev.createBindGroupLayout({
          label: name,
          entries: buffers[name].map((_, i) => ({
            binding: i,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: (i === 0 ? 'uniform' : 'storage') as GPUBufferBindingType },
          })),
        })
        this.g0Layouts.set(name, g0)
        const layout = dev.createPipelineLayout({
          bindGroupLayouts: colourKernels.includes(name) ? [g0, this.colorLayout] : [g0],
        })
        this.pipelines.set(
          name,
          dev.createComputePipeline({
            label: name,
            layout,
            compute: { module, entryPoint: 'main' },
          }),
        )
      }
    }
    // (Re)build bind groups against the current buffers.
    this.bindGroups.clear()
    this.colorGroups.clear()
    for (const [name, bufs] of Object.entries(buffers) as [KernelName, GPUBuffer[]][]) {
      this.bindGroups.set(
        name,
        dev.createBindGroup({
          label: name,
          layout: this.g0Layouts.get(name)!,
          entries: bufs.map((buffer, i) => ({ binding: i, resource: { buffer } })),
        }),
      )
    }
  }

  /** Colour-range uniforms + per-kernel bind groups for each colour slot. */
  private ensureColorMeta(ranges: { start: number; count: number }[]): void {
    const dev = this.device
    for (let slot = 0; slot < ranges.length; slot++) {
      if (!this.colorMetaBufs[slot]) {
        this.colorMetaBufs[slot] = dev.createBuffer({
          size: 16,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
      }
      const data = new Uint32Array([ranges[slot]!.start, ranges[slot]!.count, 0, 0])
      dev.queue.writeBuffer(this.colorMetaBufs[slot]!, 0, data)
    }
    const colourKernels: KernelName[] = ['distSolve', 'distDamp', 'bendSolve', 'bendDamp']
    for (const k of colourKernels) {
      for (let slot = 0; slot < ranges.length; slot++) {
        const key = `${k}:${slot}`
        if (this.colorGroups.has(key)) continue
        this.colorGroups.set(
          key,
          dev.createBindGroup({
            layout: this.colorLayout,
            entries: [{ binding: 0, resource: { buffer: this.colorMetaBufs[slot]! } }],
          }),
        )
      }
    }
  }
}
