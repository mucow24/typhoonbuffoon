import { FixedStepper } from '../core/stepper'
import { Conditions } from '../game/conditions'
import { Session } from '../game/session'
import { WaterEmitter } from '../game/waterEmitter'
import { allIds, claimIds, defaultLevel, migrateLevel, migrateSolution } from '../model/level'
import { buildBeam, buildLoadTest } from '../scenes/demos'
import { GpuSolver } from '../sim/gpu/gpuSolver'
import { materialAt } from '../sim/materials'
import { CpuSolver } from '../sim/solver'
import { SimWorld } from '../sim/world'
import { Field } from '../world/field'
import { encodeSnapshot } from './snapshot'
import type {
  BreakEventVM,
  ClusterVM,
  Command,
  HostMessage,
  ProbeName,
  SnapshotScalars,
  StructureVM,
} from './protocol'

/**
 * The sim side of the worker boundary: owns the world, the game layer, and
 * the fixed-timestep clock, applies commands, and emits snapshots.
 *
 * A plain class, deliberately free of Worker APIs - the shell (worker.ts)
 * wires `self` to it, Node tests wire a capture function, and the loopback
 * tests wire it straight to a SimClient. Everything sim-coupled lives on this
 * side: Session's id->index bookkeeping stays in the same thread that
 * recycles the indices (the drainDestroyed contract), Conditions' seeded
 * admission RNG ticks in lockstep with the steps, and the paused water
 * splash lands on the sim's own recent-spawn guard.
 */
export class SimHost {
  readonly field: Field
  readonly sim: SimWorld
  readonly session: Session
  readonly conditions: Conditions
  readonly emitter = new WaterEmitter()
  private readonly stepper: FixedStepper

  /** Starts paused: you build first, then run it. */
  private paused = true
  private breakCount = 0
  private stepMs = 0
  private simFps = 0
  private rateLastNow = 0
  private rateLastSteps = 0
  private backendName = 'cpu'
  /**
   * Commands queue and drain only at frame boundaries: the GPU backend's
   * step is async, so worker messages can arrive MID-STEP - and a topology
   * edit landing between substeps would corrupt the frame.
   */
  private readonly commandQueue: Command[] = []
  private ticking = false
  /** Acks held until AFTER the snapshot reflecting their command has been
   *  posted, so `await pump()` then reading client.latest sees the result. */
  private pendingAcks: number[] = []
  /** Held-cursor position while the water tool streams, or null. */
  private stream: { x: number; y: number } | null = null
  private pendingEvents: BreakEventVM[] = []
  /** Something changed while paused; the next frame must emit a snapshot. */
  private dirty = true
  private readonly bufferPool: ArrayBuffer[] = []

  constructor(
    private readonly post: (msg: HostMessage, transfer?: Transferable[]) => void,
    opts: { widthM?: number; seedStarter?: boolean } = {},
  ) {
    this.field = new Field(opts.widthM ?? 120)
    this.sim = new SimWorld()
    this.sim.terrain = this.field.terrain
    this.syncBounds()
    this.conditions = new Conditions(this.sim, this.field)
    this.session = new Session(defaultLevel(this.field.widthM), this.sim, this.field)

    this.field.onChange(() => {
      this.sim.terrain = this.field.terrain
      this.syncBounds()
      this.session.syncWidth()
      this.session.rebuild()
      this.dirty = true
    })

    if (opts.seedStarter !== false) this.seedStarterLevel()

    this.stepper = new FixedStepper({
      fixedHz: 60,
      fixedUpdate: (dt) => this.fixedUpdate(dt),
      frameEnd: () => this.frameEnd(),
    })
  }

  /**
   * Construct a host and pick the solver backend: WebGPU when a device
   * exists (the 10x path - docs/GPU_PLAN.md), the CPU reference otherwise.
   */
  static async create(
    post: (msg: HostMessage, transfer?: Transferable[]) => void,
    opts: { widthM?: number; seedStarter?: boolean; backend?: 'auto' | 'cpu' } = {},
  ): Promise<SimHost> {
    const host = new SimHost(post, opts)
    if (opts.backend !== 'cpu') {
      const gpu = await GpuSolver.create(host.sim)
      if (gpu) {
        host.sim.solver = gpu
        host.backendName = 'webgpu'
      } else {
        host.backendName = 'cpu (no webgpu)'
      }
    }
    return host
  }

  /** Reset frame timing; call once before the first tick, with the same clock. */
  start(now: number): void {
    this.stepper.beginFrames(now)
  }

  /** Queue a command for the next frame boundary (the worker's entry point). */
  enqueue(cmd: Command): void {
    this.commandQueue.push(cmd)
  }

  /** Advance the fixed-step clock to wall time `now` (ms). */
  async tick(now: number): Promise<void> {
    if (this.ticking) return
    this.ticking = true
    this.tickNow = now
    try {
      // Land whatever GPU frames have COMPLETED - never waiting on one that
      // hasn't (reap): real-browser fences take ~17-21 ms to observe, so a
      // loop that waits for the current frame caps below 60 Hz at idle.
      // Host-created particles are write-stamped, so frames still in flight
      // cannot clobber anything commands create next.
      await this.reapSolver()
      await this.drainCommands()
      await this.stepper.advanceAsync(now)
    } finally {
      this.ticking = false
    }
  }

  private reapSolver(): Promise<void> {
    const solver = this.sim.solver
    return solver.reap ? solver.reap() : solver.readback()
  }

  private tickNow = 0

  /**
   * Achieved sim steps per wall second - the slow-mo readout. Measured over
   * fixed ~250 ms windows of wall time, NOT as an average of per-tick
   * instantaneous rates: ticks that run steps systematically follow longer
   * gaps, so the per-tick average under-reads (a healthy 60 Hz sim showed
   * 46 on the HUD - a lying gauge on exactly the number this row exists to
   * make trustworthy).
   */
  private trackSimRate(now: number): void {
    if (this.paused) {
      this.simFps = 0
      this.rateLastNow = 0
      return
    }
    const steps = this.stepper.stats.totalSteps
    if (this.rateLastNow === 0) {
      this.rateLastNow = now
      this.rateLastSteps = steps
      return
    }
    const elapsed = now - this.rateLastNow
    if (elapsed < 250) return
    const windowed = ((steps - this.rateLastSteps) * 1000) / elapsed
    this.simFps = this.simFps === 0 ? windowed : this.simFps * 0.7 + windowed * 0.3
    this.rateLastNow = now
    this.rateLastSteps = steps
  }

  private async drainCommands(): Promise<void> {
    while (this.commandQueue.length > 0) {
      const cmd = this.commandQueue.shift()!
      try {
        await this.applyCommand(cmd)
      } catch (err) {
        // A failed command must never hang a client promise or kill the
        // drain: nack it (with its request id when it has one) and keep
        // going. The snapshot after this drain shows whatever state stands.
        const requestId = 'requestId' in cmd ? cmd.requestId : null
        this.post({ type: 'nack', requestId, error: String(err) })
        this.dirty = true
      }
    }
  }

  // ---------------------------------------------------------------- stepping

  /**
   * One fixed step, PIPELINED: land the previous frame's GPU results first
   * (they have had a whole frame to complete, so this await is ~free), then
   * mutate on that consistent state, then submit this frame WITHOUT waiting
   * for it. Awaiting the readback in line instead put a full GPU round trip
   * - 1 to 25 ms depending on browser/adapter path - inside every step,
   * which capped even an IDLE scene below 60 Hz on some machines. Host
   * logic (forces, damage, snapshots, spawn guards) sees positions one
   * frame old, exactly the lag the plan budgeted for. On the CPU backend
   * readback is a no-op and this is the same synchronous step as ever.
   */
  private async fixedUpdate(dt: number): Promise<void> {
    if (this.paused) return
    const t0 = performance.now()
    await this.reapSolver()
    this.conditions.update(dt)
    if (this.stream) this.emitter.update(this.sim, dt, this.stream.x, this.stream.y)
    this.sim.step(dt)
    // Every frame, before any command can act on member records: breakage
    // frees constraint slots, and the session must forget those indices
    // before the free list recycles them into someone else's constraints.
    this.session.syncBreaks()
    for (const e of this.sim.breakEvents) {
      this.pendingEvents.push({ x: e.x, y: e.y, strain: e.strain, material: e.material })
    }
    if (this.sim.breakEvents.length > 0) {
      this.breakCount += this.sim.breakEvents.length
      this.sim.breakEvents.length = 0
    }
    const cost = performance.now() - t0
    this.stepMs = this.stepMs === 0 ? cost : this.stepMs * 0.9 + cost * 0.1
  }

  private frameEnd(): void {
    // Rate first, so the snapshot leaving this frame carries THIS frame's
    // achieved sim rate (and 0 the instant a pause lands, not a frame late).
    this.trackSimRate(this.tickNow)
    if ((!this.paused && this.stepper.stats.stepsLastFrame > 0) || this.dirty) {
      this.emitSnapshot()
      this.dirty = false
    }
    if (this.pendingAcks.length > 0) {
      for (const requestId of this.pendingAcks) this.post({ type: 'ack', requestId })
      this.pendingAcks = []
    }
  }

  // ---------------------------------------------------------------- commands

  private async applyCommand(cmd: Command): Promise<void> {
    switch (cmd.type) {
      // -- session flow
      case 'togglePause':
        // First unpause doubles as Play: take the snapshot Reset restores to.
        if (this.paused && !this.session.running) this.session.play()
        this.paused = !this.paused
        break
      case 'setPaused':
        if (!this.paused || cmd.paused || this.session.running) {
          this.paused = cmd.paused
        } else {
          // Unpausing through the explicit setter still means Play.
          this.session.play()
          this.paused = false
        }
        break
      case 'reset':
        this.session.reset()
        this.breakCount = 0
        this.paused = true
        break
      case 'pump': {
        for (let i = 0; i < cmd.steps; i++) await this.stepper.stepAsync()
        this.pendingAcks.push(cmd.requestId)
        break
      }

      // -- build edits
      case 'buildMember': {
        // Snapped ids come from the main thread's snapshot-time picks; a
        // command queued ahead of this one (undo, delete) can have removed
        // them. A stale id must fall back to the gesture coordinates - not
        // mint a doc member whose endpoints do not exist.
        const fromValid = cmd.fromNode !== null && this.session.nodePosition(cmd.fromNode) !== null
        const toValid = cmd.toNode !== null && this.session.nodePosition(cmd.toNode) !== null
        const a = fromValid ? cmd.fromNode! : this.session.addNode(cmd.fromX, cmd.fromY)
        const b = toValid ? cmd.toNode! : this.session.addNode(cmd.toX, cmd.toY)
        this.session.addMember(a, b, cmd.material)
        break
      }
      case 'addAnchor':
        this.session.addAnchor(cmd.x, cmd.y, cmd.attachedTo)
        break
      case 'addObject':
        this.session.addWorldObject({
          x: cmd.x,
          y: cmd.y,
          width: cmd.width,
          height: cmd.height,
          density: cmd.density,
        })
        break
      case 'removeMember':
        this.session.removeMember(cmd.id)
        break
      case 'removeAnchor':
        this.session.removeAnchor(cmd.id)
        break
      case 'removeObject':
        this.session.removeObject(cmd.id)
        break
      case 'undo':
        this.session.undo()
        break
      case 'redo':
        this.session.redo()
        break
      case 'clearBuild':
        this.session.clearBuild()
        break
      case 'clearAll':
        this.session.clearAll()
        break

      // -- conditions
      case 'setWind':
        this.conditions.windKph = cmd.kph
        break
      case 'setFlood':
        this.conditions.floodLevelM = cmd.level
        break
      case 'setWaves':
        this.conditions.waveStrength = cmd.strength
        break
      case 'calm':
        this.conditions.reset()
        this.sim.clearFluid()
        break

      // -- water tool
      case 'splash':
        this.emitter.splash(this.sim, cmd.x, cmd.y)
        break
      case 'setStream':
        this.stream = { x: cmd.x, y: cmd.y }
        break
      case 'clearStream':
        this.stream = null
        break
      case 'setFlow':
        this.emitter.flow = cmd.flow
        break
      case 'clearFluid':
        this.sim.clearFluid()
        break

      // -- solver tuning
      case 'setSubsteps':
        this.sim.substeps = cmd.substeps
        break
      case 'setLinearDamping':
        this.sim.linearDamping = cmd.value
        break
      case 'setFluidIterations':
        this.sim.fluid.iterations = cmd.iterations
        break
      case 'setFluidSpacing':
        this.sim.setFluidSpacing(cmd.spacing)
        break
      case 'setBackend': {
        // Both backends treat the SoA arrays as canonical, so a swap at a
        // frame boundary migrates nothing - but the outgoing backend's
        // device resources must be released promptly, and a no-op request
        // must not mint a second device.
        if (cmd.backend === 'cpu') {
          if (this.backendName === 'cpu') break
          this.sim.solver.dispose?.()
          this.sim.solver = new CpuSolver(this.sim)
          this.backendName = 'cpu'
        } else {
          if (this.backendName === 'webgpu') break
          const gpu = await GpuSolver.create(this.sim)
          if (gpu) {
            this.sim.solver.dispose?.()
            this.sim.solver = gpu
            this.backendName = 'webgpu'
          } else {
            this.backendName = 'cpu (no webgpu)'
          }
        }
        break
      }

      // -- level
      case 'setFieldWidth':
        this.field.setWidth(cmd.widthM)
        break
      case 'loadProbe':
        this.loadProbe(cmd.probe)
        break
      case 'loadDoc': {
        // Migrate BOTH before assigning EITHER: a throw mid-way (newer
        // solution version, malformed field) must leave the session's
        // doc/solution pair untouched, not half-replaced.
        const level = migrateLevel(cmd.level)
        const solution = migrateSolution(cmd.solution)
        this.session.doc = level
        this.session.solution = solution
        // The id counter restarts at zero each worker start; without claiming
        // the loaded ids, the next placed anchor collides with a saved one.
        claimIds(allIds(level, solution))
        this.field.setWidth(level.widthM)
        this.session.rebuild()
        this.pendingAcks.push(cmd.requestId)
        break
      }
      case 'requestSave':
        this.post({
          type: 'saveData',
          requestId: cmd.requestId,
          level: this.session.doc,
          solution: this.session.solution,
        })
        return // a save reads nothing and dirties nothing

      // -- plumbing
      case 'recycleBuffer':
        if (this.bufferPool.length < 4) this.bufferPool.push(cmd.buffer)
        return
    }
    this.dirty = true
  }

  // ------------------------------------------------------------------ scenes

  /** A couple of anchors and a house, so there is something to build onto. */
  private seedStarterLevel(): void {
    const t = this.field.terrain
    const x = -8
    const ground = t.heightAt(x)
    this.session.addAnchor(x - 3, t.heightAt(x - 3))
    this.session.addAnchor(x + 3, t.heightAt(x + 3))

    const houseY = ground + 9
    const houseId = this.session.addWorldObject({
      x,
      y: houseY,
      width: 8,
      height: 4.5,
      // Light enough that a wood truss is a real option, not just steel.
      density: 150,
      label: 'house',
    })
    // Anchors underneath the house: build stilts to them, or watch it fall.
    this.session.addAnchor(x - 3, houseY - 2.25, houseId)
    this.session.addAnchor(x + 3, houseY - 2.25, houseId)
    this.session.rebuild()
  }

  private loadProbe(which: ProbeName): void {
    this.session.clearAll()
    this.conditions.reset()
    this.sim.clearFluid()
    const t = this.field.terrain
    const x = -8

    if (which === 'stilts') {
      this.seedStarterLevel()
      return
    }
    if (which === 'loadtest') {
      buildLoadTest(this.sim, { x, y: t.heightAt(x) + 14, material: 'wood', tipMassKg: 8000 })
      return
    }
    buildBeam(this.sim, {
      x0: x,
      y0: t.heightAt(x),
      x1: x,
      y1: t.heightAt(x) + 13,
      material: 'wood',
      clampStart: true,
    })
  }

  // --------------------------------------------------------------- snapshots

  private emitSnapshot(): void {
    const buffer = encodeSnapshot(this.sim, this.bufferPool.pop() ?? null)
    this.post(
      {
        type: 'snapshot',
        buffer,
        scalars: this.buildScalars(),
        structure: this.buildStructureVM(),
        clusters: this.buildClusterVMs(),
        events: this.pendingEvents,
      },
      [buffer],
    )
    this.pendingEvents = []
  }

  private buildScalars(): SnapshotScalars {
    const stats = this.stepper.stats
    return {
      frame: stats.totalSteps,
      simTime: stats.simTime,
      stepsLastFrame: stats.stepsLastFrame,
      starved: stats.starved,
      paused: this.paused,
      running: this.session.running,
      particleCount: this.sim.particles.count,
      fluidCount: this.sim.fluidCount,
      objectCount: this.sim.objectCount,
      memberCount: this.session.solution.members.length,
      substeps: this.sim.substeps,
      fluidSpacing: this.sim.fluid.spacing,
      windGustKph: this.sim.wind.gustFactor() * this.conditions.windKph,
      severity: this.conditions.severity(),
      peakLoad: this.peakLoad(),
      maxDamage: this.maxDamage(),
      breakCount: this.breakCount,
      cost: this.session.cost(),
      budget: this.session.doc.budget,
      widthM: this.field.widthM,
      stepMs: this.stepMs,
      simFps: this.simFps,
      backend: this.backendName,
    }
  }

  private buildStructureVM(): StructureVM {
    const doc = this.session.doc
    const nodes: StructureVM['nodes'] = []
    const anchors: StructureVM['anchors'] = []
    for (const a of doc.anchors) {
      const pos = this.session.nodePosition(a.id) ?? a
      nodes.push({ id: a.id, x: pos.x, y: pos.y })
      anchors.push({ id: a.id, x: pos.x, y: pos.y, attachedTo: a.attachedTo })
    }
    for (const n of this.session.solution.nodes) {
      const pos = this.session.nodePosition(n.id) ?? n
      nodes.push({ id: n.id, x: pos.x, y: pos.y })
    }
    return {
      nodes,
      anchors,
      members: this.session.memberPolylines(),
      objects: doc.objects.map((o) => ({
        id: o.id,
        x: o.x,
        y: o.y,
        width: o.width,
        height: o.height,
      })),
    }
  }

  private buildClusterVMs(): ClusterVM[] {
    const out: ClusterVM[] = []
    for (const c of this.sim.clusters) {
      if (!c.alive) continue
      const { hw, hh } = c.restExtent()
      const density = c.totalMass / Math.max(4 * hw * hh, 1e-6)
      out.push({ cx: c.cx, cy: c.cy, angle: c.angle, hw, hh, light: density < 1000 })
    }
    return out
  }

  // -------------------------------------------------------------- reductions

  private peakLoad(): number {
    const d = this.sim.distance
    let peak = 0
    for (let i = 0; i < d.highWater; i++) {
      if (d.slots.alive[i] !== 1) continue
      if (d.unbreakable[i] === 1) continue // joinery: welds, mount links
      const m = materialAt(d.material[i]!)
      if (m.breakStrain <= 0) continue
      peak = Math.max(peak, Math.abs(d.strain[i]!) / m.breakStrain)
    }
    // Bending is a failure mode too - a wall carrying hydrostatic load shows
    // almost no axial strain, and the HUD reading 0% on a wall about to snap
    // is the legibility failure the genre cannot afford.
    const b = this.sim.bend
    for (let i = 0; i < b.highWater; i++) {
      if (b.slots.alive[i] !== 1) continue
      const m = materialAt(b.material[i]!)
      if (!(m.breakAngle > 0) || !Number.isFinite(m.breakAngle)) continue
      peak = Math.max(peak, Math.abs(b.angle[i]!) / m.breakAngle)
    }
    return peak
  }

  private maxDamage(): number {
    const d = this.sim.distance
    let peak = 0
    for (let i = 0; i < d.highWater; i++) {
      if (d.slots.alive[i] === 1) peak = Math.max(peak, d.damage[i]!)
    }
    return peak
  }

  private syncBounds(): void {
    this.sim.boundsX0 = this.field.left
    this.sim.boundsX1 = this.field.right
  }
}
