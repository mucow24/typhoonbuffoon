import { EditorController, type EditorGateway, type ToolName } from './editor/tools'
import {
  emptyStructureVM,
  nodePosition,
  pickAnchor,
  pickMember,
  pickNode,
  pickObject,
} from './editor/viewModel'
import { FLOOD_MAX_M, WIND_MAX_KPH, type WaveStrength } from './game/conditions'
import { WATER_FLOW_MAX } from './game/waterEmitter'
import { CameraController } from './input/cameraController'
import { createRenderer, type Renderer } from './render/app'
import { Camera } from './render/camera'
import { EditorView } from './render/editorView'
import { FluidView } from './render/fluidView'
import { Scenery } from './render/scenery'
import { SimView } from './render/simView'
import type { SimClient } from './runtime/client'
import type { ProbeName, SnapshotScalars } from './runtime/protocol'
import type { MaterialId } from './sim/materials'
import { Choice, NumberField, Panel, Slider, button, toggle } from './ui/controls'
import { DebugOverlay } from './ui/debug'
import { Field } from './world/field'

/**
 * Panel initial values. The worker's classes are authoritative (Conditions,
 * WaterEmitter, SimWorld defaults); these mirror them so the sliders start
 * where the sim starts. Sliders are write-only - they send commands and never
 * read back.
 */
const DEFAULTS = {
  windKph: 0,
  floodLevelM: 0,
  waveStrength: 'none' as WaveStrength,
  fluidSpacing: 0.25,
  flow: 15,
  substeps: 12,
  linearDamping: 0.35,
  fluidIterations: 1,
  widthM: 120,
}

/**
 * Main-thread composition root: renderer, camera, UI, editor - and a
 * SimClient where the sim used to be. Everything the frame draws comes from
 * the latest snapshot; everything the user does becomes a command. The sim
 * itself lives in the worker (runtime/host.ts) and cannot jank the camera no
 * matter how heavy the water gets.
 */
export class Game {
  readonly renderer: Renderer
  readonly hud: DebugOverlay
  readonly camera: Camera
  /**
   * Main-thread terrain mirror. Terrain derives deterministically from the
   * field width (seeded generator), so both sides regenerate the same ground
   * from the widthM scalar instead of shipping height arrays.
   */
  readonly field: Field
  readonly scenery: Scenery
  readonly cameraController: CameraController
  readonly simView: SimView
  readonly fluidView: FluidView
  readonly editorView: EditorView
  readonly editor: EditorController

  private rafHandle = 0
  private running = false
  private lastTime = 0
  private lastDraw = 0
  private smoothedFrameMs = 0
  private playButton: HTMLButtonElement | null = null
  private budgetLabel: HTMLDivElement | null = null
  private toolChoice: Choice<ToolName> | null = null

  private constructor(renderer: Renderer, readonly client: SimClient) {
    this.renderer = renderer
    this.camera = new Camera()
    this.field = new Field(DEFAULTS.widthM)
    this.scenery = new Scenery(renderer.world, renderer.background, this.field)

    this.simView = new SimView(renderer.world, client)
    this.fluidView = new FluidView(renderer.world, renderer.app.renderer, client)

    this.cameraController = new CameraController(this.camera, renderer.app.canvas, renderer)
    this.editor = new EditorController(
      renderer.app.canvas,
      this.camera,
      renderer,
      this.buildGateway(),
    )
    // Build tools claim the left button, so panning is middle-drag or the
    // dedicated pan tool. Space is the play/pause key, not a pan modifier.
    this.cameraController.panWithLeft = false
    this.editorView = new EditorView(renderer.world, client, this.editor)

    this.camera.fitWidth(this.field.widthM, renderer.width)
    this.camera.y = 6
    renderer.app.renderer.on('resize', () => this.scenery.invalidate())

    window.addEventListener('keydown', (e) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT') return
      if (e.code === 'Space') {
        e.preventDefault()
        this.client.send({ type: 'togglePause' })
      }
    })

    this.hud = new DebugOverlay()
    const buildPanel = this.buildBuildPanel()
    const fieldPanel = this.buildFieldPanel()
    fieldPanel.below(buildPanel)
    const conditionsPanel = this.buildConditionsPanel()
    conditionsPanel.below(this.hud.panel)
    this.buildSolverPanel().below(conditionsPanel)

    const s = () => this.client.latest?.scalars ?? null
    this.hud
      .add('render', () => `${(1000 / Math.max(this.smoothedFrameMs, 0.001)).toFixed(0)} fps`)
      .add('sim rate', () => {
        const sc = s()
        if (!sc || sc.paused) return '—'
        // Render fps and sim rate are DIFFERENT clocks: 140 fps rendering of
        // a 35 Hz sim is smooth slow motion, not a healthy sim.
        return `${sc.simFps.toFixed(0)} / 60 Hz`
      })
      .add('sim step', () => `${(s()?.stepMs ?? 0).toFixed(1)} ms`)
      // WHERE the step goes. "wait" is the GPU frame-queue backpressure -
      // when it dominates, the GPU (or whatever else is using it, like an
      // uncapped renderer) is the bottleneck, not the host code.
      .add('step split', () => {
        const sc = s()
        if (!sc) return '—'
        return `wait ${sc.reapMs.toFixed(1)} · game ${sc.condMs.toFixed(1)} · sim ${sc.simMs.toFixed(1)}`
      })
      .add('backend', () => s()?.backend ?? 'starting')
      .add('state', () => {
        const sc = s()
        if (!sc || sc.paused) return 'PAUSED'
        // The worker drops sim time rather than spiralling when a step costs
        // more than the frame budget; say by how much, or heavy scenes read
        // as broken instead of as slow motion.
        if (!sc.starved) return 'running'
        const factor = Math.min(sc.simFps / 60, 1)
        return `running (slow-mo ${factor.toFixed(2)}x)`
      })
      .add('substeps', () => String(s()?.substeps ?? DEFAULTS.substeps))
      .add('field', () => `${(s()?.widthM ?? this.field.widthM).toFixed(0)} m`)
      .add('particles', () => String(s()?.particleCount ?? 0))
      .add('water', () => String(s()?.fluidCount ?? 0))
      .add('members', () => String(s()?.memberCount ?? 0))
      .add('objects', () => String(s()?.objectCount ?? 0))
      .add('peak load', () => `${((s()?.peakLoad ?? 0) * 100).toFixed(0)}%`)
      .add('max damage', () => `${((s()?.maxDamage ?? 0) * 100).toFixed(0)}%`)
      .add('broken', () => String(s()?.breakCount ?? 0))
      .add('wind', () => `${(s()?.windGustKph ?? 0).toFixed(0)} kph`)
  }

  static async create(client: SimClient): Promise<Game> {
    return new Game(await createRenderer(), client)
  }

  // ----------------------------------------------------------------- gateway

  /** The editor's synchronous world-view: picks answer from the latest
   *  snapshot, mutations become commands. */
  private buildGateway(): EditorGateway {
    const vm = () => this.client.latest?.structure ?? emptyStructureVM()
    const send = this.clientSend
    return {
      pickNode: (x, y, r) => pickNode(vm(), x, y, r),
      pickMember: (x, y, r) => pickMember(vm(), x, y, r),
      pickAnchor: (x, y, r) => pickAnchor(vm(), x, y, r),
      pickObject: (x, y) => pickObject(vm(), x, y),
      nodePosition: (id) => nodePosition(vm(), id),
      groundHeight: (x) => this.field.terrain.heightAt(x),
      buildMember: (fromNode, from, toNode, to, material) =>
        send({
          type: 'buildMember',
          fromNode,
          fromX: from.x,
          fromY: from.y,
          toNode,
          toX: to.x,
          toY: to.y,
          material,
        }),
      addAnchor: (x, y, attachedTo) => send({ type: 'addAnchor', x, y, attachedTo }),
      addObject: (x, y, width, height, density) =>
        send({ type: 'addObject', x, y, width, height, density }),
      removeMember: (id) => send({ type: 'removeMember', id }),
      removeAnchor: (id) => send({ type: 'removeAnchor', id }),
      removeObject: (id) => send({ type: 'removeObject', id }),
      undo: () => send({ type: 'undo' }),
      redo: () => send({ type: 'redo' }),
      splash: (x, y) => send({ type: 'splash', x, y }),
      setStream: (x, y) => send({ type: 'setStream', x, y }),
      clearStream: () => send({ type: 'clearStream' }),
    }
  }

  private clientSend = (cmd: Parameters<SimClient['send']>[0]): void => {
    this.client.send(cmd)
  }

  private fitView(): void {
    this.camera.fitWidth(this.field.widthM, this.renderer.width)
    this.camera.x = 0
    this.camera.y = this.field.terrain.maxHeight * 0.4
  }

  // ------------------------------------------------------------------ panels

  private buildBuildPanel(): Panel {
    const panel = new Panel({ title: 'build', side: 'left', width: 215 })

    this.toolChoice = new Choice<ToolName>(panel.body, {
      label: 'tool',
      value: this.editor.tool,
      options: [
        { value: 'build', label: 'member (1)' },
        { value: 'anchor', label: 'anchor (2)' },
        { value: 'object', label: 'object (3)' },
        { value: 'delete', label: 'delete (4)' },
        { value: 'pan', label: 'pan (5)' },
        { value: 'water', label: 'water (6)' },
      ],
      onChange: (v) => {
        this.editor.tool = v
      },
    })

    new Choice<MaterialId>(panel.body, {
      label: 'material',
      value: this.editor.material,
      options: [
        { value: 'wood', label: 'wood (q)' },
        { value: 'steel', label: 'steel (w)' },
      ],
      onChange: (v) => {
        this.editor.material = v
      },
    })

    new Slider(panel.body, {
      label: 'grid snap',
      min: 0,
      max: 4,
      step: 0.25,
      value: this.editor.gridSnap,
      format: (v) => (v === 0 ? 'off' : `${v.toFixed(2)} m`),
      onInput: (v) => {
        this.editor.gridSnap = v
      },
    })

    this.budgetLabel = panel.note('')

    button(panel.body, 'undo (ctrl+z)', () => this.client.send({ type: 'undo' }))
    button(panel.body, 'redo (ctrl+shift+z)', () => this.client.send({ type: 'redo' }))
    button(panel.body, 'clear build', () => this.client.send({ type: 'clearBuild' }))

    panel.section('session')
    this.playButton = button(panel.body, 'play (space)', () =>
      this.client.send({ type: 'togglePause' }),
    )
    button(panel.body, 'reset to snapshot', () => this.client.send({ type: 'reset' }))

    panel.section('level')
    button(panel.body, 'save level + build', () => void this.saveToDisk())
    button(panel.body, 'load level + build', () => this.loadFromDisk())
    button(panel.body, 'clear everything', () => this.client.send({ type: 'clearAll' }))
    return panel
  }

  private buildFieldPanel(): Panel {
    const panel = new Panel({ title: 'field', side: 'left', width: 215, collapsed: true })

    new NumberField(panel.body, {
      label: 'width',
      value: this.field.widthM,
      min: 5,
      step: 5,
      suffix: 'm',
      onCommit: (v) => {
        // The mirror updates when the widthM scalar comes back - one code
        // path whether the width changed here or through a loaded level.
        this.client.send({ type: 'setFieldWidth', widthM: v })
      },
    })
    button(panel.body, 'fit view', () => this.fitView())

    panel.section('probe scenes')
    const probe = (p: ProbeName) => this.client.send({ type: 'loadProbe', probe: p })
    button(panel.body, 'stilt house', () => probe('stilts'))
    button(panel.body, 'load test', () => probe('loadtest'))
    button(panel.body, 'palm', () => probe('palm'))

    panel.note('middle-drag to pan, or the pan tool (5)')
    return panel
  }

  private buildConditionsPanel(): Panel {
    const panel = new Panel({ title: 'conditions', side: 'right', width: 205 })

    new Slider(panel.body, {
      label: 'wind',
      min: 0,
      max: WIND_MAX_KPH,
      step: 5,
      value: DEFAULTS.windKph,
      format: (v) => `${v.toFixed(0)} kph`,
      onInput: (v) => this.client.send({ type: 'setWind', kph: v }),
    })

    new Slider(panel.body, {
      label: 'flood',
      min: 0,
      max: FLOOD_MAX_M,
      step: 0.5,
      value: DEFAULTS.floodLevelM,
      format: (v) => `${v.toFixed(1)} m`,
      onInput: (v) => this.client.send({ type: 'setFlood', level: v }),
    })

    new Choice<WaveStrength>(panel.body, {
      label: 'waves',
      value: DEFAULTS.waveStrength,
      options: [
        { value: 'none', label: 'none' },
        { value: 'light', label: 'light' },
        { value: 'moderate', label: 'moderate' },
        { value: 'heavy', label: 'heavy' },
        { value: 'extreme', label: 'extreme' },
      ],
      onChange: (v) => this.client.send({ type: 'setWaves', strength: v }),
    })

    panel.section('water')
    new Slider(panel.body, {
      label: 'resolution',
      min: 0.15,
      max: 1.5,
      step: 0.05,
      value: DEFAULTS.fluidSpacing,
      format: (v) => `${v.toFixed(2)} m`,
      onInput: (v) => this.client.send({ type: 'setFluidSpacing', spacing: v }),
    })

    new Slider(panel.body, {
      label: 'tool flow',
      min: 1,
      max: WATER_FLOW_MAX,
      step: 1,
      value: DEFAULTS.flow,
      format: (v) => `${v.toFixed(0)} m²/s`,
      onInput: (v) => this.client.send({ type: 'setFlow', flow: v }),
    })
    panel.note('water tool (6): click splashes, hold pours')

    button(panel.body, 'clear water', () => this.client.send({ type: 'clearFluid' }))
    button(panel.body, 'calm', () => this.client.send({ type: 'calm' }))
    return panel
  }

  private buildSolverPanel(): Panel {
    const panel = new Panel({ title: 'solver', side: 'right', width: 205, collapsed: true })

    new Choice<'webgpu' | 'cpu'>(panel.body, {
      label: 'backend',
      value: 'webgpu',
      options: [
        { value: 'webgpu', label: 'webgpu' },
        { value: 'cpu', label: 'cpu (reference)' },
      ],
      onChange: (v) => this.client.send({ type: 'setBackend', backend: v }),
    })

    new Slider(panel.body, {
      label: 'substeps',
      min: 1,
      max: 32,
      step: 1,
      value: DEFAULTS.substeps,
      format: (v) => v.toFixed(0),
      onInput: (v) => this.client.send({ type: 'setSubsteps', substeps: v }),
    })

    new Slider(panel.body, {
      label: 'global damping',
      min: 0,
      max: 2,
      step: 0.05,
      value: DEFAULTS.linearDamping,
      format: (v) => `${v.toFixed(2)}/s`,
      onInput: (v) => this.client.send({ type: 'setLinearDamping', value: v }),
    })

    new Slider(panel.body, {
      label: 'fluid iters',
      min: 1,
      max: 6,
      step: 1,
      value: DEFAULTS.fluidIterations,
      format: (v) => v.toFixed(0),
      onInput: (v) => this.client.send({ type: 'setFluidIterations', iterations: v }),
    })

    toggle(panel.body, 'stress colours', this.simView.showStress, (v) => {
      this.simView.showStress = v
    })
    toggle(panel.body, 'show nodes', this.simView.showNodes, (v) => {
      this.simView.showNodes = v
    })
    return panel
  }

  // ------------------------------------------------------------ save / load

  private async saveToDisk(): Promise<void> {
    const data = await this.client.save()
    const payload = JSON.stringify({ level: data.level, solution: data.solution }, null, 2)
    const blob = new Blob([payload], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${data.level.name || 'level'}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  private loadFromDisk(): void {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'application/json'
    input.addEventListener('change', async () => {
      const file = input.files?.[0]
      if (!file) return
      try {
        const parsed = JSON.parse(await file.text()) as { level?: unknown; solution?: unknown }
        await this.client.load(parsed.level, parsed.solution)
        this.fitView()
      } catch (err) {
        console.error('failed to load level', err)
      }
    })
    input.click()
  }

  // -------------------------------------------------------------- loop hooks

  private tick = (now: number): void => {
    if (!this.running) return
    this.rafHandle = requestAnimationFrame(this.tick)

    // Cap drawing at ~60 fps. The sim produces 60 states a second and the
    // renderer interpolates between the last two, so frames beyond that add
    // no information - but on high-refresh displays rAF runs at 120-165 Hz,
    // and every extra draw burns the SAME iGPU the WebGPU solver needs:
    // measured live, uncapped rendering stretched the solver's readback
    // fences from ~20 ms to ~46 ms and stalled the sim itself to 43 Hz.
    // The half-frame slack keeps a 60 Hz display's own rAF from beating.
    if (now - this.lastDraw < 1000 / 60 - 8) return
    this.lastDraw = now

    const frameMs = this.lastTime === 0 ? 1000 / 60 : now - this.lastTime
    this.lastTime = now
    this.smoothedFrameMs =
      this.smoothedFrameMs === 0 ? frameMs : this.smoothedFrameMs * 0.9 + frameMs * 0.1

    const scalars = this.client.latest?.scalars ?? null
    if (scalars) this.syncFromScalars(scalars)

    this.scenery.update(this.camera, this.renderer.width, this.renderer.height)
    this.simView.update(this.camera, this.renderer.width, this.renderer.height)
    this.fluidView.update(this.camera, this.renderer.width, this.renderer.height)
    this.editorView.update(this.camera, this.renderer.width, this.renderer.height)
    this.renderer.app.render()
    this.hud.update()

    if (this.toolChoice) this.toolChoice.set(this.editor.tool)
  }

  private syncFromScalars(scalars: SnapshotScalars): void {
    // The width scalar is authoritative: a typed width and a loaded level
    // both come back through here, regenerating the same deterministic
    // terrain the worker regenerated.
    if (scalars.widthM !== this.field.widthM) {
      this.field.setWidth(scalars.widthM)
      this.fitView()
    }

    this.scenery.setSeverity(scalars.severity)

    if (this.playButton) {
      const label = scalars.paused ? 'play (space)' : 'pause (space)'
      if (this.playButton.textContent !== label) this.playButton.textContent = label
    }
    if (this.budgetLabel) {
      const remaining = scalars.budget - scalars.cost
      const text = `spent $${scalars.cost.toFixed(0)} / $${scalars.budget.toFixed(0)}`
      if (this.budgetLabel.textContent !== text) {
        this.budgetLabel.textContent = text
        this.budgetLabel.style.color = remaining < 0 ? '#e2483c' : 'var(--text-dim)'
      }
    }
  }

  start(): void {
    if (this.running) return
    this.running = true
    this.lastTime = 0
    this.rafHandle = requestAnimationFrame(this.tick)
  }

  stop(): void {
    this.running = false
    if (this.rafHandle) cancelAnimationFrame(this.rafHandle)
    this.rafHandle = 0
  }

  /** Advance the sim synchronously in the worker, for headless verification. */
  pump(steps = 1): Promise<void> {
    return this.client.pump(steps)
  }
}
