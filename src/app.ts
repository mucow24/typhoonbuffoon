import { GameLoop } from './core/loop'
import { EditorController, type ToolName } from './editor/tools'
import {
  Conditions,
  FLOOD_MAX_M,
  WIND_MAX_KPH,
  type WaveStrength,
} from './game/conditions'
import { Session } from './game/session'
import { CameraController } from './input/cameraController'
import { allIds, claimIds, defaultLevel, migrateLevel, migrateSolution } from './model/level'
import { createRenderer, type Renderer } from './render/app'
import { Camera } from './render/camera'
import { EditorView } from './render/editorView'
import { FluidView } from './render/fluidView'
import { Scenery } from './render/scenery'
import { SimView } from './render/simView'
import { buildBeam, buildLoadTest } from './scenes/demos'
import { materialAt, type MaterialId } from './sim/materials'
import { SimWorld } from './sim/world'
import { Choice, NumberField, Panel, Slider, button, toggle } from './ui/controls'
import { DebugOverlay } from './ui/debug'
import { Field } from './world/field'

/**
 * Composition root. Owns the renderer, the fixed-timestep loop, the sim, and
 * the editor session.
 */
export class Game {
  readonly renderer: Renderer
  readonly loop: GameLoop
  readonly hud: DebugOverlay
  readonly camera: Camera
  readonly field: Field
  readonly scenery: Scenery
  readonly cameraController: CameraController
  readonly sim: SimWorld
  readonly simView: SimView
  readonly fluidView: FluidView
  readonly editorView: EditorView
  readonly conditions: Conditions
  readonly session: Session
  readonly editor: EditorController

  /** Starts paused: you build first, then run it. */
  paused = true
  private breakCount = 0
  private playButton: HTMLButtonElement | null = null
  private budgetLabel: HTMLDivElement | null = null
  private toolChoice: Choice<ToolName> | null = null

  private constructor(renderer: Renderer) {
    this.renderer = renderer
    this.camera = new Camera()
    this.field = new Field(120)
    this.scenery = new Scenery(renderer.world, renderer.background, this.field)

    this.sim = new SimWorld()
    this.sim.terrain = this.field.terrain
    this.simView = new SimView(renderer.world, this.sim)
    this.fluidView = new FluidView(renderer.world, renderer.app.renderer, this.sim)
    this.syncBounds()

    this.conditions = new Conditions(this.sim, this.field)
    this.session = new Session(defaultLevel(this.field.widthM), this.sim, this.field)

    this.cameraController = new CameraController(this.camera, renderer.app.canvas, renderer)
    this.editor = new EditorController(
      renderer.app.canvas,
      this.camera,
      renderer,
      this.session,
      this.field,
    )
    // Build tools claim the left button, so panning is middle-drag or the
    // dedicated pan tool. Space is the play/pause key, not a pan modifier.
    this.cameraController.panWithLeft = false
    this.editorView = new EditorView(renderer.world, this.session, this.editor)

    this.field.onChange(() => {
      this.sim.terrain = this.field.terrain
      this.syncBounds()
      this.session.syncWidth()
      this.session.rebuild()
    })

    this.camera.fitWidth(this.field.widthM, renderer.width)
    this.camera.y = 6
    renderer.app.renderer.on('resize', () => this.scenery.invalidate())

    window.addEventListener('keydown', (e) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT') return
      if (e.code === 'Space') {
        e.preventDefault()
        this.togglePause()
      }
    })

    this.hud = new DebugOverlay()
    const buildPanel = this.buildBuildPanel()
    const fieldPanel = this.buildFieldPanel()
    fieldPanel.below(buildPanel)
    const conditionsPanel = this.buildConditionsPanel()
    conditionsPanel.below(this.hud.panel)
    this.buildSolverPanel().below(conditionsPanel)

    this.seedStarterLevel()

    this.loop = new GameLoop({
      fixedHz: 60,
      fixedUpdate: (dt) => this.fixedUpdate(dt),
      render: (alpha) => this.render(alpha),
    })

    this.hud
      .add('frame', () => `${this.loop.stats.smoothedFrameMs.toFixed(1)} ms`)
      .add('fps', () => `${(1000 / Math.max(this.loop.stats.smoothedFrameMs, 0.001)).toFixed(0)}`)
      .add('state', () => {
        if (this.paused) return 'PAUSED'
        // The loop drops sim time rather than spiralling when a step costs
        // more than the frame budget; say so, or heavy scenes read as broken.
        return this.loop.stats.starved ? 'running (slow-mo)' : 'running'
      })
      .add('substeps', () => String(this.sim.substeps))
      .add('field', () => `${this.field.widthM.toFixed(0)} m`)
      .add('particles', () => String(this.sim.particles.count))
      .add('water', () => String(this.sim.fluidCount))
      .add('members', () => String(this.session.solution.members.length))
      .add('objects', () => String(this.sim.objectCount))
      .add('peak load', () => `${(this.peakLoad() * 100).toFixed(0)}%`)
      .add('max damage', () => `${(this.maxDamage() * 100).toFixed(0)}%`)
      .add('broken', () => String(this.breakCount))
      .add('wind', () => `${(this.sim.wind.gustFactor() * this.conditions.windKph).toFixed(0)} kph`)
  }

  static async create(): Promise<Game> {
    return new Game(await createRenderer())
  }

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
      // Light enough that a wood truss is a real option, not just steel. A
      // 12.6t house needed 400kN members; at this weight the choice is a
      // trade rather than a foregone conclusion.
      density: 150,
      label: 'house',
    })
    // Anchors underneath the house: build stilts to them, or watch it fall.
    this.session.addAnchor(x - 3, houseY - 2.25, houseId)
    this.session.addAnchor(x + 3, houseY - 2.25, houseId)
    this.session.rebuild()
  }

  peakLoad(): number {
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

  maxDamage(): number {
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

    button(panel.body, 'undo (ctrl+z)', () => this.session.undo())
    button(panel.body, 'redo (ctrl+shift+z)', () => this.session.redo())
    button(panel.body, 'clear build', () => this.session.clearBuild())

    panel.section('session')
    this.playButton = button(panel.body, 'play (space)', () => this.togglePause())
    button(panel.body, 'reset to snapshot', () => {
      this.session.reset()
      this.breakCount = 0
      this.setPaused(true)
    })

    panel.section('level')
    button(panel.body, 'save level + build', () => this.saveToDisk())
    button(panel.body, 'load level + build', () => this.loadFromDisk())
    button(panel.body, 'clear everything', () => this.session.clearAll())
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
        this.field.setWidth(v)
        this.fitView()
      },
    })
    button(panel.body, 'fit view', () => this.fitView())

    panel.section('probe scenes')
    button(panel.body, 'stilt house', () => this.loadProbe('stilts'))
    button(panel.body, 'load test', () => this.loadProbe('loadtest'))
    button(panel.body, 'palm', () => this.loadProbe('palm'))

    panel.note('middle-drag to pan, or the pan tool (5)')
    return panel
  }

  private buildConditionsPanel(): Panel {
    const panel = new Panel({ title: 'conditions', side: 'right', width: 205 })

    // Signed: the slider spans a storm blowing either way, with a detent at
    // dead calm so the middle is reachable, and a 50 kph scale that is marks
    // only - snapping to those would put 175 kph out of reach.
    const windTicks: number[] = []
    for (let v = -WIND_MAX_KPH; v <= WIND_MAX_KPH; v += 50) windTicks.push(v)

    new Slider(panel.body, {
      label: 'wind',
      min: -WIND_MAX_KPH,
      max: WIND_MAX_KPH,
      step: 5,
      ticks: windTicks,
      detents: [0],
      value: this.conditions.windKph,
      format: (v) => `${v.toFixed(0)} kph`,
      onInput: (v) => {
        this.conditions.windKph = v
      },
    })

    new Slider(panel.body, {
      label: 'flood',
      min: 0,
      max: FLOOD_MAX_M,
      step: 0.5,
      value: this.conditions.floodLevelM,
      format: (v) => `${v.toFixed(1)} m`,
      onInput: (v) => {
        this.conditions.floodLevelM = v
      },
    })

    new Choice<WaveStrength>(panel.body, {
      label: 'waves',
      value: this.conditions.waveStrength,
      options: [
        { value: 'none', label: 'none' },
        { value: 'light', label: 'light' },
        { value: 'moderate', label: 'moderate' },
        { value: 'heavy', label: 'heavy' },
        { value: 'extreme', label: 'extreme' },
      ],
      onChange: (v) => {
        this.conditions.waveStrength = v
      },
    })

    panel.section('water')
    new Slider(panel.body, {
      label: 'resolution',
      min: 0.15,
      max: 1.5,
      step: 0.05,
      value: this.sim.fluid.spacing,
      format: (v) => `${v.toFixed(2)} m`,
      onInput: (v) => {
        // Restamps live water to the new mass/radius - setting fluid.spacing
        // directly left existing particles at the old mass under a solver
        // that assumes one uniform mass.
        this.sim.setFluidSpacing(v)
      },
    })

    button(panel.body, 'dump water here', () => {
      const t = this.field.terrain
      this.sim.spawnBlock(this.camera.x, t.heightAt(this.camera.x) + 12, 10, 8)
    })
    button(panel.body, 'clear water', () => this.sim.clearFluid())
    button(panel.body, 'calm', () => {
      this.conditions.reset()
      this.sim.clearFluid()
    })
    return panel
  }

  private buildSolverPanel(): Panel {
    const panel = new Panel({ title: 'solver', side: 'right', width: 205, collapsed: true })

    new Slider(panel.body, {
      label: 'substeps',
      min: 1,
      max: 32,
      step: 1,
      value: this.sim.substeps,
      format: (v) => v.toFixed(0),
      onInput: (v) => {
        this.sim.substeps = v
      },
    })

    new Slider(panel.body, {
      label: 'global damping',
      min: 0,
      max: 2,
      step: 0.05,
      value: this.sim.linearDamping,
      format: (v) => `${v.toFixed(2)}/s`,
      onInput: (v) => {
        this.sim.linearDamping = v
      },
    })

    new Slider(panel.body, {
      label: 'fluid iters',
      min: 1,
      max: 6,
      step: 1,
      value: this.sim.fluid.iterations,
      format: (v) => v.toFixed(0),
      onInput: (v) => {
        this.sim.fluid.iterations = v
      },
    })

    toggle(panel.body, 'stress colours', this.simView.showStress, (v) => {
      this.simView.showStress = v
    })
    toggle(panel.body, 'show nodes', this.simView.showNodes, (v) => {
      this.simView.showNodes = v
    })
    return panel
  }

  // ------------------------------------------------------------------ probes

  private loadProbe(which: 'stilts' | 'loadtest' | 'palm'): void {
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

  // ------------------------------------------------------------ save / load

  private saveToDisk(): void {
    const payload = JSON.stringify(
      { level: this.session.doc, solution: this.session.solution },
      null,
      2,
    )
    const blob = new Blob([payload], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${this.session.doc.name || 'level'}.json`
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
        this.session.doc = migrateLevel(parsed.level)
        this.session.solution = migrateSolution(parsed.solution)
        // The id counter restarts at zero each page load; without claiming
        // the loaded ids, the next placed anchor collides with a saved one.
        claimIds(allIds(this.session.doc, this.session.solution))
        this.field.setWidth(this.session.doc.widthM)
        this.session.rebuild()
        this.fitView()
      } catch (err) {
        console.error('failed to load level', err)
      }
    })
    input.click()
  }

  // -------------------------------------------------------------- loop hooks

  /**
   * Pause freezes the simulation only. Rendering, the camera and every build
   * tool keep working, because building while paused is the normal way to lay
   * a structure out - gravity yanking each piece the moment you place it is not
   * a useful editor.
   */
  setPaused(paused: boolean): void {
    this.paused = paused
  }

  togglePause(): void {
    // First unpause doubles as Play: take the snapshot Reset restores to.
    if (this.paused && !this.session.running) this.session.play()
    this.paused = !this.paused
  }

  private fixedUpdate(dt: number): void {
    if (this.paused) return
    this.conditions.update(dt)
    this.sim.step(dt)
    // Every frame, before any tool can act on member records: breakage frees
    // constraint slots, and the session must forget those indices before the
    // free list recycles them into someone else's constraints.
    this.session.syncBreaks()
    if (this.sim.breakEvents.length > 0) {
      this.breakCount += this.sim.breakEvents.length
      this.sim.breakEvents.length = 0
    }
  }

  private render(_alpha: number): void {
    this.scenery.setSeverity(this.conditions.severity())
    this.scenery.update(this.camera, this.renderer.width, this.renderer.height)
    this.simView.update(this.camera, this.renderer.width, this.renderer.height)
    this.fluidView.update(this.camera, this.renderer.width, this.renderer.height)
    this.editorView.update(this.camera, this.renderer.width, this.renderer.height)
    this.renderer.app.render()
    this.hud.update()
    this.updateBudgetLabel()
    if (this.toolChoice) this.toolChoice.set(this.editor.tool)
    if (this.playButton) {
      const label = this.paused ? 'play (space)' : 'pause (space)'
      if (this.playButton.textContent !== label) this.playButton.textContent = label
    }
  }

  private updateBudgetLabel(): void {
    if (!this.budgetLabel) return
    const cost = this.session.cost()
    const remaining = this.session.doc.budget - cost
    const text = `spent $${cost.toFixed(0)} / $${this.session.doc.budget.toFixed(0)}`
    if (this.budgetLabel.textContent !== text) {
      this.budgetLabel.textContent = text
      this.budgetLabel.style.color = remaining < 0 ? '#e2483c' : 'var(--text-dim)'
    }
  }

  start(): void {
    this.loop.start()
  }

  stop(): void {
    this.loop.stop()
  }

  /** Advance the sim synchronously, for headless verification. */
  pump(steps = 1): void {
    for (let i = 0; i < steps; i++) this.loop.step()
  }
}
