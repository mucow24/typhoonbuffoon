import { GameLoop } from './core/loop'
import { Rng } from './core/rng'
import { CameraController } from './input/cameraController'
import { createRenderer, type Renderer } from './render/app'
import { Camera } from './render/camera'
import { Scenery } from './render/scenery'
import { SimView } from './render/simView'
import { buildChain } from './scenes/demos'
import { SimWorld } from './sim/world'
import { DebugOverlay } from './ui/debug'
import { NumberField, Panel, Slider, button } from './ui/controls'
import { Field } from './world/field'

/**
 * Composition root. Owns the renderer, the fixed-timestep loop and the sim, and
 * grows as the steps in docs/PLAN.md land.
 */
export class Game {
  readonly renderer: Renderer
  readonly loop: GameLoop
  readonly hud: DebugOverlay
  readonly rng: Rng
  readonly camera: Camera
  readonly field: Field
  readonly scenery: Scenery
  readonly cameraController: CameraController
  readonly sim: SimWorld
  readonly simView: SimView

  /** Tuning state for the step 3 probe scene. */
  private probeCompliance = 1e-7
  private probeZeta = 0.9

  private constructor(renderer: Renderer) {
    this.renderer = renderer
    this.rng = new Rng(0xf100d)
    this.camera = new Camera()
    this.field = new Field(120)
    this.scenery = new Scenery(renderer.world, renderer.screen, this.field)

    this.sim = new SimWorld()
    this.sim.terrain = this.field.terrain
    this.simView = new SimView(renderer.world, this.sim)

    this.cameraController = new CameraController(this.camera, renderer.app.canvas, renderer)

    this.field.onChange(() => {
      this.sim.terrain = this.field.terrain
      this.rebuildScene()
    })

    this.camera.fitWidth(this.field.widthM, renderer.width)
    this.camera.y = 6

    renderer.app.renderer.on('resize', () => this.scenery.invalidate())

    this.hud = new DebugOverlay()
    this.buildFieldPanel()
    this.buildSimPanel()
    this.rebuildScene()

    this.loop = new GameLoop({
      fixedHz: 60,
      fixedUpdate: (dt) => this.fixedUpdate(dt),
      render: (alpha) => this.render(alpha),
    })

    this.hud
      .add('frame', () => `${this.loop.stats.smoothedFrameMs.toFixed(1)} ms`)
      .add('fps', () => `${(1000 / Math.max(this.loop.stats.smoothedFrameMs, 0.001)).toFixed(0)}`)
      .add('steps/frame', () => String(this.loop.stats.stepsLastFrame))
      .add('substeps', () => String(this.sim.substeps))
      .add('sim time', () => `${this.loop.stats.simTime.toFixed(1)} s`)
      .add('field', () => `${this.field.widthM.toFixed(0)} m`)
      .add('zoom', () => `${this.camera.zoom.toFixed(2)}x`)
      .add('particles', () => String(this.sim.particles.count))
      .add('constraints', () => String(this.sim.distance.count))
  }

  static async create(): Promise<Game> {
    return new Game(await createRenderer())
  }

  private fitView(): void {
    this.camera.fitWidth(this.field.widthM, this.renderer.width)
    this.camera.x = 0
    this.camera.y = this.field.terrain.maxHeight * 0.4
  }

  private buildFieldPanel(): void {
    const panel = new Panel({ title: 'field', side: 'left', width: 205 })

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
    panel.note('drag to pan, wheel to zoom')
  }

  private buildSimPanel(): void {
    const panel = new Panel({ title: 'solver', side: 'left', width: 205 })
    panel.root.style.top = '150px'

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

    // Compliance spans orders of magnitude, so the slider is the exponent.
    new Slider(panel.body, {
      label: 'compliance',
      min: -9,
      max: -2,
      step: 0.1,
      value: Math.log10(this.probeCompliance),
      format: (v) => `1e${v.toFixed(1)}`,
      onInput: (v) => {
        this.probeCompliance = Math.pow(10, v)
        this.sim.distance.compliance.fill(this.probeCompliance, 0, this.sim.distance.highWater)
      },
    })

    new Slider(panel.body, {
      label: 'damping zeta',
      min: 0,
      max: 2,
      step: 0.05,
      value: this.probeZeta,
      format: (v) => v.toFixed(2),
      onInput: (v) => {
        this.probeZeta = v
        this.sim.distance.zeta.fill(v, 0, this.sim.distance.highWater)
      },
    })

    button(panel.body, 'reset scene', () => this.rebuildScene())
    panel.note('step 3 probe: hanging chain')
  }

  /** Rebuild the sim from scratch. Cheap, and keeps reset honest. */
  rebuildScene(): void {
    this.sim.clear()
    const t = this.field.terrain
    buildChain(this.sim, {
      x: t.x0 + this.field.widthM * 0.18,
      y: t.maxHeight + 16,
      links: 14,
      spacing: 0.9,
      compliance: this.probeCompliance,
      zeta: this.probeZeta,
    })
  }

  private fixedUpdate(dt: number): void {
    this.sim.step(dt)
  }

  private render(_alpha: number): void {
    this.scenery.update(this.camera, this.renderer.width, this.renderer.height)
    this.simView.update(this.camera, this.renderer.width, this.renderer.height)
    this.renderer.app.render()
    this.hud.update()
  }

  start(): void {
    this.loop.start()
  }

  stop(): void {
    this.loop.stop()
  }

  /**
   * Advance the sim synchronously, without waiting on requestAnimationFrame.
   * This is how the sim gets verified numerically (does the chain sag, does the
   * wood float, does the beam break) rather than by squinting at pixels.
   */
  pump(steps = 1): void {
    for (let i = 0; i < steps; i++) this.loop.step()
  }
}
