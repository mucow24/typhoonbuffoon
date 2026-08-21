import { GameLoop } from './core/loop'
import { Rng } from './core/rng'
import { CameraController } from './input/cameraController'
import { createRenderer, type Renderer } from './render/app'
import { Camera } from './render/camera'
import { Scenery } from './render/scenery'
import { DebugOverlay } from './ui/debug'
import { NumberField, Panel, button } from './ui/controls'
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

  private constructor(renderer: Renderer) {
    this.renderer = renderer
    this.rng = new Rng(0xf100d)
    this.camera = new Camera()
    this.field = new Field(120)
    this.scenery = new Scenery(renderer.world, renderer.screen, this.field)
    this.cameraController = new CameraController(this.camera, renderer.app.canvas, renderer)

    this.camera.fitWidth(this.field.widthM, renderer.width)
    this.camera.y = 6

    renderer.app.renderer.on('resize', () => {
      this.scenery.invalidate()
    })

    this.hud = new DebugOverlay()
    this.buildFieldPanel()

    this.loop = new GameLoop({
      fixedHz: 60,
      fixedUpdate: (dt) => this.fixedUpdate(dt),
      render: (alpha) => this.render(alpha),
    })

    this.hud
      .add('frame', () => `${this.loop.stats.smoothedFrameMs.toFixed(1)} ms`)
      .add('fps', () => `${(1000 / Math.max(this.loop.stats.smoothedFrameMs, 0.001)).toFixed(0)}`)
      .add('steps/frame', () => String(this.loop.stats.stepsLastFrame))
      .add('sim time', () => `${this.loop.stats.simTime.toFixed(1)} s`)
      .add('field', () => `${this.field.widthM.toFixed(0)} m`)
      .add('zoom', () => `${this.camera.zoom.toFixed(2)}x`)
      .add('bodies', () => '0')
      .add('particles', () => '0')
  }

  static async create(): Promise<Game> {
    return new Game(await createRenderer())
  }

  private buildFieldPanel(): void {
    const panel = new Panel({ title: 'field', side: 'left', width: 200 })

    new NumberField(panel.body, {
      label: 'width',
      value: this.field.widthM,
      min: 5,
      step: 5,
      suffix: 'm',
      onCommit: (v) => {
        this.field.setWidth(v)
        this.camera.fitWidth(this.field.widthM, this.renderer.width)
        this.camera.x = 0
        this.camera.y = this.field.terrain.maxHeight * 0.4
      },
    })

    button(panel.body, 'fit view', () => {
      this.camera.fitWidth(this.field.widthM, this.renderer.width)
      this.camera.x = 0
      this.camera.y = this.field.terrain.maxHeight * 0.4
    })

    panel.note('drag to pan, wheel to zoom')
  }

  private fixedUpdate(_dt: number): void {
    // Sim subsystems land here from step 3 onward.
  }

  private render(_alpha: number): void {
    this.scenery.update(this.camera, this.renderer.width, this.renderer.height)
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
