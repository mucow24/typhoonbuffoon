import { Graphics } from 'pixi.js'
import { GameLoop } from './core/loop'
import { Rng } from './core/rng'
import { createRenderer, type Renderer } from './render/app'
import { DebugOverlay } from './ui/debug'

/**
 * Composition root. Owns the renderer, the fixed-timestep loop and the sim, and
 * grows as the steps in docs/PLAN.md land.
 */
export class Game {
  readonly renderer: Renderer
  readonly loop: GameLoop
  readonly hud: DebugOverlay
  readonly rng: Rng

  private readonly probe: Graphics
  private phase = 0

  private constructor(renderer: Renderer) {
    this.renderer = renderer
    this.rng = new Rng(0xf100d)
    this.hud = new DebugOverlay()

    // Step 1 placeholder: a deterministic mover, proving the loop and the
    // sim/render split before anything real is built on top.
    this.probe = new Graphics().rect(-20, -20, 40, 40).fill(0x6fd3ff)
    this.renderer.world.addChild(this.probe)

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
      .add('bodies', () => '0')
      .add('particles', () => '0')
  }

  static async create(): Promise<Game> {
    return new Game(await createRenderer())
  }

  private fixedUpdate(dt: number): void {
    this.phase += dt * 0.75
  }

  private render(_alpha: number): void {
    this.probe.position.set(
      this.renderer.width * 0.5 + Math.cos(this.phase) * 160,
      this.renderer.height * 0.5 + Math.sin(this.phase * 1.7) * 90,
    )
    this.probe.rotation = this.phase
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
