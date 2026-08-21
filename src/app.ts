import { GameLoop } from './core/loop'
import { Rng } from './core/rng'
import { CameraController } from './input/cameraController'
import { createRenderer, type Renderer } from './render/app'
import { Camera } from './render/camera'
import { Scenery } from './render/scenery'
import { FluidView } from './render/fluidView'
import { SimView } from './render/simView'
import { buildBeam, buildChain, buildLoadTest } from './scenes/demos'
import { materialAt, type MaterialId } from './sim/materials'
import { SimWorld } from './sim/world'
import { DebugOverlay } from './ui/debug'
import { Choice, NumberField, Panel, Slider, button } from './ui/controls'
import { Field } from './world/field'

type SceneName = 'loadtest' | 'cantilever' | 'palm' | 'chain' | 'float'

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
  readonly fluidView: FluidView

  materialId: MaterialId = 'wood'
  segments = 0 // 0 means derive from the material
  tipMassKg = 0
  sceneName: SceneName = 'loadtest'

  /** Breakages since the last scene reset, for the HUD. */
  private breakCount = 0

  private constructor(renderer: Renderer) {
    this.renderer = renderer
    this.rng = new Rng(0xf100d)
    this.camera = new Camera()
    this.field = new Field(120)
    this.scenery = new Scenery(renderer.world, renderer.screen, this.field)

    this.sim = new SimWorld()
    this.sim.terrain = this.field.terrain
    this.simView = new SimView(renderer.world, this.sim)
    this.fluidView = new FluidView(renderer.world, renderer.app.renderer, this.sim)
    this.syncBounds()

    this.cameraController = new CameraController(this.camera, renderer.app.canvas, renderer)

    this.field.onChange(() => {
      this.sim.terrain = this.field.terrain
      this.syncBounds()
      this.rebuildScene()
    })

    this.camera.fitWidth(this.field.widthM, renderer.width)
    this.camera.y = 6

    renderer.app.renderer.on('resize', () => this.scenery.invalidate())

    this.hud = new DebugOverlay()
    this.buildFieldPanel()
    this.buildSolverPanel()
    this.buildScenePanel()
    this.buildWaterPanel()
    this.rebuildScene()

    this.loop = new GameLoop({
      fixedHz: 60,
      fixedUpdate: (dt) => this.fixedUpdate(dt),
      render: (alpha) => this.render(alpha),
    })

    this.hud
      .add('frame', () => `${this.loop.stats.smoothedFrameMs.toFixed(1)} ms`)
      .add('fps', () => `${(1000 / Math.max(this.loop.stats.smoothedFrameMs, 0.001)).toFixed(0)}`)
      .add('substeps', () => String(this.sim.substeps))
      .add('sim time', () => `${this.loop.stats.simTime.toFixed(1)} s`)
      .add('field', () => `${this.field.widthM.toFixed(0)} m`)
      .add('zoom', () => `${this.camera.zoom.toFixed(2)}x`)
      .add('particles', () => String(this.sim.particles.count))
      .add('members', () => String(this.sim.distance.count))
      .add('bends', () => String(this.sim.bend.count))
      .add('peak load', () => `${(this.peakLoad() * 100).toFixed(0)}%`)
      .add('max damage', () => `${(this.maxDamage() * 100).toFixed(0)}%`)
      .add('broken', () => String(this.breakCount))
      .add('water', () => String(this.sim.fluidCount))
      .add('objects', () => String(this.sim.objectCount))
  }

  static async create(): Promise<Game> {
    return new Game(await createRenderer())
  }

  /** Highest fraction-of-break-threshold across all live members. */
  peakLoad(): number {
    const d = this.sim.distance
    let peak = 0
    for (let i = 0; i < d.highWater; i++) {
      if (d.slots.alive[i] !== 1) continue
      const m = materialAt(d.material[i]!)
      if (m.breakStrain <= 0) continue
      peak = Math.max(peak, Math.abs(d.strain[i]!) / m.breakStrain)
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

  private buildWaterPanel(): void {
    const panel = new Panel({ title: 'water', side: 'right', width: 200 })
    panel.root.style.top = '320px'

    // Resolution is a control rather than a constant: field width is unclamped
    // and flood depth is up to 20 m, so spacing is the lever that keeps the
    // particle count affordable. Deliberately not clamped - see docs/PLAN.md 8.
    new Slider(panel.body, {
      label: 'resolution',
      min: 0.1,
      max: 1.5,
      step: 0.05,
      value: this.sim.fluid.spacing,
      format: (v) => `${v.toFixed(2)} m`,
      onInput: (v) => {
        this.sim.fluid.spacing = v
      },
    })

    new Slider(panel.body, {
      label: 'viscosity',
      min: 0,
      max: 0.4,
      step: 0.01,
      value: this.sim.fluid.viscosity,
      format: (v) => v.toFixed(2),
      onInput: (v) => {
        this.sim.fluid.viscosity = v
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

    button(panel.body, 'dump water', () => {
      const t = this.field.terrain
      const x = this.camera.x
      this.sim.spawnBlock(x, t.heightAt(x) + 14, 10, 8)
    })

    button(panel.body, 'clear water', () => this.sim.clearFluid())

    panel.section('objects')
    button(panel.body, 'drop wood crate', () => {
      const t = this.field.terrain
      this.sim.addObject({
        cx: this.camera.x, cy: t.heightAt(this.camera.x) + 12,
        width: 2.4, height: 1.6, density: 500,
      })
    })
    button(panel.body, 'drop steel crate', () => {
      const t = this.field.terrain
      this.sim.addObject({
        cx: this.camera.x + 3, cy: t.heightAt(this.camera.x) + 12,
        width: 2.0, height: 1.4, density: 7850,
      })
    })
    button(panel.body, 'clear objects', () => this.sim.clearObjects())
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

  private buildSolverPanel(): void {
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
  }

  private buildScenePanel(): void {
    const panel = new Panel({ title: 'probe', side: 'left', width: 205 })
    panel.root.style.top = '272px'

    new Choice<MaterialId>(panel.body, {
      label: 'material',
      value: this.materialId,
      options: [
        { value: 'wood', label: 'wood' },
        { value: 'steel', label: 'steel' },
      ],
      onChange: (v) => {
        this.materialId = v
        this.rebuildScene()
      },
    })

    new Choice<SceneName>(panel.body, {
      label: 'scene',
      value: this.sceneName,
      options: [
        { value: 'loadtest', label: 'load test' },
        { value: 'float', label: 'float test' },
        { value: 'cantilever', label: 'cantilever' },
        { value: 'palm', label: 'palm' },
        { value: 'chain', label: 'chain' },
      ],
      onChange: (v) => {
        this.sceneName = v
        this.rebuildScene()
      },
    })

    new Slider(panel.body, {
      label: 'segments',
      min: 0,
      max: 12,
      step: 1,
      value: this.segments,
      format: (v) => (v === 0 ? 'auto' : v.toFixed(0)),
      onInput: (v) => {
        this.segments = v
        this.rebuildScene()
      },
    })

    new Slider(panel.body, {
      label: 'tip load',
      min: 0,
      max: 60000,
      step: 500,
      value: this.tipMassKg,
      format: (v) => `${(v / 1000).toFixed(1)} t`,
      onInput: (v) => {
        this.tipMassKg = v
        this.rebuildScene()
      },
    })

    button(panel.body, 'reset scene', () => this.rebuildScene())
    panel.note('material constants live in sim/materials.ts')
  }

  /** Rebuild the sim from scratch. Cheap, and keeps reset honest. */
  rebuildScene(): void {
    this.sim.clear()
    this.breakCount = 0

    const t = this.field.terrain
    const x = t.x0 + this.field.widthM * 0.22
    const segments = this.segments > 0 ? this.segments : undefined

    switch (this.sceneName) {
      case 'chain':
        buildChain(this.sim, { x, y: t.maxHeight + 16, links: 14, spacing: 0.9 })
        break

      case 'cantilever': {
        const y = t.heightAt(x) + 14
        buildBeam(this.sim, {
          x0: x,
          y0: y,
          x1: x + 12,
          y1: y,
          material: this.materialId,
          segments,
          clampStart: true,
        })
        break
      }

      case 'palm': {
        const y = t.heightAt(x)
        buildBeam(this.sim, {
          x0: x,
          y0: y,
          x1: x,
          y1: y + 13,
          material: this.materialId,
          segments,
          clampStart: true,
        })
        break
      }

      case 'float': {
        // Wood floats, steel sinks - same code path, different density.
        this.sim.fillTo(0)
        this.sim.addObject({ cx: 6, cy: 6, width: 3, height: 1.6, density: 500 })
        this.sim.addObject({ cx: 12, cy: 6, width: 2.4, height: 1.4, density: 7850 })
        break
      }

      case 'loadtest':
        buildLoadTest(this.sim, {
          x,
          y: t.heightAt(x) + 14,
          material: this.materialId,
          segments,
          tipMassKg: this.tipMassKg,
        })
        break
    }
  }

  private fixedUpdate(dt: number): void {
    this.sim.step(dt)
    if (this.sim.breakEvents.length > 0) {
      this.breakCount += this.sim.breakEvents.length
      this.sim.breakEvents.length = 0
    }
  }

  private render(_alpha: number): void {
    this.scenery.update(this.camera, this.renderer.width, this.renderer.height)
    this.simView.update(this.camera, this.renderer.width, this.renderer.height)
    this.fluidView.update(this.camera, this.renderer.width, this.renderer.height)
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
