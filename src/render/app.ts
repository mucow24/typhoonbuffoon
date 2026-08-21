import { Application, Container } from 'pixi.js'

export interface Renderer {
  app: Application
  /** Screen-space backdrop drawn BEHIND the world, in CSS pixels. */
  background: Container
  /** Everything that lives in world space (pan/zoom/parallax applies). */
  world: Container
  /** Screen-space overlay drawn on top of the world, in CSS pixels. */
  screen: Container
  /** Viewport size in CSS pixels. */
  width: number
  height: number
}

/**
 * Pixi bootstrap. We drive rendering from our own fixed-timestep loop rather
 * than Pixi's ticker, so autoStart is off and we call render() ourselves.
 */
export async function createRenderer(): Promise<Renderer> {
  const host = document.getElementById('stage')
  if (!host) throw new Error('#stage host missing from index.html')

  const app = new Application()
  await app.init({
    background: '#0b0f14',
    resizeTo: window,
    antialias: true,
    autoStart: false,
    resolution: Math.min(window.devicePixelRatio || 1, 2),
    autoDensity: true,
    preference: 'webgl',
  })
  app.ticker.stop()
  host.appendChild(app.canvas)

  // Z-order matters: the backdrop has to be added BEFORE the world, or the sky
  // paints straight over the entire scene.
  const background = new Container()
  const world = new Container()
  const screen = new Container()
  app.stage.addChild(background)
  app.stage.addChild(world)
  app.stage.addChild(screen)

  // Pixi v8's renderer.width/height are already in CSS pixels - they are the
  // screen rect, not the backing store. Dividing by resolution shrinks the
  // viewport, which throws off both the screen-space backdrop and the camera
  // that centres the world on viewW/2.
  const renderer: Renderer = {
    app,
    background,
    world,
    screen,
    width: app.renderer.width,
    height: app.renderer.height,
  }

  const syncSize = () => {
    renderer.width = app.renderer.width
    renderer.height = app.renderer.height
  }
  app.renderer.on('resize', syncSize)
  syncSize()

  return renderer
}
