import { Application, Container } from 'pixi.js'

export interface Renderer {
  app: Application
  /** Everything that lives in world space (pan/zoom/parallax applies). */
  world: Container
  /** Screen-space overlay drawn on top of the world, in CSS pixels. */
  screen: Container
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

  const world = new Container()
  const screen = new Container()
  app.stage.addChild(world)
  app.stage.addChild(screen)

  const renderer: Renderer = {
    app,
    world,
    screen,
    width: app.renderer.width / app.renderer.resolution,
    height: app.renderer.height / app.renderer.resolution,
  }

  const syncSize = () => {
    renderer.width = app.renderer.width / app.renderer.resolution
    renderer.height = app.renderer.height / app.renderer.resolution
  }
  app.renderer.on('resize', syncSize)
  syncSize()

  return renderer
}
