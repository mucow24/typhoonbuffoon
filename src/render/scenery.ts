import { Container, Graphics } from 'pixi.js'
import { Rng } from '../core/rng'
import { clamp, lerp } from '../core/math'
import type { Field } from '../world/field'
import type { Camera } from './camera'

/** Flat-shape palette. Index 0 is calm, index 1 is full storm. */
const PALETTE = {
  skyBands: [
    [0x7fb8dd, 0x4a5560],
    [0x9ccbe6, 0x5b6673],
    [0xb9dcef, 0x6d7885],
    [0xd8ecf7, 0x828d99],
  ],
  cloud: [0xf2f7fb, 0x8e99a5],
  cloudDark: [0xdce7f0, 0x6b7681],
  farSea: [0x4e86ab, 0x3c4b58],
  nearSea: [0x5c9cc0, 0x455663],
  farDune: [0xc7b68d, 0x8a8272],
  nearDune: [0xd8c79e, 0x9b9280],
  sand: [0xe3d2a4, 0xa89f87],
  wetSand: [0xc4b184, 0x8d856f],
} as const

const mix = (pair: readonly [number, number], t: number): number => {
  const a = pair[0]
  const b = pair[1]
  const ar = (a >> 16) & 0xff
  const ag = (a >> 8) & 0xff
  const ab = a & 0xff
  const br = (b >> 16) & 0xff
  const bg = (b >> 8) & 0xff
  const bb = b & 0xff
  return (
    (Math.round(lerp(ar, br, t)) << 16) |
    (Math.round(lerp(ag, bg, t)) << 8) |
    Math.round(lerp(ab, bb, t))
  )
}

interface ParallaxLayer {
  container: Container
  factor: number
}

interface Rect {
  x0: number
  x1: number
  y0: number
  y1: number
}

/** World-space spacing between dune samples. Fixed, so dunes tile as you pan. */
const DUNE_STEP = 11
/** Horizontal period for repeating clouds. */
const CLOUD_PERIOD = 180
/**
 * Sea horizon in the far-sea layer's own space. Deliberately ABOVE the dune
 * silhouettes: every backdrop layer fills downward from its own top edge, so
 * each nearer layer hides the one behind it below that edge. The ocean is only
 * visible at all because its horizon sits higher than the dunes in front of it.
 */
const SEA_HORIZON = 15

/**
 * Background scenery: flat bands and silhouettes, parallaxed.
 *
 * Every layer is drawn to cover the region actually visible in ITS OWN parallax
 * space, recomputed whenever the camera moves. Drawing them across a fixed
 * world span instead meant that zooming out past that span left each layer
 * ending at a different screen position - hard-edged boxes with sky showing
 * through underneath. Backdrops have to be unbounded; they are the thing you
 * can never reach the edge of.
 *
 * Dunes and clouds tile by index rather than being generated across a fixed
 * width, so they extend forever without needing more data.
 */
export class Scenery {
  private readonly layers: ParallaxLayer[] = []
  private severity = 0
  private dirty = true
  private lastKey = ''

  private readonly sky = new Graphics()
  private readonly clouds = new Graphics()
  private readonly farSea = new Graphics()
  private readonly dunesFar = new Graphics()
  private readonly dunesNear = new Graphics()
  private readonly ground = new Graphics()

  private cloudSpecs: { x: number; y: number; w: number; h: number; dark: boolean }[] = []
  private duneFar: number[] = []
  private duneNear: number[] = []

  constructor(
    private readonly world: Container,
    private readonly background: Container,
    private readonly field: Field,
  ) {
    // Sky is screen space and must sit BEHIND the world, not in the overlay
    // container - putting it in the overlay painted it over the whole scene.
    this.background.addChild(this.sky)

    this.addLayer(this.clouds, 0.1)
    this.addLayer(this.farSea, 0.25)
    this.addLayer(this.dunesFar, 0.45)
    this.addLayer(this.dunesNear, 0.7)
    this.addLayer(this.ground, 1)

    this.buildSpecs()
    this.field.onChange(() => {
      this.dirty = true
    })
  }

  private addLayer(g: Graphics, factor: number): void {
    const c = new Container()
    c.addChild(g)
    this.world.addChild(c)
    this.layers.push({ container: c, factor })
  }

  setSeverity(v: number): void {
    const next = clamp(v, 0, 1)
    if (Math.abs(next - this.severity) < 0.002) return
    this.severity = next
    this.dirty = true
  }

  invalidate(): void {
    this.dirty = true
  }

  /** Deterministic, and independent of field width so it never needs rebuilding. */
  private buildSpecs(): void {
    const rng = new Rng(0x5c3ee)

    this.cloudSpecs = []
    for (let i = 0; i < 14; i++) {
      this.cloudSpecs.push({
        x: rng.range(0, CLOUD_PERIOD),
        y: rng.range(16, 54),
        w: rng.range(7, 20),
        h: rng.range(1.6, 4.4),
        dark: rng.next() < 0.4,
      })
    }

    this.duneFar = []
    this.duneNear = []
    for (let i = 0; i < 64; i++) {
      this.duneFar.push(rng.range(3.5, 9))
      this.duneNear.push(rng.range(2, 6))
    }
  }

  /** The world rect visible in a layer's own parallax space, plus margin. */
  private visibleRect(camera: Camera, factor: number, viewW: number, viewH: number): Rect {
    const halfW = viewW / (2 * camera.scale)
    const halfH = viewH / (2 * camera.scale)
    const cx = camera.x * factor
    const cy = camera.y * factor
    const m = Math.max(halfW, halfH) * 0.25 + 5
    return { x0: cx - halfW - m, x1: cx + halfW + m, y0: cy - halfH - m, y1: cy + halfH + m }
  }

  update(camera: Camera, viewW: number, viewH: number): void {
    // Redraw when the view actually changed. Backdrop geometry depends on the
    // camera now, so a dirty flag alone is not enough.
    const key =
      `${camera.x.toFixed(2)}|${camera.y.toFixed(2)}|${camera.zoom.toFixed(4)}|` +
      `${Math.round(viewW)}|${Math.round(viewH)}|${this.severity.toFixed(3)}|${this.field.widthM}`
    if (this.dirty || key !== this.lastKey) {
      this.redraw(camera, viewW, viewH)
      this.lastKey = key
      this.dirty = false
    }
    for (const layer of this.layers) {
      camera.applyTo(layer.container, layer.factor, viewW, viewH)
    }
  }

  private redraw(camera: Camera, viewW: number, viewH: number): void {
    const s = this.severity

    this.drawSky(viewW, viewH, s)
    this.drawClouds(this.visibleRect(camera, 0.1, viewW, viewH), s)
    this.drawSea(this.visibleRect(camera, 0.25, viewW, viewH), s)
    this.drawDunes(
      this.dunesFar,
      this.duneFar,
      this.visibleRect(camera, 0.45, viewW, viewH),
      mix(PALETTE.farDune, s),
      1,
    )
    this.drawDunes(
      this.dunesNear,
      this.duneNear,
      this.visibleRect(camera, 0.7, viewW, viewH),
      mix(PALETTE.nearDune, s),
      1,
    )
    this.drawGround(this.visibleRect(camera, 1, viewW, viewH), s)
  }

  private drawSky(viewW: number, viewH: number, s: number): void {
    const g = this.sky
    g.clear()
    const bands = PALETTE.skyBands.length
    for (let i = 0; i < bands; i++) {
      const y = (viewH / bands) * i
      g.rect(0, y, viewW, viewH / bands + 1).fill(mix(PALETTE.skyBands[i]!, s))
    }
  }

  private drawClouds(r: Rect, s: number): void {
    const g = this.clouds
    g.clear()
    const k0 = Math.floor(r.x0 / CLOUD_PERIOD) - 1
    const k1 = Math.ceil(r.x1 / CLOUD_PERIOD) + 1
    const alpha = lerp(0.85, 1, s)

    for (let k = k0; k <= k1; k++) {
      const offset = k * CLOUD_PERIOD
      for (const c of this.cloudSpecs) {
        const x = c.x + offset
        if (x < r.x0 - c.w || x > r.x1 + c.w) continue
        if (c.y < r.y0 || c.y > r.y1) continue
        const colour = mix(c.dark ? PALETTE.cloudDark : PALETTE.cloud, s)
        g.roundRect(x - c.w * 0.5, c.y - c.h * 0.5, c.w, c.h, c.h * 0.5).fill({ color: colour, alpha })
        g.roundRect(x - c.w * 0.22, c.y - c.h * 0.1, c.w * 0.55, c.h * 1.35, c.h * 0.6)
          .fill({ color: colour, alpha })
      }
    }
  }

  /** Distant sea, filled from the horizon all the way down past the viewport. */
  private drawSea(r: Rect, s: number): void {
    const g = this.farSea
    g.clear()
    const top = SEA_HORIZON
    if (r.y0 >= top) return
    g.rect(r.x0, r.y0, r.x1 - r.x0, top - r.y0).fill(mix(PALETTE.farSea, s))
    g.rect(r.x0, top - 0.6, r.x1 - r.x0, 1.2).fill(mix(PALETTE.nearSea, s))
  }

  private drawDunes(g: Graphics, spec: number[], r: Rect, colour: number, scale: number): void {
    g.clear()
    const n = spec.length
    const i0 = Math.floor(r.x0 / DUNE_STEP) - 1
    const i1 = Math.ceil(r.x1 / DUNE_STEP) + 1

    const pts: number[] = [i0 * DUNE_STEP, r.y0]
    for (let i = i0; i <= i1; i++) {
      // Wrap the index so the profile tiles instead of running out.
      const h = spec[((i % n) + n) % n]!
      pts.push(i * DUNE_STEP, h * scale)
    }
    pts.push(i1 * DUNE_STEP, r.y0)
    g.poly(pts).fill(colour)
  }

  private drawGround(r: Rect, s: number): void {
    const t = this.field.terrain
    const g = this.ground
    g.clear()

    // The ground spans the FIELD only. Extending it to the viewport would bury
    // the distant sea and dunes behind it; beyond the field edges the backdrop
    // layers are meant to show through.
    const left = t.x0
    const right = t.x1
    const floor = Math.min(r.y0, t.minHeight - 5)

    const pts: number[] = [left, floor, left, t.heightAt(t.x0)]
    for (let i = 0; i < t.heights.length; i++) {
      pts.push(t.x0 + i * t.spacing, t.heights[i]!)
    }
    pts.push(right, t.heightAt(t.x1), right, floor)
    g.poly(pts).fill(mix(PALETTE.sand, s))

    // Everything below sea level reads as wet/submerged ground.
    const wet: number[] = []
    let started = false
    for (let i = 0; i < t.heights.length; i++) {
      const x = t.x0 + i * t.spacing
      const h = t.heights[i]!
      if (h < 0) {
        if (!started) {
          wet.push(x, 0)
          started = true
        }
        wet.push(x, h)
      } else if (started) {
        wet.push(x, 0)
        started = false
      }
    }
    if (started) wet.push(t.x1, 0)
    if (wet.length >= 6) g.poly(wet).fill(mix(PALETTE.wetSand, s))

    // Field edges, so the authored width stays legible.
    const top = t.maxHeight + 14
    g.rect(t.x0 - 0.25, floor, 0.5, top - floor).fill({ color: 0x000000, alpha: 0.22 })
    g.rect(t.x1 - 0.25, floor, 0.5, top - floor).fill({ color: 0x000000, alpha: 0.22 })
  }
}
