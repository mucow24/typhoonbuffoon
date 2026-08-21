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
  seabed: [0x9c8f6d, 0x70695a],
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

/**
 * Background scenery: flat bands and silhouettes, parallaxed. Severity greys
 * the whole palette down in one place, which is all the sky ramp needs to be.
 */
export class Scenery {
  private readonly layers: ParallaxLayer[] = []
  private severity = 0
  private dirty = true

  private readonly sky = new Graphics()
  private readonly clouds = new Graphics()
  private readonly farSea = new Graphics()
  private readonly dunesFar = new Graphics()
  private readonly dunesNear = new Graphics()
  private readonly ground = new Graphics()

  private cloudSpecs: { x: number; y: number; w: number; h: number; dark: boolean }[] = []
  private duneFarSpec: number[] = []
  private duneNearSpec: number[] = []

  constructor(
    private readonly world: Container,
    private readonly screen: Container,
    private readonly field: Field,
  ) {
    // Sky is screen space - it should not pan at all.
    this.screen.addChildAt(this.sky, 0)

    this.addLayer(this.clouds, 0.1)
    this.addLayer(this.farSea, 0.25)
    this.addLayer(this.dunesFar, 0.45)
    this.addLayer(this.dunesNear, 0.7)
    this.addLayer(this.ground, 1)

    this.rebuildSpecs()
    this.field.onChange(() => {
      this.rebuildSpecs()
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

  private rebuildSpecs(): void {
    const rng = new Rng(0x5c3ee)
    const w = this.field.widthM
    const span = w * 2.2

    this.cloudSpecs = []
    const cloudCount = Math.max(6, Math.round(w / 9))
    for (let i = 0; i < cloudCount; i++) {
      this.cloudSpecs.push({
        x: rng.range(-span * 0.5, span * 0.5),
        y: rng.range(18, 52),
        w: rng.range(w * 0.06, w * 0.17),
        h: rng.range(1.6, 4.4),
        dark: rng.next() < 0.4,
      })
    }

    const duneSamples = 26
    this.duneFarSpec = []
    this.duneNearSpec = []
    for (let i = 0; i < duneSamples; i++) {
      this.duneFarSpec.push(rng.range(3.5, 9))
      this.duneNearSpec.push(rng.range(2, 6))
    }
  }

  /** Redraw only when something actually changed. */
  private redraw(viewW: number, viewH: number): void {
    const s = this.severity
    const w = this.field.widthM
    const span = w * 2.2

    // --- sky: flat horizontal bands, screen space ---
    this.sky.clear()
    const bands = PALETTE.skyBands.length
    for (let i = 0; i < bands; i++) {
      const y = (viewH / bands) * i
      this.sky.rect(0, y, viewW, viewH / bands + 1).fill(mix(PALETTE.skyBands[i]!, s))
    }

    // --- clouds ---
    this.clouds.clear()
    for (const c of this.cloudSpecs) {
      const colour = mix(c.dark ? PALETTE.cloudDark : PALETTE.cloud, s)
      const alpha = lerp(0.85, 1, s)
      this.clouds
        .roundRect(c.x - c.w * 0.5, c.y - c.h * 0.5, c.w, c.h, c.h * 0.5)
        .fill({ color: colour, alpha })
      this.clouds
        .roundRect(c.x - c.w * 0.22, c.y - c.h * 0.1, c.w * 0.55, c.h * 1.35, c.h * 0.6)
        .fill({ color: colour, alpha })
    }

    // --- distant sea band, sitting on the horizon ---
    this.farSea.clear()
    this.farSea.rect(-span * 0.5, -40, span, 40).fill(mix(PALETTE.farSea, s))
    this.farSea.rect(-span * 0.5, -0.9, span, 1.6).fill(mix(PALETTE.nearSea, s))

    // --- dune silhouettes ---
    this.drawDunes(this.dunesFar, this.duneFarSpec, span, mix(PALETTE.farDune, s), 1.35)
    this.drawDunes(this.dunesNear, this.duneNearSpec, span, mix(PALETTE.nearDune, s), 1)

    // --- the actual ground, from the terrain polyline ---
    this.drawGround(s)

    this.dirty = false
  }

  private drawDunes(g: Graphics, spec: number[], span: number, colour: number, scale: number): void {
    g.clear()
    const n = spec.length
    const step = span / (n - 1)
    const pts: number[] = [-span * 0.5, -60]
    for (let i = 0; i < n; i++) {
      pts.push(-span * 0.5 + i * step, spec[i]! * scale)
    }
    pts.push(span * 0.5, -60)
    g.poly(pts).fill(colour)
  }

  private drawGround(s: number): void {
    const t = this.field.terrain
    const g = this.ground
    g.clear()

    const floor = t.minHeight - 30
    const pts: number[] = [t.x0, floor]
    for (let i = 0; i < t.heights.length; i++) {
      pts.push(t.x0 + i * t.spacing, t.heights[i]!)
    }
    pts.push(t.x1, floor)
    g.poly(pts).fill(mix(PALETTE.sand, s))

    // Everything below sea level reads as wet/submerged ground.
    g.poly(pts).fill(mix(PALETTE.sand, s))
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
    if (wet.length >= 6) {
      g.poly(wet).fill(mix(PALETTE.wetSand, s))
    }

    // Field edges, so the authored width is always legible.
    const top = t.maxHeight + 14
    g.rect(t.x0 - 0.25, floor, 0.5, top - floor).fill({ color: 0x000000, alpha: 0.22 })
    g.rect(t.x1 - 0.25, floor, 0.5, top - floor).fill({ color: 0x000000, alpha: 0.22 })
  }

  update(camera: Camera, viewW: number, viewH: number): void {
    if (this.dirty) this.redraw(viewW, viewH)
    for (const layer of this.layers) {
      camera.applyTo(layer.container, layer.factor, viewW, viewH)
    }
  }

  /** Force a redraw, e.g. after the viewport resizes. */
  invalidate(): void {
    this.dirty = true
  }
}
