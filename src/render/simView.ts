import { Container, Graphics } from 'pixi.js'
import { clamp, lerp } from '../core/math'
import { materialAt } from '../sim/materials'
import { KIND_NODE } from '../sim/particles'
import { FLAG_ALIVE, FLAG_PINNED, kindOfFlags } from '../runtime/snapshot'
import type { SimClient } from '../runtime/client'
import type { Camera } from './camera'

const mixRgb = (a: number, b: number, t: number): number => {
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

const STRESS_MID = 0xf2c14e
const STRESS_HIGH = 0xe2483c
const DAMAGE_DARK = 0x3a2b22

/**
 * Stress colouring: the genre's core legibility device, and doubly important
 * here because failure unfolds over a long storm rather than in one snap.
 *
 * The material's own colour is kept at low load so wood and steel stay
 * distinguishable, then blended toward amber and red as the member approaches
 * its break threshold. Accumulated damage darkens it permanently, so a member
 * that has been overloaded and survived still reads as compromised.
 */
export function stressColour(base: number, load: number, damage: number): number {
  const l = clamp(load, 0, 1)
  const hot = l < 0.5 ? mixRgb(base, STRESS_MID, l * 2) : mixRgb(STRESS_MID, STRESS_HIGH, (l - 0.5) * 2)
  return mixRgb(hot, DAMAGE_DARK, clamp(damage, 0, 1) * 0.55)
}

/**
 * Structure rendering from snapshots: member segments arrive with world
 * endpoints, stress and material already resolved worker-side; clusters as
 * fitted frames; structure nodes from the particle block (interpolated
 * between the last two snapshots, like the fluid).
 */
export class SimView {
  readonly container = new Container()
  private readonly g = new Graphics()

  showNodes = true
  showStress = true

  constructor(
    parent: Container,
    private readonly client: SimClient,
  ) {
    this.container.addChild(this.g)
    parent.addChild(this.container)
  }

  update(camera: Camera, viewW: number, viewH: number): void {
    camera.applyTo(this.container, 1, viewW, viewH)
    this.draw()
  }

  private draw(): void {
    const g = this.g
    g.clear()
    const snap = this.client.latest
    if (!snap) return
    const body = snap.body

    for (let k = 0; k < body.segmentCount; k++) {
      const mat = materialAt(body.segMaterial[k]!)
      const load = mat.breakStrain > 0 ? Math.abs(body.segStrain[k]!) / mat.breakStrain : 0
      const colour = this.showStress
        ? stressColour(mat.colour, load, body.segDamage[k]!)
        : mat.colour

      g.moveTo(body.segAx[k]!, body.segAy[k]!)
      g.lineTo(body.segBx[k]!, body.segBy[k]!)
      g.stroke({ width: mat.section, color: colour, cap: 'round' })
    }

    // Physics objects: shape-matched clusters, drawn from their best-fit frame
    // rather than from the particles, so they read as solid things.
    for (const c of snap.clusters) {
      const cos = Math.cos(c.angle)
      const sin = Math.sin(c.angle)
      const corners = [
        [-c.hw, -c.hh],
        [c.hw, -c.hh],
        [c.hw, c.hh],
        [-c.hw, c.hh],
      ]
      const pts: number[] = []
      for (const [lx, ly] of corners) {
        pts.push(c.cx + cos * lx! - sin * ly!, c.cy + sin * lx! + cos * ly!)
      }
      const colour = c.light ? 0xc99a5b : 0x8792a0
      g.poly(pts).fill(colour)
      g.poly(pts).stroke({ width: 0.08, color: 0x2f3944, alpha: 0.7 })
    }

    if (!this.showNodes) return
    const prev = this.client.previous?.body ?? null
    const alpha = this.client.renderAlpha()
    const nBoth = prev ? Math.min(body.particleCount, prev.particleCount) : 0
    for (let i = 0; i < body.particleCount; i++) {
      const flags = body.flags[i]!
      if ((flags & FLAG_ALIVE) === 0 || kindOfFlags(flags) !== KIND_NODE) continue
      let x = body.posX[i]!
      let y = body.posY[i]!
      if (prev && i < nBoth && prev.flags[i] === flags) {
        const px = prev.posX[i]!
        const py = prev.posY[i]!
        // A recycled slot can hold a different particle in the two snapshots;
        // a long lerp streak across the field gives it away. Snap instead.
        if ((px - x) * (px - x) + (py - y) * (py - y) < 4) {
          x = px + (x - px) * alpha
          y = py + (y - py) * alpha
        }
      }
      const pinned = (flags & FLAG_PINNED) !== 0
      g.circle(x, y, pinned ? 0.26 : 0.13)
      g.fill(pinned ? 0xff9d5c : 0x2f3944)
    }
  }
}
