import { Container, Graphics } from 'pixi.js'
import { clamp, lerp } from '../core/math'
import { materialAt } from '../sim/materials'
import { KIND_NODE } from '../sim/particles'
import type { SimWorld } from '../sim/world'
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

export class SimView {
  readonly container = new Container()
  private readonly g = new Graphics()

  showNodes = true
  showStress = true

  constructor(
    parent: Container,
    private readonly sim: SimWorld,
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
    const p = this.sim.particles
    const d = this.sim.distance
    g.clear()

    const alive = d.slots.alive
    for (let i = 0; i < d.highWater; i++) {
      if (alive[i] !== 1) continue
      const ia = d.a[i]!
      const ib = d.b[i]!
      const mat = materialAt(d.material[i]!)
      const load = mat.breakStrain > 0 ? Math.abs(d.strain[i]!) / mat.breakStrain : 0
      const colour = this.showStress
        ? stressColour(mat.colour, load, d.damage[i]!)
        : mat.colour

      g.moveTo(p.posX[ia]!, p.posY[ia]!)
      g.lineTo(p.posX[ib]!, p.posY[ib]!)
      g.stroke({ width: mat.section, color: colour, cap: 'round' })
    }

    // Physics objects: shape-matched clusters, drawn from their best-fit frame
    // rather than from the particles, so they read as solid things.
    for (const c of this.sim.clusters) {
      if (!c.alive) continue
      const { hw, hh } = c.restExtent()
      const cos = Math.cos(c.angle)
      const sin = Math.sin(c.angle)
      const corners = [
        [-hw, -hh],
        [hw, -hh],
        [hw, hh],
        [-hw, hh],
      ]
      const pts: number[] = []
      for (const [lx, ly] of corners) {
        pts.push(c.cx + cos * lx! - sin * ly!, c.cy + sin * lx! + cos * ly!)
      }
      const density = c.totalMass / Math.max(4 * hw * hh, 1e-6)
      const colour = density < 1000 ? 0xc99a5b : 0x8792a0
      g.poly(pts).fill(colour)
      g.poly(pts).stroke({ width: 0.08, color: 0x2f3944, alpha: 0.7 })
    }

    if (!this.showNodes) return
    const palive = p.slots.alive
    for (let i = 0; i < p.highWater; i++) {
      if (palive[i] !== 1) continue
      // Structure nodes only. This used to draw a circle for EVERY particle,
      // including thousands of fluid particles the fluid renderer had already
      // drawn - 13.75ms a frame of pure redundancy at 6k particles.
      if (p.kind[i] !== KIND_NODE) continue
      const pinned = p.invMass[i] === 0
      g.circle(p.posX[i]!, p.posY[i]!, pinned ? 0.26 : 0.13)
      g.fill(pinned ? 0xff9d5c : 0x2f3944)
    }
  }
}
