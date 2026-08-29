import { Container, Graphics, Particle, ParticleContainer, type Renderer, type Texture } from 'pixi.js'
import { KIND_FLUID } from '../sim/particles'
import { FLAG_ALIVE, kindOfFlags } from '../runtime/snapshot'
import type { SimClient } from '../runtime/client'
import type { Camera } from './camera'

const TEX_SIZE = 32
/** Somewhere no camera can reach. */
const PARKED = 1e7

/**
 * Fluid rendering via ParticleContainer, from snapshots.
 *
 * A Graphics redraw with thousands of circle() calls rebuilds its geometry
 * every frame and will not hold 60fps at the particle counts this game needs.
 * ParticleContainer batches a shared texture into one draw call, which is the
 * difference between a few thousand particles being affordable and not.
 *
 * Positions are interpolated between the last two snapshots: sim frames are
 * not phase-locked to rAF, and at heavy particle counts the sim deliberately
 * runs slower than render - exactly when smooth drawing matters most.
 */
export class FluidView {
  readonly container: ParticleContainer
  private readonly pool: Particle[] = []
  private readonly texture: Texture
  private shown = 0

  constructor(parent: Container, renderer: Renderer, private readonly client: SimClient) {
    const g = new Graphics().circle(TEX_SIZE / 2, TEX_SIZE / 2, TEX_SIZE / 2 - 1).fill(0xffffff)
    this.texture = renderer.generateTexture({
      target: g,
      resolution: 2,
    })
    g.destroy()

    this.container = new ParticleContainer({
      dynamicProperties: { position: true, scale: true, rotation: false, color: false },
    })
    parent.addChild(this.container)
  }

  update(camera: Camera, viewW: number, viewH: number): void {
    camera.applyTo(this.container, 1, viewW, viewH)

    const snap = this.client.latest
    if (!snap) return
    const body = snap.body
    const prev = this.client.previous?.body ?? null
    const alpha = this.client.renderAlpha()
    const nBoth = prev ? Math.min(body.particleCount, prev.particleCount) : 0

    // Slight overlap so the surface reads as a body of water rather than beads.
    const drawSize = snap.scalars.fluidSpacing * 1.7
    const scale = drawSize / TEX_SIZE

    let n = 0
    for (let i = 0; i < body.particleCount; i++) {
      const flags = body.flags[i]!
      if ((flags & FLAG_ALIVE) === 0 || kindOfFlags(flags) !== KIND_FLUID) continue
      let x = body.posX[i]!
      let y = body.posY[i]!
      if (prev && i < nBoth && prev.flags[i] === flags) {
        const px = prev.posX[i]!
        const py = prev.posY[i]!
        // A recycled slot may hold a different particle in the two snapshots;
        // a teleport-length lerp gives it away. Snap to current instead.
        if ((px - x) * (px - x) + (py - y) * (py - y) < 4) {
          x = px + (x - px) * alpha
          y = py + (y - py) * alpha
        }
      }
      const particle = this.ensure(n)
      particle.x = x
      particle.y = y
      particle.scaleX = scale
      particle.scaleY = scale
      n++
    }

    // Park the surplus rather than reallocating the container every frame.
    //
    // Moved far out of the world AS WELL AS scaled to zero. Hiding by scale
    // alone assumes the scale attribute is re-uploaded every frame; if the
    // container treats it as static - written once when the particle is created
    // - the assignment silently does nothing and the sprite stays wherever it
    // was last drawn. That is what left a cloud of blue dots hanging over a map
    // with no water left in it. Position is dynamic by necessity, so parking
    // works whatever the container does with scale.
    for (let k = n; k < this.shown; k++) {
      const particle = this.pool[k]
      if (particle) {
        particle.x = PARKED
        particle.y = PARKED
        particle.scaleX = 0
        particle.scaleY = 0
      }
    }
    this.shown = Math.max(n, this.shown)
  }

  private ensure(index: number): Particle {
    let particle = this.pool[index]
    if (!particle) {
      particle = new Particle({
        texture: this.texture,
        anchorX: 0.5,
        anchorY: 0.5,
        tint: 0x4ea8d8,
        alpha: 0.92,
        x: PARKED,
        y: PARKED,
      })
      this.pool[index] = particle
      this.container.addParticle(particle)
    }
    return particle
  }
}
