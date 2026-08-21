import { Container, Graphics } from 'pixi.js'
import type { SimWorld } from '../sim/world'
import type { Camera } from './camera'

/**
 * Draws the simulated structure in world space. Redrawn every frame - the
 * geometry is entirely dynamic, so retained-mode display objects would buy
 * nothing.
 */
export class SimView {
  readonly container = new Container()
  private readonly g = new Graphics()

  /** Line half-width in metres for constraint rendering. */
  memberThickness = 0.18
  showNodes = true

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
      g.moveTo(p.posX[ia]!, p.posY[ia]!)
      g.lineTo(p.posX[ib]!, p.posY[ib]!)
      g.stroke({ width: this.memberThickness, color: 0xd7c9a8, cap: 'round' })
    }

    if (!this.showNodes) return
    const palive = p.slots.alive
    for (let i = 0; i < p.highWater; i++) {
      if (palive[i] !== 1) continue
      const pinned = p.invMass[i] === 0
      g.circle(p.posX[i]!, p.posY[i]!, pinned ? 0.28 : 0.16)
      g.fill(pinned ? 0xff9d5c : 0x6fd3ff)
    }
  }
}
