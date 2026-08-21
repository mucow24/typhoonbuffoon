import type { Terrain } from '../world/terrain'
import { BendConstraints } from './constraints/bending'
import { DistanceConstraints } from './constraints/distance'
import { ParticleStore } from './particles'

export interface SimStats {
  substeps: number
  particles: number
  constraints: number
  bends: number
}

/**
 * The simulated world: one particle table, the constraints over it, and the
 * substepped solve.
 *
 * Substepping rather than iteration count is deliberate - at equal cost,
 * substepping converges far better for stiff constraints (Macklin et al., Small
 * Steps in Physics Simulation, 2019). One constraint iteration per substep.
 */
export class SimWorld {
  readonly particles = new ParticleStore(4096)
  readonly distance = new DistanceConstraints(2048)
  readonly bend = new BendConstraints(2048)

  gravity = -9.81
  substeps = 12
  /** Tangential velocity retained per contact frame. 1 is frictionless. */
  groundFriction = 0.65
  /**
   * Mass-proportional (Rayleigh alpha) damping, as a rate per second.
   *
   * The constraint dampers are stiffness-proportional and so target local,
   * high-frequency modes; they barely touch a whole structure swinging bodily
   * about its base, which is a low-frequency global mode with far more inertia
   * than the local estimate. Without a little of this, an unloaded structure
   * wobbles forever instead of settling.
   *
   * Keep it small. It is the term that would flatten the slow lean if it grew,
   * and the lean is the point.
   */
  linearDamping = 0.35
  terrain: Terrain | null = null

  step(dt: number): void {
    const h = dt / this.substeps
    for (let s = 0; s < this.substeps; s++) {
      this.predict(h)
      this.bend.resetLambda()
      this.distance.resetLambda()
      // Bending first, axial last: in Gauss-Seidel the last solve dominates
      // locally, and members must hold their length before they hold their
      // shape. Stiff axially, compliant in bending.
      this.bend.solve(this.particles, h)
      this.distance.solve(this.particles, h)
      this.solveTerrain()
      this.updateVelocities(h)
      this.bend.dampVelocities(this.particles, h)
      this.distance.dampVelocities(this.particles, h)
      this.applyLinearDamping(h)
      this.applyGroundFriction()
    }
    this.clearAccelerations()
  }

  private predict(h: number): void {
    const p = this.particles
    const alive = p.slots.alive
    const n = p.highWater
    for (let i = 0; i < n; i++) {
      if (alive[i] !== 1) continue
      if (p.invMass[i] === 0) {
        p.prevX[i] = p.posX[i]!
        p.prevY[i] = p.posY[i]!
        continue
      }
      p.prevX[i] = p.posX[i]!
      p.prevY[i] = p.posY[i]!
      p.velX[i]! += p.accX[i]! * h
      p.velY[i]! += (this.gravity + p.accY[i]!) * h
      p.posX[i]! += p.velX[i]! * h
      p.posY[i]! += p.velY[i]! * h
    }
  }

  private updateVelocities(h: number): void {
    const p = this.particles
    const alive = p.slots.alive
    const n = p.highWater
    const invH = 1 / h
    for (let i = 0; i < n; i++) {
      if (alive[i] !== 1 || p.invMass[i] === 0) continue
      p.velX[i] = (p.posX[i]! - p.prevX[i]!) * invH
      p.velY[i] = (p.posY[i]! - p.prevY[i]!) * invH
    }
  }

  /** Positional push out of the ground. Vertical projection; the beach is gentle. */
  private solveTerrain(): void {
    const t = this.terrain
    if (!t) return
    const p = this.particles
    const alive = p.slots.alive
    const n = p.highWater
    for (let i = 0; i < n; i++) {
      if (alive[i] !== 1 || p.invMass[i] === 0) continue
      const floor = t.heightAt(p.posX[i]!) + p.radius[i]!
      if (p.posY[i]! < floor) p.posY[i] = floor
    }
  }

  private applyLinearDamping(h: number): void {
    if (this.linearDamping <= 0) return
    const p = this.particles
    const alive = p.slots.alive
    const n = p.highWater
    const keep = Math.exp(-this.linearDamping * h)
    for (let i = 0; i < n; i++) {
      if (alive[i] !== 1 || p.invMass[i] === 0) continue
      p.velX[i]! *= keep
      p.velY[i]! *= keep
    }
  }

  private applyGroundFriction(): void {
    const t = this.terrain
    if (!t || this.groundFriction >= 1) return
    const p = this.particles
    const alive = p.slots.alive
    const n = p.highWater
    for (let i = 0; i < n; i++) {
      if (alive[i] !== 1 || p.invMass[i] === 0) continue
      const floor = t.heightAt(p.posX[i]!) + p.radius[i]!
      if (p.posY[i]! <= floor + 1e-4) {
        p.velX[i]! *= this.groundFriction
      }
    }
  }

  private clearAccelerations(): void {
    const p = this.particles
    p.accX.fill(0, 0, p.highWater)
    p.accY.fill(0, 0, p.highWater)
  }

  clear(): void {
    this.particles.clear()
    this.distance.clear()
    this.bend.clear()
  }

  stats(): SimStats {
    return {
      substeps: this.substeps,
      particles: this.particles.count,
      constraints: this.distance.count,
      bends: this.bend.count,
    }
  }
}
