import type { Terrain } from '../world/terrain'
import { BendConstraints } from './constraints/bending'
import { DistanceConstraints } from './constraints/distance'
import { FluidSolver } from './fluid'
import { materialAt } from './materials'
import { KIND_FLUID, ParticleStore } from './particles'
import { Rng } from '../core/rng'

export interface BreakEvent {
  a: number
  b: number
  strain: number
  x: number
  y: number
  material: number
}

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
  readonly fluid = new FluidSolver()

  gravity = -9.81
  substeps = 12
  /** Tangential velocity retained per contact frame. 1 is frictionless. */
  groundFriction = 0.65
  /**
   * Normal velocity retained on ground contact. Near zero on purpose.
   *
   * The terrain pass is positional: it pushes a penetrating particle back out,
   * and the velocity derivation (pos - prev)/h then hands back the full impact
   * speed, so the collision is perfectly elastic. Left alone, a pile of water
   * pumps itself into orbit. Water does not bounce.
   */
  groundRestitution = 0.02
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
  /** Drained by the renderer each frame for splinters and sound. */
  readonly breakEvents: BreakEvent[] = []
  /** Field edges. Water is contained; it does not run off the world. */
  boundsX0 = -60
  boundsX1 = 60
  private readonly jitter = new Rng(0x9a7e2)
  /**
   * Hard speed cap, m/s. A positional correction that teleports a particle -
   * terrain pushout being the usual culprit - becomes velocity when it is
   * differentiated as (pos - prev)/h, and a 5 m push over a 1.4 ms substep
   * reads as 3600 m/s. Stability rule 4 in docs/PLAN.md.
   */
  maxSpeed = 45

  step(dt: number): void {
    const h = dt / this.substeps
    this.fluid.beginFrame(this.particles)
    for (let s = 0; s < this.substeps; s++) {
      this.predict(h)
      this.bend.resetLambda()
      this.distance.resetLambda()
      // Bending first, axial last: in Gauss-Seidel the last solve dominates
      // locally, and members must hold their length before they hold their
      // shape. Stiff axially, compliant in bending.
      this.bend.solve(this.particles, h)
      this.distance.solve(this.particles, h)
      if (s % this.fluid.substepsPerProjection === 0) this.fluid.project(this.particles)
      this.solveContacts()
      this.updateVelocities(h)
      this.bend.dampVelocities(this.particles, h)
      this.distance.dampVelocities(this.particles, h)
    }
    this.fluid.applyViscosity(this.particles)
    this.clearAccelerations()
    this.updateDamage(dt)
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
    const keep = this.linearDamping > 0 ? Math.exp(-this.linearDamping * h) : 1
    const maxSpeed = this.maxSpeed
    const maxSpeed2 = maxSpeed * maxSpeed
    const friction = this.groundFriction
    const restitution = this.groundRestitution
    const grounded = this.grounded

    for (let i = 0; i < n; i++) {
      if (alive[i] !== 1 || p.invMass[i] === 0) continue
      let vx = (p.posX[i]! - p.prevX[i]!) * invH
      let vy = (p.posY[i]! - p.prevY[i]!) * invH

      const sp2 = vx * vx + vy * vy
      if (sp2 > maxSpeed2) {
        const k = maxSpeed / Math.sqrt(sp2)
        vx *= k
        vy *= k
      }

      vx *= keep
      vy *= keep

      if (grounded[i] === 1) {
        vx *= friction
        if (vy > 0) vy *= restitution
      }

      p.velX[i] = vx
      p.velY[i] = vy
    }
  }

  /**
   * Terrain and field-edge contacts in one pass, recording which particles are
   * grounded so the velocity pass can reuse it. These were three separate loops
   * over every particle per substep, each recomputing terrain.heightAt; at 12
   * substeps that dominated the frame.
   */
  private solveContacts(): void {
    const p = this.particles
    const alive = p.slots.alive
    const n = p.highWater
    const t = this.terrain
    const grounded = this.grounded
    if (grounded.length < n) this.grounded = new Uint8Array(Math.max(n, 1024))

    for (let i = 0; i < n; i++) {
      this.grounded[i] = 0
      if (alive[i] !== 1 || p.invMass[i] === 0) continue
      const r = p.radius[i]!

      if (p.posX[i]! < this.boundsX0 + r) p.posX[i] = this.boundsX0 + r
      else if (p.posX[i]! > this.boundsX1 - r) p.posX[i] = this.boundsX1 - r

      if (t) {
        const floor = t.heightAt(p.posX[i]!) + r
        if (p.posY[i]! < floor) {
          p.posY[i] = floor
          this.grounded[i] = 1
        }
      }
    }
  }

  private grounded = new Uint8Array(4096)

  private clearAccelerations(): void {
    const p = this.particles
    p.accX.fill(0, 0, p.highWater)
    p.accY.fill(0, 0, p.highWater)
  }

  /**
   * Damage, plastic set, and breakage. Runs once per frame rather than per
   * substep - these are slow, irreversible processes, not part of the solve.
   *
   * Damage and plasticity are deliberately separate. Damage is a scalar that
   * lowers the break threshold and changes no geometry; it applies to every
   * material and is what makes a long siege feel like it is being lost.
   * Plastic set actually moves the rest length, and only steel does it - wood
   * is near-linear-elastic to fracture and never takes a set.
   */
  updateDamage(dt: number): void {
    const d = this.distance
    const alive = d.slots.alive
    const n = d.highWater

    for (let i = 0; i < n; i++) {
      if (alive[i] !== 1) continue
      const m = materialAt(d.material[i]!)
      const signed = d.strain[i]!
      const s = Math.abs(signed)

      const load = m.breakStrain > 0 ? s / m.breakStrain : 0
      if (load > m.damageOnset && m.damageOnset < 1) {
        const over = (load - m.damageOnset) / (1 - m.damageOnset)
        d.damage[i] = Math.min(0.9, d.damage[i]! + m.damageRate * over * over * dt)
      }

      // Permanent set past yield. Wood's yieldStrain is Infinity, so this is
      // not merely disabled for wood - it never executes.
      if (m.plasticRate > 0 && s > m.yieldStrain) {
        const excess = signed - Math.sign(signed) * m.yieldStrain
        d.rest[i]! += d.rest[i]! * excess * m.plasticRate * dt
      }

      if (s > m.breakStrain * (1 - d.damage[i]!)) {
        this.breakConstraint(i, signed)
      }
    }
  }

  private breakConstraint(i: number, strain: number): void {
    const d = this.distance
    const p = this.particles
    const ia = d.a[i]!
    const ib = d.b[i]!

    this.breakEvents.push({
      a: ia,
      b: ib,
      strain,
      x: (p.posX[ia]! + p.posX[ib]!) * 0.5,
      y: (p.posY[ia]! + p.posY[ib]!) * 0.5,
      material: d.material[i]!,
    })

    d.destroy(i)
    this.severBendsSpanning(ia, ib)
  }

  /** A severed link cannot carry a bending moment across itself. */
  private severBendsSpanning(ia: number, ib: number): void {
    const b = this.bend
    const alive = b.slots.alive
    for (let j = 0; j < b.highWater; j++) {
      if (alive[j] !== 1) continue
      const x = b.a[j]!
      const y = b.b[j]!
      const z = b.c[j]!
      if ((x === ia && y === ib) || (y === ia && x === ib) ||
          (y === ia && z === ib) || (z === ia && y === ib)) {
        b.destroy(j)
      }
    }
  }

  clear(): void {
    this.particles.clear()
    this.distance.clear()
    this.bend.clear()
    this.breakEvents.length = 0
  }

  /**
   * Fill the region between the terrain and `level` with water, on a jittered
   * grid at the current fluid resolution. Jitter matters: a perfect lattice is
   * a metastable state that takes a while to relax into a natural surface.
   */
  fillTo(level: number, x0 = this.boundsX0, x1 = this.boundsX1, maxParticles = 40000): number {
    const t = this.terrain
    const spacing = this.fluid.spacing
    const p = this.particles
    let spawned = 0

    for (let x = x0 + spacing * 0.5; x < x1 && spawned < maxParticles; x += spacing) {
      const ground = t ? t.heightAt(x) : 0
      for (let y = ground + spacing * 0.5; y < level && spawned < maxParticles; y += spacing) {
        p.create({
          x: x + (this.jitter.next() - 0.5) * spacing * 0.25,
          y: y + (this.jitter.next() - 0.5) * spacing * 0.25,
          invMass: 1 / this.fluid.particleMass,
          radius: spacing * 0.5,
          kind: KIND_FLUID,
        })
        spawned++
      }
    }
    return spawned
  }

  /** Drop a block of water, for the sandbox dump tool. */
  spawnBlock(cx: number, cy: number, w: number, h: number, maxParticles = 40000): number {
    const spacing = this.fluid.spacing
    const p = this.particles
    let spawned = 0
    const t = this.terrain
    for (let x = cx - w * 0.5; x < cx + w * 0.5 && spawned < maxParticles; x += spacing) {
      const ground = t ? t.heightAt(x) : -Infinity
      for (let y = cy - h * 0.5; y < cy + h * 0.5 && spawned < maxParticles; y += spacing) {
        // Spawning below ground would be teleported out by the terrain pass and
        // differentiate into an enormous velocity.
        if (y < ground + spacing) continue
        p.create({
          x: x + (this.jitter.next() - 0.5) * spacing * 0.25,
          y: y + (this.jitter.next() - 0.5) * spacing * 0.25,
          invMass: 1 / this.fluid.particleMass,
          radius: spacing * 0.5,
          kind: KIND_FLUID,
        })
        spawned++
      }
    }
    return spawned
  }

  clearFluid(): void {
    const p = this.particles
    for (let i = 0; i < p.highWater; i++) {
      if (p.slots.alive[i] === 1 && p.kind[i] === KIND_FLUID) p.destroy(i)
    }
  }

  get fluidCount(): number {
    return this.particles.countOfKind(KIND_FLUID)
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
