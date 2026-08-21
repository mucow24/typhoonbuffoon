import type { Terrain } from '../world/terrain'
import { BendConstraints } from './constraints/bending'
import { DistanceConstraints } from './constraints/distance'
import { Cluster, buildObject, type ObjectSpec } from './clusters'
import { FluidSolver } from './fluid'
import { WaterField } from './water'
import { AIR_DENSITY, WindField } from './wind'
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
  readonly water = new WaterField()
  readonly wind = new WindField()
  readonly clusters: Cluster[] = []

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
  /** Water density for buoyancy, kg/m^3. */
  waterDensity = 1000
  /** Linear drag from submersion, per second. Stops floaters bobbing forever. */
  waterDrag = 2.2
  private contactStamp = new Int32Array(4096)
  private stampCounter = 0
  private readonly tmpVel = { x: 0, y: 0 }

  step(dt: number): void {
    const h = dt / this.substeps
    this.water.build(
      this.particles,
      this.boundsX0,
      this.boundsX1,
      this.fluid.spacing * this.fluid.spacing,
    )
    this.applyBuoyancy()
    this.wind.advance(dt)
    this.applyWind()
    this.fluid.beginFrame(this.particles)
    for (let s = 0; s < this.substeps; s++) {
      this.predict(h)
      this.bend.resetLambda()
      this.distance.resetLambda()
      // Bending first, axial last: in Gauss-Seidel the last solve dominates
      // locally, and members must hold their length before they hold their
      // shape. Stiff axially, compliant in bending.
      // Clusters FIRST, then the structural constraints. In Gauss-Seidel the
      // last solve wins locally: with shape matching last it snapped object
      // particles back to the rigid formation every substep, undoing the welds
      // that attach a structure to the object. The two then fought and pumped
      // energy in - a supported house rose and flipped. Structure last means
      // load actually transfers into the object.
      for (const c of this.clusters) if (c.alive) c.solve(this.particles)
      this.bend.solve(this.particles, h)
      this.distance.solve(this.particles, h)
      if (s % this.fluid.substepsPerProjection === 0) this.fluid.project(this.particles)
      this.solveFluidAgainstMembers()
      this.solveFluidAgainstObjects()
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

  /**
   * Buoyancy and water drag, from REST volume. Computed once per frame into the
   * acceleration accumulators.
   *
   * Net acceleration is g * (rhoWater/rhoBody - 1), so wood at 500 kg/m^3 rises
   * and steel at 7850 sinks with no flag anywhere - density alone decides it.
   */
  private applyBuoyancy(): void {
    const p = this.particles
    const alive = p.slots.alive
    const g = -this.gravity
    for (let i = 0; i < p.highWater; i++) {
      if (alive[i] !== 1 || p.invMass[i] === 0) continue
      if (p.kind[i] === KIND_FLUID) continue
      const vol = p.volume[i]!
      if (vol <= 0) continue

      const frac = this.water.submergedFraction(p.posX[i]!, p.posY[i]!, p.radius[i]!)
      if (frac <= 0) continue

      const mass = 1 / p.invMass[i]!
      p.accY[i]! += g * ((this.waterDensity * vol) / mass) * frac

      // Drag against the LOCAL WATER velocity, not absolute velocity, so a
      // current or a passing wave actually pushes a body along instead of only
      // slowing it down.
      this.water.velocityAt(p.posX[i]!, this.tmpVel)
      const drag = this.waterDrag * frac
      p.accX[i]! -= (p.velX[i]! - this.tmpVel.x) * drag
      p.accY[i]! -= (p.velY[i]! - this.tmpVel.y) * drag
    }
  }

  /**
   * Fluid against structural members. Members are line segments, so they are
   * treated as capsules and fluid particles are pushed out with a mass-weighted
   * reaction onto the member's endpoints - which is what lets a flood wall
   * actually hold water back instead of being decorative.
   *
   * The capsule radius has a floor of 0.75 * particle spacing. A member thinner
   * than about one particle spacing is not watertight and fluid tunnels through
   * it, which on a flood wall reads as a bug rather than a near miss.
   */
  private solveFluidAgainstMembers(): void {
    const d = this.distance
    if (d.count === 0) return
    const p = this.particles
    const hash = this.fluid.hash
    const spacing = this.fluid.spacing
    const alive = d.slots.alive

    if (this.contactStamp.length < p.highWater) {
      this.contactStamp = new Int32Array(Math.max(p.highWater, 4096))
    }

    for (let m = 0; m < d.highWater; m++) {
      if (alive[m] !== 1) continue
      const ia = d.a[m]!
      const ib = d.b[m]!
      const ax = p.posX[ia]!
      const ay = p.posY[ia]!
      const bx = p.posX[ib]!
      const by = p.posY[ib]!
      const mat = materialAt(d.material[m]!)
      const radius = Math.max(mat.section * 0.5, spacing * 0.75)

      const ex = bx - ax
      const ey = by - ay
      const len2 = ex * ex + ey * ey
      if (len2 < 1e-12) continue
      const segLen = Math.sqrt(len2)

      const stamp = ++this.stampCounter
      const steps = Math.max(1, Math.ceil(segLen / Math.max(spacing, 0.05)))
      for (let sIdx = 0; sIdx <= steps; sIdx++) {
        const t = sIdx / steps
        const sx = ax + ex * t
        const sy = ay + ey * t
        const buckets = hash.collectBuckets(sx, sy)
        const starts = hash.starts
        const entries = hash.entries
        const scratch = hash.scratch

        for (let s = 0; s < buckets; s++) {
          const b = scratch[s]!
          const end = starts[b + 1]!
          for (let k = starts[b]!; k < end; k++) {
            const j = entries[k]!
            if (this.contactStamp[j] === stamp) continue
            this.contactStamp[j] = stamp
            if (p.slots.alive[j] !== 1 || p.invMass[j] === 0) continue

            const px = p.posX[j]! - ax
            const py = p.posY[j]! - ay
            let u = (px * ex + py * ey) / len2
            u = u < 0 ? 0 : u > 1 ? 1 : u
            const cx = ax + ex * u
            const cy = ay + ey * u
            let nx = p.posX[j]! - cx
            let ny = p.posY[j]! - cy
            const dist = Math.sqrt(nx * nx + ny * ny)
            const minDist = radius + p.radius[j]!
            if (dist >= minDist) continue

            if (dist < 1e-6) {
              nx = -ey / segLen
              ny = ex / segLen
            } else {
              nx /= dist
              ny /= dist
            }

            // One-way: the fluid particle is pushed out, the member is not
            // pushed back. All fluid -> body force is mediated by the water
            // field (buoyancy + drag). Doing both double-counts buoyancy, and
            // since a wood crate's particles are lighter than fluid particles
            // the collision term dominated and launched objects tens of metres.
            const pen = minDist - dist
            p.posX[j]! += nx * pen
            p.posY[j]! += ny * pen
          }
        }
      }
    }
  }

  /**
   * Fluid against object cluster particles.
   *
   * Without this, water flows straight through a crate: the object displaces
   * nothing, so the height field reads water where the hull is, buoyancy is
   * computed against a surface the object itself should have raised, and the
   * result is a crate that launches. Objects have no distance constraints, so
   * they are not covered by the member pass and need their own.
   */
  private solveFluidAgainstObjects(): void {
    const p = this.particles
    if (this.clusters.length === 0) return
    const hash = this.fluid.hash
    const starts = hash.starts
    const entries = hash.entries
    const scratch = hash.scratch

    for (const c of this.clusters) {
      if (!c.alive) continue
      for (let k = 0; k < c.particles.length; k++) {
        const i = c.particles[k]!
        if (p.slots.alive[i] !== 1) continue
        const xi = p.posX[i]!
        const yi = p.posY[i]!
        const ri = p.radius[i]!

        const buckets = hash.collectBuckets(xi, yi)
        for (let s = 0; s < buckets; s++) {
          const b = scratch[s]!
          const end = starts[b + 1]!
          for (let q = starts[b]!; q < end; q++) {
            const j = entries[q]!
            if (p.slots.alive[j] !== 1) continue
            if (p.invMass[j]! <= 0) continue

            let nx = p.posX[j]! - xi
            let ny = p.posY[j]! - yi
            const dist = Math.sqrt(nx * nx + ny * ny)
            const minDist = ri + p.radius[j]!
            if (dist >= minDist) continue

            if (dist < 1e-6) {
              nx = 0
              ny = 1
            } else {
              nx /= dist
              ny /= dist
            }

            // One-way, for the same reason as the member pass above.
            const pen = minDist - dist
            p.posX[j]! += nx * pen
            p.posY[j]! += ny * pen
          }
        }
      }
    }
  }

  /**
   * Wind drag, applied once per frame into the acceleration accumulators.
   *
   * Members are treated as segments and loaded by their frontal length - the
   * projection of the segment perpendicular to the wind, plus its thickness.
   * That makes a broadside wall catch far more wind than an edge-on strut, and
   * because the force is split between the two endpoints it produces torque
   * about the member naturally rather than needing a separate moment term.
   *
   * Frontal length comes from REST geometry: computing it from the deformed
   * shape is the same runaway as buoyancy from deformed volume.
   */
  private applyWind(): void {
    if (this.wind.baseSpeed <= 0) return
    const p = this.particles
    const d = this.distance
    const alive = d.slots.alive
    const half = 0.5 * AIR_DENSITY

    for (let m = 0; m < d.highWater; m++) {
      if (alive[m] !== 1) continue
      const ia = d.a[m]!
      const ib = d.b[m]!
      const wa = p.invMass[ia]!
      const wb = p.invMass[ib]!
      if (wa === 0 && wb === 0) continue

      const mat = materialAt(d.material[m]!)
      const rest = d.rest[m]!
      const ex = p.posX[ib]! - p.posX[ia]!
      const ey = p.posY[ib]! - p.posY[ia]!
      const len = Math.sqrt(ex * ex + ey * ey)
      if (len < 1e-9) continue

      const midX = (p.posX[ia]! + p.posX[ib]!) * 0.5
      const midY = (p.posY[ia]! + p.posY[ib]!) * 0.5

      // Submerged members are shielded from wind; the water has them instead.
      const submerged = this.water.submergedFraction(midX, midY, mat.section * 0.5)
      const exposure = 1 - submerged
      if (exposure <= 0.01) continue

      const windU = this.wind.velocityAt(midX)
      const relX = windU - (p.velX[ia]! + p.velX[ib]!) * 0.5
      const relY = -(p.velY[ia]! + p.velY[ib]!) * 0.5
      const relSpeed = Math.sqrt(relX * relX + relY * relY)
      if (relSpeed < 1e-6) continue

      // Frontal length: the segment projected across the flow, from rest
      // length, plus the member's own thickness.
      const dirX = relX / relSpeed
      const dirY = relY / relSpeed
      const cross = Math.abs((ex * dirY - ey * dirX) / len)
      const frontal = rest * cross + mat.section

      const force = half * mat.dragCoefficient * frontal * relSpeed * relSpeed * exposure
      const fx = force * dirX
      const fy = force * dirY

      // Split evenly; each endpoint converts to acceleration by its own mass.
      if (wa > 0) {
        p.accX[ia]! += fx * 0.5 * wa
        p.accY[ia]! += fy * 0.5 * wa
      }
      if (wb > 0) {
        p.accX[ib]! += fx * 0.5 * wb
        p.accY[ib]! += fy * 0.5 * wb
      }
    }

    // Objects: frontal length is the particle's own width.
    for (const c of this.clusters) {
      if (!c.alive) continue
      for (let k = 0; k < c.particles.length; k++) {
        const i = c.particles[k]!
        if (p.slots.alive[i] !== 1 || p.invMass[i] === 0) continue
        const exposure = 1 - this.water.submergedFraction(p.posX[i]!, p.posY[i]!, p.radius[i]!)
        if (exposure <= 0.01) continue
        const relX = this.wind.velocityAt(p.posX[i]!) - p.velX[i]!
        const relY = -p.velY[i]!
        const relSpeed = Math.sqrt(relX * relX + relY * relY)
        if (relSpeed < 1e-6) continue
        const force = half * 1.2 * (2 * p.radius[i]!) * relSpeed * relSpeed * exposure
        p.accX[i]! += ((force * relX) / relSpeed) * p.invMass[i]!
        p.accY[i]! += ((force * relY) / relSpeed) * p.invMass[i]!
      }
    }
  }

  /** Add a rectangular physics object. Returns its cluster. */
  addObject(spec: ObjectSpec): Cluster {
    const c = buildObject(this.particles, spec)
    this.clusters.push(c)
    return c
  }

  clearObjects(): void {
    const p = this.particles
    for (const c of this.clusters) {
      for (const i of c.particles) if (p.slots.alive[i] === 1) p.destroy(i)
      c.alive = false
    }
    this.clusters.length = 0
  }

  get objectCount(): number {
    return this.clusters.filter((c) => c.alive).length
  }

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
    this.clusters.length = 0
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
