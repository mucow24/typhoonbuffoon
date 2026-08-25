import type { Terrain } from '../world/terrain'
import { BendConstraints } from './constraints/bending'
import { DistanceConstraints } from './constraints/distance'
import { Cluster, buildObject, type ObjectSpec } from './clusters'
import { FluidSolver } from './fluid'
import { WaterField } from './water'
import { AIR_DENSITY, WindField } from './wind'
import { materialAt } from './materials'
import { KIND_BOUNDARY, KIND_FLUID, KIND_NODE, KIND_OBJECT, ParticleStore } from './particles'
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
  /**
   * Tangential velocity retained per contact FRAME for solids. 1 is
   * frictionless. Applied per substep as friction^(1/substeps), so the frame
   * semantics hold whatever the substep count - the old code applied the raw
   * factor every substep, which compounds to 0.65^12 ~ 0.6% retention and
   * glued anything touching the ground to it.
   */
  groundFriction = 0.65
  /**
   * Same, for FLUID. Water slides over a sandy bed; its dissipation should
   * come from viscosity, not a no-slip clamp. Mild, or dam-break fronts crawl.
   */
  fluidBedFriction = 0.9
  /**
   * Mass-proportional damping for FLUID, per second. PBF's projection is not
   * energy-conserving - violent scenes pump a little energy each bounce, and
   * the literature's answer is viscosity. Most of ours comes from XSPH; this
   * is a last-resort bulk term and should stay well below the structural
   * damping or waves die crossing the field. Zero disables.
   *
   * 0.02/s costs a wave ~18% over a 10-second crossing and takes the settled
   * pool's residual jiggle from 0.059 to 0.040 m/s - measured on the beach
   * rest probe; the settle tests depend on it.
   */
  fluidDamping = 0.02
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
   * STRUCTURE NODES ONLY - see updateVelocities.
   *
   * The constraint dampers are stiffness-proportional and so target local,
   * high-frequency modes; they barely touch a whole structure swinging bodily
   * about its base, which is a low-frequency global mode with far more inertia
   * than the local estimate. Without a little of this, an unloaded structure
   * wobbles forever instead of settling.
   *
   * Keep it small, and keep it OFF the fluid: applied to water it bled ~30% of
   * every wave and current per second, which is most of why the flood felt
   * like syrup. The plan's own prescription is alpha ~ 0; this is the one
   * structural exception.
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
  /** Fraction of a contact penetration resolved per substep. */
  contactRelaxation = 0.35
  /** How much of the inbound normal velocity a contact removes, 0..1. */
  contactNormalDamping = 0.6
  /**
   * Hard cap on one substep's contact push, metres.
   *
   * Stated in metres but the constraint that matters is velocity: a push is
   * differentiated by the substep, so 0.05 m over 1.4 ms implies 36 m/s and
   * 0.02 m implies 14. The larger value was letting a floating box work the
   * water around it up to 19 m/s.
   */
  maxContactCorrection = 0.02
  /** Hard cap on one substep's terrain push, metres. */
  maxTerrainPush = 0.35
  private contactDX = new Float32Array(4096)
  private contactDY = new Float32Array(4096)
  private contactHits = new Int32Array(4096)
  private contactStamp = new Int32Array(4096)
  private stampCounter = 0
  /**
   * Reactions owed to member ENDPOINTS from capsule contacts, kept apart from
   * contactDX because they are applied with different semantics: position
   * only, no prev carry, so sustained water pressure becomes velocity and then
   * force through the constraint solve - which is what loads a flood wall.
   */
  private memberDX = new Float32Array(4096)
  private memberDY = new Float32Array(4096)
  private memberHits = new Int32Array(4096)
  private readonly tmpVel = { x: 0, y: 0 }
  /**
   * Constraint indices destroyed by breakage since the last drain. The session
   * MUST drain these and drop them from its member records - a freed index is
   * recycled by the next build action, and destroying it a second time through
   * a stale record kills whatever innocent constraint lives there now.
   */
  readonly destroyedDistance: number[] = []
  readonly destroyedBends: number[] = []
  /**
   * Sample the terrain as static solid particles so the ground exerts pressure.
   *
   * Without this the fluid has no neighbours below it near the bed: density
   * reads low, lambda is zero, and the bottom layer gets no pressure support.
   * On a flat floor that is survivable, but on a slope the unopposed component
   * of gravity makes the bottom layer creep downhill forever - a measured net
   * drift of +0.087 m/s that decayed only as the pile slowly rearranged. It is
   * the same omission that stopped objects floating, one layer down.
   *
   * Two rows deep, because a single row leaves the kernel half empty and only
   * half-fixes the deficit.
   */
  rebuildTerrainBoundary(): void {
    const p = this.particles
    for (let i = 0; i < p.highWater; i++) {
      if (p.slots.alive[i] === 1 && p.kind[i] === KIND_BOUNDARY) p.destroy(i)
    }
    const t = this.terrain
    if (!t) return

    const spacing = this.fluid.spacing
    const area = spacing * spacing
    for (let x = this.boundsX0; x <= this.boundsX1; x += spacing) {
      const surface = t.heightAt(x)
      for (let row = 0; row < 2; row++) {
        p.create({
          x,
          y: surface - spacing * (0.5 + row),
          invMass: 0,
          radius: spacing * 0.5,
          kind: KIND_BOUNDARY,
          volume: area,
        })
      }
    }
  }

  /** Rebuilt automatically when the terrain, bounds or resolution change. */
  private boundarySignature = ''

  private ensureTerrainBoundary(): void {
    const t = this.terrain
    const sig = t
      ? `${t.x0}:${t.x1}:${t.heights.length}:${t.heights[0]}:${t.heights[t.heights.length - 1]}:` +
        `${this.fluid.spacing}:${this.boundsX0}:${this.boundsX1}`
      : ''
    if (sig === this.boundarySignature) return
    this.boundarySignature = sig
    this.rebuildTerrainBoundary()
  }

  step(dt: number): void {
    this.ensureTerrainBoundary()
    const h = dt / this.substeps
    this.water.build(
      this.particles,
      this.boundsX0,
      this.boundsX1,
      this.fluid.spacing * this.fluid.spacing,
    )
    this.applyBuoyancy()
    this.applyHydrostaticLoad()
    this.applyWaterDrag(dt)
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
      // EVERY substep. Projecting on every third one let density error build
      // across two unpressurised substeps and then discharged it in a single
      // correction, which (x - x_prev)/h turns into a velocity three times too
      // large before the tiny substep h multiplies it again. Small steps only
      // work when every step is corrected.
      this.fluid.project(this.particles, h)
      this.beginContacts()
      this.solveSolidContacts()
      this.solveMemberContacts()
      this.resolveContacts()
      this.resolveMemberReactions()
      this.solveContacts()
      this.updateVelocities(h)
      this.bend.dampVelocities(this.particles, h)
      this.distance.dampVelocities(this.particles, h)
      this.fluid.applyHullViscosity(this.particles)
      if (this.fluid.viscosityEverySubstep) this.fluid.applyViscosity(this.particles)
    }
    if (!this.fluid.viscosityEverySubstep) this.fluid.applyViscosity(this.particles)
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
    const kind = p.kind
    const n = p.highWater
    const invH = 1 / h
    // Structure-only: water is not a damped oscillator to be settled, and
    // 0.35/s applied globally cost the fluid ~30% of its momentum per second.
    const keep = this.linearDamping > 0 ? Math.exp(-this.linearDamping * h) : 1
    const keepFluid = this.fluidDamping > 0 ? Math.exp(-this.fluidDamping * h) : 1
    const maxSpeed = this.maxSpeed
    const maxSpeed2 = maxSpeed * maxSpeed
    // Per-FRAME friction semantics, distributed over the substeps.
    const invSub = 1 / this.substeps
    const frictionSolid = Math.pow(this.groundFriction, invSub)
    const frictionFluid = Math.pow(this.fluidBedFriction, invSub)
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

      if (kind[i] === KIND_NODE) {
        vx *= keep
        vy *= keep
      } else if (kind[i] === KIND_FLUID && keepFluid < 1) {
        vx *= keepFluid
        vy *= keepFluid
      }

      if (grounded[i] === 1) {
        const friction = kind[i] === KIND_FLUID ? frictionFluid : frictionSolid
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

      // Field edges are contacts too, and need the same prev carry as terrain.
      // Snapping posX back inside without it makes every wall touch a velocity
      // spike - which is why water admitted AT the edges detonated on arrival.
      if (p.posX[i]! < this.boundsX0 + r || p.posX[i]! > this.boundsX1 - r) {
        p.posX[i] = p.posX[i]! < this.boundsX0 + r ? this.boundsX0 + r : this.boundsX1 - r
        p.prevX[i] = p.posX[i]!
      }

      if (t) {
        const floor = t.heightAt(p.posX[i]!) + r
        if (p.posY[i]! < floor) {
          // Resolve the penetration fully, but carry prevY along with it so the
          // correction is never differentiated into velocity.
          //
          // The projection is vertical, so on steep ground - a basin wall, a
          // cliff - a buried particle is moved metres in one 1.4 ms substep,
          // and (pos - prev)/h reads that as hundreds of m/s. Easing the push
          // instead just leaves particles buried and the terrain stops holding
          // water. Moving prev with the correction gives the right thing: an
          // inelastic contact, position resolved and normal velocity killed.
          // Cap how far one substep may lift a particle, and carry prevY with
          // it so the correction is never differentiated into velocity. Without
          // the cap, steep ground teleports a buried particle metres upward and
          // invents the potential energy to match; without the prev carry, the
          // push reads as hundreds of m/s.
          // Push ALONG THE NORMAL, not straight up. For a heightfield the
          // distance to the surface along the normal is the vertical gap times
          // ny, so a vertical push overshoots on any slope - and repeatedly
          // overshooting at the foot of a bank is what kept flicking particles
          // out of an otherwise still pool.
          const nrm = t.normalAt(p.posX[i]!)
          const gap = Math.min(floor - p.posY[i]!, this.maxTerrainPush)
          const d = gap * nrm.ny
          p.posX[i]! += nrm.nx * d
          p.posY[i]! += nrm.ny * d
          p.prevX[i]! += nrm.nx * d
          p.prevY[i]! += nrm.ny * d

          // Remove the velocity going INTO the surface and leave the along
          // slope part alone. Zeroing the whole vertical component instead is a
          // discontinuity on any slope: it destroys downhill motion as well as
          // impact.
          const rx = p.posX[i]! - p.prevX[i]!
          const ry = p.posY[i]! - p.prevY[i]!
          const vn = rx * nrm.nx + ry * nrm.ny
          if (vn < 0) {
            p.prevX[i] = p.posX[i]! - (rx - vn * nrm.nx)
            p.prevY[i] = p.posY[i]! - (ry - vn * nrm.ny)
          }
          this.grounded[i] = 1
        }
      }
    }
  }

  private grounded = new Uint8Array(4096)

  /**
   * Buoyancy from REST volume, for nodes AND object particles, once per frame
   * into the acceleration accumulators.
   *
   * Net acceleration is g * (rhoWater/rhoBody - 1), so wood at 500 kg/m^3 rises
   * and steel at 7850 sinks with no flag anywhere - density alone decides it.
   *
   * Objects briefly had buoyancy from the pressure field alone (they are in
   * the density estimate, so the water pushes on them directly). Measured:
   * that lift exists only near the FREE SURFACE - PBF's per-substep lambda
   * carries no depth gradient in the bulk, so a wood crate released 5 m down
   * hovered there forever at any interface stiffness. The analytic term works
   * at any depth and is resolution-independent; the pressure interaction
   * stays for what it is actually good at - displacing water around the hull
   * and transmitting waves and slams.
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

      let frac = this.water.submergedFraction(p.posX[i]!, p.posY[i]!, p.radius[i]!)
      if (frac <= 0) continue

      // Object particles scale by measured wetness (last frame's neighbour
      // census). The column field says "below the local waterline"; only
      // actual fluid contact says "in water" - a house standing on a wet
      // slope is beside the puddle, not in it, and lifting it on the column
      // reading flickered it into orbit.
      if (p.kind[i] === KIND_OBJECT) {
        const wet = this.fluid.solidWetCount[i] ?? 0
        frac *= Math.min(1, wet / 8)
        if (frac <= 0) continue
      }

      const mass = 1 / p.invMass[i]!
      p.accY[i]! += g * ((this.waterDensity * vol) / mass) * frac
    }
  }

  /**
   * Static water pressure on members, from the column height field.
   *
   * Positional capsule contacts transmit the *dynamic* exchange - impacts,
   * currents shoving - but at rest the water stands off the capsule and the
   * per-substep penetration flux is tiny, so contacts alone deliver a few
   * percent of the true hydrostatic load. The static load is computed the way
   * the rest of the member regime computes water effects: from the height
   * field, resolution-independent by construction.
   *
   * Per member: sample the water surface a little to each side, take the
   * pressure difference at the member's depth, and load its VERTICAL extent
   * with the net horizontal force. Only the horizontal component is applied -
   * the vertical pressure difference across a member IS buoyancy, which
   * applyBuoyancy already provides from rest volume. Integrated over a wall
   * this reproduces the 1/2 rho g H^2 resultant, which is the load the whole
   * flood-wall archetype is about.
   *
   * Known limit: a sealed vessel (water inside a dome vs outside) shares one
   * column, so interior/exterior pressure cannot differ at the same x. The
   * dome archetype is deferred in the plan for exactly this kind of reason.
   */
  private applyHydrostaticLoad(): void {
    const p = this.particles
    const d = this.distance
    const alive = d.slots.alive
    const g = -this.gravity
    const rhoG = this.waterDensity * g
    const sampleOffset = Math.max(1.5 * this.water.resolution, 1.0)

    for (let m = 0; m < d.highWater; m++) {
      if (alive[m] !== 1) continue
      const rest = d.rest[m]!
      if (rest <= 1e-6) continue
      const ia = d.a[m]!
      const ib = d.b[m]!
      const wa = p.invMass[ia]!
      const wb = p.invMass[ib]!
      if (wa === 0 && wb === 0) continue

      const vertical = Math.abs(p.posY[ib]! - p.posY[ia]!)
      if (vertical < 0.05) continue

      const midX = (p.posX[ia]! + p.posX[ib]!) * 0.5
      const midY = (p.posY[ia]! + p.posY[ib]!) * 0.5

      const surfL = this.water.surfaceAt(midX - sampleOffset)
      const surfR = this.water.surfaceAt(midX + sampleOffset)
      const pL = surfL === -Infinity ? 0 : Math.max(0, rhoG * (surfL - midY))
      const pR = surfR === -Infinity ? 0 : Math.max(0, rhoG * (surfR - midY))
      const net = pL - pR
      if (net === 0) continue

      const fx = net * vertical
      if (wa > 0) p.accX[ia]! += fx * 0.5 * wa
      if (wb > 0) p.accX[ib]! += fx * 0.5 * wb
    }
  }

  /**
   * Water drag: the plan's F = 1/2 rho Cd A v_rel^2 law, against the LOCAL
   * water velocity so a current or a passing wave pushes a body along instead
   * of only slowing it down. Frontal geometry comes from REST shape - the same
   * rule as buoyancy and wind, and for the same runaway reason.
   *
   * Members are loaded like applyWind loads them: per member, projected across
   * the flow, split to the endpoints. The old implementation was a
   * mass-proportional relaxation (accel = -2.2/s * v_rel), which made drag
   * scale with MASS instead of area - physically backwards, and two orders of
   * magnitude below a real wave load on a wall.
   */
  private applyWaterDrag(dt: number): void {
    const p = this.particles
    const half = 0.5 * this.waterDensity
    const d = this.distance
    const alive = d.slots.alive

    for (let m = 0; m < d.highWater; m++) {
      if (alive[m] !== 1) continue
      const rest = d.rest[m]!
      if (rest <= 1e-6) continue // welds have no frontal area
      const ia = d.a[m]!
      const ib = d.b[m]!
      const wa = p.invMass[ia]!
      const wb = p.invMass[ib]!
      if (wa === 0 && wb === 0) continue

      const midX = (p.posX[ia]! + p.posX[ib]!) * 0.5
      const midY = (p.posY[ia]! + p.posY[ib]!) * 0.5
      const mat = materialAt(d.material[m]!)
      const submerged = this.water.submergedFraction(midX, midY, mat.section * 0.5)
      if (submerged <= 0.01) continue

      this.water.velocityAt(midX, this.tmpVel)
      const relX = this.tmpVel.x - (p.velX[ia]! + p.velX[ib]!) * 0.5
      const relY = this.tmpVel.y - (p.velY[ia]! + p.velY[ib]!) * 0.5
      const relSpeed = Math.sqrt(relX * relX + relY * relY)
      if (relSpeed < 1e-6) continue

      const ex = p.posX[ib]! - p.posX[ia]!
      const ey = p.posY[ib]! - p.posY[ia]!
      const len = Math.sqrt(ex * ex + ey * ey)
      if (len < 1e-9) continue
      const dirX = relX / relSpeed
      const dirY = relY / relSpeed
      const cross = Math.abs((ex * dirY - ey * dirX) / len)
      const frontal = rest * cross + mat.section

      let force = half * mat.dragCoefficient * frontal * relSpeed * relSpeed * submerged
      // Drag may slow the relative motion, never reverse it: explicit
      // quadratic drag on a light node at high v_rel overshoots the shared
      // velocity and oscillates. Cap the impulse at what stops the motion.
      const massAB = (wa > 0 ? 1 / wa : 0) + (wb > 0 ? 1 / wb : 0)
      const maxForce = (massAB * relSpeed) / dt
      if (force > maxForce) force = maxForce
      const fx = force * dirX
      const fy = force * dirY

      if (wa > 0) {
        p.accX[ia]! += fx * 0.5 * wa
        p.accY[ia]! += fy * 0.5 * wa
      }
      if (wb > 0) {
        p.accX[ib]! += fx * 0.5 * wb
        p.accY[ib]! += fy * 0.5 * wb
      }
    }

    // Objects get no force-based drag here. Their drag is the pairwise hull
    // viscosity in the fluid solver (fluid.applyHullViscosity): quadratic
    // drag against the column mean misfired in spray, and drag against the
    // measured local flow read the hull's own entrained wake and vanished.
  }

  /**
   * Particles against structural members, TWO-WAY. Members are line segments
   * treated as capsules; a penetrating particle is pushed out and the member's
   * endpoints receive the equal-and-opposite share, split by inverse mass with
   * the standard PBD barycentric weights. This is the channel that loads a
   * flood wall: settled water leans on the capsule every substep, and the
   * sustained pushback becomes strain in the wall's constraints. It is also
   * what lets a crate rest ON a platform and bend it.
   *
   * The particle side keeps the inelastic prev-carry semantics of
   * resolveContacts; the member side is position-only (resolveMemberReactions)
   * so sustained pressure reads as force rather than being silently cancelled.
   *
   * The capsule radius has a floor of 0.75 * particle spacing. A member thinner
   * than about one particle spacing is not watertight and fluid tunnels through
   * it, which on a flood wall reads as a bug rather than a near miss.
   */
  private solveMemberContacts(): void {
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
      const rest = d.rest[m]!
      if (rest <= 1e-6) continue // welds are points, not surfaces
      const ia = d.a[m]!
      const ib = d.b[m]!
      const ax = p.posX[ia]!
      const ay = p.posY[ia]!
      const bx = p.posX[ib]!
      const by = p.posY[ib]!
      const mat = materialAt(d.material[m]!)
      const radius = Math.max(mat.section * 0.5, spacing * 0.75)
      const noCollide = d.noCollideCluster[m]!
      const wa = p.invMass[ia]!
      const wb = p.invMass[ib]!

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
            if (j === ia || j === ib) continue
            // A member bolted to an object must not collide with that object,
            // or the capsule fights the weld and pumps energy.
            if (noCollide >= 0 && p.cluster[j] === noCollide) continue

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

            // Split the pushout across the pair by inverse mass, with the
            // segment's effective inverse mass at the contact point. Gathered,
            // not applied here - see resolveContacts / resolveMemberReactions.
            // Resolved over several substeps rather than in one jump: a full
            // penetration of one particle spacing discharged in a single
            // 1.4 ms substep is a 280 m/s impulse.
            const wj = p.invMass[j]!
            const u1 = 1 - u
            const wSeg = wa * u1 * u1 + wb * u * u
            const wSum = wj + wSeg
            if (wSum <= 0) continue
            const pen = Math.min(minDist - dist, this.maxContactCorrection)
            const push = pen * this.contactRelaxation
            const scale = push / wSum

            this.contactDX[j]! += nx * scale * wj
            this.contactDY[j]! += ny * scale * wj
            this.contactHits[j]!++

            if (wa > 0) {
              this.memberDX[ia]! -= nx * scale * wa * u1
              this.memberDY[ia]! -= ny * scale * wa * u1
              this.memberHits[ia]!++
            }
            if (wb > 0) {
              this.memberDX[ib]! -= nx * scale * wb * u
              this.memberDY[ib]! -= ny * scale * wb * u
              this.memberHits[ib]!++
            }
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

  /**
   * Is there already fluid within `radius` of this point?
   *
   * Uses last frame's hash, which is plenty for deciding whether there is room
   * to admit a particle. Spawning water into occupied space produces an
   * enormous density error, and the correction that follows saturates the
   * velocity clamp - the flood slider detonating on contact was exactly this.
   */
  hasFluidNear(x: number, y: number, radius: number): boolean {
    const hash = this.fluid.hash
    const starts = hash.starts
    const entries = hash.entries
    const scratch = hash.scratch
    const p = this.particles
    const r2 = radius * radius
    const buckets = hash.collectBuckets(x, y)
    for (let s = 0; s < buckets; s++) {
      const b = scratch[s]!
      const end = starts[b + 1]!
      for (let k = starts[b]!; k < end; k++) {
        const j = entries[k]!
        if (p.slots.alive[j] !== 1 || p.kind[j] !== KIND_FLUID) continue
        const dx = p.posX[j]! - x
        const dy = p.posY[j]! - y
        if (dx * dx + dy * dy < r2) return true
      }
    }
    return false
  }

  private beginContacts(): void {
    const n = this.particles.highWater
    if (this.contactDX.length < n) {
      const cap = Math.max(n, 4096)
      this.contactDX = new Float32Array(cap)
      this.contactDY = new Float32Array(cap)
      this.contactHits = new Int32Array(cap)
    }
    if (this.memberDX.length < n) {
      const cap = Math.max(n, 4096)
      this.memberDX = new Float32Array(cap)
      this.memberDY = new Float32Array(cap)
      this.memberHits = new Int32Array(cap)
    }
    this.contactDX.fill(0, 0, n)
    this.contactDY.fill(0, 0, n)
    this.contactHits.fill(0, 0, n)
    this.memberDX.fill(0, 0, n)
    this.memberDY.fill(0, 0, n)
    this.memberHits.fill(0, 0, n)
  }

  /**
   * Object-object contact: particles of DIFFERENT clusters push apart, split
   * by inverse mass, through the same gather machinery as everything else.
   * Without this two crates pass through each other and nothing can ever
   * stack, dam, or pile up. Shape matching then keeps each cluster rigid
   * against the dent.
   */
  private solveSolidContacts(): void {
    if (this.clusters.length < 2) return
    const p = this.particles
    const hash = this.fluid.hash
    const cluster = p.cluster
    const alive = p.slots.alive

    for (const c of this.clusters) {
      if (!c.alive) continue
      for (let k = 0; k < c.particles.length; k++) {
        const i = c.particles[k]!
        if (alive[i] !== 1) continue
        const xi = p.posX[i]!
        const yi = p.posY[i]!
        const ri = p.radius[i]!
        const wi = p.invMass[i]!
        const ci = cluster[i]!

        const buckets = hash.collectBuckets(xi, yi)
        const starts = hash.starts
        const entries = hash.entries
        const scratch = hash.scratch
        for (let s = 0; s < buckets; s++) {
          const b = scratch[s]!
          const end = starts[b + 1]!
          for (let e = starts[b]!; e < end; e++) {
            const j = entries[e]!
            // Each unordered pair once, and only across clusters.
            if (j <= i) continue
            if (alive[j] !== 1 || p.kind[j] !== KIND_OBJECT) continue
            if (cluster[j] === ci) continue

            const dx = xi - p.posX[j]!
            const dy = yi - p.posY[j]!
            const minDist = ri + p.radius[j]!
            const d2 = dx * dx + dy * dy
            if (d2 >= minDist * minDist || d2 < 1e-12) continue
            const dist = Math.sqrt(d2)
            const nx = dx / dist
            const ny = dy / dist

            const wj = p.invMass[j]!
            const wSum = wi + wj
            if (wSum <= 0) continue
            const pen = Math.min(minDist - dist, this.maxContactCorrection)
            const scale = (pen * this.contactRelaxation) / wSum

            if (wi > 0) {
              this.contactDX[i]! += nx * scale * wi
              this.contactDY[i]! += ny * scale * wi
              this.contactHits[i]!++
            }
            if (wj > 0) {
              this.contactDX[j]! -= nx * scale * wj
              this.contactDY[j]! -= ny * scale * wj
              this.contactHits[j]!++
            }
          }
        }
      }
    }
  }

  /**
   * Apply the substep's member-endpoint reactions. SUMMED, not averaged:
   * thirty particles of settled water leaning on a wall are thirty separate
   * loads, and averaging them would cap the hydrostatic force at one
   * contact's worth however deep the water gets. The per-substep cap bounds
   * the transient case (a wave slamming every sample point at once).
   *
   * Position only - prev is deliberately NOT carried, so sustained contact
   * pressure differentiates into velocity, the constraint solve answers it
   * with strain, and the near-critical zeta damping keeps the exchange from
   * ringing. That chain is precisely "the water loads the wall".
   */
  private resolveMemberReactions(): void {
    const p = this.particles
    const n = p.highWater
    for (let i = 0; i < n; i++) {
      if (this.memberHits[i] === 0) continue
      if (p.slots.alive[i] !== 1 || p.invMass[i] === 0) continue
      let dx = this.memberDX[i]!
      let dy = this.memberDY[i]!
      const mag = Math.sqrt(dx * dx + dy * dy)
      if (mag < 1e-12) continue
      if (mag > this.maxContactCorrection) {
        const k = this.maxContactCorrection / mag
        dx *= k
        dy *= k
      }
      p.posX[i]! += dx
      p.posY[i]! += dy
    }
  }

  /**
   * Apply the substep's contacts once per particle, not once per contact.
   *
   * Applying each contact as it is found lets them STACK: a fluid particle
   * caught between several particles of the same object collects a push from
   * every one of them inside a single substep, and five pushes of 0.0175 m over
   * 1.4 ms is 63 m/s. That is the spray that erupts the instant anything
   * touches the water.
   *
   * Averaging the corrections instead of summing them also makes the result
   * independent of the order contacts happen to be visited in, which is what
   * makes it reproducible.
   */
  private resolveContacts(): void {
    const p = this.particles
    const n = p.highWater
    for (let j = 0; j < n; j++) {
      const hits = this.contactHits[j]!
      if (hits === 0) continue
      if (p.slots.alive[j] !== 1 || p.invMass[j] === 0) continue

      let dx = this.contactDX[j]! / hits
      let dy = this.contactDY[j]! / hits
      const mag = Math.sqrt(dx * dx + dy * dy)
      if (mag < 1e-12) continue
      if (mag > this.maxContactCorrection) {
        const k = this.maxContactCorrection / mag
        dx *= k
        dy *= k
      }

      // BOTH halves, together. Each alone is wrong, and I tried each alone:
      //
      //  - move pos only: the push is differentiated into velocity and erupts
      //    as spray the instant anything enters the water;
      //  - move pos and prev only: no spray, but the velocity driving the
      //    particle into the solid survives, so it burrows through and a flood
      //    wall leaks a third of its water.
      //
      // Moving prev with the push makes the correction itself contribute no
      // velocity. (pos - prev) is then exactly the motion the particle already
      // had, and the inbound normal part of THAT is what a contact should kill.
      p.posX[j]! += dx
      p.posY[j]! += dy
      p.prevX[j]! += dx
      p.prevY[j]! += dy

      const inv = 1 / Math.hypot(dx, dy)
      const nx = dx * inv
      const ny = dy * inv
      const rx = p.posX[j]! - p.prevX[j]!
      const ry = p.posY[j]! - p.prevY[j]!
      const vn = rx * nx + ry * ny
      if (vn < 0) {
        // Damp the inbound normal velocity rather than annihilating it.
        // Removing it outright freezes the water against a hull: the level
        // beneath a floating object stops replenishing, the height field reads
        // a lower surface there, buoyancy falls with it and the object sinks.
        const keep = 1 - this.contactNormalDamping
        p.prevX[j] = p.posX[j]! - (rx - vn * nx * (1 - keep))
        p.prevY[j] = p.posY[j]! - (ry - vn * ny * (1 - keep))
      }
    }
  }

  /** Add a rectangular physics object. Returns its cluster. */
  addObject(spec: ObjectSpec): Cluster {
    // An object dropped into existing water must DISPLACE it - materialising
    // solid particles inside fluid ones is a teleport, and the density error
    // it creates discharges as spray however the solver is tuned. The flood
    // driver replaces the volume at its own admission rate.
    this.clearFluidInRect(
      spec.cx - spec.width * 0.5,
      spec.cx + spec.width * 0.5,
      spec.cy - spec.height * 0.5,
      spec.cy + spec.height * 0.5,
    )
    // Sampled at the fluid's own resolution unless told otherwise. Coarser than
    // that and water simply flows between the object's particles: the density
    // estimate never sees a solid, so nothing floats.
    const c = buildObject(this.particles, {
      spacing: this.fluid.spacing,
      clusterIndex: this.clusters.length,
      ...spec,
    })
    this.clusters.push(c)
    return c
  }

  private clearFluidInRect(x0: number, x1: number, y0: number, y1: number): void {
    const p = this.particles
    const pad = this.fluid.spacing * 0.5
    for (let i = 0; i < p.highWater; i++) {
      if (p.slots.alive[i] !== 1 || p.kind[i] !== KIND_FLUID) continue
      const x = p.posX[i]!
      const y = p.posY[i]!
      if (x > x0 - pad && x < x1 + pad && y > y0 - pad && y < y1 + pad) p.destroy(i)
    }
  }

  /** Index of a cluster in this world's list, for contact exclusions. -1 if absent. */
  clusterIndexOf(cluster: Cluster): number {
    return this.clusters.indexOf(cluster)
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

    // Bending failure. Transverse load - wind on a mast, a wave against a
    // stilt - rotates segments without elongating them, so a member under the
    // genre's primary load case accumulates no axial strain at all. Without a
    // break path on the ANGLE, a beam could fold double and never snap.
    const b = this.bend
    const bAlive = b.slots.alive
    const bn = b.highWater
    for (let i = 0; i < bn; i++) {
      if (bAlive[i] !== 1) continue
      const m = materialAt(b.material[i]!)
      const signed = b.angle[i]!
      const a = Math.abs(signed)

      const load = m.breakAngle > 0 && Number.isFinite(m.breakAngle) ? a / m.breakAngle : 0
      if (load > m.damageOnset && m.damageOnset < 1) {
        const over = (load - m.damageOnset) / (1 - m.damageOnset)
        b.damage[i] = Math.min(0.9, b.damage[i]! + m.damageRate * over * over * dt)
      }

      // Steel takes a permanent set in bending - the rest angle migrates
      // toward the deformed shape, so it stays bent when the load lifts.
      if (m.plasticRate > 0 && Number.isFinite(m.yieldAngle) && a > m.yieldAngle) {
        const excess = signed - Math.sign(signed) * m.yieldAngle
        b.restAngle[i]! += excess * m.plasticRate * dt
      }

      if (a > m.breakAngle * (1 - b.damage[i]!)) {
        this.breakBend(i, signed)
      }
    }
  }

  /**
   * A joint snapping in bending severs the member there: the bend goes, and so
   * does the weaker of the two distance segments meeting at the joint -
   * a fracture is a break in the material, not just a freed hinge.
   */
  private breakBend(i: number, angle: number): void {
    const b = this.bend
    const ib = b.b[i]!
    const ia = b.a[i]!
    const ic = b.c[i]!

    let victim = -1
    let worst = -1
    const d = this.distance
    for (let j = 0; j < d.highWater; j++) {
      if (d.slots.alive[j] !== 1) continue
      const x = d.a[j]!
      const y = d.b[j]!
      const spansAB = (x === ia && y === ib) || (x === ib && y === ia)
      const spansBC = (x === ib && y === ic) || (x === ic && y === ib)
      if (!spansAB && !spansBC) continue
      const s = Math.abs(d.strain[j]!)
      if (s > worst) {
        worst = s
        victim = j
      }
    }

    b.destroy(i)
    this.destroyedBends.push(i)
    if (victim >= 0) {
      this.breakConstraint(victim, Math.sign(angle) * worst)
    } else {
      // No adjacent segment found (already broken): still report the event.
      const p = this.particles
      this.breakEvents.push({
        a: ia,
        b: ic,
        strain: angle,
        x: p.posX[ib]!,
        y: p.posY[ib]!,
        material: b.material[i]!,
      })
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
    this.destroyedDistance.push(i)
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
        this.destroyedBends.push(j)
      }
    }
  }

  /**
   * Constraint indices freed by breakage since the last call. Whoever keeps
   * external records of constraint indices (the Session's member table) must
   * call this every frame and drop the listed indices, BEFORE acting on those
   * records - freed slots are recycled by the very next create.
   */
  drainDestroyed(): { distance: number[]; bends: number[] } {
    const out = { distance: [...this.destroyedDistance], bends: [...this.destroyedBends] }
    this.destroyedDistance.length = 0
    this.destroyedBends.length = 0
    return out
  }

  clear(): void {
    this.boundarySignature = ''
    this.particles.clear()
    this.distance.clear()
    this.bend.clear()
    this.clusters.length = 0
    this.breakEvents.length = 0
    this.destroyedDistance.length = 0
    this.destroyedBends.length = 0
  }

  /**
   * Clear the BUILT world - structure nodes, members, objects - while leaving
   * fluid and the terrain boundary alone. This is what a live-edit rebuild
   * wants: pressing Ctrl+Z during a flood should revert the build, not
   * vaporise several thousand water particles and un-flood the level.
   */
  clearStructures(): void {
    const p = this.particles
    for (let i = 0; i < p.highWater; i++) {
      if (p.slots.alive[i] !== 1) continue
      const kind = p.kind[i]
      if (kind === KIND_FLUID || kind === KIND_BOUNDARY) continue
      p.destroy(i)
    }
    this.distance.clear()
    this.bend.clear()
    this.clusters.length = 0
    this.breakEvents.length = 0
    this.destroyedDistance.length = 0
    this.destroyedBends.length = 0
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
        // Never spawn into occupied space - same rule as the flood inflow, for
        // the same reason: an overlapped pair is a density error the solver
        // can only answer with a violent correction. Dumping water onto water
        // fills the gaps and skips the rest.
        if (this.hasFluidNear(x, y, spacing * 0.85)) continue
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

  /**
   * Change the fluid resolution, keeping existing water consistent. The PBF
   * solve hoists ONE particle mass for all fluid, so existing particles must
   * be restamped to the new mass and radius or the solver mixes two masses
   * under one assumption - the old slider left them stale. The water's
   * represented volume reinterprets with the new spacing; the flood driver
   * tops it up or drains it toward the target level on the next frames.
   */
  setFluidSpacing(spacing: number): void {
    if (!(spacing > 0) || spacing === this.fluid.spacing) return
    this.fluid.spacing = spacing
    const p = this.particles
    const invMass = 1 / this.fluid.particleMass
    for (let i = 0; i < p.highWater; i++) {
      if (p.slots.alive[i] !== 1 || p.kind[i] !== KIND_FLUID) continue
      p.invMass[i] = invMass
      p.radius[i] = spacing * 0.5
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
