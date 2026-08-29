import { materialAt } from './materials'
import { KIND_FLUID, KIND_NODE, KIND_OBJECT } from './particles'
import type { SimWorld } from './world'

/**
 * The solver seam (docs/GPU_PLAN.md P2): everything per-substep, plus the
 * per-frame neighbour build and XSPH, behind a backend interface. The CPU
 * implementation below is the REFERENCE - the physics the whole test suite
 * pins down - and is a pure code motion of SimWorld's substep pipeline. The
 * WebGPU backend implements the same lifecycle against device buffers.
 *
 * SimWorld keeps everything frame-rate and host-authoritative: topology,
 * force passes, the water column field, damage/breakage. The backend owns
 * particle STATE evolution within a frame.
 */
export interface SolverBackend {
  /**
   * Reconcile host-side changes (topology creates/destroys, tuning params)
   * into backend-owned state, once per frame before step. CPU: nothing to
   * do - it works on the SoA arrays in place.
   */
  sync(): void
  /** Run one full frame: wave drive, neighbour build, all substeps, XSPH. */
  step(dt: number): void
  /**
   * FLUSH: make every submitted frame's outputs (positions, velocities,
   * strain, density...) visible to host logic, waiting as long as that
   * takes. CPU: already visible. This is the synchronous contract the
   * parity tests and SimWorld.stepAsync rely on.
   */
  readback(): Promise<void>
  /**
   * PIPELINED consume: apply whichever submitted frames are ALREADY
   * complete, without waiting - blocking only for backpressure when the
   * in-flight queue is full. The host loop calls this instead of
   * readback(), because a GPU fence takes ~17-21 ms to observe in real
   * browsers regardless of workload: any loop that waits for the CURRENT
   * frame's fence caps below 60 Hz with an empty scene. Absent on backends
   * with nothing in flight (CPU).
   */
  reap?(): Promise<void>
  /** Release device resources on backend swap. CPU backends have none. */
  dispose?(): void
}

/** Frame-level wave forcing, set by Conditions, applied by the backend -
 *  whichever device owns the velocities. */
export interface WaveDrive {
  /** Zone start: fluid with x >= x0 is driven. */
  x0: number
  /** Paddle velocity target, m/s (signed). */
  push: number
  /** Per-frame blend toward the target, 0..1. Nudges, never teleports. */
  blend: number
}

export class CpuSolver implements SolverBackend {
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
  private grounded = new Uint8Array(4096)

  constructor(private readonly w: SimWorld) {}

  sync(): void {
    // CPU reference: state lives in the world's SoA arrays already.
  }

  readback(): Promise<void> {
    return Promise.resolve()
  }

  step(dt: number): void {
    const w = this.w
    const prof = w.profileEnabled ? w.profile : null
    let t0 = 0
    const mark = prof ? () => (t0 = performance.now()) : () => {}
    const lap = prof
      ? (key: string) => {
          const t1 = performance.now()
          prof[key] = (prof[key] ?? 0) + (t1 - t0)
          t0 = t1
        }
      : () => {}

    this.applyWaveDrive()

    mark()
    const h = dt / w.substeps
    w.fluid.beginFrame(w.particles)
    lap('neighbours')

    for (let s = 0; s < w.substeps; s++) {
      mark()
      this.predict(h)
      lap('predict')
      w.bend.resetLambda()
      w.distance.resetLambda()
      // Bending first, axial last: in Gauss-Seidel the last solve dominates
      // locally, and members must hold their length before they hold their
      // shape. Stiff axially, compliant in bending.
      // Clusters FIRST, then the structural constraints. In Gauss-Seidel the
      // last solve wins locally: with shape matching last it snapped object
      // particles back to the rigid formation every substep, undoing the welds
      // that attach a structure to the object. The two then fought and pumped
      // energy in - a supported house rose and flipped. Structure last means
      // load actually transfers into the object.
      for (const c of w.clusters) if (c.alive) c.solve(w.particles)
      w.bend.solve(w.particles, h)
      w.distance.solve(w.particles, h)
      lap('structure')
      // EVERY substep. Projecting on every third one let density error build
      // across two unpressurised substeps and then discharged it in a single
      // correction, which (x - x_prev)/h turns into a velocity three times too
      // large before the tiny substep h multiplies it again. Small steps only
      // work when every step is corrected.
      w.fluid.project(w.particles, h)
      lap('project')
      this.beginContacts()
      this.solveSolidContacts()
      lap('solidContacts')
      this.solveMemberContacts()
      lap('memberContacts')
      this.resolveContacts()
      this.resolveMemberReactions()
      this.solveContacts()
      lap('terrain')
      this.updateVelocities(h)
      w.bend.dampVelocities(w.particles, h)
      w.distance.dampVelocities(w.particles, h)
      w.fluid.applyHullViscosity(w.particles)
      if (w.fluid.viscosityEverySubstep) w.fluid.applyViscosity(w.particles)
      lap('velocities')
    }
    mark()
    if (!w.fluid.viscosityEverySubstep) w.fluid.applyViscosity(w.particles)
    lap('xsph')
  }

  /**
   * Frame-level wave forcing (see Conditions.driveWaves for the paddle).
   * Lives in the backend because it mutates velocities, which the GPU path
   * owns device-side. Blend rather than set, so the paddle nudges the water
   * instead of teleporting its velocity and injecting a shock.
   */
  private applyWaveDrive(): void {
    const w = this.w
    const wd = w.waveDrive
    if (!wd) return
    const p = w.particles
    for (let i = 0; i < p.highWater; i++) {
      if (p.slots.alive[i] !== 1 || p.kind[i] !== KIND_FLUID) continue
      if (p.posX[i]! < wd.x0) continue
      p.velX[i]! += (wd.push - p.velX[i]!) * wd.blend
    }
  }

  private predict(h: number): void {
    const w = this.w
    const p = w.particles
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
      p.velY[i]! += (w.gravity + p.accY[i]!) * h
      p.posX[i]! += p.velX[i]! * h
      p.posY[i]! += p.velY[i]! * h
    }
  }

  private updateVelocities(h: number): void {
    const w = this.w
    const p = w.particles
    const alive = p.slots.alive
    const kind = p.kind
    const n = p.highWater
    const invH = 1 / h
    // Structure-only: water is not a damped oscillator to be settled, and
    // 0.35/s applied globally cost the fluid ~30% of its momentum per second.
    const keep = w.linearDamping > 0 ? Math.exp(-w.linearDamping * h) : 1
    const keepFluid = w.fluidDamping > 0 ? Math.exp(-w.fluidDamping * h) : 1
    const maxSpeed = w.maxSpeed
    const maxSpeed2 = maxSpeed * maxSpeed
    // Per-FRAME friction semantics, distributed over the substeps.
    const invSub = 1 / w.substeps
    const frictionSolid = Math.pow(w.groundFriction, invSub)
    const frictionFluid = Math.pow(w.fluidBedFriction, invSub)
    const restitution = w.groundRestitution
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
    const w = this.w
    const p = w.particles
    const alive = p.slots.alive
    const kind = p.kind
    const n = p.highWater
    const t = w.terrain
    const grounded = this.grounded
    if (grounded.length < n) this.grounded = new Uint8Array(Math.max(n, 1024))

    for (let i = 0; i < n; i++) {
      this.grounded[i] = 0
      if (alive[i] !== 1 || p.invMass[i] === 0) continue
      const r = p.radius[i]!

      // Field edges are contacts too, and need the same prev carry as terrain.
      // Snapping posX back inside without it makes every wall touch a velocity
      // spike - which is why water admitted AT the edges detonated on arrival.
      if (p.posX[i]! < w.boundsX0 + r || p.posX[i]! > w.boundsX1 - r) {
        p.posX[i] = p.posX[i]! < w.boundsX0 + r ? w.boundsX0 + r : w.boundsX1 - r
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
          // STRUCTURE nodes get a far gentler per-substep push than anything
          // else. A node buried under a dune sits in series with near-rigid
          // axial constraints, and a 0.35 m heave per substep against
          // EA ~ 1e8 does work against the spring every substep - a measured
          // energy pump that took a resting steel tower from 0 to the speed
          // cap and 18 snapped members inside a quarter of a second.
          // Build-time spawning keeps nodes out of the ground; this cap keeps
          // runtime burial (collapse debris, dragged joints) resolving gently
          // enough that the dampers win. Fluid needs the fast push to hold
          // water, and OBJECTS keep it too: their fight is bounded by the
          // cluster's own correction cap, and the gentle push just made a
          // landing house grind on the slope ten times longer.
          const pushCap = kind[i] === KIND_NODE ? w.maxTerrainPush * 0.1 : w.maxTerrainPush
          const nrm = t.normalAt(p.posX[i]!)
          const gap = Math.min(floor - p.posY[i]!, pushCap)
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

  private beginContacts(): void {
    const n = this.w.particles.highWater
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
    const w = this.w
    if (w.clusters.length < 2) return
    const p = w.particles
    const hash = w.fluid.hash
    const cluster = p.cluster
    const alive = p.slots.alive

    for (const c of w.clusters) {
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
            const pen = Math.min(minDist - dist, w.maxContactCorrection)
            const scale = (pen * w.contactRelaxation) / wSum

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
    const w = this.w
    const d = w.distance
    if (d.count === 0) return
    const p = w.particles
    const hash = w.fluid.hash
    const spacing = w.fluid.spacing
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
            const pen = Math.min(minDist - dist, w.maxContactCorrection)
            const push = pen * w.contactRelaxation
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
    const w = this.w
    const p = w.particles
    const n = p.highWater
    for (let i = 0; i < n; i++) {
      if (this.memberHits[i] === 0) continue
      if (p.slots.alive[i] !== 1 || p.invMass[i] === 0) continue
      let dx = this.memberDX[i]!
      let dy = this.memberDY[i]!
      const mag = Math.sqrt(dx * dx + dy * dy)
      if (mag < 1e-12) continue
      if (mag > w.maxContactCorrection) {
        const k = w.maxContactCorrection / mag
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
    const w = this.w
    const p = w.particles
    const n = p.highWater
    for (let j = 0; j < n; j++) {
      const hits = this.contactHits[j]!
      if (hits === 0) continue
      if (p.slots.alive[j] !== 1 || p.invMass[j] === 0) continue

      let dx = this.contactDX[j]! / hits
      let dy = this.contactDY[j]! / hits
      const mag = Math.sqrt(dx * dx + dy * dy)
      if (mag < 1e-12) continue
      if (mag > w.maxContactCorrection) {
        const k = w.maxContactCorrection / mag
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
        const keep = 1 - w.contactNormalDamping
        p.prevX[j] = p.posX[j]! - (rx - vn * nx * (1 - keep))
        p.prevY[j] = p.posY[j]! - (ry - vn * ny * (1 - keep))
      }
    }
  }
}
