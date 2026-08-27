import type { ParticleStore } from '../particles'
import { SlotAllocator } from '../slots'

export interface DistanceSpec {
  a: number
  b: number
  rest: number
  /** XPBD compliance (inverse stiffness). Small = stiff. */
  compliance: number
  /** Damping ratio, as a fraction of critical. ~0.8-1.0 settles without ringing. */
  zeta: number
  /** Index into MATERIAL_IDS. */
  material?: number
  /**
   * Cluster index this member's capsule must NOT collide with, -1 for none.
   * A stilt bolted to a house would otherwise fight its own weld: the capsule
   * pushes the house particles out while the weld pulls them back, and the
   * loop pumps energy until the house flips.
   */
  noCollideCluster?: number
  /**
   * Joinery, not structure: welds and anchor-mount links never break, take no
   * damage, and are invisible to load readouts. Their rest lengths are
   * centimetres, so ordinary strain arithmetic would rate a 1 cm flex on a
   * mount link as instant fracture.
   */
  unbreakable?: boolean
}

/**
 * XPBD distance constraints.
 *
 * Compliance enters the solve as alphaTilde = alpha / h^2, which is the whole
 * reason for XPBD over hand-rolled springs: effective stiffness stays constant
 * when the substep count changes, so retuning substeps does not retune the game,
 * and genuinely stiff constraints do not explode.
 */
export class DistanceConstraints {
  readonly slots: SlotAllocator

  a: Int32Array
  b: Int32Array
  rest: Float32Array
  /** Rest length AT CREATION - the ductility baseline. Plastic set migrates
   *  `rest`; how far it has migrated from here is the permanent deformation,
   *  and past the material's budget the member breaks instead of creeping
   *  forever into an inconsistent tangle. */
  rest0: Float32Array
  compliance: Float32Array
  zeta: Float32Array
  lambda: Float32Array
  /** (length - rest) / rest from the most recent solve. Drives colour and damage. */
  strain: Float32Array
  /** Index into MATERIAL_IDS. */
  material: Uint8Array
  /** Accumulated, irreversible. Lowers the effective break threshold. */
  damage: Float32Array
  /** Cluster index excluded from this member's capsule contacts, -1 for none. */
  noCollideCluster: Int32Array
  /** 1 = joinery (weld / mount link): never breaks, never rates as load. */
  unbreakable: Uint8Array

  constructor(capacity = 2048) {
    this.slots = new SlotAllocator(capacity)
    this.a = new Int32Array(capacity)
    this.b = new Int32Array(capacity)
    this.rest = new Float32Array(capacity)
    this.rest0 = new Float32Array(capacity)
    this.compliance = new Float32Array(capacity)
    this.zeta = new Float32Array(capacity)
    this.lambda = new Float32Array(capacity)
    this.strain = new Float32Array(capacity)
    this.material = new Uint8Array(capacity)
    this.damage = new Float32Array(capacity)
    this.noCollideCluster = new Int32Array(capacity)
    this.unbreakable = new Uint8Array(capacity)
    this.slots.onGrow = (cap) => this.grow(cap)
  }

  private grow(cap: number): void {
    this.a = SlotAllocator.growI32(this.a, cap)
    this.b = SlotAllocator.growI32(this.b, cap)
    this.rest = SlotAllocator.growF32(this.rest, cap)
    this.rest0 = SlotAllocator.growF32(this.rest0, cap)
    this.compliance = SlotAllocator.growF32(this.compliance, cap)
    this.zeta = SlotAllocator.growF32(this.zeta, cap)
    this.lambda = SlotAllocator.growF32(this.lambda, cap)
    this.strain = SlotAllocator.growF32(this.strain, cap)
    this.material = SlotAllocator.growU8(this.material, cap)
    this.damage = SlotAllocator.growF32(this.damage, cap)
    this.noCollideCluster = SlotAllocator.growI32(this.noCollideCluster, cap)
    this.unbreakable = SlotAllocator.growU8(this.unbreakable, cap)
  }

  get count(): number {
    return this.slots.liveCount
  }

  get highWater(): number {
    return this.slots.highWater
  }

  isAlive(i: number): boolean {
    return this.slots.isAlive(i)
  }

  create(spec: DistanceSpec): number {
    const i = this.slots.alloc()
    this.a[i] = spec.a
    this.b[i] = spec.b
    this.rest[i] = spec.rest
    this.rest0[i] = spec.rest
    this.compliance[i] = spec.compliance
    this.zeta[i] = spec.zeta
    this.lambda[i] = 0
    this.strain[i] = 0
    this.material[i] = spec.material ?? 0
    this.damage[i] = 0
    this.noCollideCluster[i] = spec.noCollideCluster ?? -1
    this.unbreakable[i] = spec.unbreakable ? 1 : 0
    return i
  }

  destroy(i: number): void {
    this.slots.release(i)
  }

  clear(): void {
    this.slots.clear()
  }

  resetLambda(): void {
    this.lambda.fill(0, 0, this.highWater)
  }

  solve(p: ParticleStore, h: number): void {
    const alive = this.slots.alive
    const invH2 = 1 / (h * h)
    const n = this.highWater

    for (let i = 0; i < n; i++) {
      if (alive[i] !== 1) continue
      const ia = this.a[i]!
      const ib = this.b[i]!
      const w1 = p.invMass[ia]!
      const w2 = p.invMass[ib]!
      const wSum = w1 + w2
      if (wSum === 0) continue

      const dx = p.posX[ia]! - p.posX[ib]!
      const dy = p.posY[ia]! - p.posY[ib]!
      const len = Math.sqrt(dx * dx + dy * dy)
      if (len < 1e-9) continue

      const rest = this.rest[i]!
      const C = len - rest
      const nx = dx / len
      const ny = dy / len

      const alphaTilde = this.compliance[i]! * invH2
      const dLambda = (-C - alphaTilde * this.lambda[i]!) / (wSum + alphaTilde)
      this.lambda[i]! += dLambda

      p.posX[ia]! += w1 * dLambda * nx
      p.posY[ia]! += w1 * dLambda * ny
      p.posX[ib]! -= w2 * dLambda * nx
      p.posY[ib]! -= w2 * dLambda * ny

      this.strain[i] = rest > 1e-9 ? C / rest : 0
    }
  }

  /**
   * Stiffness-proportional damping, applied to velocity.
   *
   * Only the component of relative velocity ALONG the constraint is removed, so
   * bulk translation and rotation of the structure are untouched while the
   * constraint's own vibration mode is killed. The rate is 2*zeta*omega with
   * omega = sqrt(k/m_eff), so stiffer constraints are damped harder - which is
   * the Rayleigh beta*K behaviour the design calls for: kill the buzz, keep the
   * slow lean.
   *
   * Removing a fraction of relative velocity is momentum conserving and can
   * never inject energy, so this cannot destabilise however hard it is driven.
   */
  dampVelocities(p: ParticleStore, h: number): void {
    const alive = this.slots.alive
    const n = this.highWater

    for (let i = 0; i < n; i++) {
      if (alive[i] !== 1) continue
      const zeta = this.zeta[i]!
      if (zeta <= 0) continue

      const ia = this.a[i]!
      const ib = this.b[i]!
      const w1 = p.invMass[ia]!
      const w2 = p.invMass[ib]!
      const wSum = w1 + w2
      if (wSum === 0) continue

      const dx = p.posX[ia]! - p.posX[ib]!
      const dy = p.posY[ia]! - p.posY[ib]!
      const len = Math.sqrt(dx * dx + dy * dy)
      if (len < 1e-9) continue
      const nx = dx / len
      const ny = dy / len

      const compliance = this.compliance[i]!
      const k = compliance > 1e-12 ? 1 / compliance : 1e12
      const omega = Math.sqrt(k * wSum)
      const factor = 1 - Math.exp(-2 * zeta * omega * h)

      const vrel = (p.velX[ia]! - p.velX[ib]!) * nx + (p.velY[ia]! - p.velY[ib]!) * ny
      const dv = vrel * factor

      const s1 = w1 / wSum
      const s2 = w2 / wSum
      p.velX[ia]! -= s1 * dv * nx
      p.velY[ia]! -= s1 * dv * ny
      p.velX[ib]! += s2 * dv * nx
      p.velY[ib]! += s2 * dv * ny
    }
  }

  /** Current length of a constraint, for diagnostics and stress readouts. */
  lengthOf(p: ParticleStore, i: number): number {
    const ia = this.a[i]!
    const ib = this.b[i]!
    return Math.hypot(p.posX[ia]! - p.posX[ib]!, p.posY[ia]! - p.posY[ib]!)
  }
}
