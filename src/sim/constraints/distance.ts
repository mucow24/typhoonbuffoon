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
  compliance: Float32Array
  zeta: Float32Array
  lambda: Float32Array
  /** (length - rest) / rest from the most recent solve. Drives colour and damage. */
  strain: Float32Array
  /** Index into MATERIAL_IDS. */
  material: Uint8Array
  /** Accumulated, irreversible. Lowers the effective break threshold. */
  damage: Float32Array

  constructor(capacity = 2048) {
    this.slots = new SlotAllocator(capacity)
    this.a = new Int32Array(capacity)
    this.b = new Int32Array(capacity)
    this.rest = new Float32Array(capacity)
    this.compliance = new Float32Array(capacity)
    this.zeta = new Float32Array(capacity)
    this.lambda = new Float32Array(capacity)
    this.strain = new Float32Array(capacity)
    this.material = new Uint8Array(capacity)
    this.damage = new Float32Array(capacity)
    this.slots.onGrow = (cap) => this.grow(cap)
  }

  private grow(cap: number): void {
    this.a = SlotAllocator.growI32(this.a, cap)
    this.b = SlotAllocator.growI32(this.b, cap)
    this.rest = SlotAllocator.growF32(this.rest, cap)
    this.compliance = SlotAllocator.growF32(this.compliance, cap)
    this.zeta = SlotAllocator.growF32(this.zeta, cap)
    this.lambda = SlotAllocator.growF32(this.lambda, cap)
    this.strain = SlotAllocator.growF32(this.strain, cap)
    this.material = SlotAllocator.growU8(this.material, cap)
    this.damage = SlotAllocator.growF32(this.damage, cap)
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
    this.compliance[i] = spec.compliance
    this.zeta[i] = spec.zeta
    this.lambda[i] = 0
    this.strain[i] = 0
    this.material[i] = spec.material ?? 0
    this.damage[i] = 0
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
