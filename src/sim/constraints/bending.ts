import type { ParticleStore } from '../particles'
import { SlotAllocator } from './../slots'

export interface BendSpec {
  a: number
  b: number
  c: number
  /** Rest turn angle in radians. 0 for a straight member. */
  restAngle: number
  /** XPBD compliance in rad/(N*m). Large = bendy. This is where softness lives. */
  compliance: number
  zeta: number
}

const TWO_PI = Math.PI * 2

/** Shortest signed difference between two angles. */
function angleDelta(a: number, b: number): number {
  let d = (a - b) % TWO_PI
  if (d > Math.PI) d -= TWO_PI
  else if (d < -Math.PI) d += TWO_PI
  return d
}

/**
 * XPBD angle constraint over three consecutive particles.
 *
 * A distance constraint between the outer two particles is the usual cheap
 * "bend spring", but its response goes quadratic near straight: d|ac|/dtheta is
 * zero at theta = 0, so small deflections meet almost no resistance and the
 * member reads as floppy before it reads as stiff. A real angle constraint is
 * linear in curvature, which is what makes a trunk bow proportionally instead
 * of hinging.
 *
 * theta = angle(bc) - angle(ab), and with n1 = perp(ab)/|ab|^2,
 * n2 = perp(bc)/|bc|^2 the gradients are dtheta/da = n1, dtheta/dc = n2,
 * dtheta/db = -(n1 + n2). Those sum to zero, so the constraint is momentum
 * conserving and cannot translate the structure.
 */
export class BendConstraints {
  readonly slots: SlotAllocator

  a: Int32Array
  b: Int32Array
  c: Int32Array
  restAngle: Float32Array
  compliance: Float32Array
  zeta: Float32Array
  lambda: Float32Array
  /** Signed deflection from rest, radians, from the most recent solve. */
  angle: Float32Array

  constructor(capacity = 2048) {
    this.slots = new SlotAllocator(capacity)
    this.a = new Int32Array(capacity)
    this.b = new Int32Array(capacity)
    this.c = new Int32Array(capacity)
    this.restAngle = new Float32Array(capacity)
    this.compliance = new Float32Array(capacity)
    this.zeta = new Float32Array(capacity)
    this.lambda = new Float32Array(capacity)
    this.angle = new Float32Array(capacity)
    this.slots.onGrow = (cap) => this.grow(cap)
  }

  private grow(cap: number): void {
    this.a = SlotAllocator.growI32(this.a, cap)
    this.b = SlotAllocator.growI32(this.b, cap)
    this.c = SlotAllocator.growI32(this.c, cap)
    this.restAngle = SlotAllocator.growF32(this.restAngle, cap)
    this.compliance = SlotAllocator.growF32(this.compliance, cap)
    this.zeta = SlotAllocator.growF32(this.zeta, cap)
    this.lambda = SlotAllocator.growF32(this.lambda, cap)
    this.angle = SlotAllocator.growF32(this.angle, cap)
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

  create(spec: BendSpec): number {
    const i = this.slots.alloc()
    this.a[i] = spec.a
    this.b[i] = spec.b
    this.c[i] = spec.c
    this.restAngle[i] = spec.restAngle
    this.compliance[i] = spec.compliance
    this.zeta[i] = spec.zeta
    this.lambda[i] = 0
    this.angle[i] = 0
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

  /** Current turn angle at b, radians. */
  static turnAngle(p: ParticleStore, ia: number, ib: number, ic: number): number {
    const abx = p.posX[ib]! - p.posX[ia]!
    const aby = p.posY[ib]! - p.posY[ia]!
    const bcx = p.posX[ic]! - p.posX[ib]!
    const bcy = p.posY[ic]! - p.posY[ib]!
    return Math.atan2(abx * bcy - aby * bcx, abx * bcx + aby * bcy)
  }

  solve(p: ParticleStore, h: number): void {
    const alive = this.slots.alive
    const invH2 = 1 / (h * h)
    const n = this.highWater

    for (let i = 0; i < n; i++) {
      if (alive[i] !== 1) continue
      const ia = this.a[i]!
      const ib = this.b[i]!
      const ic = this.c[i]!
      const w0 = p.invMass[ia]!
      const w1 = p.invMass[ib]!
      const w2 = p.invMass[ic]!
      if (w0 + w1 + w2 === 0) continue

      const abx = p.posX[ib]! - p.posX[ia]!
      const aby = p.posY[ib]! - p.posY[ia]!
      const bcx = p.posX[ic]! - p.posX[ib]!
      const bcy = p.posY[ic]! - p.posY[ib]!
      const lab2 = abx * abx + aby * aby
      const lbc2 = bcx * bcx + bcy * bcy
      if (lab2 < 1e-12 || lbc2 < 1e-12) continue

      const theta = Math.atan2(abx * bcy - aby * bcx, abx * bcx + aby * bcy)
      const C = angleDelta(theta, this.restAngle[i]!)
      this.angle[i] = C

      // Gradients.
      const n1x = -aby / lab2
      const n1y = abx / lab2
      const n2x = -bcy / lbc2
      const n2y = bcx / lbc2
      const gbx = -(n1x + n2x)
      const gby = -(n1y + n2y)

      const denom =
        w0 * (n1x * n1x + n1y * n1y) +
        w1 * (gbx * gbx + gby * gby) +
        w2 * (n2x * n2x + n2y * n2y)
      if (denom < 1e-12) continue

      const alphaTilde = this.compliance[i]! * invH2
      const dLambda = (-C - alphaTilde * this.lambda[i]!) / (denom + alphaTilde)
      this.lambda[i]! += dLambda

      p.posX[ia]! += w0 * dLambda * n1x
      p.posY[ia]! += w0 * dLambda * n1y
      p.posX[ib]! += w1 * dLambda * gbx
      p.posY[ib]! += w1 * dLambda * gby
      p.posX[ic]! += w2 * dLambda * n2x
      p.posY[ic]! += w2 * dLambda * n2y
    }
  }

  /**
   * Damp the rate of change of the angle only, leaving bulk motion alone.
   * Same construction as the distance damper: remove a fraction of the
   * generalised velocity along the constraint gradient, which is momentum
   * conserving and cannot inject energy.
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
      const ic = this.c[i]!
      const w0 = p.invMass[ia]!
      const w1 = p.invMass[ib]!
      const w2 = p.invMass[ic]!
      if (w0 + w1 + w2 === 0) continue

      const abx = p.posX[ib]! - p.posX[ia]!
      const aby = p.posY[ib]! - p.posY[ia]!
      const bcx = p.posX[ic]! - p.posX[ib]!
      const bcy = p.posY[ic]! - p.posY[ib]!
      const lab2 = abx * abx + aby * aby
      const lbc2 = bcx * bcx + bcy * bcy
      if (lab2 < 1e-12 || lbc2 < 1e-12) continue

      const n1x = -aby / lab2
      const n1y = abx / lab2
      const n2x = -bcy / lbc2
      const n2y = bcx / lbc2
      const gbx = -(n1x + n2x)
      const gby = -(n1y + n2y)

      const denom =
        w0 * (n1x * n1x + n1y * n1y) +
        w1 * (gbx * gbx + gby * gby) +
        w2 * (n2x * n2x + n2y * n2y)
      if (denom < 1e-12) continue

      const thetaDot =
        n1x * p.velX[ia]! +
        n1y * p.velY[ia]! +
        gbx * p.velX[ib]! +
        gby * p.velY[ib]! +
        n2x * p.velX[ic]! +
        n2y * p.velY[ic]!

      const compliance = this.compliance[i]!
      const k = compliance > 1e-12 ? 1 / compliance : 1e12
      const omega = Math.sqrt(k * denom)
      const factor = 1 - Math.exp(-2 * zeta * omega * h)
      const s = (thetaDot * factor) / denom

      p.velX[ia]! -= w0 * s * n1x
      p.velY[ia]! -= w0 * s * n1y
      p.velX[ib]! -= w1 * s * gbx
      p.velY[ib]! -= w1 * s * gby
      p.velX[ic]! -= w2 * s * n2x
      p.velY[ic]! -= w2 * s * n2y
    }
  }
}
