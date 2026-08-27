import { KIND_OBJECT, type ParticleStore } from './particles'

export interface ClusterFrame {
  cx: number
  cy: number
  angle: number
}

/**
 * Shape-matched particle clusters (Muller et al., Meshless Deformations Based
 * on Shape Matching, 2005) - the physics objects from docs/PLAN.md 5.8.
 *
 * An object is a handful of particles held in formation by a best-fit rigid
 * transform, which keeps it a first-class citizen of the same position-based
 * solver as everything else. Floating, spinning, wind drag and getting shoved
 * by a wave all fall out of the per-particle forces that already exist; there
 * is no coupling layer between a rigid-body world and a fluid world because
 * there is only one world.
 *
 * The trade is that rigidity is approximate rather than exact. For debris,
 * furniture and boats that is the right amount of looseness.
 */
export class Cluster {
  readonly particles: Int32Array
  private readonly restX: Float32Array
  private readonly restY: Float32Array
  private readonly mass: Float32Array
  readonly totalMass: number
  /** 0..1. 1 holds shape hard; lower is squashier. */
  stiffness = 1
  /** Per-substep correction cap, as a multiple of particle radius. */
  maxCorrection = 0.35
  alive = true

  cx = 0
  cy = 0
  angle = 0

  constructor(p: ParticleStore, indices: number[], stiffness = 1) {
    this.particles = Int32Array.from(indices)
    const n = indices.length
    this.restX = new Float32Array(n)
    this.restY = new Float32Array(n)
    this.mass = new Float32Array(n)
    this.stiffness = stiffness

    let total = 0
    let comX = 0
    let comY = 0
    for (let k = 0; k < n; k++) {
      const i = indices[k]!
      const m = p.invMass[i]! > 0 ? 1 / p.invMass[i]! : 1
      this.mass[k] = m
      total += m
      comX += p.posX[i]! * m
      comY += p.posY[i]! * m
    }
    this.totalMass = total
    comX /= total
    comY /= total
    this.cx = comX
    this.cy = comY

    for (let k = 0; k < n; k++) {
      const i = indices[k]!
      this.restX[k] = p.posX[i]! - comX
      this.restY[k] = p.posY[i]! - comY
    }
  }

  /** Project the cluster's particles back toward their rigid formation. */
  solve(p: ParticleStore): void {
    const idx = this.particles
    const n = idx.length
    if (n === 0) return

    let comX = 0
    let comY = 0
    let total = 0
    for (let k = 0; k < n; k++) {
      const i = idx[k]!
      const m = this.mass[k]!
      comX += p.posX[i]! * m
      comY += p.posY[i]! * m
      total += m
    }
    if (total <= 0) return
    comX /= total
    comY /= total

    // Best-fit rotation in 2D reduces to a single atan2: maximising
    // sum(p' . R q) over theta gives theta = atan2(B, A) with
    // A = sum m (p'x qx + p'y qy), B = sum m (p'y qx - p'x qy).
    let A = 0
    let B = 0
    for (let k = 0; k < n; k++) {
      const i = idx[k]!
      const m = this.mass[k]!
      const px = p.posX[i]! - comX
      const py = p.posY[i]! - comY
      const qx = this.restX[k]!
      const qy = this.restY[k]!
      A += m * (px * qx + py * qy)
      B += m * (py * qx - px * qy)
    }

    const theta = Math.atan2(B, A)
    const cos = Math.cos(theta)
    const sin = Math.sin(theta)

    this.cx = comX
    this.cy = comY
    this.angle = theta

    const k0 = this.stiffness
    for (let k = 0; k < n; k++) {
      const i = idx[k]!
      if (p.invMass[i] === 0) continue
      const qx = this.restX[k]!
      const qy = this.restY[k]!
      const gx = comX + cos * qx - sin * qy
      const gy = comY + sin * qx + cos * qy

      let dx = (gx - p.posX[i]!) * k0
      let dy = (gy - p.posY[i]!) * k0

      // Bound the correction. Shape matching and terrain contact fight each
      // other - contact pushes a particle out of the ground, this pulls it
      // back in - and because (pos - prev)/h turns position corrections into
      // velocity, an unbounded fight pumps energy in. An unsupported house rose
      // 6 m and flipped over. Same failure mode as the fluid corrections.
      const maxCorr = this.maxCorrection * p.radius[i]!
      const mag = Math.sqrt(dx * dx + dy * dy)
      if (mag > maxCorr && mag > 1e-12) {
        const s = maxCorr / mag
        dx *= s
        dy *= s
      }

      p.posX[i]! += dx
      p.posY[i]! += dy
    }
  }

  /** Transform a point given in object-local space into world space. */
  localToWorld(lx: number, ly: number): { x: number; y: number } {
    const cos = Math.cos(this.angle)
    const sin = Math.sin(this.angle)
    return { x: this.cx + cos * lx - sin * ly, y: this.cy + sin * lx + cos * ly }
  }

  /** Inverse of localToWorld, for binding an anchor at its current position. */
  worldToLocal(x: number, y: number): { x: number; y: number } {
    const cos = Math.cos(-this.angle)
    const sin = Math.sin(-this.angle)
    const dx = x - this.cx
    const dy = y - this.cy
    return { x: cos * dx - sin * dy, y: sin * dx + cos * dy }
  }

  /** Axis-aligned extent of the rest shape, for drawing. */
  restExtent(): { hw: number; hh: number } {
    let hw = 0
    let hh = 0
    for (let k = 0; k < this.restX.length; k++) {
      hw = Math.max(hw, Math.abs(this.restX[k]!))
      hh = Math.max(hh, Math.abs(this.restY[k]!))
    }
    return { hw, hh }
  }
}

export interface ObjectSpec {
  cx: number
  cy: number
  width: number
  height: number
  /** kg/m^3. Below 1000 floats, above sinks - not a flag. */
  density: number
  /**
   * Particle spacing for this object, independent of the fluid resolution.
   * A house resolved at fluid resolution would be hundreds of wasted
   * particles, and shape matching does not need density to stay rigid.
   */
  spacing?: number
  stiffness?: number
  /** Index of this cluster in the world's cluster list, stamped per particle. */
  clusterIndex?: number
}

/** Build a rectangular physics object as a shape-matched cluster. */
export function buildObject(p: ParticleStore, spec: ObjectSpec): Cluster {
  const spacing = spec.spacing ?? Math.max(0.4, Math.min(spec.width, spec.height) / 3)
  const nx = Math.max(2, Math.round(spec.width / spacing))
  const ny = Math.max(2, Math.round(spec.height / spacing))
  const dx = spec.width / (nx - 1)
  const dy = spec.height / (ny - 1)

  // Mass is the object's real mass shared between however many particles
  // represent it, so density behaves the same at any object resolution.
  const totalMass = spec.density * spec.width * spec.height
  const perParticle = totalMass / (nx * ny)
  const volumePer = (spec.width * spec.height) / (nx * ny)

  const indices: number[] = []
  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const i = p.create({
        x: spec.cx - spec.width * 0.5 + ix * dx,
        y: spec.cy - spec.height * 0.5 + iy * dy,
        invMass: 1 / perParticle,
        radius: Math.max(dx, dy) * 0.55,
        kind: KIND_OBJECT,
        // Cluster membership tag; the world overwrites this with the cluster's
        // index so contact passes can tell "same object" from "other object".
        cluster: spec.clusterIndex ?? -1,
      })
      p.volume[i] = volumePer
      indices.push(i)
    }
  }

  return new Cluster(p, indices, spec.stiffness ?? 1)
}
