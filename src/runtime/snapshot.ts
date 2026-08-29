import type { SimWorld } from '../sim/world'

/**
 * Binary snapshot body: the bulk data of one frame, packed into a single
 * transferable ArrayBuffer. Everything small or string-keyed (scalars, the
 * structure view-model, events) rides the message as structured clone; this
 * buffer carries only what scales with the world - per-particle state and
 * per-member-segment render data.
 *
 * Layout is parallel typed arrays with f32 blocks first so every view is
 * 4-byte aligned, then the u8 blocks:
 *
 *   u32 magic, u32 particleCount n, u32 segmentCount m, u32 reserved
 *   f32 posX[n], f32 posY[n]
 *   f32 segAx[m], segAy[m], segBx[m], segBy[m], segStrain[m], segDamage[m]
 *   u8  flags[n]        alive | kind<<1 | pinned
 *   u8  segMaterial[m]
 *
 * The particle block is SLOT-INDEXED and sparse, exactly like the sim's SoA
 * tables: dead slots carry flags 0. Index stability is what lets the fluid
 * renderer pool sprites and the client interpolate between two snapshots
 * without any identity bookkeeping.
 *
 * Member segments are packed DENSE in table order (per live distance
 * constraint, welds included - the render draws them all today), with world
 * endpoint positions resolved worker-side so no particle index ever crosses
 * the boundary.
 */

const MAGIC = 0x7b5eab01

export const FLAG_ALIVE = 1
export const FLAG_PINNED = 8
export const kindOfFlags = (flags: number): number => (flags >> 1) & 0x3

export interface SnapshotBody {
  particleCount: number
  posX: Float32Array
  posY: Float32Array
  flags: Uint8Array
  segmentCount: number
  segAx: Float32Array
  segAy: Float32Array
  segBx: Float32Array
  segBy: Float32Array
  segStrain: Float32Array
  segDamage: Float32Array
  segMaterial: Uint8Array
}

const HEADER_BYTES = 16

const byteSize = (n: number, m: number): number =>
  HEADER_BYTES + 8 * n + 24 * m + n + m

/**
 * Pack the world into `pool` if it is large enough, else a fresh buffer.
 * Returns the buffer written (which is then transferred to the main thread;
 * the client posts it back for reuse once it rotates out of the interpolation
 * pair).
 */
export function encodeSnapshot(sim: SimWorld, pool: ArrayBuffer | null): ArrayBuffer {
  const p = sim.particles
  const d = sim.distance
  const n = p.highWater

  let m = 0
  for (let i = 0; i < d.highWater; i++) if (d.slots.alive[i] === 1) m++

  const need = byteSize(n, m)
  const buffer = pool && pool.byteLength >= need ? pool : new ArrayBuffer(need)

  const header = new Uint32Array(buffer, 0, 4)
  header[0] = MAGIC
  header[1] = n
  header[2] = m
  header[3] = 0

  let off = HEADER_BYTES
  const posX = new Float32Array(buffer, off, n)
  off += 4 * n
  const posY = new Float32Array(buffer, off, n)
  off += 4 * n
  const segAx = new Float32Array(buffer, off, m)
  off += 4 * m
  const segAy = new Float32Array(buffer, off, m)
  off += 4 * m
  const segBx = new Float32Array(buffer, off, m)
  off += 4 * m
  const segBy = new Float32Array(buffer, off, m)
  off += 4 * m
  const segStrain = new Float32Array(buffer, off, m)
  off += 4 * m
  const segDamage = new Float32Array(buffer, off, m)
  off += 4 * m
  const flags = new Uint8Array(buffer, off, n)
  off += n
  const segMaterial = new Uint8Array(buffer, off, m)

  posX.set(p.posX.subarray(0, n))
  posY.set(p.posY.subarray(0, n))
  const alive = p.slots.alive
  const kind = p.kind
  const invMass = p.invMass
  for (let i = 0; i < n; i++) {
    flags[i] = alive[i] === 1
      ? FLAG_ALIVE | (kind[i]! << 1) | (invMass[i] === 0 ? FLAG_PINNED : 0)
      : 0
  }

  let k = 0
  const dAlive = d.slots.alive
  for (let i = 0; i < d.highWater; i++) {
    if (dAlive[i] !== 1) continue
    const ia = d.a[i]!
    const ib = d.b[i]!
    segAx[k] = p.posX[ia]!
    segAy[k] = p.posY[ia]!
    segBx[k] = p.posX[ib]!
    segBy[k] = p.posY[ib]!
    segStrain[k] = d.strain[i]!
    segDamage[k] = d.damage[i]!
    segMaterial[k] = d.material[i]!
    k++
  }

  return buffer
}

/** Wrap a received buffer in typed-array views. Zero copies. */
export function decodeSnapshot(buffer: ArrayBuffer): SnapshotBody {
  if (buffer.byteLength < HEADER_BYTES) throw new Error('snapshot buffer too small')
  const header = new Uint32Array(buffer, 0, 4)
  if (header[0] !== MAGIC) throw new Error('not a snapshot buffer')
  const n = header[1]!
  const m = header[2]!
  if (buffer.byteLength < byteSize(n, m)) throw new Error('snapshot buffer truncated')

  let off = HEADER_BYTES
  const posX = new Float32Array(buffer, off, n)
  off += 4 * n
  const posY = new Float32Array(buffer, off, n)
  off += 4 * n
  const segAx = new Float32Array(buffer, off, m)
  off += 4 * m
  const segAy = new Float32Array(buffer, off, m)
  off += 4 * m
  const segBx = new Float32Array(buffer, off, m)
  off += 4 * m
  const segBy = new Float32Array(buffer, off, m)
  off += 4 * m
  const segStrain = new Float32Array(buffer, off, m)
  off += 4 * m
  const segDamage = new Float32Array(buffer, off, m)
  off += 4 * m
  const flags = new Uint8Array(buffer, off, n)
  off += n
  const segMaterial = new Uint8Array(buffer, off, m)

  return {
    particleCount: n,
    posX,
    posY,
    flags,
    segmentCount: m,
    segAx,
    segAy,
    segBx,
    segBy,
    segStrain,
    segDamage,
    segMaterial,
  }
}
