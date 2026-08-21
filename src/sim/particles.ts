import { SlotAllocator } from './slots'

export const KIND_NODE = 0
export const KIND_FLUID = 1
export const KIND_OBJECT = 2

export interface ParticleSpec {
  x: number
  y: number
  /** 0 means pinned - an anchor welded to the world. */
  invMass?: number
  radius?: number
  kind?: number
  /** Cluster id for shape-matched objects, -1 otherwise. */
  cluster?: number
}

/**
 * One struct-of-arrays store for every particle in the world: structure nodes,
 * fluid, and object cluster particles alike. Keeping them in a single table is
 * what makes "there aren't two worlds" true in the code and not just in the
 * plan - the spatial hash and the collision passes do not care which is which.
 */
export class ParticleStore {
  readonly slots: SlotAllocator

  posX: Float32Array
  posY: Float32Array
  prevX: Float32Array
  prevY: Float32Array
  velX: Float32Array
  velY: Float32Array
  /** Accumulated external acceleration for the current substep. */
  accX: Float32Array
  accY: Float32Array
  invMass: Float32Array
  radius: Float32Array
  kind: Uint8Array
  cluster: Int32Array

  constructor(capacity = 4096) {
    this.slots = new SlotAllocator(capacity)
    this.posX = new Float32Array(capacity)
    this.posY = new Float32Array(capacity)
    this.prevX = new Float32Array(capacity)
    this.prevY = new Float32Array(capacity)
    this.velX = new Float32Array(capacity)
    this.velY = new Float32Array(capacity)
    this.accX = new Float32Array(capacity)
    this.accY = new Float32Array(capacity)
    this.invMass = new Float32Array(capacity)
    this.radius = new Float32Array(capacity)
    this.kind = new Uint8Array(capacity)
    this.cluster = new Int32Array(capacity)

    this.slots.onGrow = (cap) => this.grow(cap)
  }

  private grow(cap: number): void {
    this.posX = SlotAllocator.growF32(this.posX, cap)
    this.posY = SlotAllocator.growF32(this.posY, cap)
    this.prevX = SlotAllocator.growF32(this.prevX, cap)
    this.prevY = SlotAllocator.growF32(this.prevY, cap)
    this.velX = SlotAllocator.growF32(this.velX, cap)
    this.velY = SlotAllocator.growF32(this.velY, cap)
    this.accX = SlotAllocator.growF32(this.accX, cap)
    this.accY = SlotAllocator.growF32(this.accY, cap)
    this.invMass = SlotAllocator.growF32(this.invMass, cap)
    this.radius = SlotAllocator.growF32(this.radius, cap)
    this.kind = SlotAllocator.growU8(this.kind, cap)
    this.cluster = SlotAllocator.growI32(this.cluster, cap)
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

  create(spec: ParticleSpec): number {
    const i = this.slots.alloc()
    this.posX[i] = spec.x
    this.posY[i] = spec.y
    this.prevX[i] = spec.x
    this.prevY[i] = spec.y
    this.velX[i] = 0
    this.velY[i] = 0
    this.accX[i] = 0
    this.accY[i] = 0
    this.invMass[i] = spec.invMass ?? 1
    this.radius[i] = spec.radius ?? 0.12
    this.kind[i] = spec.kind ?? KIND_NODE
    this.cluster[i] = spec.cluster ?? -1
    return i
  }

  destroy(i: number): void {
    this.slots.release(i)
  }

  clear(): void {
    this.slots.clear()
  }

  setMassFromDensity(i: number, densityKgM3: number, volumeM3: number): void {
    const m = Math.max(1e-6, densityKgM3 * volumeM3)
    this.invMass[i] = 1 / m
  }

  massOf(i: number): number {
    const w = this.invMass[i]!
    return w > 0 ? 1 / w : Infinity
  }

  pin(i: number): void {
    this.invMass[i] = 0
    this.velX[i] = 0
    this.velY[i] = 0
  }

  countOfKind(kind: number): number {
    let n = 0
    for (let i = 0; i < this.highWater; i++) {
      if (this.slots.alive[i] === 1 && this.kind[i] === kind) n++
    }
    return n
  }
}
