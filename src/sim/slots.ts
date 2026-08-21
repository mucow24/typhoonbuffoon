/**
 * Index allocator with a free list. Live entries are never moved, so anything
 * holding an index stays valid across insert/remove - which is the requirement
 * that falls out of allowing building during the sim (docs/PLAN.md 5.2).
 *
 * Holes are tolerated rather than compacted. Compaction would have to fix up
 * every mapping that references an index, and nothing yet needs it.
 */
export class SlotAllocator {
  private free: number[] = []
  private _capacity: number
  private _highWater = 0
  private _live = 0
  alive: Uint8Array

  /** Called with the new capacity when the table has to grow. */
  onGrow?: (capacity: number) => void

  constructor(capacity = 1024) {
    this._capacity = capacity
    this.alive = new Uint8Array(capacity)
  }

  get capacity(): number {
    return this._capacity
  }

  /** One past the highest index ever allocated. Iterate to this, skipping holes. */
  get highWater(): number {
    return this._highWater
  }

  get liveCount(): number {
    return this._live
  }

  isAlive(i: number): boolean {
    return i >= 0 && i < this._highWater && this.alive[i] === 1
  }

  alloc(): number {
    let i = this.free.pop()
    if (i === undefined) {
      if (this._highWater >= this._capacity) {
        this.grow(this._capacity * 2)
      }
      i = this._highWater++
    }
    this.alive[i] = 1
    this._live++
    return i
  }

  release(i: number): void {
    if (!this.isAlive(i)) return
    this.alive[i] = 0
    this._live--
    this.free.push(i)
  }

  clear(): void {
    this.alive.fill(0)
    this.free.length = 0
    this._highWater = 0
    this._live = 0
  }

  private grow(capacity: number): void {
    const next = new Uint8Array(capacity)
    next.set(this.alive)
    this.alive = next
    this._capacity = capacity
    this.onGrow?.(capacity)
  }

  /** Grow a Float32Array to the allocator's capacity, preserving contents. */
  static growF32(arr: Float32Array, capacity: number): Float32Array {
    const next = new Float32Array(capacity)
    next.set(arr)
    return next
  }

  static growI32(arr: Int32Array, capacity: number): Int32Array {
    const next = new Int32Array(capacity)
    next.set(arr)
    return next
  }

  static growU8(arr: Uint8Array, capacity: number): Uint8Array {
    const next = new Uint8Array(capacity)
    next.set(arr)
    return next
  }
}
