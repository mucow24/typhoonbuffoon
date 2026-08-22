import { generateBeach, type Terrain } from './terrain'

/**
 * The playable field. Width is typed in by the author and deliberately
 * unclamped - flexibility to test concepts matters more than guardrails, and
 * the HUD surfaces the cost (docs/PLAN.md 8).
 */
export class Field {
  private _widthM: number
  private _terrain: Terrain
  private readonly listeners = new Set<(field: Field) => void>()

  constructor(widthM = 120) {
    this._widthM = widthM
    this._terrain = generateBeach({ widthM })
  }

  get widthM(): number {
    return this._widthM
  }

  get terrain(): Terrain {
    return this._terrain
  }

  get left(): number {
    return -this._widthM * 0.5
  }

  get right(): number {
    return this._widthM * 0.5
  }

  setWidth(widthM: number): void {
    const w = Math.max(5, widthM)
    if (w === this._widthM) return
    this._widthM = w
    this._terrain = generateBeach({ widthM: w })
    for (const fn of this.listeners) fn(this)
  }

  onChange(fn: (field: Field) => void): () => void {
    this.listeners.add(fn)
    return () => this.listeners.delete(fn)
  }
}
