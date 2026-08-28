import type { SimWorld } from '../sim/world'

export const WATER_FLOW_MAX = 60

/**
 * The water tool's tap.
 *
 * The slider is a FLOW in m²/s - area per second, the 2D analogue of volume
 * flow - not a spawn-region size. Flow survives resolution changes: the same
 * slider position pours the same amount of water at any particle spacing,
 * because the per-frame particle budget is flow / spacing². The emitter disc
 * only exists to give that budget somewhere to land, and its radius grows
 * with sqrt(flow): particles leave the disc no faster than they fall, so a
 * fixed-size disc would silently cap high flows at whatever its footprint
 * happens to drain.
 */
export class WaterEmitter {
  /** m²/s of water while the button is held. */
  flow = 15
  /** One click's splash is this many seconds of flow, in a single burst. */
  splashSeconds = 0.35
  /** Fractional particles owed from previous frames. */
  private carry = 0

  /** Emitter footprint: holds roughly half a second of flow at rest packing. */
  radius(spacing: number): number {
    return Math.max(0.6, spacing * 2.5, 0.42 * Math.sqrt(this.flow))
  }

  /** A single click: one burst of splashSeconds worth of flow. */
  splash(sim: SimWorld, x: number, y: number): number {
    const spacing = sim.fluid.spacing
    const area = this.flow * this.splashSeconds
    const budget = Math.max(1, Math.round(area / (spacing * spacing)))
    // Sized so the disc can hold the whole burst: πr² ≥ area, with margin for
    // jitter and terrain clipping.
    const r = Math.max(this.radius(spacing), Math.sqrt(area / Math.PI) * 1.3)
    return sim.spawnDisc(x, y, r, budget)
  }

  /**
   * One fixed step of held-button streaming. Whatever the disc cannot accept
   * this frame is DROPPED, not banked: an emitter held under water for ten
   * seconds owes nothing, rather than discharging a ten-second burst the
   * moment it reaches open air.
   */
  update(sim: SimWorld, dt: number, x: number, y: number): number {
    const spacing = sim.fluid.spacing
    this.carry += (this.flow * dt) / (spacing * spacing)
    const budget = Math.floor(this.carry)
    if (budget <= 0) return 0
    this.carry -= budget
    return sim.spawnDisc(x, y, this.radius(spacing), budget)
  }
}
