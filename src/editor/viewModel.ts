import type { StructureVM } from '../runtime/protocol'
import type { Vec2 } from '../core/math'

/**
 * Editor picking over the snapshot's structure view-model.
 *
 * These are Session's hit rules, ported to the main-thread side of the worker
 * boundary: the editor picks against the latest snapshot (positions at most
 * one sim frame old) and sends id-based commands back. Same-frame accuracy
 * was never part of the contract - picking against the live arrays was
 * already stale-by-a-step whenever the sim was paused.
 */

/** Nearest connectable point (anchor or graph node) within `radius`. */
export function pickNode(vm: StructureVM, x: number, y: number, radius: number): string | null {
  let best: string | null = null
  let bestD = radius * radius
  for (const n of vm.nodes) {
    const d = (n.x - x) ** 2 + (n.y - y) ** 2
    if (d < bestD) {
      bestD = d
      best = n.id
    }
  }
  return best
}

/** Nearest member polyline within `radius`, by point-segment distance. */
export function pickMember(vm: StructureVM, x: number, y: number, radius: number): string | null {
  let best: string | null = null
  let bestD = radius * radius
  for (const m of vm.members) {
    const pts = m.points
    for (let k = 0; k + 3 < pts.length; k += 2) {
      const ax = pts[k]!
      const ay = pts[k + 1]!
      const ex = pts[k + 2]! - ax
      const ey = pts[k + 3]! - ay
      const len2 = ex * ex + ey * ey
      if (len2 < 1e-9) continue
      let u = ((x - ax) * ex + (y - ay) * ey) / len2
      u = u < 0 ? 0 : u > 1 ? 1 : u
      const d = (ax + ex * u - x) ** 2 + (ay + ey * u - y) ** 2
      if (d < bestD) {
        bestD = d
        best = m.id
      }
    }
  }
  return best
}

/** Nearest anchor within `radius`. Anchors ride their live (mount) positions. */
export function pickAnchor(vm: StructureVM, x: number, y: number, radius: number): string | null {
  let best: string | null = null
  let bestD = radius * radius
  for (const a of vm.anchors) {
    const d = (a.x - x) ** 2 + (a.y - y) ** 2
    if (d < bestD) {
      bestD = d
      best = a.id
    }
  }
  return best
}

/** Which object contains this point - topmost (last-listed) first. */
export function pickObject(vm: StructureVM, x: number, y: number): string | null {
  for (let i = vm.objects.length - 1; i >= 0; i--) {
    const o = vm.objects[i]!
    if (Math.abs(x - o.x) <= o.width * 0.5 + 0.4 && Math.abs(y - o.y) <= o.height * 0.5 + 0.4) {
      return o.id
    }
  }
  return null
}

export function nodePosition(vm: StructureVM, id: string): Vec2 | null {
  for (const n of vm.nodes) if (n.id === id) return { x: n.x, y: n.y }
  return null
}

/** An empty view-model, for the frames before the first snapshot arrives. */
export const emptyStructureVM = (): StructureVM => ({
  nodes: [],
  anchors: [],
  members: [],
  objects: [],
})
