import { MATERIALS, segmentsFor, type MaterialId } from '../sim/materials'
import type { Cluster } from '../sim/clusters'
import type { SimWorld } from '../sim/world'
import { buildBeam } from '../scenes/demos'
import {
  cloneLevel,
  cloneSolution,
  emptySolution,
  nextId,
  type AnchorDoc,
  type LevelDoc,
  type MemberDoc,
  type Solution,
  type WorldObjectDoc,
} from '../model/level'
import type { Field } from '../world/field'

/**
 * Undo captures the LEVEL as well as the solution. Anchors and world objects
 * live in the document, so snapshotting only the solution meant placing an
 * anchor was invisible to undo - Ctrl+Z would silently revert some earlier,
 * unrelated member edit instead, which is worse than doing nothing.
 */
interface EditSnapshot {
  solution: Solution
  doc: LevelDoc
}

interface MemberSim {
  nodes: number[]
  distances: number[]
  bends: number[]
}

/**
 * Projects the level document and the player's solution into the sim, and keeps
 * the two in step while the sim runs.
 *
 * The document is authoritative and the sim world is a projection of it; sim
 * state never writes back. Play SNAPSHOTS the solution rather than treating the
 * world as one-shot, because building is allowed during the sim - edits mutate
 * the live world and the working solution together, and the snapshot is what
 * makes Reset still mean something.
 */
export class Session {
  solution: Solution = emptySolution()

  private snapshot: Solution | null = null
  private readonly nodeSim = new Map<string, number>()
  private readonly memberSim = new Map<string, MemberSim>()
  private readonly objectSim = new Map<string, Cluster>()
  private undoStack: EditSnapshot[] = []
  private redoStack: EditSnapshot[] = []

  running = false

  constructor(
    public doc: LevelDoc,
    private readonly sim: SimWorld,
    private readonly field: Field,
  ) {}

  // ---------------------------------------------------------------- building

  /** Rebuild the whole sim world from the document plus the current solution. */
  rebuild(): void {
    this.sim.clear()
    this.nodeSim.clear()
    this.memberSim.clear()
    this.objectSim.clear()

    for (const obj of this.doc.objects) this.spawnObject(obj)
    for (const anchor of this.doc.anchors) this.spawnAnchor(anchor)
    for (const node of this.solution.nodes) this.spawnNode(node.id, node.x, node.y)
    for (const member of this.solution.members) this.spawnMember(member)
  }

  private spawnObject(obj: WorldObjectDoc): void {
    const cluster = this.sim.addObject({
      cx: obj.x,
      cy: obj.y,
      width: obj.width,
      height: obj.height,
      density: obj.density,
    })
    this.objectSim.set(obj.id, cluster)
  }

  /**
   * A terrain anchor is a pinned particle. An object anchor is NOT a new
   * particle - it reuses the nearest particle of the object's cluster, so load
   * genuinely transfers into the object. Pinning a separate particle beside the
   * object would let a house hover with no stilts under it.
   */
  private spawnAnchor(anchor: AnchorDoc): void {
    if (anchor.attachedTo) {
      const cluster = this.objectSim.get(anchor.attachedTo)
      if (cluster) {
        const i = this.nearestClusterParticle(cluster, anchor.x, anchor.y)
        if (i >= 0) {
          this.nodeSim.set(anchor.id, i)
          return
        }
      }
    }
    const i = this.sim.particles.create({ x: anchor.x, y: anchor.y, invMass: 0, radius: 0.3 })
    this.nodeSim.set(anchor.id, i)
  }

  private nearestClusterParticle(cluster: Cluster, x: number, y: number): number {
    const p = this.sim.particles
    let best = -1
    let bestD = Infinity
    for (const i of cluster.particles) {
      if (p.slots.alive[i] !== 1) continue
      const d = (p.posX[i]! - x) ** 2 + (p.posY[i]! - y) ** 2
      if (d < bestD) {
        bestD = d
        best = i
      }
    }
    return best
  }

  private spawnNode(id: string, x: number, y: number): number {
    const i = this.sim.particles.create({ x, y, invMass: 1 / 40, radius: 0.16 })
    this.nodeSim.set(id, i)
    return i
  }

  private spawnMember(member: MemberDoc): void {
    const p = this.sim.particles
    const ia = this.nodeSim.get(member.a)
    const ib = this.nodeSim.get(member.b)
    if (ia === undefined || ib === undefined) return

    const mat = MATERIALS[member.material]
    const beam = buildBeam(this.sim, {
      x0: p.posX[ia]!,
      y0: p.posY[ia]!,
      x1: p.posX[ib]!,
      y1: p.posY[ib]!,
      material: member.material,
      segments:
        member.segments ??
        segmentsFor(mat, Math.hypot(p.posX[ib]! - p.posX[ia]!, p.posY[ib]! - p.posY[ia]!)),
    })

    // Weld the beam's own endpoints onto the graph nodes with stiff links,
    // rather than trying to reuse the node particles directly - it keeps the
    // beam's internal spacing uniform whatever the node positions are.
    const stiff = { compliance: 1e-9, zeta: 0.95 }
    const first = beam.nodes[0]!
    const last = beam.nodes[beam.nodes.length - 1]!
    this.sim.distance.create({ a: ia, b: first, rest: 0, ...stiff })
    this.sim.distance.create({ a: ib, b: last, rest: 0, ...stiff })

    this.memberSim.set(member.id, {
      nodes: beam.nodes,
      distances: beam.distances,
      bends: beam.bends,
    })
  }

  // ------------------------------------------------------------ live editing

  addNode(x: number, y: number): string {
    this.pushUndo()
    const id = nextId('n')
    this.solution.nodes.push({ id, x, y })
    this.spawnNode(id, x, y)
    return id
  }

  /** Add a member between two existing graph nodes or anchors. */
  addMember(a: string, b: string, material: MaterialId): string | null {
    if (a === b) return null
    if (this.solution.members.some((m) => (m.a === a && m.b === b) || (m.a === b && m.b === a))) {
      return null
    }
    this.pushUndo()
    const member: MemberDoc = { id: nextId('m'), a, b, material }
    this.solution.members.push(member)
    this.spawnMember(member)
    return member.id
  }

  removeMember(id: string): void {
    const idx = this.solution.members.findIndex((m) => m.id === id)
    if (idx < 0) return
    this.pushUndo()
    this.solution.members.splice(idx, 1)
    this.despawnMember(id)
    this.pruneOrphanNodes()
  }

  private despawnMember(id: string): void {
    const sim = this.memberSim.get(id)
    if (!sim) return
    for (const d of sim.distances) this.sim.distance.destroy(d)
    for (const b of sim.bends) this.sim.bend.destroy(b)
    for (const n of sim.nodes) this.sim.particles.destroy(n)
    this.memberSim.delete(id)
  }

  /** Graph nodes with no members left are noise; anchors always stay. */
  private pruneOrphanNodes(): void {
    const used = new Set<string>()
    for (const m of this.solution.members) {
      used.add(m.a)
      used.add(m.b)
    }
    this.solution.nodes = this.solution.nodes.filter((n) => {
      if (used.has(n.id)) return true
      const i = this.nodeSim.get(n.id)
      if (i !== undefined) this.sim.particles.destroy(i)
      this.nodeSim.delete(n.id)
      return false
    })
  }

  // ------------------------------------------------------------------ lookup

  /** Nearest graph node or anchor to a world point, within `radius`. */
  pickNode(x: number, y: number, radius: number): string | null {
    const p = this.sim.particles
    let best: string | null = null
    let bestD = radius * radius
    for (const [id, i] of this.nodeSim) {
      if (p.slots.alive[i] !== 1) continue
      const d = (p.posX[i]! - x) ** 2 + (p.posY[i]! - y) ** 2
      if (d < bestD) {
        bestD = d
        best = id
      }
    }
    return best
  }

  pickMember(x: number, y: number, radius: number): string | null {
    const p = this.sim.particles
    let best: string | null = null
    let bestD = radius * radius
    for (const [id, sim] of this.memberSim) {
      for (let k = 0; k + 1 < sim.nodes.length; k++) {
        const ia = sim.nodes[k]!
        const ib = sim.nodes[k + 1]!
        if (p.slots.alive[ia] !== 1 || p.slots.alive[ib] !== 1) continue
        const ax = p.posX[ia]!
        const ay = p.posY[ia]!
        const ex = p.posX[ib]! - ax
        const ey = p.posY[ib]! - ay
        const len2 = ex * ex + ey * ey
        if (len2 < 1e-9) continue
        let u = ((x - ax) * ex + (y - ay) * ey) / len2
        u = u < 0 ? 0 : u > 1 ? 1 : u
        const d = (ax + ex * u - x) ** 2 + (ay + ey * u - y) ** 2
        if (d < bestD) {
          bestD = d
          best = id
        }
      }
    }
    return best
  }

  nodePosition(id: string): { x: number; y: number } | null {
    const i = this.nodeSim.get(id)
    if (i === undefined || this.sim.particles.slots.alive[i] !== 1) return null
    return { x: this.sim.particles.posX[i]!, y: this.sim.particles.posY[i]! }
  }

  isAnchor(id: string): boolean {
    return this.doc.anchors.some((a) => a.id === id)
  }

  // ------------------------------------------------------------------- money

  memberLength(m: MemberDoc): number {
    const a = this.solution.nodes.find((n) => n.id === m.a) ?? this.doc.anchors.find((n) => n.id === m.a)
    const b = this.solution.nodes.find((n) => n.id === m.b) ?? this.doc.anchors.find((n) => n.id === m.b)
    if (!a || !b) return 0
    return Math.hypot(b.x - a.x, b.y - a.y)
  }

  cost(): number {
    let total = 0
    for (const m of this.solution.members) {
      total += this.memberLength(m) * MATERIALS[m.material].costPerMetre
    }
    return total
  }

  remainingBudget(): number {
    return this.doc.budget - this.cost()
  }

  // ------------------------------------------------------------ play / reset

  play(): void {
    this.snapshot = cloneSolution(this.solution)
    this.running = true
  }

  /**
   * Restore the snapshot taken at Play and rebuild. Exact by construction: the
   * document is never mutated by the sim, so there is no drift to reconcile.
   */
  reset(): void {
    if (this.snapshot) this.solution = cloneSolution(this.snapshot)
    this.running = false
    this.rebuild()
  }

  // ------------------------------------------------------------- undo / redo

  /**
   * Snapshot of whole state rather than a command log. The solution is a small
   * JSON document, so this is far less code and far less to get wrong than
   * inverse operations for every edit.
   */
  private pushUndo(): void {
    this.undoStack.push({ solution: cloneSolution(this.solution), doc: cloneLevel(this.doc) })
    if (this.undoStack.length > 100) this.undoStack.shift()
    this.redoStack.length = 0
  }

  private snapshotNow(): EditSnapshot {
    return { solution: cloneSolution(this.solution), doc: cloneLevel(this.doc) }
  }

  private restore(snap: EditSnapshot): void {
    this.solution = snap.solution
    this.doc = snap.doc
    this.rebuild()
  }

  canUndo(): boolean {
    return this.undoStack.length > 0
  }

  canRedo(): boolean {
    return this.redoStack.length > 0
  }

  undo(): void {
    const prev = this.undoStack.pop()
    if (!prev) return
    this.redoStack.push(this.snapshotNow())
    this.restore(prev)
  }

  redo(): void {
    const next = this.redoStack.pop()
    if (!next) return
    this.undoStack.push(this.snapshotNow())
    this.restore(next)
  }

  // --------------------------------------------------------------- authoring

  addAnchor(x: number, y: number, attachedTo: string | null = null): string {
    this.pushUndo()
    const id = nextId('a')
    this.doc.anchors.push({ id, x, y, attachedTo })
    this.spawnAnchor({ id, x, y, attachedTo })
    return id
  }

  addWorldObject(obj: Omit<WorldObjectDoc, 'id'>): string {
    this.pushUndo()
    const id = nextId('o')
    const doc = { id, ...obj }
    this.doc.objects.push(doc)
    this.spawnObject(doc)
    return id
  }

  /** Nearest anchor to a world point, within `radius`. */
  pickAnchor(x: number, y: number, radius: number): string | null {
    let best: string | null = null
    let bestD = radius * radius
    for (const a of this.doc.anchors) {
      const pos = this.nodePosition(a.id) ?? a
      const d = (pos.x - x) ** 2 + (pos.y - y) ** 2
      if (d < bestD) {
        bestD = d
        best = a.id
      }
    }
    return best
  }

  /**
   * Remove an anchor, and any members that hung off it. Leaving dangling
   * members would put the solution in a state rebuild() silently drops, which
   * is how you get a member you are billed for but cannot see.
   */
  removeAnchor(id: string): void {
    const idx = this.doc.anchors.findIndex((a) => a.id === id)
    if (idx < 0) return
    this.pushUndo()
    this.doc.anchors.splice(idx, 1)
    this.dropMembersReferencing(new Set([id]))
    this.rebuild()
  }

  /** Remove an object, the anchors bound to it, and anything built on those. */
  removeObject(id: string): void {
    const idx = this.doc.objects.findIndex((o) => o.id === id)
    if (idx < 0) return
    this.pushUndo()
    const orphanedAnchors = new Set(
      this.doc.anchors.filter((a) => a.attachedTo === id).map((a) => a.id),
    )
    this.doc.objects.splice(idx, 1)
    this.doc.anchors = this.doc.anchors.filter((a) => a.attachedTo !== id)
    this.dropMembersReferencing(orphanedAnchors)
    this.rebuild()
  }

  private dropMembersReferencing(ids: Set<string>): void {
    this.solution.members = this.solution.members.filter(
      (m) => !ids.has(m.a) && !ids.has(m.b),
    )
    const used = new Set<string>()
    for (const m of this.solution.members) {
      used.add(m.a)
      used.add(m.b)
    }
    this.solution.nodes = this.solution.nodes.filter((n) => used.has(n.id))
  }

  /** Which object, if any, contains this point. */
  pickObject(x: number, y: number): string | null {
    for (let i = this.doc.objects.length - 1; i >= 0; i--) {
      const o = this.doc.objects[i]!
      if (
        Math.abs(x - o.x) <= o.width * 0.5 + 0.4 &&
        Math.abs(y - o.y) <= o.height * 0.5 + 0.4
      ) {
        return o.id
      }
    }
    return null
  }

  clearBuild(): void {
    this.pushUndo()
    this.solution = emptySolution()
    this.rebuild()
  }

  clearAll(): void {
    this.pushUndo()
    this.solution = emptySolution()
    this.doc.anchors = []
    this.doc.objects = []
    this.rebuild()
  }

  syncWidth(): void {
    this.doc.widthM = this.field.widthM
  }
}
