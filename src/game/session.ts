import { MATERIALS, massPerMetre, segmentsFor, type MaterialId } from '../sim/materials'
import { KIND_NODE } from '../sim/particles'
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
  /**
   * The two stiff zero-rest welds binding the beam's ends to the graph nodes.
   * Tracked so despawn can destroy them: a leaked weld outlives its member,
   * and the freed particle slots it references are recycled by the next
   * create - during a flood, into water particles, leaving a water particle
   * rigidly and unbreakably welded to the structure.
   */
  welds: number[]
}

/**
 * Projects the level document and the player's solution into the sim, and keeps
 * the two in step while the sim runs.
 *
 * The document is authoritative and the sim world is a projection of it; sim
 * state never writes back. Play SNAPSHOTS the whole editable state - solution
 * AND document - rather than treating the world as one-shot, because building
 * is allowed during the sim: edits mutate the live world and the working
 * state together, and the snapshot is what makes Reset still mean something.
 */
export class Session {
  solution: Solution = emptySolution()

  /**
   * Play snapshots the WHOLE editable state, doc included - anchors and world
   * objects live in the document and the player's tools stay live during the
   * sim, so a solution-only snapshot let mid-run anchor edits survive Reset
   * and left members billed for but unable to spawn.
   */
  private snapshot: EditSnapshot | null = null
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

  /**
   * Rebuild the built world from the document plus the current solution.
   *
   * FLUID SURVIVES. Undo, redo and anchor/object deletion all come through
   * here, and they are build edits - clearing the water too meant Ctrl+Z
   * during a flood deleted thousands of particles and un-flooded the level,
   * with a multi-second refill at the admission rate. Accumulated damage and
   * plastic set do reset with the structures; a rebuild is a state jump for
   * the build, exact water history is not part of its contract.
   */
  rebuild(): void {
    this.sim.clearStructures()
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
   * A terrain anchor is a pinned particle. An object anchor is a MOUNT: a
   * dedicated node tied to the object's three nearest cluster particles by
   * triangulated stiff links, so load genuinely transfers into the object.
   *
   * Earlier versions reused a cluster particle directly, which put that one
   * particle under two rigid masters - the member's weld and the shape
   * matcher - and the sequential fight between them rang a steel-braced
   * house at 2-5 m/s indefinitely. The mount keeps every master inside the
   * one constraint solver (the plan's 5.8 always said anchors bind to the
   * object's FRAME, not to an individual particle), and spreads the load
   * across several particles instead of gouging one.
   */
  private spawnAnchor(anchor: AnchorDoc): void {
    if (anchor.attachedTo) {
      const cluster = this.objectSim.get(anchor.attachedTo)
      if (cluster) {
        const i = this.spawnMount(cluster, anchor.x, anchor.y)
        if (i >= 0) {
          this.nodeSim.set(anchor.id, i)
          return
        }
      }
    }
    const i = this.sim.particles.create({ x: anchor.x, y: anchor.y, invMass: 0, radius: 0.3 })
    this.nodeSim.set(anchor.id, i)
  }

  private spawnMount(cluster: Cluster, x: number, y: number): number {
    const p = this.sim.particles
    const near: { i: number; d: number }[] = []
    for (const i of cluster.particles) {
      if (p.slots.alive[i] !== 1) continue
      near.push({ i, d: (p.posX[i]! - x) ** 2 + (p.posY[i]! - y) ** 2 })
    }
    if (near.length === 0) return -1
    near.sort((a, b) => a.d - b.d)
    const chosen = near.slice(0, Math.min(3, near.length))

    // The mount weighs like the bit of object it bolts to.
    const mount = this.sim.particles.create({
      x,
      y,
      invMass: p.invMass[chosen[0]!.i]!,
      radius: 0.16,
    })
    const clusterIdx = this.sim.clusterIndexOf(cluster)
    for (const { i } of chosen) {
      this.sim.distance.create({
        a: mount,
        b: i,
        rest: Math.hypot(p.posX[i]! - x, p.posY[i]! - y),
        compliance: 2e-8,
        zeta: 0.95,
        unbreakable: true,
        // A mount link is joinery INSIDE the object's footprint; its capsule
        // must not shove the very particles it ties together.
        noCollideCluster: clusterIdx,
      })
    }
    return mount
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
    const length = Math.hypot(p.posX[ib]! - p.posX[ia]!, p.posY[ib]! - p.posY[ia]!)
    const segments = member.segments ?? segmentsFor(mat, length)
    const beam = buildBeam(this.sim, {
      x0: p.posX[ia]!,
      y0: p.posY[ia]!,
      x1: p.posX[ib]!,
      y1: p.posY[ib]!,
      material: member.material,
      segments,
    })

    // The joint must weigh at least what the member ends bolted to it weigh.
    // A 40 kg default node welded rigidly between 400 kg steel segment ends
    // is the classic XPBD mass-ratio instability: sequential stiff constraints
    // tug the light puppet between their targets and PUMP energy - measured
    // on the reported tower, self-exciting at ~40% per frame from rest (in
    // vacuum, with gravity off) until 13 members snapped. Matching the joint
    // mass to its members kills the instability outright; physically, the
    // gusset weighs like the steel it joins. Mass only ever rises, and never
    // for pinned anchors.
    //
    // NEVER for cluster particles either: the shape-matching caches per-
    // particle masses at construction, and mutating p.invMass under it makes
    // constraints and cluster disagree about the same particle's inertia -
    // measured as the house-anchored variant of the same pump. An object
    // anchor does not need the bump anyway: the cluster's rigidity gives its
    // particles the whole object's effective inertia.
    const endMass = Math.max(0.01, massPerMetre(mat) * (length / Math.max(1, segments)))
    for (const end of [ia, ib]) {
      if (p.kind[end] !== KIND_NODE) continue
      const w = p.invMass[end]!
      if (w > 0) p.invMass[end] = Math.min(w, 1 / endMass)
    }

    // Weld the beam's own endpoints onto the graph nodes with stiff links,
    // rather than trying to reuse the node particles directly - it keeps the
    // beam's internal spacing uniform whatever the node positions are.
    const stiff = { compliance: 1e-9, zeta: 0.95, unbreakable: true }
    const first = beam.nodes[0]!
    const last = beam.nodes[beam.nodes.length - 1]!
    const weldA = this.sim.distance.create({ a: ia, b: first, rest: 0, ...stiff })
    const weldB = this.sim.distance.create({ a: ib, b: last, rest: 0, ...stiff })

    // A member bolted to an object must not collide with that object - the
    // capsule would fight the weld and pump energy until the object flipped.
    // (A member bridging two different objects keeps only one exclusion; that
    // configuration is rare enough to accept the fight on the second joint.)
    let noCollide = -1
    for (const end of [member.a, member.b]) {
      const anchor = this.doc.anchors.find((a) => a.id === end)
      if (anchor?.attachedTo) {
        const cluster = this.objectSim.get(anchor.attachedTo)
        if (cluster) {
          const idx = this.sim.clusterIndexOf(cluster)
          if (idx >= 0) noCollide = idx
        }
      }
    }
    if (noCollide >= 0) {
      for (const di of beam.distances) this.sim.distance.noCollideCluster[di] = noCollide
    }

    this.memberSim.set(member.id, {
      nodes: beam.nodes,
      distances: beam.distances,
      bends: beam.bends,
      welds: [weldA, weldB],
    })
  }

  /**
   * Drop constraint indices the sim destroyed by breakage from our records.
   *
   * MUST run every frame (and defensively before any despawn): a freed index
   * is recycled by the next create, and a stale record then points at an
   * unrelated live constraint - destroying "our" indices on member delete was
   * silently killing segments of newer members, with no break event.
   */
  syncBreaks(): void {
    const destroyed = this.sim.drainDestroyed()
    if (destroyed.distance.length === 0 && destroyed.bends.length === 0) return
    const dSet = new Set(destroyed.distance)
    const bSet = new Set(destroyed.bends)
    for (const sim of this.memberSim.values()) {
      if (dSet.size > 0) {
        sim.distances = sim.distances.filter((i) => !dSet.has(i))
        sim.welds = sim.welds.filter((i) => !dSet.has(i))
      }
      if (bSet.size > 0) {
        sim.bends = sim.bends.filter((i) => !bSet.has(i))
      }
    }
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
    // Records may hold indices the sim freed by breakage this frame; acting
    // on them would destroy whatever recycled those slots.
    this.syncBreaks()
    const sim = this.memberSim.get(id)
    if (!sim) return
    for (const d of sim.distances) this.sim.distance.destroy(d)
    for (const w of sim.welds) this.sim.distance.destroy(w)
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
    this.snapshot = this.snapshotNow()
    this.running = true
  }

  /**
   * Restore the snapshot taken at Play and rebuild. Exact by construction: the
   * document is never mutated by the sim, and Play snapshots doc AND solution,
   * so every edit made while running - members, anchors, objects - reverts.
   * Reset also discards the water: it ends the storm run, unlike a build edit.
   */
  reset(): void {
    if (this.snapshot) {
      this.solution = cloneSolution(this.snapshot.solution)
      this.doc = cloneLevel(this.snapshot.doc)
    }
    this.running = false
    this.sim.clear()
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
