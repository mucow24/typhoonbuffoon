import type { WaveStrength } from '../game/conditions'
import type { LevelDoc, Solution } from '../model/level'
import type { MaterialId } from '../sim/materials'

/**
 * The worker boundary, in types. Two invariants (docs/GPU_PLAN.md):
 *
 * 1. Raw sim slot indices NEVER cross this boundary. The particle and
 *    constraint stores recycle freed indices on the very next create, so the
 *    only safe place to hold one is next to the code that frees it. Commands
 *    speak document ids and world positions; snapshots speak positions and
 *    flags.
 * 2. Commands are user INTENTS, not bookkeeping steps. `buildMember` carries
 *    the whole drag gesture (both endpoints, snapped-node ids if any) so the
 *    worker can mint ids and create nodes atomically - the main thread never
 *    needs an id back mid-gesture.
 *
 * Transport: structured clone for commands and the small snapshot parts; one
 * transferred ArrayBuffer per snapshot for the bulk (particles + member
 * segments - see runtime/snapshot.ts). Consumed buffers are posted back via
 * `recycleBuffer` so steady-state rendering allocates nothing.
 */

export type ProbeName = 'stilts' | 'loadtest' | 'palm'

export type Command =
  // -- session flow
  | { type: 'togglePause' }
  | { type: 'setPaused'; paused: boolean }
  | { type: 'reset' }
  | { type: 'pump'; steps: number; requestId: number }
  // -- build edits (worker mints all ids)
  | {
      type: 'buildMember'
      fromNode: string | null
      fromX: number
      fromY: number
      toNode: string | null
      toX: number
      toY: number
      material: MaterialId
    }
  | { type: 'addAnchor'; x: number; y: number; attachedTo: string | null }
  | { type: 'addObject'; x: number; y: number; width: number; height: number; density: number }
  | { type: 'removeMember'; id: string }
  | { type: 'removeAnchor'; id: string }
  | { type: 'removeObject'; id: string }
  | { type: 'undo' }
  | { type: 'redo' }
  | { type: 'clearBuild' }
  | { type: 'clearAll' }
  // -- conditions
  | { type: 'setWind'; kph: number }
  | { type: 'setFlood'; level: number }
  | { type: 'setWaves'; strength: WaveStrength }
  | { type: 'calm' }
  // -- water tool
  | { type: 'splash'; x: number; y: number }
  | { type: 'setStream'; x: number; y: number }
  | { type: 'clearStream' }
  | { type: 'setFlow'; flow: number }
  | { type: 'clearFluid' }
  // -- solver tuning
  | { type: 'setSubsteps'; substeps: number }
  | { type: 'setLinearDamping'; value: number }
  | { type: 'setFluidIterations'; iterations: number }
  | { type: 'setFluidSpacing'; spacing: number }
  | { type: 'setBackend'; backend: 'cpu' | 'webgpu' }
  // -- level
  | { type: 'setFieldWidth'; widthM: number }
  | { type: 'loadProbe'; probe: ProbeName }
  | { type: 'loadDoc'; level: unknown; solution: unknown; requestId: number }
  | { type: 'requestSave'; requestId: number }
  // -- plumbing
  | { type: 'recycleBuffer'; buffer: ArrayBuffer }

/** Scalar block: everything the HUD and panels read, reduced worker-side. */
export interface SnapshotScalars {
  frame: number
  simTime: number
  stepsLastFrame: number
  starved: boolean
  paused: boolean
  /** Session has taken its play snapshot (first unpause happened). */
  running: boolean
  particleCount: number
  fluidCount: number
  objectCount: number
  memberCount: number
  substeps: number
  fluidSpacing: number
  /** Instantaneous gusting wind at the HUD, kph. */
  windGustKph: number
  /** 0..1 storm severity, for the sky tint. */
  severity: number
  /** Worst member load fraction across axial and bending, 0..1+. */
  peakLoad: number
  maxDamage: number
  /** Total members broken since the last reset. */
  breakCount: number
  cost: number
  budget: number
  widthM: number
  /** Smoothed wall-clock cost of one sim step, ms. */
  stepMs: number
  /**
   * Achieved sim rate, steps per wall second, smoothed - 60 means real time,
   * lower means the drop-debt slow-mo contract is active. Zero while paused.
   * This is the number that answers "the HUD says slow-mo but fps reads
   * 140": render rate and sim rate are different clocks.
   */
  simFps: number
  /** Active solver backend: 'webgpu', 'cpu', or 'cpu (no webgpu)'. */
  backend: string
}

/**
 * Structure view-model: what the editor needs to pick and overlay, keyed by
 * document id. Positions are live sim positions (anchors ride their mounts,
 * members sag), falling back to document positions where nothing is spawned.
 */
export interface StructureVM {
  /** Every connectable point: anchors AND graph nodes, for snapping. */
  nodes: { id: string; x: number; y: number }[]
  /** Anchors only (subset of nodes), with their binding, for the overlay. */
  anchors: { id: string; x: number; y: number; attachedTo: string | null }[]
  /** Per member, the live beam polyline as flat [x0,y0, x1,y1, ...]. */
  members: { id: string; points: number[] }[]
  /** Document rects, for pickObject. */
  objects: { id: string; x: number; y: number; width: number; height: number }[]
}

export interface ClusterVM {
  cx: number
  cy: number
  angle: number
  hw: number
  hh: number
  /** Density below water: floats, drawn as timber rather than masonry. */
  light: boolean
}

export interface BreakEventVM {
  x: number
  y: number
  strain: number
  material: number
}

export interface SnapshotMessage {
  type: 'snapshot'
  /** Particles + member segments; layout in runtime/snapshot.ts. Transferred. */
  buffer: ArrayBuffer
  scalars: SnapshotScalars
  structure: StructureVM
  clusters: ClusterVM[]
  /** Break events since the last snapshot. At-most-once; FX-grade delivery. */
  events: BreakEventVM[]
}

export type HostMessage =
  | SnapshotMessage
  | { type: 'saveData'; requestId: number; level: LevelDoc; solution: Solution }
  /** Completion of a pump or loadDoc request - posted AFTER the snapshot
   *  that reflects the change, so an awaiting caller reading client.latest
   *  observes post-command state. */
  | { type: 'ack'; requestId: number }
  /** A command failed. Carries the request id when the command had one (the
   *  client rejects that promise); fire-and-forget commands surface the
   *  error without hanging anything. */
  | { type: 'nack'; requestId: number | null; error: string }

/** Both ends of the wire, shaped so Node tests can join them with a loopback. */
export interface CommandPort {
  postMessage(msg: HostMessage, transfer?: Transferable[]): void
}
export interface HostHandle {
  postMessage(msg: Command, transfer?: Transferable[]): void
  onMessage(cb: (msg: HostMessage) => void): void
}
