import type { LevelDoc, Solution } from '../model/level'
import { decodeSnapshot, type SnapshotBody } from './snapshot'
import type { Command, HostHandle, HostMessage, SnapshotMessage } from './protocol'

/** One decoded frame, as the render side consumes it. */
export interface WorldSnapshot {
  body: SnapshotBody
  scalars: SnapshotMessage['scalars']
  structure: SnapshotMessage['structure']
  clusters: SnapshotMessage['clusters']
  events: SnapshotMessage['events']
  arrivedAt: number
}

/**
 * Main-thread end of the worker boundary. Sends commands, keeps the last TWO
 * snapshots for interpolated drawing (sim frames are not phase-locked to rAF,
 * and at heavy particle counts the sim deliberately runs slower than render),
 * and recycles outgoing snapshot buffers back to the worker so steady-state
 * rendering allocates nothing.
 *
 * A plain class over an injected port: the loopback tests join it directly to
 * a SimHost with no Worker involved.
 */
export class SimClient {
  latest: WorldSnapshot | null = null
  previous: WorldSnapshot | null = null

  private requestCounter = 0
  private readonly pending = new Map<number, (msg: HostMessage) => void>()
  private readonly snapshotListeners = new Set<(snap: WorldSnapshot) => void>()

  constructor(
    private readonly port: HostHandle,
    private readonly now: () => number = () => performance.now(),
  ) {
    port.onMessage((msg) => this.handleMessage(msg))
  }

  send(cmd: Command): void {
    this.port.postMessage(cmd)
  }

  /** Fires on every snapshot, after rotation - the app's frame-data hook. */
  onSnapshot(cb: (snap: WorldSnapshot) => void): () => void {
    this.snapshotListeners.add(cb)
    return () => this.snapshotListeners.delete(cb)
  }

  /**
   * 0..1 blend from `previous` toward `latest` for the current wall time,
   * from snapshot arrival cadence. Clamped: a stalled sim holds the last
   * frame rather than extrapolating into nonsense.
   */
  renderAlpha(): number {
    if (!this.latest || !this.previous) return 1
    const interval = this.latest.arrivedAt - this.previous.arrivedAt
    if (interval <= 1e-3) return 1
    const alpha = (this.now() - this.latest.arrivedAt) / interval
    return alpha < 0 ? 0 : alpha > 1 ? 1 : alpha
  }

  // ------------------------------------------------------- request / reply

  /** Advance the sim n fixed steps synchronously, for headless verification. */
  pump(steps: number): Promise<void> {
    return this.request((requestId) => ({ type: 'pump', steps, requestId })).then(() => {})
  }

  save(): Promise<{ level: LevelDoc; solution: Solution }> {
    return this.request((requestId) => ({ type: 'requestSave', requestId })).then((msg) => {
      if (msg.type !== 'saveData') throw new Error('expected saveData reply')
      return { level: msg.level, solution: msg.solution }
    })
  }

  load(level: unknown, solution: unknown): Promise<void> {
    return this.request((requestId) => ({ type: 'loadDoc', level, solution, requestId })).then(
      () => {},
    )
  }

  private request(make: (requestId: number) => Command): Promise<HostMessage> {
    const requestId = ++this.requestCounter
    return new Promise((resolve) => {
      this.pending.set(requestId, resolve)
      this.port.postMessage(make(requestId))
    })
  }

  // ----------------------------------------------------------------- intake

  private handleMessage(msg: HostMessage): void {
    switch (msg.type) {
      case 'snapshot': {
        const retiring = this.previous
        this.previous = this.latest
        this.latest = {
          body: decodeSnapshot(msg.buffer),
          scalars: msg.scalars,
          structure: msg.structure,
          clusters: msg.clusters,
          events: msg.events,
          arrivedAt: this.now(),
        }
        // The retiring snapshot's buffer goes home for reuse. Its views die
        // with the transfer, which is safe: render only ever reads
        // latest/previous, both rotated before anything is posted back.
        if (retiring) {
          this.port.postMessage(
            { type: 'recycleBuffer', buffer: retiring.body.posX.buffer as ArrayBuffer },
            [retiring.body.posX.buffer as ArrayBuffer],
          )
        }
        for (const cb of this.snapshotListeners) cb(this.latest)
        break
      }
      case 'saveData':
      case 'ack': {
        const resolve = this.pending.get(msg.requestId)
        if (resolve) {
          this.pending.delete(msg.requestId)
          resolve(msg)
        }
        break
      }
    }
  }
}
