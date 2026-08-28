import { describe, it, expect } from 'vitest'
import { SimClient } from '../../src/runtime/client'
import { SimHost } from '../../src/runtime/host'
import type { Command, HostMessage } from '../../src/runtime/protocol'

/**
 * Client and host joined by a synchronous loopback wire - the whole protocol
 * exercised end to end with no Worker involved. What these prove: requests
 * correlate to their replies, snapshots rotate into an interpolation pair,
 * and retired buffers really do flow back into the host's pool.
 */

function loopback() {
  let toClient: ((msg: HostMessage) => void) | null = null
  let host: SimHost | null = null
  let clock = 0

  const client = new SimClient(
    {
      postMessage: (cmd: Command) => host!.handleCommand(cmd),
      onMessage: (cb) => {
        toClient = cb
      },
    },
    () => clock,
  )
  host = new SimHost((msg) => toClient?.(msg), { seedStarter: false })
  host.start(0)

  let t = 0
  const frame = () => {
    t += 1000 / 60
    clock = t
    host!.tick(t)
  }
  return { client, host, frame, setClock: (ms: number) => (clock = ms) }
}

describe('SimClient over loopback', () => {
  it('resolves pump() and reflects the advance in the next snapshot', async () => {
    const { client, frame } = loopback()
    frame()
    const before = client.latest!.scalars.frame
    const done = client.pump(5)
    await expect(done).resolves.toBeUndefined()
    frame()
    expect(client.latest!.scalars.frame).toBe(before + 5 + 1)
  })

  it('save() returns the live document', async () => {
    const { client } = loopback()
    client.send({
      type: 'buildMember',
      fromNode: null,
      fromX: 0,
      fromY: 6,
      toNode: null,
      toX: 4,
      toY: 6,
      material: 'wood',
    })
    const saved = await client.save()
    expect(saved.level.widthM).toBe(120)
    expect(saved.solution.members.length).toBe(1)
  })

  it('keeps two snapshots and computes a clamped renderAlpha between them', () => {
    const { client, frame, setClock } = loopback()
    frame()
    client.send({ type: 'setWind', kph: 10 }) // dirty -> next tick emits
    frame()
    expect(client.previous).not.toBeNull()
    expect(client.latest).not.toBeNull()
    expect(client.latest).not.toBe(client.previous)

    const arrived = client.latest!.arrivedAt
    const interval = arrived - client.previous!.arrivedAt
    setClock(arrived) // exactly at arrival: alpha 0
    expect(client.renderAlpha()).toBeCloseTo(0, 5)
    setClock(arrived + interval / 2)
    expect(client.renderAlpha()).toBeCloseTo(0.5, 5)
    setClock(arrived + interval * 3) // stalled sim: hold, never extrapolate
    expect(client.renderAlpha()).toBe(1)
  })

  it('recycles retired snapshot buffers back into the host pool', () => {
    const { client, frame } = loopback()
    const buffers: ArrayBuffer[] = []
    client.onSnapshot((snap) => buffers.push(snap.body.posX.buffer as ArrayBuffer))
    for (let i = 0; i < 4; i++) {
      client.send({ type: 'setWind', kph: i + 1 })
      frame()
    }
    expect(buffers.length).toBe(4)
    // Snapshot 1's buffer retires when snapshot 3 arrives, goes home, and the
    // host reuses it for snapshot 4. (No structured clone in the loopback, so
    // identity survives the wire.)
    expect(buffers[3]).toBe(buffers[0])
  })
})
