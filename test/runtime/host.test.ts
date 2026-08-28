import { describe, it, expect } from 'vitest'
import { SimHost } from '../../src/runtime/host'
import { FLAG_ALIVE, decodeSnapshot, kindOfFlags } from '../../src/runtime/snapshot'
import type { HostMessage, SnapshotMessage } from '../../src/runtime/protocol'
import { KIND_FLUID } from '../../src/sim/particles'

/**
 * SimHost is the whole game core behind the protocol: these tests drive it
 * exactly the way the worker shell does (commands in, ticks, messages out)
 * and assert on what crosses the wire - the same data the render and editor
 * will trust. No reaching into sim internals except to state ground truth.
 */

function makeHost(opts: { seedStarter?: boolean } = {}) {
  const messages: HostMessage[] = []
  const host = new SimHost((msg) => messages.push(msg), { seedStarter: false, ...opts })
  host.start(0)
  let t = 0
  // Commands drain at the next frame boundary (tick) - the same contract the
  // worker gives them now that GPU frames are async.
  const frame = async () => {
    t += 1000 / 60
    await host.tick(t)
  }
  const snapshots = () => messages.filter((m): m is SnapshotMessage => m.type === 'snapshot')
  const last = () => {
    const all = snapshots()
    expect(all.length).toBeGreaterThan(0)
    return all[all.length - 1]!
  }
  return { host, messages, frame, snapshots, last }
}

function fluidStats(snap: SnapshotMessage): { count: number; meanY: number } {
  const body = decodeSnapshot(snap.buffer)
  let count = 0
  let sumY = 0
  for (let i = 0; i < body.particleCount; i++) {
    const f = body.flags[i]!
    if ((f & FLAG_ALIVE) === 0 || kindOfFlags(f) !== KIND_FLUID) continue
    count++
    sumY += body.posY[i]!
  }
  return { count, meanY: count > 0 ? sumY / count : NaN }
}

describe('SimHost', () => {
  it('emits an initial snapshot, paused, before any command', async () => {
    const { frame, last } = makeHost()
    await frame()
    const snap = last()
    expect(snap.scalars.paused).toBe(true)
    expect(snap.scalars.frame).toBeGreaterThan(0) // the stepper ran, gated
    expect(snap.scalars.memberCount).toBe(0)
  })

  it('builds a member from one drag gesture and undoes/redoes it', async () => {
    const { host, frame, last } = makeHost()
    host.enqueue({
      type: 'buildMember',
      fromNode: null,
      fromX: -2,
      fromY: 6,
      toNode: null,
      toX: 3,
      toY: 6,
      material: 'wood',
    })
    await frame()
    let snap = last()
    expect(snap.scalars.memberCount).toBe(1)
    expect(snap.scalars.cost).toBeGreaterThan(0)
    expect(snap.structure.nodes.length).toBe(2)
    expect(snap.structure.members.length).toBe(1)
    // The polyline is a real beam chain in world space, ends near the gesture.
    const pts = snap.structure.members[0]!.points
    expect(pts.length).toBeGreaterThanOrEqual(4)
    expect(pts[0]).toBeCloseTo(-2, 0)
    expect(pts[pts.length - 2]).toBeCloseTo(3, 0)

    host.enqueue({ type: 'undo' })
    await frame()
    snap = last()
    expect(snap.scalars.memberCount).toBe(0)
    expect(snap.structure.members.length).toBe(0)

    host.enqueue({ type: 'redo' })
    await frame()
    snap = last()
    expect(snap.scalars.memberCount).toBe(1)
  })

  it('paused splashes saturate the disc instead of stacking particles', async () => {
    const { host, frame, last } = makeHost()
    await frame()
    expect(fluidStats(last()).count).toBe(0)

    // The sim never steps here, so the spatial hash never sees these spawns:
    // only the recent-spawn guard stands between repeat clicks and particles
    // stacked on top of each other (which detonate on unpause). A repeat
    // click may fill cells the first click's budget left empty - what it must
    // NEVER do is keep adding forever.
    host.enqueue({ type: 'splash', x: 0, y: 8 })
    await frame()
    const afterFirst = fluidStats(last()).count
    expect(afterFirst).toBeGreaterThan(10)

    host.enqueue({ type: 'splash', x: 0, y: 8 })
    await frame()
    const afterSecond = fluidStats(last()).count

    host.enqueue({ type: 'splash', x: 0, y: 8 })
    await frame()
    // Saturated: the disc is full, further identical clicks add zero.
    expect(fluidStats(last()).count).toBe(afterSecond)
  })

  it('pump respects pause: counters advance, the world does not', async () => {
    const { host, frame, last } = makeHost()
    // Splash well clear of the beach so half a second of fall stays airborne
    // (a splash near the surface lands and sloshes UP, which is physics, not
    // a pause bug). The host's terrain is deterministic from the width.
    const ground = host.field.terrain.heightAt(0)
    const dropY = ground + 12
    host.enqueue({ type: 'splash', x: 0, y: dropY })
    await frame()
    const spawned = fluidStats(last())
    const frame0 = last().scalars.frame

    host.enqueue({ type: 'pump', steps: 5, requestId: 1 })
    await frame()
    const paused = last()
    expect(paused.scalars.frame).toBe(frame0 + 5 + 1) // 5 pumped + 1 ticked
    // Paused means the water hangs exactly where it spawned.
    expect(fluidStats(paused).meanY).toBeCloseTo(spawned.meanY, 6)

    host.enqueue({ type: 'togglePause' })
    host.enqueue({ type: 'pump', steps: 31, requestId: 2 })
    await frame()
    // Ballistic check on the CENTRE OF MASS: a fresh splash carries some
    // pair-repulsion fizz (see the spawnDisc jitter-after-guards issue), but
    // that fizz is momentum-conserving, so the blob's mean must fall like a
    // stone: g*t^2/2 = 1.31 m over 32 steps (31 pumped + 1 ticked).
    const t = 32 / 60
    const expectedDrop = 0.5 * 9.81 * t * t
    const drop = spawned.meanY - fluidStats(last()).meanY
    expect(drop).toBeGreaterThan(expectedDrop * 0.8)
    expect(drop).toBeLessThan(expectedDrop * 1.2)
  })

  it('round-trips save -> clearAll -> load through the protocol', async () => {
    const { host, messages, frame, last } = makeHost()
    host.enqueue({
      type: 'buildMember',
      fromNode: null,
      fromX: -2,
      fromY: 6,
      toNode: null,
      toX: 3,
      toY: 6,
      material: 'steel',
    })
    host.enqueue({ type: 'requestSave', requestId: 7 })
    await frame()
    const saved = messages.find((m) => m.type === 'saveData')
    expect(saved).toBeDefined()
    if (saved?.type !== 'saveData') throw new Error('unreachable')
    expect(saved.requestId).toBe(7)
    expect(saved.solution.members.length).toBe(1)
    expect(saved.solution.members[0]!.material).toBe('steel')

    // Serialise like the save dialog does, so what we reload is what a file holds.
    const onDisk = JSON.parse(JSON.stringify({ level: saved.level, solution: saved.solution }))

    host.enqueue({ type: 'clearAll' })
    await frame()
    expect(last().scalars.memberCount).toBe(0)

    host.enqueue({ type: 'loadDoc', level: onDisk.level, solution: onDisk.solution, requestId: 8 })
    await frame()
    expect(messages.some((m) => m.type === 'ack' && m.requestId === 8)).toBe(true)
    expect(last().scalars.memberCount).toBe(1)
    expect(last().structure.members.length).toBe(1)
  })

  it('removes a member picked by its wire id', async () => {
    const { host, frame, last } = makeHost()
    host.enqueue({
      type: 'buildMember',
      fromNode: null,
      fromX: 0,
      fromY: 6,
      toNode: null,
      toX: 4,
      toY: 6,
      material: 'wood',
    })
    await frame()
    const id = last().structure.members[0]!.id
    host.enqueue({ type: 'removeMember', id })
    await frame()
    expect(last().scalars.memberCount).toBe(0)
    // Orphaned graph nodes are pruned with the member.
    expect(last().structure.nodes.length).toBe(0)
  })

  it('reuses a recycled snapshot buffer', async () => {
    const { host, frame, snapshots } = makeHost()
    await frame()
    const first = snapshots()[0]!.buffer
    host.enqueue({ type: 'recycleBuffer', buffer: first })
    host.enqueue({ type: 'setWind', kph: 50 }) // dirty, forces a snapshot
    await frame()
    const next = snapshots()[snapshots().length - 1]!.buffer
    expect(next).toBe(first)
  })

  it('reports breakage as events and a rising break count', async () => {
    const { host, frame, last, snapshots } = makeHost()
    host.enqueue({ type: 'loadProbe', probe: 'loadtest' })
    host.enqueue({ type: 'togglePause' })
    // 8 tonnes on a wood cantilever: it must fail within a few seconds.
    host.enqueue({ type: 'pump', steps: 600, requestId: 1 })
    await frame()
    const snap = last()
    expect(snap.scalars.breakCount).toBeGreaterThan(0)
    // Every break delivered exactly once across the snapshot stream, and
    // each event carries a real position for FX: finite, on the field.
    const everyEvent = snapshots().flatMap((s) => s.events)
    expect(everyEvent.length).toBe(snap.scalars.breakCount)
    for (const e of everyEvent) {
      expect(Number.isFinite(e.x) && Number.isFinite(e.y)).toBe(true)
      expect(Math.abs(e.x)).toBeLessThanOrEqual(60)
      expect(Number.isFinite(e.strain)).toBe(true)
    }
  })

  it('setBackend swaps to cpu, and webgpu falls back loudly where none exists', async () => {
    const { host, frame, last } = makeHost()
    await frame()
    expect(last().scalars.backend).toBe('cpu') // Node construction default

    // Requesting webgpu in Node (no navigator.gpu) must degrade, not crash,
    // and SAY SO in the scalar the HUD shows.
    host.enqueue({ type: 'setBackend', backend: 'webgpu' })
    await frame()
    expect(last().scalars.backend).toBe('cpu (no webgpu)')

    host.enqueue({ type: 'setBackend', backend: 'cpu' })
    await frame()
    expect(last().scalars.backend).toBe('cpu')
    // The sim still steps after the swap.
    host.enqueue({ type: 'splash', x: 0, y: host.field.terrain.heightAt(0) + 10 })
    host.enqueue({ type: 'togglePause' })
    host.enqueue({ type: 'pump', steps: 5, requestId: 9 })
    await frame()
    expect(fluidStats(last()).count).toBeGreaterThan(10)
  })

  it('a failing command nacks instead of hanging, and leaves state whole', async () => {
    const { host, messages, frame, last } = makeHost()
    host.enqueue({
      type: 'buildMember',
      fromNode: null,
      fromX: 0,
      fromY: 6,
      toNode: null,
      toX: 4,
      toY: 6,
      material: 'wood',
    })
    await frame()

    // A level from a NEWER build: migrateLevel throws. The doc/solution pair
    // must survive untouched and the requester must get a nack, not silence.
    host.enqueue({
      type: 'loadDoc',
      level: { version: 999 },
      solution: { version: 1, nodes: [], members: [] },
      requestId: 42,
    })
    await frame()
    const nack = messages.find((m) => m.type === 'nack')
    expect(nack).toBeDefined()
    if (nack?.type !== 'nack') throw new Error('unreachable')
    expect(nack.requestId).toBe(42)
    expect(nack.error).toContain('newer')
    // The pre-load world is intact - not half-replaced.
    expect(last().scalars.memberCount).toBe(1)

    // And the queue keeps draining afterwards.
    host.enqueue({ type: 'setWind', kph: 30 })
    await frame()
    expect(messages.filter((m) => m.type === 'snapshot').length).toBeGreaterThan(2)
  })

  it('commands arriving mid-step apply at the next frame boundary, never inside the frame', async () => {
    const { host, frame, last } = makeHost()
    // A deliberately slow async backend: readbacks park until released, the
    // way a GPU frame parks the worker on its staging map. The latch frees
    // every parked AND future readback - one tick can run several catch-up
    // steps, each with its own readback.
    let released = false
    const parked: (() => void)[] = []
    let stepsEntered = 0
    const release = () => {
      released = true
      for (const r of parked.splice(0)) r()
    }
    const before = host.sim.solver
    host.sim.solver = {
      sync: () => {},
      step: () => {
        stepsEntered++
      },
      readback: () =>
        released ? Promise.resolve() : new Promise<void>((res) => parked.push(res)),
    }
    const flush = async () => {
      for (let i = 0; i < 50; i++) await Promise.resolve()
    }

    // Start a tick; it drains the unpause, then parks inside the fake
    // readback mid-frame. (No awaited frame() before this - the fake solver
    // parks EVERY unpaused frame until released.)
    host.enqueue({ type: 'togglePause' })
    const ticking = host.tick(100)
    await flush()
    expect(stepsEntered).toBeGreaterThan(0)
    expect(parked.length).toBe(1)

    // A command lands mid-step. If it applied now, the splash would mutate
    // the particle store while the "GPU" owns the frame.
    const fluidBefore = host.sim.fluidCount
    host.enqueue({ type: 'splash', x: 0, y: host.field.terrain.heightAt(0) + 10 })
    await flush()
    const appliedDuringStep = host.sim.fluidCount !== fluidBefore

    release()
    await ticking
    expect(appliedDuringStep).toBe(false)

    // Next frame boundary: the queued splash applies.
    host.sim.solver = before
    host.enqueue({ type: 'setPaused', paused: true })
    await frame()
    expect(fluidStats(last()).count).toBeGreaterThan(fluidBefore)
  })

  it('reports the achieved sim rate: ~60 when healthy, 0 while paused', async () => {
    const { host, frame, last } = makeHost()
    await frame()
    expect(last().scalars.simFps).toBe(0) // paused

    host.enqueue({ type: 'togglePause' })
    // Enough frames for the smoothing to converge: per-tick step counts
    // oscillate 0/1/2 at float boundaries of the accumulator, so instant
    // samples swing while the mean is 60.
    for (let i = 0; i < 30; i++) await frame()
    const running = last().scalars.simFps
    expect(running).toBeGreaterThan(35)
    expect(running).toBeLessThan(85)

    host.enqueue({ type: 'togglePause' })
    await frame()
    await frame()
    expect(last().scalars.simFps).toBe(0)
  })

  it('reset restores the play snapshot and pauses', async () => {
    const { host, frame, last } = makeHost()
    host.enqueue({
      type: 'buildMember',
      fromNode: null,
      fromX: 0,
      fromY: 6,
      toNode: null,
      toX: 4,
      toY: 6,
      material: 'wood',
    })
    host.enqueue({ type: 'togglePause' }) // Play: snapshots the build
    host.enqueue({ type: 'clearBuild' }) // mid-run edit
    await frame()
    expect(last().scalars.memberCount).toBe(0)

    host.enqueue({ type: 'reset' })
    await frame()
    const snap = last()
    expect(snap.scalars.paused).toBe(true)
    expect(snap.scalars.memberCount).toBe(1) // the play snapshot came back
    expect(snap.scalars.breakCount).toBe(0)
  })
})
