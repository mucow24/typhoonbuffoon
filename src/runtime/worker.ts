import { SimHost } from './host'
import type { Command } from './protocol'

/**
 * The worker shell: the only file that runs inside the Worker global. Wires
 * `self` to a SimHost and drives its clock from a short interval - the
 * stepper's accumulator turns however irregular the timer fires into exact
 * fixed steps, and the host's own reentrance guard makes overlapping timer
 * fires harmless while a GPU frame is in flight. Nothing here is testable
 * and nothing here should grow: logic belongs in SimHost (exempt-layer entry
 * in CLAUDE.md).
 */

const scope = self as unknown as {
  postMessage(msg: unknown, options?: { transfer?: Transferable[] }): void
  onmessage: ((e: MessageEvent) => void) | null
}

const early: Command[] = []
let host: SimHost | null = null

scope.onmessage = (e: MessageEvent) => {
  const cmd = e.data as Command
  if (host) host.enqueue(cmd)
  else early.push(cmd)
}

void SimHost.create((msg, transfer) =>
  transfer ? scope.postMessage(msg, { transfer }) : scope.postMessage(msg),
).then((h) => {
  host = h
  for (const cmd of early) h.enqueue(cmd)
  early.length = 0
  h.start(performance.now())
  setInterval(() => void h.tick(performance.now()), 4)
})
