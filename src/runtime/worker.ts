import { SimHost } from './host'
import type { Command } from './protocol'

/**
 * The worker shell: the only file that runs inside the Worker global. Wires
 * `self` to a SimHost and drives its clock from a short interval - the
 * stepper's accumulator turns however irregular the timer fires into exact
 * fixed steps. Nothing here is testable and nothing here should grow: logic
 * belongs in SimHost (exempt-layer entry in CLAUDE.md).
 */

const scope = self as unknown as {
  postMessage(msg: unknown, options?: { transfer?: Transferable[] }): void
  onmessage: ((e: MessageEvent) => void) | null
}

const host = new SimHost((msg, transfer) =>
  transfer ? scope.postMessage(msg, { transfer }) : scope.postMessage(msg),
)

scope.onmessage = (e: MessageEvent) => host.handleCommand(e.data as Command)

host.start(performance.now())
setInterval(() => host.tick(performance.now()), 4)
