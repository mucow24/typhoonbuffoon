import { Game } from './app'
import { SimClient } from './runtime/client'
import type { HostMessage } from './runtime/protocol'

const worker = new Worker(new URL('./runtime/worker.ts', import.meta.url), { type: 'module' })

const client = new SimClient({
  postMessage: (msg, transfer) =>
    transfer ? worker.postMessage(msg, transfer) : worker.postMessage(msg),
  onMessage: (cb) => {
    worker.onmessage = (e: MessageEvent) => cb(e.data as HostMessage)
  },
})

const game = await Game.create(client)
game.start()

if (import.meta.env.DEV) {
  ;(window as unknown as { __tb: unknown }).__tb = {
    game,
    client,
    pump: (steps = 1) => client.pump(steps),
  }
}
