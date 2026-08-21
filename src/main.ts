import { Game } from './app'

const game = await Game.create()
game.start()

if (import.meta.env.DEV) {
  ;(window as unknown as { __tb: Game }).__tb = game
}
