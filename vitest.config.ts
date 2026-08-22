import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    // Simulation tests run thousands of steps - some settle two minutes of world
    // time - and CI runners are slower than a dev machine. The default 5s is not
    // in the right order of magnitude.
    testTimeout: 300_000,
    hookTimeout: 300_000,
    include: ['test/**/*.test.ts'],
  },
})
