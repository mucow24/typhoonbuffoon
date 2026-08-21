import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    // Simulation tests run thousands of steps; the default 5s is far too tight.
    testTimeout: 120_000,
    hookTimeout: 120_000,
    include: ['test/**/*.test.ts'],
  },
})
