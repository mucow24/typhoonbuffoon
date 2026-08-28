import { playwright } from '@vitest/browser-playwright'
import { defineConfig } from 'vitest/config'

// Simulation tests run thousands of steps - some settle two minutes of world
// time - and CI runners are slower than a dev machine. The default 5s is not
// in the right order of magnitude.
const TIMEOUTS = {
  testTimeout: 300_000,
  hookTimeout: 300_000,
}

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          ...TIMEOUTS,
          name: 'unit',
          include: ['test/**/*.test.ts'],
          exclude: ['test/gpu/**', '**/node_modules/**'],
        },
      },
      {
        test: {
          ...TIMEOUTS,
          name: 'gpu',
          include: ['test/gpu/**/*.test.ts'],
          browser: {
            enabled: true,
            headless: true,
            screenshotFailures: false,
            provider: playwright({
              launchOptions: {
                // channel 'chromium' selects the FULL Chromium build in new
                // headless mode - the default headless shell has no GPU
                // adapter at all. Playwright injects --disable-gpu into
                // headless launches, which nulls every WebGPU adapter; strip
                // it and opt headless into WebGPU explicitly.
                channel: 'chromium',
                ignoreDefaultArgs: ['--disable-gpu'],
                args: ['--enable-unsafe-webgpu', '--enable-gpu'],
              },
            }),
            instances: [{ browser: 'chromium' }],
          },
        },
      },
    ],
  },
})
