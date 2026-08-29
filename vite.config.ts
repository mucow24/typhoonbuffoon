import { defineConfig } from 'vite'

// PORT comes from the preview harness when several sessions/worktrees run dev
// servers side by side; the fixed defaults stay for humans typing `npm run dev`.
const envPort = Number(process.env.PORT)

export default defineConfig({
  server: {
    port: envPort > 0 ? envPort : 5173,
    strictPort: true,
  },
  preview: {
    port: envPort > 0 ? envPort : 4173,
    strictPort: true,
  },
  build: {
    target: 'es2022',
  },
})
