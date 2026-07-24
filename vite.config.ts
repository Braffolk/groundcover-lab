import path from 'node:path'
import { defineConfig } from 'vite'
import { devSink } from './tools/vite-plugin-dev-sink.ts'
import { wgsl } from './tools/vite-plugin-wgsl.ts'

export default defineConfig({
  plugins: [wgsl(), devSink()],
  resolve: {
    alias: {
      '@harness': path.resolve(import.meta.dirname, 'src/harness/index.ts'),
    },
  },
  server: {
    port: 5173,
    watch: {
      // Artifacts written by the dev-sink endpoints (thumbnails, goldens,
      // bench results, bakes) must not trigger page reloads mid-session.
      ignored: [
        '**/experiments/*/thumbnail.png',
        '**/goldens/**',
        '**/results/**',
        '**/mesh/baked/**',
        '**/mesh/raw/**',
      ],
    },
  },
})
