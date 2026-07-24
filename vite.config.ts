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
  },
})
