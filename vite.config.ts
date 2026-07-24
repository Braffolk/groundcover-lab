import path from 'node:path'
import { defineConfig } from 'vite'
import { devSink } from './tools/vite-plugin-dev-sink.ts'
import { staticDeploy } from './tools/vite-plugin-static-deploy.ts'
import { wgsl } from './tools/vite-plugin-wgsl.ts'

/**
 * Dev serves at '/', production builds are deployed under /groundcover-lab/
 * (S3/CloudFront/Pages — see README "Static deploy"). The app is hash-routed,
 * so no server rewrite rules are needed; runtime asset fetches resolve
 * through src/util/paths.ts → assetUrl().
 */
export default defineConfig(({ command, isPreview }) => ({
  // `vite preview` reports command 'serve' but serves the built site, so it
  // needs the deploy base too — only the real dev server stays at '/'.
  base: command === 'build' || isPreview ? '/groundcover-lab/' : '/',
  plugins: [wgsl(), devSink(), staticDeploy()],
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
        '**/experiments/*/rating.json',
        '**/goldens/**',
        '**/results/**',
        '**/mesh/baked/**',
        '**/mesh/raw/**',
      ],
    },
  },
}))
