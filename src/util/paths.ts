/**
 * Runtime asset URLs.
 *
 * Production builds are served under a base path (`base` in vite.config.ts,
 * e.g. /groundcover-lab/), so every repo-root-absolute runtime path —
 * `/mesh/...`, `/experiments/...`, `/results/...` — has to be resolved
 * against import.meta.env.BASE_URL. In dev BASE_URL is '/' and assetUrl is
 * the identity, so dev behaviour is unchanged.
 *
 * Assets that go through the module graph (WGSL, manifests, CSS) are rewritten
 * by Vite itself; assetUrl is only for things fetched by hand at runtime.
 */

const BASE = import.meta.env.BASE_URL

/** '/mesh/baked/x.bin' -> '/groundcover-lab/mesh/baked/x.bin' (identity in dev). */
export function assetUrl(path: string): string {
  if (/^[a-z][a-z0-9+.-]*:/i.test(path)) return path // already absolute URL
  return `${BASE.replace(/\/+$/, '')}/${path.replace(/^\/+/, '')}`
}

