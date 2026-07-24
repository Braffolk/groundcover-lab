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

/** Repo trees that are copied verbatim into the build and fetched by path. */
const RUNTIME_ASSET_PREFIXES = ['/mesh/', '/experiments/', '/results/', '/goldens/']

/**
 * Compatibility net for code the harness does not own: experiments are free
 * to fetch their own committed bakes with a plain `fetch('/mesh/baked/...')`
 * (several do), and CLAUDE.md forbids agents from touching shared code — so
 * under a base path we rewrite those root-absolute asset requests here rather
 * than breaking every experiment that took the obvious route. No-op in dev
 * (BASE is '/'), and it only ever touches the four prefixes above.
 */
export function installAssetBaseShim(): void {
  if (BASE === '/' || BASE === '') return
  const original = globalThis.fetch.bind(globalThis)
  const rewrite = (url: string): string =>
    RUNTIME_ASSET_PREFIXES.some((p) => url.startsWith(p)) ? assetUrl(url) : url
  globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    if (typeof input === 'string') return original(rewrite(input), init)
    if (input instanceof URL) return original(new URL(rewrite(input.pathname) + input.search, input), init)
    if (input instanceof Request && input.url.startsWith(location.origin)) {
      const url = new URL(input.url)
      const path = rewrite(url.pathname)
      if (path !== url.pathname) return original(new Request(new URL(path + url.search, url), input), init)
    }
    return original(input, init)
  }) as typeof fetch
}
