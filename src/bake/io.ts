import { HAS_DEV_SINK } from '../util/env.ts'
import { assetUrl } from '../util/paths.ts'
import { bakeCacheGet, bakeCachePut } from './cache.ts'

/**
 * Per-experiment bake flow. An experiment's baked artifact format is entirely
 * its own — the harness only moves bytes:
 *
 *   1. OPFS cache hit?                         -> use it
 *   2. committed file in mesh/baked/<exp>/?    -> use it (and warm the cache)
 *   3. run `bake()` in-browser                 -> cache it
 *
 * `commitBake()` POSTs an artifact to the dev server, which writes it into
 * mesh/baked/<exp>/ for committing to git (no-op fallback on static builds:
 * the promise rejects and the caller should offer a download instead).
 */

/** File extension the artifact is stored under. Default 'bin'. */
export type BakeExt = 'bin' | 'png' | 'json'

export interface BakeContext {
  /** Experiment id — namespaces the artifact. */
  expId: string
  /** Cache key; fold your manifest bakeVersion + params into it. */
  key: string
  /**
   * Artifact extension (default 'bin'). Put it HERE, never inside `key` — the
   * dev sink's path validator allows '.' in a key but writes `${key}.${ext}`,
   * so an extension baked into the key produces `foo.png.bin`.
   */
  ext?: BakeExt
  onProgress?: (fraction: number, note?: string) => void
}

/** OPFS key. 'bin' keeps the historical spelling so no cache needs migrating. */
function cacheKey(ctx: BakeContext): string {
  const ext = ctx.ext ?? 'bin'
  return ext === 'bin' ? `${ctx.expId}__${ctx.key}` : `${ctx.expId}__${ctx.key}__${ext}`
}

export async function bakedArtifact(ctx: BakeContext, bake: () => Promise<ArrayBuffer>): Promise<ArrayBuffer> {
  const fullKey = cacheKey(ctx)
  const cached = await bakeCacheGet(fullKey)
  if (cached) return cached

  // Vite's SPA fallback answers missing files with index.html at HTTP 200 —
  // never treat an HTML response as a baked artifact (found by three
  // experiment agents independently; their caches got poisoned).
  const committed = await fetch(assetUrl(`/mesh/baked/${ctx.expId}/${ctx.key}.${ctx.ext ?? 'bin'}`)).catch(() => null)
  if (committed?.ok && !(committed.headers.get('content-type') ?? '').includes('text/html')) {
    const data = await committed.arrayBuffer()
    void bakeCachePut(fullKey, data)
    return data
  }

  const data = await bake()
  void bakeCachePut(fullKey, data)
  return data
}

export async function commitBake(expId: string, key: string, data: ArrayBuffer, ext: BakeExt = 'bin'): Promise<string> {
  // Static deployment: there is no repo to commit into. Reject without ever
  // issuing the request — every caller already treats this as best-effort.
  if (!HAS_DEV_SINK) throw new Error('commitBake needs the dev server (static build — artifact stays in the OPFS cache)')
  const res = await fetch(
    `/__bake?exp=${encodeURIComponent(expId)}&key=${encodeURIComponent(key)}&ext=${encodeURIComponent(ext)}`,
    { method: 'POST', body: data },
  )
  if (!res.ok) throw new Error(`commit bake failed: ${res.status} ${await res.text()}`)
  const { saved } = (await res.json()) as { saved: string }
  return saved
}
