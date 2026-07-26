import { HAS_DEV_SINK } from '../util/env.ts'
import { assetUrl } from '../util/paths.ts'
import { bakeCacheGet, bakeCachePruneOther, bakeCachePut } from './cache.ts'

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

/**
 * Revision of the SHARED code every bake reads through — bump it and every
 * OPFS-cached artifact in every browser is invalidated at once.
 *
 * An experiment folds its own `bakeVersion` into `ctx.key`, but it cannot fold
 * in a harness change, and a stale cache silently wins over corrected code:
 * that is exactly what happened when the GCMESH1 octahedral normal decode was
 * corrected (2026-07-26) — every bake that had ever decoded a source normal was
 * wrong, and every one of them would have been served from OPFS anyway.
 * Committed files under mesh/baked/ are NOT namespaced by this (their paths are
 * quoted in NOTES.md and bench results); invalidate those by deleting them.
 *
 *   r1  original
 *   r2  GCMESH1 source normals: stored pair is (x, y) and the derived component
 *       is z, not y (src/wgsl/gcmesh.wgsl, decodeGcMeshNormal)
 */
const SHARED_BAKE_REV = 'r2'

/** OPFS key. 'bin' keeps the historical spelling so no cache needs migrating. */
function cacheKey(ctx: BakeContext): string {
  const ext = ctx.ext ?? 'bin'
  const base = ext === 'bin' ? `${ctx.expId}__${ctx.key}` : `${ctx.expId}__${ctx.key}__${ext}`
  return `${SHARED_BAKE_REV}__${base}`
}

let pruned: Promise<void> | null = null

export async function bakedArtifact(ctx: BakeContext, bake: () => Promise<ArrayBuffer>): Promise<ArrayBuffer> {
  // Once per session, evict artifacts from earlier shared revisions.
  pruned ??= bakeCachePruneOther(`${SHARED_BAKE_REV}__`)
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
