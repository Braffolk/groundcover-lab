/**
 * Owner's visual ratings (1-5), stored INSIDE each experiment as
 * <dir>/rating.json — deleting an experiment deletes its rating, new
 * experiments start unrated, nothing central to go stale. Written through
 * the dev-sink; read via plain fetch so static builds still display them.
 * Agents never touch rating.json (CLAUDE.md rule).
 *
 * Keyed by the experiment's DIRECTORY, not its id: once experiments gained a
 * hierarchy the two stopped being the same string. The returned maps are still
 * keyed by id, because that is what the UI and the scoring index use.
 */

import { HAS_DEV_SINK } from '../util/env.ts'
import { assetUrl } from '../util/paths.ts'

/** Enough of a RegistryEntry to locate its rating file. */
export interface RatedRef {
  id: string
  dir: string
}

export async function fetchRating(dir: string): Promise<number | null> {
  try {
    const res = await fetch(assetUrl(`/${dir}/rating.json`))
    if (!res.ok) return null
    const { visual } = (await res.json()) as { visual?: number }
    return Number.isInteger(visual) && visual! >= 1 && visual! <= 5 ? visual! : null
  } catch {
    return null
  }
}

export async function fetchAllRatings(refs: readonly RatedRef[]): Promise<Map<string, number>> {
  const out = new Map<string, number>()
  await Promise.all(
    refs.map(async (ref) => {
      const v = await fetchRating(ref.dir)
      if (v !== null) out.set(ref.id, v)
    }),
  )
  return out
}

/** visual: 1-5 to set, null to clear. Requires the dev server (HAS_DEV_SINK). */
export async function saveRating(dir: string, visual: number | null): Promise<void> {
  if (!HAS_DEV_SINK) throw new Error('ratings are read-only on a static build')
  const res = await fetch(`/__rating?exp=${encodeURIComponent(dir)}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ visual }),
  })
  if (!res.ok) throw new Error(`saving rating failed: ${res.status} ${await res.text()}`)
}
