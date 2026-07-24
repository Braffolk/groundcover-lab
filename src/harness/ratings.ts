/**
 * Owner's visual ratings (1-5), stored INSIDE each experiment as
 * experiments/<id>/rating.json — deleting an experiment deletes its rating,
 * new experiments start unrated, nothing central to go stale. Written through
 * the dev-sink; read via plain fetch so static builds still display them.
 * Agents never touch rating.json (CLAUDE.md rule).
 */

import { HAS_DEV_SINK } from '../util/env.ts'
import { assetUrl } from '../util/paths.ts'

export async function fetchRating(id: string): Promise<number | null> {
  try {
    const res = await fetch(assetUrl(`/experiments/${encodeURIComponent(id)}/rating.json`))
    if (!res.ok) return null
    const { visual } = (await res.json()) as { visual?: number }
    return Number.isInteger(visual) && visual! >= 1 && visual! <= 5 ? visual! : null
  } catch {
    return null
  }
}

export async function fetchAllRatings(ids: string[]): Promise<Map<string, number>> {
  const out = new Map<string, number>()
  await Promise.all(
    ids.map(async (id) => {
      const v = await fetchRating(id)
      if (v !== null) out.set(id, v)
    }),
  )
  return out
}

/** visual: 1-5 to set, null to clear. Requires the dev server (HAS_DEV_SINK). */
export async function saveRating(id: string, visual: number | null): Promise<void> {
  if (!HAS_DEV_SINK) throw new Error('ratings are read-only on a static build')
  const res = await fetch(`/__rating?exp=${encodeURIComponent(id)}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ visual }),
  })
  if (!res.ok) throw new Error(`saving rating failed: ${res.status} ${await res.text()}`)
}
