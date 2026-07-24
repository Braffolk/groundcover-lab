import { HAS_DEV_SINK } from '../util/env.ts'
import { assetUrl } from '../util/paths.ts'
import type { BenchResult } from './schema.ts'

/**
 * Bench results live in results/ as one JSON per run, written by the
 * dev-server sink (POST /__bench). Static builds are read-only: new runs are
 * downloaded instead of saved, and the run list comes from results/index.json
 * (emitted at build time by tools/vite-plugin-static-deploy.ts) rather than
 * from the dev-only GET /__bench/list.
 */

export async function postBenchResult(result: BenchResult): Promise<string> {
  if (!HAS_DEV_SINK) throw new Error('no dev-server sink on a static build')
  const res = await fetch('/__bench', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(result),
  })
  if (!res.ok) throw new Error(`POST /__bench: ${res.status} ${await res.text()}`)
  return ((await res.json()) as { saved: string }).saved
}

export async function listResultNames(): Promise<string[]> {
  if (HAS_DEV_SINK) {
    const res = await fetch('/__bench/list')
    if (!res.ok) throw new Error(`GET /__bench/list: ${res.status}`)
    return (await res.json()) as string[]
  }
  const url = assetUrl('/results/index.json')
  const res = await fetch(url)
  if (!res.ok || (res.headers.get('content-type') ?? '').includes('text/html')) {
    throw new Error(`GET ${url}: ${res.status} (no results index in this deployment)`)
  }
  const names = (await res.json()) as unknown
  if (!Array.isArray(names)) throw new Error(`GET ${url}: not an array of result names`)
  return names as string[]
}

export async function fetchResult(name: string): Promise<BenchResult> {
  const res = await fetch(assetUrl(`/results/${encodeURIComponent(name)}`))
  if (!res.ok) throw new Error(`GET results/${name}: ${res.status}`)
  return (await res.json()) as BenchResult
}

export function downloadResult(result: BenchResult): void {
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `${result.experiment.id}__p-${result.meta.paramsHash}__${result.meta.adapterSlug}.json`
  a.click()
  URL.revokeObjectURL(a.href)
}
