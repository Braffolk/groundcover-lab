import type { BenchResult } from './schema.ts'

/**
 * Bench results live in results/ as one JSON per run, written by the
 * dev-server sink (POST /__bench). Static builds fall back to download +
 * drag-drop in the results view.
 */

export async function postBenchResult(result: BenchResult): Promise<string> {
  const res = await fetch('/__bench', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(result),
  })
  if (!res.ok) throw new Error(`POST /__bench: ${res.status} ${await res.text()}`)
  return ((await res.json()) as { saved: string }).saved
}

export async function listResultNames(): Promise<string[]> {
  const res = await fetch('/__bench/list')
  if (!res.ok) throw new Error(`GET /__bench/list: ${res.status}`)
  return (await res.json()) as string[]
}

export async function fetchResult(name: string): Promise<BenchResult> {
  const res = await fetch(`/results/${encodeURIComponent(name)}`)
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
