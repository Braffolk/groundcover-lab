import { gpuTotalP50, type BenchResult } from '../bench/schema.ts'
import { fetchResult, listResultNames } from '../bench/store.ts'

/**
 * Card ranking inputs. Perf comes from bench results in ONE reference context
 * (the most-populated stand/spline/adapter combination), latest run per
 * experiment. Scores use ABSOLUTE anchors so adding or deleting experiments
 * never reshuffles existing scores (no min-max normalization).
 */

export interface PerfIndex {
  /** "stand · spline · adapter" the numbers come from. */
  context: string
  /** Experiment id -> gpu Σp50 ms (latest run in the reference context). */
  byExperiment: Map<string, number>
}

export async function loadPerfIndex(): Promise<PerfIndex> {
  const empty: PerfIndex = { context: 'no bench results yet', byExperiment: new Map() }
  let results: BenchResult[] = []
  try {
    const names = await listResultNames()
    const loaded = await Promise.allSettled(names.map((n) => fetchResult(n)))
    results = loaded.filter((l) => l.status === 'fulfilled').map((l) => l.value)
  } catch {
    return empty
  }
  results = results.filter((r) => r.meta.timestampQuery && gpuTotalP50(r) !== null)
  if (results.length === 0) return empty

  const comboOf = (r: BenchResult): string => `${r.experiment.stand} · ${r.experiment.spline} · ${r.meta.adapterSlug}`
  const comboCounts = new Map<string, number>()
  for (const r of results) comboCounts.set(comboOf(r), (comboCounts.get(comboOf(r)) ?? 0) + 1)
  const context = [...comboCounts.entries()].sort((a, b) => b[1] - a[1])[0]![0]

  const latest = new Map<string, BenchResult>()
  for (const r of results.filter((r) => comboOf(r) === context)) {
    const held = latest.get(r.experiment.id)
    if (!held || r.meta.date > held.meta.date) latest.set(r.experiment.id, r)
  }
  return { context, byExperiment: new Map([...latest].map(([id, r]) => [id, gpuTotalP50(r)!])) }
}

/** 1-5 rating -> 0..1. */
export function visualScore(rating: number): number {
  return (rating - 1) / 4
}

/**
 * gpu Σp50 -> 0..1 on a fixed log ramp: 0.5ms => 1.0, 16ms => 0.
 * Absolute anchors keep scores stable as the experiment set changes — do not
 * retune these without accepting that every displayed balance shifts.
 */
export function perfScore(ms: number): number {
  const t = (Math.log(16) - Math.log(ms)) / (Math.log(16) - Math.log(0.5))
  return Math.min(1, Math.max(0, t))
}

/**
 * 0-100 balance of visuals and perf — geometric mean, so lopsided entries
 * (gorgeous-but-slow, fast-but-ugly) score mediocre by design.
 */
export function balanceScore(rating: number, ms: number): number {
  return Math.round(100 * Math.sqrt(visualScore(rating) * perfScore(ms)))
}
