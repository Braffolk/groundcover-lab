import type { StatSummary } from '../gpu/timing.ts'

export const BENCH_SCHEMA_VERSION = 1

/** One benchmark run. Saved as results/<exp>__p-<hash>__<adapter>__<date>.json */
export interface BenchResult {
  schemaVersion: number
  experiment: {
    id: string
    params: Record<string, unknown>
    spline: string
    seed: number
  }
  meta: {
    date: string
    paramsHash: string
    adapterSlug: string
    adapter: { vendor: string; architecture: string; description: string }
    ua: string
    canvas: [number, number]
    /** false => gpu is null; never compare across timing classes. */
    timestampQuery: boolean
    harnessApi: number
    warmup: number
    frames: number
  }
  cpu: { frameMs: StatSummary; jsMs: StatSummary }
  /** Per-pass GPU stats (ms), keyed by pass label. Null without timestamp-query. */
  gpu: Record<string, StatSummary> | null
  vram: { totalBytes: number; bySpecies: Record<string, number> }
  /** Raw measured samples for offline analysis. */
  samples: { cpuFrameMs: number[]; gpu: Record<string, number[]> }
}

/** Total of per-pass GPU p50s — the single sortable "GPU cost" number. */
export function gpuTotalP50(r: BenchResult): number | null {
  if (!r.gpu) return null
  return Object.values(r.gpu).reduce((a, s) => a + s.p50, 0)
}
