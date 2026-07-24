import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** View-bin resolve: stochastic per-pixel pick (default), nearest, or 4-tap bilinear. */
  taps: p.enum('dither', ['dither', '1', '4'] as const),
  /** Scales the aggregate wind advection of the canopy texture field. */
  windAmount: p.num(1, { min: 0, max: 2, step: 0.05 }),
  /** Multiplies aggregate coverage (alpha) before the dither test. */
  coverage: p.num(1, { min: 0.4, max: 1.5, step: 0.05 }),
  /** Per-macro-cell brightness variation to break tile repetition. */
  macroTint: p.bool(true),
  /** Draw the canopy-top shell layer (crest/sky silhouettes). */
  topShell: p.bool(true),
  /**
   * Method-local inspection the global `view` modes cannot express: which of
   * the 25 hemi-oct view bins a pixel resolved to, the distance detail fade,
   * or which proxy layer (ground grid / canopy-top shell) drew it. The global
   * debug views take precedence when both are set.
   */
  inspect: p.enum('off', ['off', 'bin', 'lod', 'layer'] as const),
}

export default defineExperiment({
  id: '009-canopy-btf',
  title: 'canopy BTF field',
  description:
    'Gives up per-plant identity: each species is baked into a periodic canopy BTF — a hemi-octahedral grid of orthographic community-tile captures parameterised by the view ray ground intersection. Runtime rasterises a plant-count-free ground+canopy shell and reconstructs aggregate appearance (albedo, coverage, hit height, normal) per pixel from precomputed lookups.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 1,
  load: () => import('./main.ts'),
})
