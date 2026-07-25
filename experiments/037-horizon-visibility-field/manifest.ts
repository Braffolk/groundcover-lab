import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the ground-conformal carrier hull that is rasterized. */
  regionRadius: p.num(120, { min: 24, max: 128, step: 4 }),
  /** Alpha reference for the baked canopy coverage (hard test, no dither). */
  alphaRef: p.num(0.45, { min: 0.1, max: 0.9, step: 0.05 }),
  /** Direction taps into the ray table: 1 = nearest, 2 = lerp elevation, 4 = bilinear. */
  dirTaps: p.enum('2', ['1', '2', '4']),
  /** Mip bias for the prefiltered ray answer (higher = softer with distance). */
  lodBias: p.num(0, { min: -2, max: 3, step: 0.25 }),
  /** Shade + depth-write the baked ground answer under thin canopy. */
  groundContact: p.bool(true),
  /** Apply the exact inverse wind shear to the ray query. */
  windShear: p.bool(true),
  /** Re-sample the canopy-height field at the entry point (grazing accuracy). */
  refineEntry: p.bool(true),
}

export default defineExperiment({
  id: '037-horizon-visibility-field',
  title: 'horizon visibility field',
  description:
    'Zero plant primitives: all that is rasterized is one instanced draw of a fixed ground-conformal polar grid on 1-6 iso-clearance shells (49k triangles per shell, identical at 557k or 134M plants), and each pixel resolves its own eye ray in 5-8 texture fetches out of a baked 4D visibility table — where the canopy first blocks that ray, plus hit depth, oct normal, coverage, albedo and baked sky occlusion. Extinction is baked as per-axis projected leaf area, so coverage rises from 0.47 straight down to 0.996 at grazing like a real blade canopy. Per-plant position, height, scale, species and wind phase come from a canopy-hull field stamped from the stand scatter; wind is the exact inverse of the harness shear applied to the query.',
  status: 'wip',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 1,
  load: () => import('./main.ts'),
})
