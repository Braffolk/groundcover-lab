import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the camera-centered region that gets materialized. */
  maxDist: p.num(80, { min: 20, max: 96, step: 4 }),
  /** Width (m) of the dithered fade band at maxDist. */
  fadeBand: p.num(14, { min: 2, max: 40, step: 1 }),
  /** Fake occlusion darkening toward the plant base (uses baked height). */
  heightAO: p.num(0.35, { min: 0, max: 0.8, step: 0.05 }),
  /** Wind shear multiplier on top of the stand's per-species sway. */
  swayMul: p.num(1, { min: 0, max: 2, step: 0.1 }),
  /**
   * Alpha reference for CARPET species (`carpet_div > 0`): a mat is tested hard
   * against this instead of being dithered, so it stays a closed depth-writing
   * occluder. Lower = more of the cushion's fringe survives.
   */
  matCov: p.num(0.3, { min: 0.05, max: 0.8, step: 0.05 }),
  /**
   * Method-specific inspection view (the global `view` dropdown covers
   * albedo/normals/lighting/coverage/depth): flat-tint every fragment by the
   * baked direction cell it answered from, exposing the 24x24 direction
   * quantization and its popping. Composes with the global debug views.
   */
  showCells: p.bool(false),
}

export default defineExperiment({
  id: '004-raycast-lut',
  title: 'ray-answer LUT impostors',
  description:
    'Bakes 576 directions x 64x64 offsets of precomputed raycast answers (albedo/coverage/depth/normal) per species; each plant is one quad whose pixels answer the true eye ray with ~2 LUT fetches, reprojected to a real world hit for true depth.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: [
    'calamagrostis-canescens',
    'elymus-repens',
    'poa-pratensis',
    'spaghnum-palustre-wet-vigorous',
    'spaghnum-palustre-late-season',
    'spaghnum-palustre-sun-exposed',
  ],
  params: PARAMS,
  bakeVersion: 1,
  load: () => import('./main.ts'),
})
