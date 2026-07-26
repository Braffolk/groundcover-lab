import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the camera-centered region that is evaluated per frame. */
  regionRadius: p.num(110, { min: 24, max: 128, step: 4 }),
  /** Alpha-test reference for the baked coverage. */
  alphaRef: p.num(0.4, { min: 0.1, max: 0.8, step: 0.05 }),
  /** Horizontal canopy card (fills the top-down view). */
  topCard: p.bool(true),
  /** Fake grounding: darken card bottoms toward the soil. */
  bottomShade: p.num(0.25, { min: 0, max: 0.6, step: 0.05 }),
  /**
   * Uniform overscale of every carpet tile (mat species only). The grid step is
   * untouched — each tile just covers more than its own cell, the way the
   * source mesh's own 0.24m of geometry overflows its 0.18m period — so
   * neighbours still agree. 1.0 (tiles abut exactly) is the default because it
   * measured BEST here: a flat ground-parallel tile has no thickness to hide an
   * overlap in, and the overscaled fringe samples the sparse overhang part of
   * the baked top view, so every tile edge draws itself as a dark line (a
   * visible lattice at grazing angles). Kept as a knob because the trade-off
   * would flip for a representation with real thickness.
   */
  carpetOverscale: p.num(1, { min: 1, max: 1.5, step: 0.05 }),
}

export default defineExperiment({
  id: '001-billboard-smoke',
  title: 'billboard cards',
  description:
    'Baseline billboards from real baked imagery: per species an 8-azimuth + top-view card atlas (albedo+coverage, oct normals) is baked from the GCMESH1 mesh. Upright plants draw one camera-facing quad plus one horizontal top card; carpet species (Sphagnum) instead draw a single ground-parallel tile-sized quad, per-vertex conformed to the terrain. Alpha-tested, compute-culled into an indirect draw.',
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
