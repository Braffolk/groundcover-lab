import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the camera-centered region that is evaluated per frame. */
  regionRadius: p.num(110, { min: 24, max: 128, step: 4 }),
  /** Alpha-test reference for the baked coverage. */
  alphaRef: p.num(0.4, { min: 0.1, max: 0.8, step: 0.05 }),
  /** How far the lattice bends out to the baked front surface. 0 = billboard. */
  curvature: p.num(1, { min: 0, max: 1.4, step: 0.05 }),
  /** Darkening of texels that sit deep inside the clump. */
  depthShade: p.num(0.15, { min: 0, max: 0.7, step: 0.05 }),
  /**
   * Carpet species only: how dark the gaps BETWEEN capitula get. The baked
   * top-view front distance is the cushion's own height field, so low texels
   * are the crevices; darkening them is the cavity shading a straight-down
   * capture has no sky occlusion for. Mean-preserving, so it adds contrast
   * rather than dimming the bog.
   */
  carpetCrevice: p.num(0.55, { min: 0, max: 1, step: 0.05 }),
  /** How much of the baked sky occlusion reaches the light term. */
  aoStrength: p.num(0.5, { min: 0, max: 1, step: 0.05 }),
  /** Ring 0 (curved 5x7 patch) radius, in metres per unit plant scale. */
  lod0Dist: p.num(7, { min: 0, max: 24, step: 1 }),
  /** Ring 1 (curved 3x5 patch) radius, in metres per unit plant scale. */
  lod1Dist: p.num(20, { min: 0, max: 70, step: 2 }),
}

export default defineExperiment({
  id: '020-curved-cards',
  title: 'curved cards',
  description:
    'Cards that are not flat: a small lattice per plant is pushed out to the baked FRONT-SURFACE DISTANCE of the mesh, and its orientation is snapped to the nearest of 8 baked azimuths instead of tracking the camera — so inside its 45deg sector each card is a fixed 3D relief patch with real parallax, a view-dependent silhouette and per-pixel depth sorting against its neighbours. A canopy patch is displaced by the baked top-view height field. Carpet species (Sphagnum) turn the same idea on its side: a ground-parallel lattice over the periodic tile, displaced by the top view'
    + ' — which for a straight-down capture IS the cushion height field — terrain-conformed per vertex, with the gaps between capitula shaded from the same channel. Three distance rings collapse 66 -> 24 -> 4 triangles, the far ring being exactly the billboard baseline. 2 texture taps per fragment, hard alpha test, no frag_depth.',
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
