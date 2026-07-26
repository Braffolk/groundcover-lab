import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the camera-centered region that is evaluated per frame. */
  regionRadius: p.num(110, { min: 24, max: 128, step: 4 }),
  /** Radius (m) inside which a plant is drawn as a 4-part card cloud. */
  nearRadius: p.num(16, { min: 0, max: 40, step: 1 }),
  /** Alpha-test reference for the baked coverage. */
  alphaRef: p.num(0.4, { min: 0.1, max: 0.8, step: 0.05 }),
  /** Crown card height as a fraction of the part box. */
  crownHeight: p.num(0.7, { min: 0.3, max: 0.95, step: 0.05 }),
  /** Horizontal crown cards (they carry the top-down view). */
  topCards: p.bool(true),
  /** Sky-occlusion strength read from the canopy cache. */
  aoStrength: p.num(0.7, { min: 0, max: 1, step: 0.05 }),
  /** Sun-transmittance (canopy self-shadow) strength from the canopy cache. */
  sunShadow: p.num(0.6, { min: 0, max: 1, step: 0.05 }),
  /** Per-plant albedo variation (0 = every plant identical). */
  tintVar: p.num(0.12, { min: 0, max: 0.4, step: 0.02 }),
  /** Canopy-cache slabs refreshed per frame (4 = full rebuild, no amortization). */
  cacheSlabs: p.num(1, { min: 1, max: 4, step: 1 }),
}

export default defineExperiment({
  id: '031-temporal-detail',
  title: 'part-cloud cards + amortised canopy cache',
  description:
    'Each plant is a cloud of 4 spatially separated sub-clump cards — mesh triangles split into quadrants at bake time, each quadrant baked as its own 8-azimuth + crown impostor with a tight coverage box — so near parts parallax against far parts and the silhouette is a real 3D union; beyond ~16m the cloud collapses to one whole-plant card. Canopy density and sun/sky transmittance live in a world-anchored toroidal 3D cache refreshed one quarter per frame and read with a single tap for self-occlusion, ground contact and inter-plant shadowing.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 3,
  load: () => import('./main.ts'),
})
