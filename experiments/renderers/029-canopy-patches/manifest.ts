import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the camera-centered region evaluated per frame. */
  regionRadius: p.num(110, { min: 24, max: 128, step: 4 }),
  /**
   * Distance (m) out to which a plant is drawn as a depth-layered patch stack.
   * Beyond it the stack has collapsed into one camera-facing composite card —
   * exactly a billboard. 0 = pure billboards, i.e. the honest A/B of what the
   * layering itself buys.
   */
  patchDist: p.num(10, { min: 0, max: 40, step: 1 }),
  /** Alpha-test reference for the baked coverage. */
  alphaRef: p.num(0.4, { min: 0.1, max: 0.8, step: 0.05 }),
  /** Multiplier on the depth-slab plane offsets (0 = flat, 1 = as baked). */
  slabSpread: p.num(1, { min: 0, max: 1.6, step: 0.05 }),
  /** Per-slab darkening deeper into the canopy (self-occlusion proxy). */
  slabShade: p.num(0.1, { min: 0, max: 0.4, step: 0.02 }),
  /** Fake grounding: darken patch bottoms toward the soil. */
  bottomShade: p.num(0.25, { min: 0, max: 0.6, step: 0.05 }),
  /** Horizontal crown patches (they carry the top-down view). */
  topPatches: p.bool(true),
  /** Tint each fragment by which patch it came from (inspection). */
  patchTint: p.bool(false),
}

export default defineExperiment({
  id: '029-canopy-patches',
  title: 'canopy patches',
  description:
    'Every plant is a stack of canopy patches instead of one card: two view-bin-fixed depth slabs (front/back half of the foliage, split at the depth median) plus two horizontal crown slabs, baked per azimuth from the GCMESH1 mesh. Front foliage parallaxes across back foliage, the silhouette is the union of real depth layers, and the stack collapses geometrically into one camera-facing composite card with distance, so far plants cost exactly a billboard.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 2,
  load: () => import('./main.ts'),
})
