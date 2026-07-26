import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Scales every screen-size LOD threshold: >1 keeps fine ribbons further out. */
  detail: p.num(1, { min: 0.35, max: 2.5, step: 0.05 }),
  /** Radius (m) of the camera-centred region that is enumerated at all. */
  regionRadius: p.num(96, { min: 32, max: 128, step: 2 }),
  /** Global ribbon width multiplier — fill vs. gaps between blades. */
  widthScale: p.num(1, { min: 0.5, max: 2, step: 0.05 }),
  /** Cross-blade normal curl (rad): 0 = flat strips, ~0.8 = channelled blades. */
  bladeCurl: p.num(0.75, { min: 0, max: 1.4, step: 0.05 }),
  /** Per-ribbon wind flutter on top of the shared per-plant sway. */
  flutter: p.num(0.6, { min: 0, max: 2, step: 0.05 }),
  /** Bias of the shading normal toward the canopy normal (scattering stand-in). */
  canopyLift: p.num(0.4, { min: 0, max: 1.2, step: 0.05 }),
  /** Method-local inspection: LOD bucket / ribbon id / variant / fluffiness. */
  show: p.enum('off', ['off', 'lod', 'ribbon', 'variant', 'fluff']),
}

export default defineExperiment({
  id: '014-ribbon-skeleton',
  title: 'ribbon skeleton',
  description:
    'Plants distilled into a nested hierarchy of curved ribbons in a 12x360 texture, unrolled as strips in the vertex shader; screen-size LOD morphs each ribbon into its parent, so levels collapse with zero popping. Carpet species (Sphagnum) reparameterise the same rows outward instead of upward: a radial fan that tiles the periodic tile square exactly, carries the cushion surface relief, and is terrain-conformed per vertex.',
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
  bakeVersion: 4,
  load: () => import('./main.ts'),
})
