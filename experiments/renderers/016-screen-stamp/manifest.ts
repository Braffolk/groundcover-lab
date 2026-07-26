import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  maxDist: p.num(80, { min: 30, max: 160, step: 5 }),
  tintStrength: p.num(0.9, { min: 0, max: 1, step: 0.05 }),
  // Method-specific inspection of the screen-tile state, on top of the global
  // `view` debug modes: `fill` = list occupancy (green empty -> red at K=64),
  // `mode` = binning strategy per tile (blue tint-only / green enumerate /
  // orange column-march).
  tileView: p.enum('off', ['off', 'fill', 'mode'] as const),
}

export default defineExperiment({
  id: '016-screen-stamp',
  title: 'screen-stamp tile binning',
  description:
    'Inverted loop: a compute pass per 16px screen tile finds the plants that project into it (depth-guided footprint, procedural scatter twin, frustum-tested, sorted front-to-back), then one fullscreen pass composites at most K precomputed hemi-octa impostor lookups per pixel with early termination. Carpet species skip the lists entirely: the mat is stamped per pixel straight from the depth buffer (ground point -> grid node -> wetness state -> that tile\'s high-res top-view bake, one parallax step for cushion relief). Per-frame cost is bounded by screen tiles, never by plant count.',
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
  load: () => import('./main.ts'),
})
