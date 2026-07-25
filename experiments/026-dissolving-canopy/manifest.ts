import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Distance (m) at which the canopy has fully dissolved into the shell when looking along it (grazing). */
  dissolveGrazing: p.num(106, { min: 16, max: 110, step: 2 }),
  /** Same, looking straight down: from above a canopy IS a continuous surface. */
  dissolveOverhead: p.num(16, { min: 6, max: 60, step: 1 }),
  /** Radius (m) inside which a plant is drawn as its 4 sub-tuft splats instead of one. */
  tuftRadius: p.num(12, { min: 4, max: 36, step: 1 }),
  /** Alpha-test reference for the baked coverage. */
  alphaRef: p.num(0.45, { min: 0.1, max: 0.8, step: 0.05 }),
  /** Far-field coverage threshold of the shell (lower = more solid canopy). */
  shellFloor: p.num(0.3, { min: 0.05, max: 0.9, step: 0.01 }),
  /** Vertical gain on the baked canopy heightfield (relief of the continuous surface). */
  shellRelief: p.num(1.3, { min: 0.4, max: 2, step: 0.05 }),
  /** Fake occlusion: darken tuft bottoms, back tufts and canopy troughs. */
  groundShade: p.num(0.3, { min: 0, max: 0.8, step: 0.05 }),
  /**
   * Carpet species only (Sphagnum): gain on the parallax offset that expresses
   * the cushion's capitulum relief. 0 = a flat printed tile, 1 = the baked
   * height taken at face value. The offset is clamped in metres regardless, so
   * this trades depth against how far a grazing ray may slide the texture.
   */
  carpetRelief: p.num(1, { min: 0, max: 2, step: 0.1 }),
  /** Draw the per-plant tuft splats. */
  tufts: p.bool(true),
  /** Draw the continuous canopy shell. */
  shell: p.bool(true),
}

export default defineExperiment({
  id: '026-dissolving-canopy',
  title: 'dissolving canopy',
  description:
    'A near plant is a cloud of 4 baked sub-tuft splats at four real depths — parallax, self-occlusion and a view-dependent silhouette inside ONE plant; receding, it collapses to one splat and then dissolves, coverage threshold by coverage threshold, into a continuous displaced canopy shell composited from tileable top-views of the same meshes. The dissolve radius follows viewing elevation, so from above the canopy closes into one surface within 16 m while at grazing it stays individual plants past 100 m; the shell then closes up via 1-(1-c)^(1/sin) path-length coverage and re-hues itself from the baked side views.',
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
  bakeVersion: 2,
  load: () => import('./main.ts'),
})
