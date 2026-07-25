import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  // Radius (m) of the camera-centred region populated each frame. Bounds the
  // per-frame work; does NOT change placement, only visibility/LOD.
  regionRadius: p.num(96, { min: 24, max: 200, step: 4 }),
  // Alpha-test cutoff on the blended 4-view coverage (hard edge, no dither).
  coverage: p.num(0.38, { min: 0.05, max: 0.9, step: 0.01 }),
  // One eye-ray reprojection tap after the first depth lookup (8 gathers
  // instead of 4) — kills the residual parallax of the ~4.3deg view grid.
  refine: p.bool(true),
  // Carpet species only (Sphagnum): how far the per-fragment normal is pulled
  // from the volume normal toward the light field's own depth gradient. A dense
  // cushion averages every leaf direction into one voxel, so its volume normal
  // is close to noise; the depth gradient is the honest mesoscale relief.
  carpetRelief: p.num(0.85, { min: 0, max: 1, step: 0.05 }),
  // Carpet distance collapse (m): where the mat stops being sampled from the
  // light field and becomes the closed slab of cushion tops it actually reads
  // as at that range. A 0.9mm texel is below one pixel past ~1m, so beyond a
  // few metres the light field can only alias; collapsing is both quieter and
  // much cheaper. Blended over [near, far], never dithered.
  collapseNear: p.num(14, { min: 1, max: 60, step: 1 }),
  collapseFar: p.num(36, { min: 2, max: 120, step: 1 }),
  // Inspection: replace albedo with a stable per-view-node colour, blended by
  // the same 4-view weights. Shows WHICH of the 576 hemi-oct views a fragment
  // is reading and how wide the blend region is (method-specific state, so it
  // lives here and not in the global debug=<mode> selector).
  showViewGrid: p.bool(false),
}

export default defineExperiment({
  id: '015-factored-lightfield',
  title: 'Factored quantized light field',
  description:
    'The 4D per-species light field factored into geometry x radiance: 576 hemi-oct views of 8-bit ortho depth + coverage (one rg8 atlas) answer "what surface do you see from where"; the reconstructed hit point indexes a view-independent 3D appearance volume, so colour cannot ghost between views.',
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
