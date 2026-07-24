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
}

export default defineExperiment({
  id: '015-factored-lightfield',
  title: 'Factored quantized light field',
  description:
    'The 4D per-species light field factored into geometry x radiance: 576 hemi-oct views of 8-bit ortho depth (one r8 atlas) answer "what surface do you see from where"; the reconstructed hit point indexes a view-independent 3D appearance volume, so colour cannot ghost between views.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  load: () => import('./main.ts'),
})
