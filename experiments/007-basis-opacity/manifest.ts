import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  // Radius (m) of the region around the camera populated each frame. Bounds
  // per-frame work; does NOT change placement, only visibility.
  regionRadius: p.num(80, { min: 24, max: 200, step: 4 }),
  // Contrast applied to the reconstructed opacity before the alpha test.
  // Fourier truncation makes silhouettes fuzzy; this sharpens them back.
  alphaGain: p.num(2.4, { min: 0.5, max: 6, step: 0.1 }),
  // 0 = hard alpha test at 0.5, 1 = fully dithered (stochastic) coverage.
  dither: p.num(0.4, { min: 0, max: 1, step: 0.05 }),
  // View-dependent luminance modulation strength (order-1 color Fourier).
  colorView: p.num(1, { min: 0, max: 2, step: 0.05 }),
}

export default defineExperiment({
  id: '007-basis-opacity',
  title: 'Fourier appearance cards',
  description:
    'Per-texel truncated Fourier series (azimuth, order 3) x elevation rings encode view-dependent opacity/color/depth/normal; one card per plant evaluates the basis in closed form per pixel — smooth from every angle, no view atlas, no per-view blending.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 1,
  load: () => import('./main.ts'),
})
