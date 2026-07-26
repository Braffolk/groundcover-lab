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
  // Inspection aid (method-specific state; the global `view` dropdown owns
  // albedo/normals/lighting/coverage/depth). Tints each fragment by the
  // elevation ring it sampled — hue = ring index 0..3, green = the lerp
  // fraction to the next ring. 0 = off; only affects the normal view.
  ringTint: p.num(0, { min: 0, max: 1, step: 0.05 }),

  // --- carpet path (stand entries with carpetDiv > 0, i.e. the Sphagnum mat) --
  // Alpha reference for mat tiles, deliberately far below the cards' 0.5: a mat
  // is a closed surface and must not dissolve with distance, but the ~16% of the
  // tile that is genuinely bare peat still opens. Hard-edged, never dithered.
  carpetAlphaRef: p.num(0.06, { min: 0.01, max: 0.6, step: 0.01 }),
  // Strength of the horizon self-shadowing of the direct sun between capitula.
  sunShadow: p.num(1, { min: 0, max: 1, step: 0.05 }),
  // Strength of the view-dependent horizon occlusion: how much a crevice that
  // the view direction cannot actually see is darkened. This is the term that
  // makes the mat close up into separate cushions as the camera drops.
  viewOcclusion: p.num(0.55, { min: 0, max: 1, step: 0.05 }),
  // Baked sky visibility (ambient occlusion) of the fitted horizon ring.
  carpetAo: p.num(1, { min: 0, max: 1, step: 0.05 }),
  // Uniform overscale of every mat tile. The grid step, the 90-degree yaw and
  // the per-tile scale are untouched, so the lattice invariant holds; 1.0
  // (exact abutment) is the default and measured best.
  carpetOverscale: p.num(1, { min: 1, max: 1.4, step: 0.05 }),
  // Inspection aid for the carpet's method-specific state: baked relief height
  // (red->green) and view visibility (blue). 0 = off; normal view only.
  reliefTint: p.num(0, { min: 0, max: 1, step: 0.05 }),
}

export default defineExperiment({
  id: '007-basis-opacity',
  title: 'Fourier appearance cards',
  description:
    'Per-texel truncated Fourier series (azimuth, order 3) x elevation rings encode view-dependent opacity/color/depth/normal; one card per plant evaluates the basis in closed form per pixel — smooth from every angle, no view atlas, no per-view blending. Carpet species swap the appearance fit for a horizon fit over a periodic ground-parallel tile.',
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
