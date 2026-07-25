import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the camera-centered region that is evaluated per frame. */
  regionRadius: p.num(110, { min: 24, max: 112, step: 4 }),
  /** Alpha-test reference for the baked coverage. */
  alphaRef: p.num(0.42, { min: 0.1, max: 0.8, step: 0.02 }),
  /** Depth-shell strength: 0 = flat billboard cards, 1 = baked front surface. */
  domeScale: p.num(1, { min: 0, max: 1.4, step: 0.05 }),
  /**
   * Screen-space canopy-thickness occlusion. Costs ONE extra fullscreen pass,
   * so it ships off — see NOTES.md ("what the extra pass buys").
   */
  canopyOcclusion: p.num(0, { min: 0, max: 0.8, step: 0.02 }),
  /** Sun transmission through thin (low baked thickness) parts. */
  translucency: p.num(0.55, { min: 0, max: 1.5, step: 0.05 }),
  /** Radius (m) inside which plants use the full 8x8 depth shell. */
  lodNear: p.num(12, { min: 4, max: 16, step: 1 }),
  /** Radius (m) inside which plants use the 4x4 depth shell (flat cards beyond). */
  lodFar: p.num(38, { min: 20, max: 48, step: 2 }),
}

export default defineExperiment({
  id: '033-screenspace-thickness',
  title: 'depth-shell impostors',
  description:
    'One camera-facing card per plant, drawn as a 9x9 lattice displaced along the view axis by a baked FRONT-DEPTH SHELL (13 views), so the card is a real 3D surface in the depth buffer: intra-silhouette parallax, curved interpenetration with neighbours and honest ground contact without frag_depth. Baked per-view burial AO + ray thickness carry the volumetric shading (thin tips transmit sun); an optional screen-space canopy-thickness pass adds inter-plant occlusion and contact shadows. Shell collapses 8x8 -> 4x4 -> flat quad with distance, so a distant plant is exactly a billboard.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 3,
  load: () => import('./main.ts'),
})
