import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Blade rows drawn per plant at full detail; falls off as rFull/d. */
  blades: p.num(320, { min: 48, max: 384, step: 8 }),
  /** Distance (m) out to which every `blades` row is live at TRUE width. */
  rFull: p.num(1.8, { min: 1.2, max: 3, step: 0.1 }),
  /** Plant region edge (m); beyond it the single canopy shell is the meadow. */
  rOuter: p.num(38, { min: 24, max: 44, step: 2 }),
  widthScale: p.num(1, { min: 0.4, max: 2, step: 0.05 }),
  /** Width-boost exponent; 1 conserves silhouette coverage exactly. */
  coverPow: p.num(0.9, { min: 0.5, max: 1, step: 0.05 }),
  /** Geometric keel of the near-field cross-section (fraction of half-width). */
  keel: p.num(0.45, { min: 0, max: 1, step: 0.05 }),
  /** Cross-blade shading-normal rotation (the channelled-blade highlight). */
  curl: p.num(0.75, { min: 0, max: 1.4, step: 0.05 }),
  /** Distance (m) at which ribbons stop using their baked plane. */
  orientFar: p.num(14, { min: 4, max: 34, step: 1 }),
  /** Minimum projected blade width in pixels (anti-shimmer floor). */
  minPx: p.num(0.9, { min: 0, max: 2, step: 0.1 }),
  /** Albedo floor of the baked ambient occlusion. */
  aoMin: p.num(0.6, { min: 0.15, max: 1, step: 0.05 }),
  /** +Y nudge of blade normals; keep small or the canopy flattens. */
  upBias: p.num(0.25, { min: 0, max: 1, step: 0.05 }),
  /** Static per-blade bend amplitude (m) — breaks the cloned-plant look. */
  bendAmp: p.num(0.035, { min: 0, max: 0.12, step: 0.005 }),
  /** Far-shell canopy height as a fraction of the mixed stand height. */
  shellHeight: p.num(0.6, { min: 0.2, max: 1, step: 0.05 }),
  debugRings: p.bool(false),
}

export default defineExperiment({
  id: '034-strand-shell-fidelity',
  title: 'blade library shell',
  description:
    'Real individual grass blades traced tip-down out of the source mesh with their own oriented frame, drawn as keeled cross-section ribbons in the blade plane (never camera-facing up close), silhouette-calibrated against the mesh, collapsing through five rings into one canopy shell.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 10,
  load: () => import('./main.ts'),
})
