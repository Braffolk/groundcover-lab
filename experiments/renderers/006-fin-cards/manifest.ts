import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  // Radius (m) of the region around the camera populated each frame. Bounds
  // per-frame work; does NOT change placement, only how far cards are drawn.
  regionRadius: p.num(56, { min: 16, max: 160, step: 4 }),
  // |cos(view, fin normal)| at which a fin reaches full opacity. Below it the
  // fin dissolves (hashed alpha) toward its edge-on orientation.
  edgeFade: p.num(0.38, { min: 0.08, max: 0.9, step: 0.02 }),
  // Vertical view component (vd.y) where vertical fins START fading out in
  // favour of the horizontal slab cards (kills the top-down star artifact).
  topBlend: p.num(0.72, { min: 0.4, max: 0.95, step: 0.01 }),
  // Coverage sharpening applied to the (premultiplied-mip) atlas alpha before
  // the hashed-alpha test; >1 compensates coverage thinning in the mip chain.
  alphaSharp: p.num(1.35, { min: 0.7, max: 2.5, step: 0.05 }),
  // --- carpet species (stand_table.carpet_div > 0, i.e. the bog's Sphagnum) ---
  // Cross-sections in the cushion stack at the closest LOD (see moss.wgsl).
  // 1 collapses the mat to a single ground quad — the useful A/B for judging
  // what the terraced stack actually buys.
  mossLayers: p.num(6, { min: 1, max: 6, step: 1 }),
  // Alpha reference for the cross-section cards ABOVE the closure card: how
  // much of each terrace survives, i.e. how pronounced the cushion relief
  // reads. The closure card has its own low reference (a mat must stay solid).
  mossRelief: p.num(0.5, { min: 0.15, max: 0.9, step: 0.05 }),
  // Inspection aid: replace albedo with a per-tile hue so you can see WHICH of
  // the 8 baked views (3 fins x 2 sides, 2 top-down slabs) covers each pixel —
  // for a carpet, which cross-section of the stack is the visible surface.
  // Complements the global `view` debug modes — it is this method's own state.
  cardTint: p.bool(false),
}

export default defineExperiment({
  id: '006-fin-cards',
  title: 'Trig-weighted crossed fins',
  description:
    'Classic crossed-cards pushed hard: 3 static fins + 2 horizontal slab cards per plant, each card showing a DIFFERENT baked view (6 azimuth + 2 top slabs); trig view weights + hashed alpha make edge-on fins and the top-down star dissolve. Carpet species instead get a stack of ground-parallel cross-sections through the cushion — a terraced height field with real thickness.',
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
