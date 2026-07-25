import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the camera-centred region the cull pass enumerates. */
  regionRadius: p.num(128, { min: 24, max: 128, step: 4 }),
  /** Alpha-test cutoff on blended coverage (hard edges, never dithered). */
  alphaRef: p.num(0.38, { min: 0.05, max: 0.9, step: 0.01 }),
  /**
   * Alpha-test cutoff for CARPET species (stand_table.carpet_div > 0) only. A
   * mat is a closed surface: at the grass reference the mip chain drags a
   * tile's mean coverage below the cutoff and whole distant tiles vanish,
   * punching tile-shaped holes in the carpet. Low is deliberate — it makes the
   * mat MORE opaque and depth-writing, not less (the taste rule's opposite of
   * dithering), while the truly empty texels down to the peat still open.
   */
  carpetAlphaRef: p.num(0.06, { min: 0.01, max: 0.6, step: 0.01 }),
  /**
   * How much of the baked cushion normal a CARPET tile keeps, as detail around
   * the ground normal. The bake flips normals toward each view, so the full
   * baked normal makes a tile's mean orientation follow the view it happened to
   * pick and the mat reads as a chequerboard; only the view-perpendicular part
   * is real geometry, so it is used as a zero-mean BUMP field around the ground
   * normal. 0 = shade the mat as bare ground (visibly flatter, like paint);
   * 0.9 measured best of 0 / 0.5 / 0.9 at a 0.45 m eye — real surface roughness,
   * no per-tile bias.
   */
  carpetNormalDetail: p.num(0.9, { min: 0, max: 1.5, step: 0.05 }),
  /**
   * How far INSIDE its own period a carpet tile samples, as a fraction of the
   * tile. The source mesh overhangs its 0.18 m period by up to 2.1 cm, so in a
   * real tiling the outer ~10% of every tile is covered partly by its
   * neighbour's overhang — which a single-tile capture cannot contain. At 0 the
   * mat therefore shows a bare strip along each tile's far edge up close and a
   * woven lattice at grazing; 0.10 skips the deficient band for a ~25%
   * magnification of a 2-5 mm noise texture. Geometry is untouched (the quad is
   * always exactly one grid step), so the lattice invariant holds either way.
   */
  carpetCropInset: p.num(0.1, { min: 0, max: 0.2, step: 0.01 }),
  /**
   * How the view set resolves per plant: 'blend3' = barycentric blend of the
   * 3 hemi-octahedral views around the viewing direction, 'nearest' = hard
   * switch to the closest view (shows what the blend is buying).
   */
  viewBlend: p.enum('blend3', ['blend3', 'nearest'] as const),
  /** Debug: paint each plant by the id of its dominant baked view. */
  viewTint: p.bool(false),
}

export default defineExperiment({
  id: '005-octa-impostors',
  title: 'Octahedral view-set impostors',
  description:
    '121 hemi-octahedral views per species in a mipped texture array; each plant picks its 3 surrounding views and reprojects the card into each one, so the same quad reads correctly from grazing to straight down. Carpet species (Sphagnum) instead lay ONE ground-parallel tile-sized quad per grid node and use the same view set as a precomputed raycast: projecting a ground point into the view of the actual direction returns the exact colour along that ray, so a flat quad reproduces the cushion relief view-dependently.',
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
