# blade strips

## Idea

A plant is not one card and not a shell — it is **~16 curved ribbons, each one a
real bundle of its blades, standing where those blades actually stand.**

Precomputed per species (in-browser, once, from the raw GCMESH1 mesh):

1. **Bundle split.** The mesh is voxelised at 2cm and the *occupied voxels* (not
   the vertices — the fluffy panicle carries most of the vertices and would
   swallow every bundle) are k-means clustered into 16 blade bundles, with the
   vertical axis weighted 0.5 so bundles come out as columns / half-columns
   rather than horizontal slabs.
2. **Merge tree.** Bundles are paired bottom-up by centroid distance into a
   perfect binary tree, and the leaves are ordered by that tree, so a cluster at
   *every* level is one CONTIGUOUS run of leaves. One reordered index buffer
   therefore serves all five levels (16 / 8 / 4 / 2 / 1 ribbons) without ever
   duplicating an index.
3. **Ribbon fit.** Per cluster of every level: a 5-node polyline through its
   height-binned centroids — literally the bundle's mean blade arc, so ribbons
   arch outward the way the real blades do — a lateral axis from the principal
   direction of the horizontal residuals, per-node half-widths that hug the real
   lateral spread (90th percentile + 20%), and the residual thickness that the
   flattening throws away (2-12cm at level 0: these are thin slices, not shells).
   The arc spans the height range of the TRIANGLES assigned to the bundle, not of
   the sampled points, so neighbouring bundles overlap instead of cutting blades
   in half.
4. **Unrolled capture.** Each cluster's real triangles are rendered into its own
   atlas tile through a curvilinear vertex transform (`shaders/bake_unroll.wgsl`):
   `u` = lateral offset along the ribbon's lateral axis / half-width, `v` = arc
   parameter, `z` = residual along the ribbon's face normal. The bundle's real
   geometry is flattened onto the ribbon surface *and straightened along its own
   arc*; re-rolling the tile onto the same curve at draw time puts the real blade
   silhouettes, colours and normals back on a curved 3D strip. Normals are stored
   in the ribbon's tangent frame, so wind bending and the runtime twist keep them
   correct. One extra tile holds the straight-down view for a canopy card.

Per frame (two passes, both bounded by the visible region):

- `cull` — one thread per scatter slot of the camera-region cell rect (clamped to
  the stand's own cell range on the CPU). Whole workgroups exit on the
  region-circle and frustum tests before touching the terrain; survivors are
  classified by **projected diameter in pixels** into five buckets and compacted
  into that bucket's instance range with atomics. The bucket's `instanceCount`
  *is* its indexed-indirect draw arg.
- `strips` — five `drawIndexedIndirect` per stand entry, **near bucket first** so
  early-z already holds the closest depth when the far buckets rasterize. Hard
  alpha test + depth write, `cullMode: none`, no blending and no dither anywhere.
  Bucket 3 draws 2-quad ribbons (nodes 0/2/4) and the last bucket one flat quad
  with the normal tap skipped — that bucket holds most of the meadow, so its
  per-plant cost is 4 vertices, 2 triangles, 1 texture tap: *less* than the
  billboard baseline's 12 vertices. A canopy card whose ring cannot pass the
  shader's elevation gate is skipped on the CPU, which at eye level means every
  bucket but the nearest.

Why it is O(1) in plant count: nothing is ever materialised for the stand. Cost
is region area, and per plant it collapses with screen size. Verified: the
`scaling-100m` stand (134.2M plants) renders at the same pass time and the same
VRAM as `default` (557k).

Fades: camera-inside collapses ribbon WIDTH to zero (no alpha ramp, no dither);
the region rim erodes coverage through the alpha reference. Wind bends every node
by `hf^1.5` and tilts the arc tangent by the wind gradient, so a ribbon bends
instead of shearing.

## VRAM budget math

Per species, HUD-verified: **19.3 MiB / 25** (calamagrostis, poa), 18.8 for
elymus.

| item | size |
|---|---|
| albedo atlas 2048x1024 rgba8 + 7 mips | 11.2 MiB |
| tangent-frame normal atlas 2048x1024 rg8 + 7 mips | 5.6 MiB |
| culled instance pool, 16 B/instance, sized for regionRadius 128 at the entry's density | 3.5 MiB @ density 3 |
| ribbon table (31 x 96 B), 5 bucket info slots, indirect args | ~6 KiB |

The atlas is 32 tiles of 128 x 512 (31 ribbons over 5 LOD levels + 1 top view).
Tile aspect and size are not free choices: at 64 x 1024 the blade *edges* went
soft (edges run across the arc, so lateral resolution sets crispness) and the
flat billboard card read as the sharper image up close — exactly the failure this
method exists to avoid. 128 x 512 is ~2mm/texel both ways on a bundle, the same
texel density the baseline gets from a 512px card, at 1/16 of the plant.

Mips are built on the CPU at load and are **coverage-preserving**: a strip tile is
only ~11% covered, so a plain box-filtered alpha drops under the alpha reference
within two levels and the whole mid-field dissolves into the terrain. Each tile's
alpha is rescaled per level to match its mip-0 alpha-test coverage (the standard
alpha-preserving-mipmap trick). This was the single biggest visual fix in the
whole experiment.

Bake transients (tagged `bake-scratch`, destroyed at the end): the raw mesh
vertex + reordered index buffers, 4096x2048 supersampled targets, two readbacks —
~250-350 MB for poa (8.5M verts).

## Bake

`bake.ts` -> `mesh/baked/032-blade-strips/strips-v7-<species>.bin`, 12.6 MiB each
(128 B header + 31 x 96 B ribbon table + 2048x1024 rgba8 albedo + rg8 normals).
Harness `bakedArtifact`/`commitBake` flow with magic+size validation on every
load (the dev server answers missing /mesh/baked files with index.html at 200,
which would poison both the OPFS cache and the committed-file path; a poisoned
entry is rebaked and the OPFS entry repaired). Full three-species bake ~40 s
including fetching 374 MB of raw meshes; afterwards the committed artifacts load
in under a second. Stale v1-v6 variants were deleted.

## Status

**working** — verified by headless screenshots at grazing / topdown /
inside-plant / far-horizon on `default`, plus `close-quality`, `dense-mixed` and
`scaling-100m`, plus `debug=normals` and `debug=lighting`, plus the A/B page
against the billboard baseline. Zero console errors, zero shader warnings, all
three species render. `npx tsc --noEmit` clean.

## Findings

### Looks vs the billboard baseline (`#/ab/001-billboard-smoke/032-blade-strips?cam=grazing&seed=42`)

Honest verdict: **better on everything the 3D bar asks for, roughly equal as a
postcard.**

- **Parallax inside the silhouette: real.** The ribbons are world-anchored (only
  their width axis twists toward the camera, by `faceCam`, and the arc never
  moves), so a plant's near ribbons genuinely shift against its far ones as the
  camera translates. Checked with a pure 0.35 m lateral translation pair at the
  grazing pose: the near bundles slide across the far bundles of the *same*
  plant. The baseline cannot do this — its card rotates, so its interior is
  rigid, and that rotation is visible as swim.
- **Silhouette genuinely changes with view direction**, because the outline is
  the union of 16 arcs at their real 3D positions, not one shape being rotated.
  In flicker mode against the baseline, the baseline's plants snap between 8
  azimuth bins; blade strips have no snapping and no discrete azimuths at all.
- **Self-occlusion and depth layering: real** — opaque alpha-tested ribbons
  depth-test against each other, and at `inside-plant` you can see sky through
  the canopy between individual blade bundles instead of through one card.
- **Per-plant variety**, which the baseline structurally lacks: every plant is
  the same imagery in the baseline, while here the plant's yaw rotates a genuine
  3D arrangement, the tile mirror flips per plant, and a +-9% albedo jitter
  breaks the wallpaper repetition. The baseline's regular repeating motif is very
  visible in a mid-field crop; this is the clearest side-by-side difference.
- **Where the baseline still wins:** its card is one 512px capture of the WHOLE
  plant, so its blades are long, continuous and slightly denser / more saturated.
  Blade strips chop the plant into 16 pieces and resample each through the
  unroll, so the mid-field reads a touch airier and greener. Top-down the two are
  near-indistinguishable (both fall back to a baked canopy card).

### Performance (all contended — 15 sibling agents shared this GPU; no absolute ms quoted)

- The A/B page has a **systematic slot bias**: rendering the *same* experiment on
  both sides measures B at 1.33x A (median of 5 samples). Raw A/B ratios must
  therefore be corrected. At grazing, `001 -> A, 032 -> B` measured 1.9-2.6x; the
  swapped ordering `032 -> A, 001 -> B` measured 0.98x. Geometric mean of the two
  orderings (which cancels the bias): **~1.2-1.6x**, and the least-contended
  samples of each ordering give ~1.1x.
- Better measurement: **bench runs normalised against the harness's own
  `composite` pass**, which is byte-identical work in both experiments and
  therefore a contention yardstick. Interleaved runs (001, 032, 001, 032):
  billboards' draw pass = 1.07 and 1.16 composite-units, blade strips' = 1.09 and
  1.05 composite-units. That is **parity within ~10%**.
- Bucket LOD is what makes that possible. Ablations (self-A/B, bias-corrected):
  `finest=8` instead of 16 changes total cost by ~4% — the 16-ribbon near level is
  nearly free because the near ring's area is bounded. Shrinking the region from
  110 m to 40 m changes it by ~3% — the far field is not the cost either, once
  the last bucket is a single flat quad with one texture tap and a deep mip chain
  (before those two fixes the far field alone cost ~40% more; a truncated mip
  chain on a 512-tall tile is a texture-cache stall per fragment).
- Bench JSONs (default stand, orbit-low, seed 42, 1600x900, contended):
  `results/032-blade-strips__default__p-64afff07__apple-metal-3__2026-07-25T08-36-47-222Z.json`
  and `…T08-36-16-923Z.json`; the baseline runs from the same minutes are
  `results/001-billboard-smoke__default__p-86e4907a__apple-metal-3__2026-07-25T08-3*.json`.
- Plant-count independence: `stand=scaling-100m` (134.2M plants) shows the same
  pass time and the same 19.3 MiB/species as `default` (557k).
- `dense-mixed` (density 13/m², ~7.7M plants) roughly doubles the draw pass, as it
  must — that is density, not plant count, and the baseline scales the same way.

### What went wrong on the way (all fixed; kept because they are the lessons)

1. **Ground-anchored ribbons.** The first fit forced every ribbon to start at
   y=0, so a canopy-top bundle got a 1.15 m ribbon with content in the top fifth
   of its tile: 12 mm-wide slivers, four fifths empty, vertical smear everywhere.
   A ribbon must own only its bundle's height band.
2. **Clustering on vertices, not space.** The panicle is over-tessellated, so
   k-means on vertices put 9 of 16 bundles inside the plumes. Voxel-occupancy
   sampling fixed the decomposition.
3. **Coverage-collapsing mips** (see VRAM section) — the mid-field looked thin and
   olive because most fragments failed the alpha test at mip >= 2.
4. **Full-height-only bundles** (`Y_WEIGHT` 0.22) were tried to avoid band seams:
   every ribbon halved its arc resolution and the panicles smeared into diagonal
   streaks. Reverted; seams are handled by fitting the arc to the assigned
   triangles instead.
5. **Tile aspect** 64x1024 vs 128x512 — see VRAM section. Lateral resolution is
   what the eye reads as sharpness on a blade.

### Known weaknesses

- The unroll flattens each bundle by 2-12cm, so a bundle's internal depth is
  lost; at very close range (< 1 m) a wide ribbon can read as a slightly soft
  sheet rather than separate blades.
- Ribbons whose lateral axis ends up near the view direction compress to slivers;
  `faceCam` 0.55 twists them enough to keep coverage stable, but a fully
  world-anchored setting (`faceCam=0`) visibly thins the canopy.
- LOD levels are discrete (16/8/4/2/1). Coverage is roughly preserved across a
  switch because a merged bundle keeps its children's silhouette, but a slow
  dolly does show a mild change at the ring boundaries. Morphing a ribbon toward
  its parent (as 014-ribbon-skeleton does) would remove it and is the obvious
  next step.
- The owner's 2/5 rating was given against an intermediate state (before fixes
  3-5); the current build is materially different from what was rated.
