# splat cloud + screen-space reconstruction

## Idea

A plant is stored as a tiny cloud of **anisotropic covariance splats** and the
screen turns that sparse cloud back into continuous foliage.

Bake (once per species): vertices of the raw GCMESH1 are sorted along a
30-bit Morton curve; each LOD level chops the sorted order into K equal-count
chunks (K = 2048 / 384 / 96 / 24 / 6). A chunk becomes one 16-byte splat:
mean position + color, a normal, and the covariance eigen-axes give an
elongation direction + two sqrt-encoded semi-radii — blades bake into slim
oriented splats, fluffy panicles into round ones, and coarser LODs get bigger
radii automatically because their chunks are bigger. Whole species = 2558
splats = 41KB.

The normal is the load-bearing part, and it depends on the plant's SHAPE.
Foliage here is two-sided, so a chunk's mean vertex normal cancels to ~0 at
every LOD of every species (measured, not assumed), and the fallback is what
actually lights the plant: an **upright tuft** takes a radial shell normal
(blades radiate out and up from the base), a low wide **mat** takes the
covariance's minor axis — the local sheet normal — flipped into the upper
hemisphere. See "Moss carpet" below for why a mat must not have a radial
normal at all.

Per frame, three window/screen-bounded passes:

1. **cull** (compute): ONE DISPATCH PER STAND ENTRY over a fixed
   camera-centered cell window (fade_end 80m -> 43x43 cells), one workgroup per
   (cell, 128-slot group) and one thread per candidate slot. The
   workgroup first rejects the whole cell (its 4x4m column, padded by terrain
   height range + tallest plant) against the frustum and the fade distance, so
   the majority of the square window — which circumscribes the fade disc while
   only the frustum wedge inside it can contribute — costs nothing. Surviving
   cells evaluate the shared `scatter_candidate` — the WGSL twin of the stand
   scatter, so placement is bit-identical to every other renderer. Frustum +
   distance culled plants are appended as 16B records into per-(entry, bucket)
   buckets. Work is bounded by the window, never by stand size. The dispatch's
   z dimension is that entry's own slot count, so a carpet enumerates all
   `carpet_div²` = 484 of its grid nodes while a scattered entry still costs one
   128-thread group per cell.
2. **splats** (render, HALF-RES): one indexed indirect draw per (entry,
   bucket), issued bucket-major so the nearest ring of every species lays down depth
   first; vertices enumerate that LOD's splats (4 corners each, 6 shared
   indices). Each splat is a quad spanned by its baked
   elongation axis (projected off the view dir, floor at end-on) and the
   view-perpendicular; elliptical footprint via dithered discard so depth
   stays exact (no OIT). Wind = shared `wind_sway` scaled by baked height
   fraction, per plant phase. Outputs albedo+heightAO and normal+viewdepth,
   depth-tested against max-downsampled scene depth. Near splats clamp to
   ~48px projected radius so camera-adjacent stems stay made of many small
   splats instead of one blur-bomb; plant fades (camera-inside, far edge)
   dissolve whole splats, never pixels.
3. **reconstruct** (render, full-res): gathers a 3x3 half-res neighborhood,
   takes min view-depth as the local surface, weights taps by spatial
   gaussian x depth affinity (`surfaceTol`) for color/normal/height, and
   takes **alpha from coverage at any depth** (a pixel wedged between a near
   leaf and the field behind must not open a sky seam). Weighted coverage ->
   smoothstepped alpha closes the dither gaps; lighting + fog computed once
   per output pixel from the reconstructed normal/depth. So the expensive
   half-res pass shades nothing, and the full-res pass shades once — that is
   the point of the reconstruction split.

O(1) in plant count: no global plant array exists anywhere; the cull window is
fixed, splat pass is capped by bucket capacities, reconstruction is pure
screen space. Verified: `stand=default` (~557k) and `stand=scaling-100m`
(~134.2M plants) both run ~10ms GPU with identical VRAM (HUD, M-series
laptop, contended machine — see Findings).

## VRAM budget math

Per species (HUD-tagged, 2.9MB each, budget 25MB):

- splat LODs: 2558 splats x 16B = 41KB
- plant record buckets: 185,344 records x 16B = 2.97MB. Six buckets, in one of
  two layouts summing to the same total, so which layout an entry uses is a pure
  re-allocation and costs nothing:
  - scattered: 1024 / 4096 / 16384 / 32768 / 131072 / 0 — the far ring is where
    nearly all of a scattered entry's plants are, and it never reaches bucket 5.
  - carpet: 256 / 1024 / 4096 / 16384 / 32768 / 131072 — a life-size mat has
    ~30 tiles/m², so its far bucket needs ~50k records per entry (measured
    ~46k at grazing on `bog`) while a metre-radius disc around the camera can
    only hold a few hundred tiles.

Shared (species-independent, screen-bounded): half-res albedo rgba8 +
normal/depth rgba16f + depth32float = 16B/half-res px ~= 3.7MB at 1280x720,
14.7MB at 4K; plus a 24KB static quad index buffer (2048 splats x 6 u16, the
largest LOD; smaller LODs use a prefix) and <10KB of counts/indirect/uniform
buffers. Species total stays ~12% of budget.

## Bake

`bakeSpeciesSplats()` (bake.ts) runs in-browser from the raw GCMESH1 (vertex
cloud only, subsampled to <=2M verts), ~1-2s per grass species and ~2min per
19.8M-tri Sphagnum mesh, committed via `commitBake` to
`mesh/baked/002-splat-reconstruct/splats-v2-<mesh>.bin` (GSP1 format, 41KB
each, all six committed; the v1 files are deleted). Community tiles
(calamagrostis, elymus, all three Sphagnum) center on the tile, poa on its
bounds center.

**v2** (bakeVersion 2) changes only what a MAT-shaped mesh bakes to —
`ey < 0.8 * max(ex, ez)` off the mesh's own bounds, which selects the three
Sphagnum meshes and nothing else. The three grass artifacts are byte-identical
to v1 (verified with `cmp` on the record payload), so `default` cannot have
moved through the bake. What changes for a mat: the fallback normal (covariance
minor axis instead of radial shell), an isotropic Morton lattice, and a tighter
radius constant. All three are explained under "Moss carpet".
Note: `bakedArtifact()`'s committed-file fetch gets the SPA index.html (200)
when the file is missing — artifacts are magic-validated and re-baked +
OPFS-repaired if poisoned (see `isValidSplatBake`).

## Status

working — verified by headless screenshots at grazing / topdown /
inside-plant / far-horizon on `default`, plus `scaling-100m` and
`close-quality`. All six species covered by the same pipeline. Post-audit
re-verified at grazing/topdown/inside-plant/far-horizon + `scaling-100m` +
`reconstruct=false` and in all five debug views; console clean.

The `bog` stand (moss carpet) works as of the carpet pass below. Before it, the
whole frame was rejected on `bog`: the pipeline was hardwired for three stand
entries, and the bog's five overran the indirect buffer, so every command buffer
came back invalid and the screen was black. Re-verified on `bog` at grazing /
carpet-close / topdown / inside-plant / far-horizon, two sloped views across the
ridges, and albedo/normals/lighting/coverage; no toasts, no console errors.

## Findings

- HUD GPU totals at 1280x720 (M-series, deterministic cam, machine shared
  with many parallel agents so treat as indicative, NOT bench numbers; no
  results/ JSON claimed yet): grazing ~9-10ms (cull 0.15-0.3, splats 4.5-7,
  reconstruct 2.6-4.6, composite 1.4-3.9), topdown ~8ms, inside-plant ~10ms,
  scaling-100m grazing ~10ms. Plant count does not move the numbers; fill
  rate (splat overdraw at half res) dominates.
- Moss round: **do not trust any absolute number below.** Identical code, the
  same frozen camera and the same machine produced `default` grazing Σp50 of
  6.1, 8.9, 10.3 and 11.4 across four runs (splats 2.5-4.3ms for the same
  draws), against a 9.2 baseline measured in the same session before any change.
  Sibling agents were rendering throughout, and the swing shows up inside the
  fill-bound passes while the harness's own `base` pass stays at 0.4ms, so it is
  bandwidth contention, not a code difference. **A solo
  `#/bench/002-splat-reconstruct?stand=default&spline=orbit-low` is still owed
  before anyone quotes a number.** Structurally, `default` gained: two extra
  cull dispatches (the same total threads, plus 2x1849 workgroup-level cell
  rejects), three zero-instance indirect draws, and a never-taken carpet branch
  in the splat VS/FS. Nothing per-splat.
- Within a single run the ratios are meaningful, and the far bucket is the whole
  story on `bog` grazing: Σp50 13-31 with splats at 6.7-12.8ms when every far
  tile drew six sub-pixel splats, versus Σp50 4.4 with splats at 1.4ms after.
  `bog` now costs less than `default` at every camera measured, because a mat is
  one closed surface with almost no overdraw while a grass field is many
  overlapping ones (bog grazing 6.5 vs default 10.3 in the same run).
- **`default` is unchanged**, checked by pixel diff and not by reasoning:
  masked (HUD/panel/toolbar excluded) before-vs-after at frozen time `det=1&t=3`
  differs in 309 of 625,399 scene pixels at `cam=grazing` and 1,383 at
  `topdown`, versus a same-code control run that differs in 0 and 1,480
  respectively. The grazing differences are ~25 isolated specks of a few pixels
  each, spread across the mid/far field — the record compaction order changed
  when the cull became one dispatch per entry, so splats at a depth tie win
  differently. VRAM per species is identical (2.9MB), and both bake artifacts for
  every grass species are byte-identical to v1.
- The reconstruction genuinely carries the method: with `reconstruct=false`
  the raw pass is visibly a dithered point cloud; the filter turns the same
  input into closed foliage (toggle in the params panel).
- Distance LOD via per-plant jittered rings (2048/384/96/24/6 splats at
  3/7/16/36/80m) shows no visible popping in stills; ring jitter decorrelates
  transitions spatially.
- Fixes that mattered: (1) fades must dissolve whole splats, not pixels —
  sparse dithered depth from a fading plant occludes the field behind it and
  punches sky-colored holes; (2) alpha from any-depth coverage kills blue
  seams between overlapping foliage layers; (3) ~48px screen-space radius
  clamp turns camera-adjacent blur-bombs back into structured stems.
- Honest limitations: painterly/posterized look — near field reads as leaf
  discs, not individual blades (2048 splats vs 2.2M tris); silhouettes are
  quantized to half-res (2px) blocks; one reconstructed depth layer per pixel
  (no true multi-layer transparency); colors are slightly flatter than
  000-ground-truth (cluster-mean albedo + luminance-variance re-injection,
  no tone matching). A/B vs 000 is structural only: 000 renders the periodic
  community tile field, not per-plant placement.
- Not done yet: bench JSONs on an uncontended machine; sub-pixel temporal
  jitter of the half-res grid (would recover some silhouette resolution).

### Harness notes (from the moss round)

Things the harness could make easier; none of them blocked the work.

- **A carpet entry's real scale is not reachable from TypeScript.** `carpetScale()`
  computes it (and `createStandBuffer` uses it for the GPU table), but it is not
  exported from `@harness`, so `ctx.stand.species[i].scaleMin/scaleMax` still
  reports the placeholder 1.7/2.5 from the stand definition while the GPU sees
  1.01. Anything CPU-side that sizes a bound from an entry's scale is silently
  wrong for a carpet. Exporting `carpetScale`, or resolving the scale in the
  `Stand` object itself, would remove a trap. (Here it only over-padded a
  conservative cull margin, so it was harmless.)
- **`standEntrySlots()` was exactly the right primitive** and the CLAUDE.md
  warning about it is worth keeping — enumerating 128 slots renders a quarter of
  the mat, and the result looks like a placement bug, not a loop bound.
- **`terrain_sample()` returning height and (nx, nz) from one fetch is the right
  shape of primitive** for per-vertex conforming; it is what made rung 3 cheap
  enough to use unconditionally. What is missing for the mat case is a *normal
  transform* helper: I needed the inverse-transpose of a vertical shear
  (`n' = (n.x + nx/ny*n.y, n.y, n.z + nz/ny*n.y)`), which is three lines but easy
  to get wrong and useful for every renderer that conforms geometry by
  displacement instead of by rotating a basis. `plant_basis_from_up()` only
  covers the rigid-tilt case.
- **`stand_table.slope_align` had no natural insertion point here** and is
  unused: a splat cloud conforms by displacing each splat, not by blending a
  basis toward vertical, and `carpet_div > 0` already says "this is a mat, follow
  the ground exactly". A partial conform (align 0.3, as the bog's calamagrostis
  has) does not have an obvious meaning for a displacement scheme — it would be a
  fractional shear, which is not the same thing as a partially-uprighted plant.
  The grass here is still conformed only through its base point, as before.
- A `footprint_m` for non-tiling species (poa has none, so the field is 0) would
  let a renderer use one code path for "how wide is this plant" instead of
  branching on carpet-ness.

## Moss carpet (bog stand)

Sphagnum palustre is a 0.18m periodic community tile, 0.07-0.09m tall, laid out
by the `bog` stand as a grid-snapped mat at LIFE SIZE: 22x22 tiles per 4m cell
(484 slots), constant scale 1.01, 90deg-only yaw, three states partitioning the
wetness field. ~1.13M tiles over ±96m.

What changed, smallest first — all of it inside this experiment:

1. **The whole pipeline was hardwired for three stand entries.** `records0..2`
   with `switch e { case 0, case 1, default }`, `counts: array<atomic<u32>, 15>`
   and a 300-byte indirect buffer. On the bog's five entries the draw loop
   indexed the indirect buffer past its end, so **every command buffer was
   invalid and the whole frame — terrain included — never rendered.** That is
   the entire "before" state on `bog`: a black screen behind three validation
   toasts. Now every buffer is sized from `ctx.stand.species.length` and the
   cull runs one dispatch per entry against that entry's own record buffer.
2. **Only 128 of the 484 carpet slots were ever evaluated**, because the cull
   was one thread per `lid.x` in a 128-thread workgroup. The dispatch's z
   dimension is now `ceil(standEntrySlots(entry) / 128)`, so a carpet visits all
   484 nodes and a scattered entry still visits exactly 128. Two numbers, kept
   apart: *slots to evaluate* is per entry, *records to store* is the expected
   survivors (see the two bucket layouts under VRAM).
3. **Per-splat terrain conforming, ladder rung 3-4.** Every splat is placed at
   `terrain_sample(splat.xz)` — one bilinear fetch that returns the height AND
   the (nx, nz) the shading needs — and the baked normal is sheared by the same
   gradient (`n' = J^-T n`, exact for a vertical shear). A vertical shear rather
   than a rigid per-tile tilt on purpose: neighbouring tiles that each fit their
   own plane crack apart along their shared edge, whereas any two splats at the
   same xz get the same ground height by construction, so the mat is exactly
   continuous. Over 9cm of moss, "vertical" vs "normal to the slope" is a
   sub-millimetre difference, so the shear costs nothing visually. Rung 1 or 2
   is not cheaper here, it is wrong.
4. **Size-relative LOD rings.** The ring constants (3/7/16/36m) are calibrated
   for a ~1.2m grass tuft, i.e. an effective size of 0.78m. A 0.18m tile reaches
   the same splats-per-pixel at a quarter of the distance, so for a carpet the
   rings scale by `footprint_m * scale / 0.78`. Without this every tile within
   3m got all 2048 LOD0 splats — thousands of overlapping splats inside a 40px
   tile. `footprint_m`, never `height_scale`: sized from height a moss tile
   would be 3.5x too small.
5. **A carpet-specific closure floor on the splat radius.** A coarse LOD's
   radii come from the mesh's point spread, not from "cover this tile", so the
   coarse levels left the mat holed. `r >= 0.87 * footprint * scale / sqrt(n)`
   is the area bound for n discs of effective radius 0.65r (where the dither
   falloff saturates) covering the tile square, plus a 2.5-pixel screen-space
   floor so a sub-pixel splat still rasterises a sample. Both bind only where
   the bake would otherwise open holes. `debug=coverage` on `bog` at grazing is
   solid white below the horizon — binary coverage, no screen door.
6. **No camera-inside fade for a carpet.** A mat you are standing on must not
   open a hole under your feet; only the far fade remains.
7. **Tighter cull sphere for a tile**: `0.75 * footprint * scale + 0.1` (the
   source mesh's own 0.24m-of-geometry-in-a-0.18m-period overflow) instead of
   the grass `max(height, 0.75*scale) + 0.35`, which accepted a 1.11m radius for
   a 0.18m tile — 6x too loose.
8. **A sixth, carpet-only distance bucket: one ground-parallel disc per tile,
   beyond ~25m.** At 25m a life-size tile is ~2px, where the six splats of the
   coarsest baked level are all clamped to the same one-pixel minimum: ~20x the
   fragments needed, and — since every tile paints the identical six-blob
   pattern — a corduroy moire across the whole mid field. The far bucket needs
   no baked data (one disc at the tile centre with the tile's mean albedo,
   computed at load from the level-4 splats) and it is **hard-edged**, not
   dithered: it is a closed ground surface, so it writes solid depth and needs
   no stochastic coverage. Ground-parallel, not view-facing, because a
   view-facing disc wide enough to cover a 0.18m tile would stand 0.2m proud of
   a 9cm mat and paint a floating fuzzy band at grazing. This one change took
   the bog splat pass from 6.7-12.8ms to 1.2-2.9ms and removed the moire.
9. **Bake v2 for mat-shaped meshes** (the biggest single visual change):
   - *The fallback normal.* The mean vertex normal cancels to ~0 for two-sided
     foliage, so every cluster of every species was being lit by the radial
     shell fallback — measured: `n.y = 0.51`, radial component `0.86`, identical
     to 2 decimals in every radial bin of every LOD. For a tiling mat that is
     the worst possible normal field: each tile lights as an outward-facing
     dome, so the mat reads as a **lattice of domes** and the tile grid is
     visible at every distance (this was the dominant artifact — see the a1/a2
     screenshots described below). Mats now take the covariance's MINOR axis
     (`e1 x e2`, which the bake was computing and throwing away) flipped into
     the upper hemisphere, plus a small up-bias for isotropic clusters. Radial
     trend after: 0.00-0.04.
   - *An isotropic Morton lattice.* The first attempt at the above produced
     `n.y = 0.97` everywhere — nearly flat. The Morton lattice is per-axis
     normalized, so for a 0.24 x 0.09 x 0.24 mesh every chunk is a 2:1:2 slab,
     and a slab's covariance minor axis is vertical *by construction*. Scaling
     the lattice to one world cell size per axis (mats only; the factors are
     exactly 1 for the grasses, which is why their bakes stay byte-identical)
     gives cube-ish chunks whose minor axis is the real local surface: `n.y`
     0.65-0.68 at LOD0, i.e. ~48deg of honest cushion-scale variation.
   - *A tighter radius constant for mats* (KR 1.5σ vs 1.9σ), so a splat stays
     smaller than a capitulum instead of soup; the closure floor from (5) is
     what lets the coarse levels get away with it.

### Rejected after measuring

- **A steeper height-AO curve for mats** (hf^2.3, on the theory that a dense
  cushion shadows its gaps harder than a grass tuft). Measured 10% darker with
  *lower* local contrast (std 11.9 -> 11.5, mean |horizontal gradient| 1.57 ->
  1.50 at `carpet-close`). The visible surface of a mat is its top, so hf is
  nearly constant there and the exponent only shifts the level. Reverted. The
  one part kept is that the far disc reports the height fraction of the cushion
  TOP (0.92) rather than the mean over all 9cm of moss — with the mean, the
  distant mat came out ~13% darker than the near field through the same AO term.
- **Overscale** was never tried: this representation's tiles already overlap by
  construction (the splat cloud spans the mesh's 0.24m of geometry inside a
  0.18m period), so the mechanism overscale exists to fix does not apply.

### What improved / what is still bad

Screenshots read at `cam=grazing`, `carpet-close`, `topdown`, `inside-plant`,
`far-horizon`, two poses across the ridged slopes at (26, -44), and
albedo/normals/lighting/coverage — before and after, same cameras.

- Improved: from *nothing at all* (invalid frames) to a closed, continuous,
  terrain-conforming mat at every distance from 1m to the fade edge; no tile
  lattice; no corduroy moire; no bare-peat holes (including along zone
  boundaries — the harness scatter fix holds); the three states read as coherent
  zones from topdown; per-fragment normals are real cushion-scale variation near
  the camera (`debug=normals`) decaying to the ground plane in the far field,
  which is honest for a 2px tile; no hole under the camera; and the bog stand
  costs less than `default` does.
- Still bad #1: **the near field reads as fine granular texture, not as
  individual capitula.** At 1m a tile spans ~126px and gets 384 splats of
  ~1cm — about the size of a capitulum, but the splats are Morton chunks, not
  rosettes, so their colour/normal variation is uncorrelated noise rather than
  structure. It reads as a convincing moss *surface* with real relief shading;
  it does not read as "cushions you could press with your hand".
- Still bad #2: **no cast shadowing between cushions.** The only occlusion cues
  are the depth test between splats and the baked height-fraction AO, and the
  visible top surface of a mat has an almost constant height fraction, so the
  AO does very little (measured above). A mat is where this method's lack of
  any inter-splat visibility term shows most.
- Still bad #3, and NOT the moss's fault: **dark elliptical blobs on the mat**,
  most visible at `cam=far-horizon` on `bog`. Traced by sampling `debug=normals`
  inside one: normal (0.98, 0.11, 0.15), i.e. horizontal. Mat normals are bounded
  at `n.y >= 0.22` by the up-flip, so it cannot be moss — it is a *grass* LOD0
  splat. Two of calamagrostis' 2048 finest clusters (and one of poa's) straddle a
  high-level boundary of the Morton curve, so they span two distant parts of the
  mesh: their covariance describes the gap, giving a 0.5-0.6m radius (clamped to
  48px on screen) and an incoherent near-horizontal normal that lights almost
  black. It has always been there; a grass canopy hid it, and a smooth mat does
  not. The fix is at bake time (detect a chunk whose covariance is much larger
  than its Morton cell and drop or split it — dropping 2 of 2048 is invisible),
  but it would change every grass bake and `default` with it, so it is out of
  scope for a moss pass. It is the cheapest next win on this renderer. Mat splats
  now get the same treatment cheaply, via a half-a-tile radius ceiling in the
  shader (the moss bake has a few 13cm outliers where the typical radius is 3cm).
- Still bad #4: the far mat is a field of flat mean-coloured discs. That is the
  right answer at 2px per tile, but the transition band (~20-30m, spatially
  dithered by the per-tile ring jitter) is where the mat loses the last of its
  volume, and on a steep slope seen from far away it is visibly a painted
  surface. No visible ring at the transition (row-mean brightness across the
  grazing frame falls smoothly, 80 -> 59 with no step).

### Is a splat cloud suited to a moss carpet?

Better than a card is, and better than I expected. A splat cloud is a volume
representation, so unlike a single flat quad it has real thickness: the near
field has genuine per-fragment normal variation from actual geometry, the mat
has a soft fuzzy top rather than a hard plane, and nothing about the technique
fights the shape — a low wide cloud of splats is just as natural as a tall
narrow one. Three things had to change to get there, and all three were the
same mistake in different places: the bake, the LOD rings and the record
budget were all calibrated for an upright metre-tall plant, and a 0.18m mat is
a quarter of the size with 30x the instance count.

Where it stops is resolution, not shape. 2558 splats per tile is a lot for a
grass tuft and not much for a 19.8M-triangle cushion: the finest level is
~1cm splats, which is exactly one capitulum, so the cushion's internal
structure — the individual rosettes and the shadowed gaps between them — is
below the representation's resolution and comes out as noise of the right
statistics rather than as form. And because there is no inter-splat visibility
term, the relief that IS there is under-lit. So: an honest, cheap, closed,
terrain-following moss volume with correct colour zoning and real normals, that
reads as moss ground from 1m and up, and never reads as individual cushions.

## Debug views

The reconstruction pass is the single place this method produces colour, so it
answers the global `view` selector (`frame.debug_mode`, URL `debug=`) there,
from the reconstructed G-buffer: albedo = weighted cluster albedo before
lighting, normals = the reconstructed splat normal actually fed to
`light_surface()`, lighting = that light term with the height-fraction AO
folded in (AO is occlusion, not albedo), depth = the reconstructed surface
distance. Fog is applied only in `off`.

`coverage` is drawn **opaque** and full-screen, unlike the other modes: this
pass is a premultiplied-alpha overlay, and blending a grey coverage value over
the terrain's own coverage (a constant 1.0 = white) would wash the map out to
uniform white and tell you nothing. As a full-screen map it reads directly —
black = no groundcover resolved, white = fully covered — and the fade_end ring
and the reconstruction's soft silhouettes are both visible in one image.

What the views exposed: coverage is essentially binary (alpha saturates to 1
wherever the field is dense at default `gapFill`, softening only on
silhouettes), which is the wanted behaviour — hard edges, not a screen-door.
Normals are genuinely per-fragment and varied (blade/shell mix rotated per
plant); the top-down normal view is correctly dominated by +y. Nothing needed
repairing in the shading itself: albedo is raw baked cluster colour (never
premultiplied by light at bake time) and lighting goes through the shared
`light_surface()` exactly once, in the full-res pass.

The old per-experiment `debug` enum param (off/coverage/normals/depth) is gone
— it was a strictly weaker duplicate of the global selector.

## Audit

Found and fixed:

- **The cull pass evaluated the entire cell window every frame.** 43x43 cells
  x 128 slots x 3 entries = ~710k `scatter_candidate` evaluations (each a
  handful of hashes plus a bilinear terrain sample) per frame, with the
  frustum/distance test applied only *after* the candidate was fully built.
  The window is a square circumscribing the fade_end disc, so at any normal
  camera the great majority of it is behind the camera or off to the side.
  Added a whole-cell reject at the top of `cs_cull`: one conservative AABB
  (the cell column, y-bounded by `terrain_height_scale * 1.15` since
  |fbm| <= 1.015, padded by `cell_pad` = 2x tallest plant + 0.8m) tested
  against the 5 frustum planes and the fade distance, in a uniform branch, so
  a rejected cell costs one box test for the whole workgroup instead of 384
  candidate evaluations. The pad is stand-derived and computed once at init,
  not per frame. Verified live (not dead code) by temporarily shrinking the
  reject distance to `fade_end * 0.25` — the field visibly stopped at ~20m —
  then reverting.
- **The splat pass shaded 6 vertices per quad.** A quad has 4 corners; the
  pass draws ~1M splats a frame, so that was ~2M wasted vertex-shader
  invocations of a not-cheap VS (splat decode, oct decodes, wind sway, two
  view transforms). Switched to a static 24KB u16 index buffer (pattern
  0,1,2,1,3,2 per splat) and `drawIndexedIndirect`; `cs_finalize` now writes
  5-u32 indexed args. Same triangles, same winding, same corner attributes.
- **Draw order was entry-major.** LOD n covers a strictly nearer ring than LOD
  n+1, so entry0's 80m ring was drawn before entry1's 3m ring — far fragments
  shaded then overwritten. The loop is now LOD-major (near-to-far across all
  species), which is what lets early-z reject the field behind the near
  plants. Free reorder; depth ties between distinct splats are measure-zero.
- **Empty pixels in the reconstruction wrote transparent black.** With
  premultiplied blending that is an exact no-op that still costs a full-screen
  read-modify-write; they `discard` now (except in the coverage view, which
  wants them as its zero level). At a grazing camera that is roughly a third
  of the frame; at far-horizon much more.
- **A dead interpolant.** `misc.x` was plumbed vertex->fragment as the
  constant 1.0 and multiplied into the coverage term. Removed (multiplying by
  exactly 1.0 is bit-exact, so the image is untouched).

Deliberately left alone:

- **The 3x3 reconstruction gather is 4x redundant.** All four full-res pixels
  of a 2x2 block load the same 9 half-res texels and differ only in their
  sub-texel weights. Fixing that means reconstructing at half res and
  upsampling, which changes the silhouette quality — a redesign, not an audit
  fix.
- **The dithered discard in the splat pass defeats early-z.** That dither *is*
  the coverage mechanism this method resolves in pass 3 (see the taste rule in
  CLAUDE.md); removing it is the technique, not waste. The LOD-major reorder
  and the depth-init seed are the honest structural mitigations available.
- **The params UBO is rewritten every frame** (64B). `base_cell` genuinely
  tracks the camera; skipping the write on unchanged frames is not worth a
  dirty-flag.
- **`ray_dir` does an inv_view_proj mat-vec per pixel** and could be an
  interpolated varying off the fullscreen triangle. ALU-level, out of scope.
- Untested suggestion for later (needs a bench, so not done here): the splat
  pass could get a cheap depth prepass for the LOD0/LOD1 rings only, since
  those are the ones with real overdraw depth.
