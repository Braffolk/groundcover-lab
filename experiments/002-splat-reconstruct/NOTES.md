# splat cloud + screen-space reconstruction

## Idea

A plant is stored as a tiny cloud of **anisotropic covariance splats** and the
screen turns that sparse cloud back into continuous foliage.

Bake (once per species): vertices of the raw GCMESH1 are sorted along a
30-bit Morton curve; each LOD level chops the sorted order into K equal-count
chunks (K = 2048 / 384 / 96 / 24 / 6). A chunk becomes one 16-byte splat:
mean position + color, mean normal blended toward a radial "shell" normal
when blade normals cancel, and the covariance eigen-axes give an elongation
direction + two sqrt-encoded semi-radii — blades bake into slim oriented
splats, fluffy panicles into round ones, and coarser LODs get bigger radii
automatically because their chunks are bigger. Whole species = 2558 splats =
41KB.

Per frame, three window/screen-bounded passes:

1. **cull** (compute): one thread per (cell, slot) in a fixed camera-centered
   cell window (fade_end 80m -> 41x41 cells), evaluating the shared
   `scatter_candidate` — the WGSL twin of the stand scatter, so placement is
   bit-identical to every other renderer. Frustum + distance culled plants are
   appended as 16B records into per-(entry, LOD) buckets. Work is bounded by
   the window, never by stand size.
2. **splats** (render, HALF-RES): one indirect draw per (entry, LOD); vertices
   enumerate that LOD's splats. Each splat is a quad spanned by its baked
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
- plant record buckets: (1024+4096+16384+32768+131072) x 16B = 2.97MB

Shared (species-independent, screen-bounded): half-res albedo rgba8 +
normal/depth rgba16f + depth32float = 16B/half-res px ~= 3.7MB at 1280x720,
14.7MB at 4K; plus <10KB of counts/indirect/uniform buffers. Species total
stays ~12% of budget.

## Bake

`bakeSpeciesSplats()` (bake.ts) runs in-browser from the raw GCMESH1 (vertex
cloud only, subsampled to <=2M verts), ~1-2s per species, committed via
`commitBake` to `mesh/baked/002-splat-reconstruct/splats-v1-<mesh>.bin`
(GSP1 format, 41KB each, all three committed). Community tiles
(calamagrostis, elymus) center on the tile, poa on its bounds center.
Note: `bakedArtifact()`'s committed-file fetch gets the SPA index.html (200)
when the file is missing — artifacts are magic-validated and re-baked +
OPFS-repaired if poisoned (see `isValidSplatBake`).

## Status

working — verified by headless screenshots at grazing / topdown /
inside-plant / far-horizon on `default`, plus `scaling-100m` and
`close-quality`. All three species covered by the same pipeline.

## Findings

- HUD GPU totals at 1280x720 (M-series, deterministic cam, machine shared
  with 15 parallel agents so treat as indicative, NOT bench numbers; no
  results/ JSON claimed yet): grazing ~9-10ms (cull 0.15-0.3, splats 4.5-7,
  reconstruct 2.6-4.6, composite 1.4-3.9), topdown ~8ms, inside-plant ~10ms,
  scaling-100m grazing ~10ms. Plant count does not move the numbers; fill
  rate (splat overdraw at half res) dominates.
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
  jitter of the half-res grid (would recover some silhouette resolution);
  per-species tuning of KR/radius floors in the bake.
