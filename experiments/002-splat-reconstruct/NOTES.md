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

1. **cull** (compute): one workgroup per cell of a fixed camera-centered cell
   window (fade_end 80m -> 43x43 cells), one thread per candidate slot. The
   workgroup first rejects the whole cell (its 4x4m column, padded by terrain
   height range + tallest plant) against the frustum and the fade distance, so
   the majority of the square window — which circumscribes the fade disc while
   only the frustum wedge inside it can contribute — costs nothing. Surviving
   cells evaluate the shared `scatter_candidate` — the WGSL twin of the stand
   scatter, so placement is bit-identical to every other renderer. Frustum +
   distance culled plants are appended as 16B records into per-(entry, LOD)
   buckets. Work is bounded by the window, never by stand size.
2. **splats** (render, HALF-RES): one indexed indirect draw per (entry, LOD),
   issued LOD-major so the nearest ring of every species lays down depth
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
- plant record buckets: (1024+4096+16384+32768+131072) x 16B = 2.97MB

Shared (species-independent, screen-bounded): half-res albedo rgba8 +
normal/depth rgba16f + depth32float = 16B/half-res px ~= 3.7MB at 1280x720,
14.7MB at 4K; plus a 24KB static quad index buffer (2048 splats x 6 u16, the
largest LOD; smaller LODs use a prefix) and <10KB of counts/indirect/uniform
buffers. Species total stays ~12% of budget.

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
`close-quality`. All three species covered by the same pipeline. Post-audit
re-verified at grazing/topdown/inside-plant/far-horizon + `scaling-100m` +
`reconstruct=false` and in all five debug views; console clean.

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
