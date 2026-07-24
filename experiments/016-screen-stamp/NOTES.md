# 016 screen-stamp — tile-frustum binning + per-pixel stamp resolve

## Idea

Invert the loop: iterate over SCREEN TILES, not plants. Per frame:

1. **near pass** (render): the constant 6x6 ring of scatter cells around the
   camera (< ~8 m) drawn as classic alpha-tested hemi-octa impostor cards
   with depth write. Rationale: a plant 1-3 m away angularly overlaps almost
   every tile, so per-tile lists would drown in near plants (verified — the
   first build had every list full of 1-3 m plants and a sparse mid-field).
   Region-bounded, plant-count independent, and it primes the depth buffer.
2. **bin pass** (compute, one 128-thread workgroup per 16x8 screen tile):
   - reduces the tile's scene-depth texels into a world footprint (visible
     ground cell bbox, max ground distance, sky presence) — the terrain base
     pass already made the depth buffer THE tile-to-world map;
   - picks a per-tile strategy: **enumerate** the few footprint cells
     (close-up / top-down), **column-march** cells front-to-back along the
     center ray's ground track (grazing wedge / above-horizon), or
     **tint-only** (footprint too large for individual plants);
   - evaluates the procedural scatter twin over those cells only (no plant
     arrays at any count), sphere-tests candidates against the tile's four
     frustum planes, applies wind sway + fades, snaps each plant to its
     nearest baked hemi-octa view, packs 32 B per plant;
   - hard work bound per tile: <=32 surviving cells x entries x 128 slot
     evals, list capped at K=64; thread 0 sorts front-to-back.
3. **stamp pass** (fullscreen fragment): each pixel walks its tile's sorted
   list, ray-intersects each impostor card, samples baked albedo/normal
   (mipmapped, premultiplied), and composites over-operator front-to-back
   with two early exits (transmittance saturated; entry behind the visible
   surface — valid because the list is depth-sorted). Beyond the per-tile
   stamp horizon a statistical meadow tint (species average albedos mixed by
   world-anchored value noise, lit by the terrain normal, fogged) takes over
   with a crossfade.

Cost is tiles x bounded constants + screen pixels x bounded walk — per-frame
work is independent of total plant count AND stand area (verified visually on
`scaling-100m`, ~134M plants, same frame cost as the default stand). No
source geometry per frame, no per-pixel marching: each pixel does at most K
precomputed lookups; the per-tile column walk is a bounded tiled-culling
walk (<=32 cells), not a raymarch.

Key trick for top-down: a per-plant **view-angle card clip** (cards shrink to
0.45R when seen from above, mirrored bit-exactly in bin+resolve+near). From
overhead, full-radius community-tile cards overlap ~40x — no bounded list
could hold them, and arbitrary overflow choices showed as tile seams. The
clip bounds overlap below K and the seams disappear; clipped fringes are
covered by neighbouring plants and the under-tint.

## VRAM budget math

Per species (HUD-confirmed 16.6 / 25 MB each):
- albedo atlas 1280x1280 rgba8, 4 mips: 6.25 MiB x 1.328 = 8.3 MiB
- normal atlas same: 8.3 MiB
- => 16.6 MiB / species for all three species.

Screen-scoped (not per species, scales with resolution, NOT with plants):
- tile lists: ceil(W/16) x ceil(H/8) x (8 + 64*8) u32 = 2080 B/tile;
  16.6 MB at 1280x800, ~66 MB at 4K-retina — flagged as a known cost, K or
  the packing could shrink it 2x if needed.
- 160 B uniform.

## Bake

`bake.ts` renders each species' GCMESH1 mesh into a 10x10 hemi-octahedral
grid of 128px orthographic captures (MRT albedo+coverage / local-frame
normal), reads back, then builds a CPU box-filtered mip chain (albedo
premultiplied — empty texels are black, so plain averaging is the correct
filter; this fixed dark silhouette fringes) and the coverage-weighted average
albedo used by the far tint. ~2-4 s for all three species, baked fresh per
session. The harness bakedArtifact/commitBake flow is intentionally bypassed
for the same reason 005 documents (dev server answers missing bake files with
200 index.html, poisoning the cache). Capture pipeline adapted from
005-octa-impostors; everything after the readback is new.

## Debug views

All three passes (`near`, `stamp`) answer the global `view` selector
(`frame.debug_mode`, URL `debug=`):

- `near.wgsl` is an alpha-tested opaque card: `debug_shade(..., coverage 1.0)`.
- `resolve.wgsl` composites, so it accumulates **albedo and world normal with
  the same front-to-back weights as the colour** (`trans * a`, including the
  aggregate tint field) and un-premultiplies at the end. albedo / normals /
  lighting therefore show the coverage-weighted average surface the pass
  actually resolved, and `depth` uses the nearest contributing stamp's hit
  point. Fog is applied only when `debug_mode() == DEBUG_OFF`.
- `coverage` is the accumulated stamp alpha, written **opaque** (the terrain
  writes coverage 1.0 everywhere, so a blended coverage map washes out to
  uniform white). Pixels the near-card pass already filled are skipped instead
  of overwritten, so the near field reads as covered rather than empty.
- Normals are real per-fragment normals: the baked normal atlas is decoded,
  coverage-weighted through the mip chain, and rotated into world space by the
  plant yaw (the bake stores mesh-local, y-up normals). Lighting is the shared
  `light_surface()`; the atlas stores raw vertex colour (never premultiplied by
  light), so there is no double-lighting. The `lighting` view saturates to
  white in bright areas because the harness model peaks at 1.15 + 0.32 — same
  for terrain, not a bug in this method.

Method-specific state has its own param, `tileView` (`off` / `fill` = list
occupancy heatmap / `mode` = which binning strategy each tile picked: blue
tint-only, green enumerate, orange column-march). `mode` is a good way to see
that at grazing MOST tiles are in enumerate mode and only the horizon band
column-marches.

## Status

working — verified by headless screenshots at grazing, topdown,
inside-plant (fades via coverage erosion, no dither), far-horizon, and
far-horizon on `scaling-100m` (134M plants). Wind verified by diffing
det=1 frames at t=2 vs t=6 (foreground moves; per-species sway from the
shared model, computed once per plant per tile in the bin pass, sheared
per-pixel in resolve). Covers all three species.

Honesty / known issues:
- All frame timings this session were contended (many agents on one GPU):
  observed bin 3.5-17 ms, stamp 4.7-21 ms p50 at 1280x800 across runs of the
  SAME code. No bench recorded — needs a quiet GPU; the structure is bounded
  but the constants deserve tuning (candidates: half-res resolve, smaller K,
  tighter cell budget).
- Grazing: lists overflow at ~35-45 m and the scene crossfades into the
  aggregate tint field earlier than ground truth would; the tint carries
  noise octaves but is visibly flatter than stamped plants.
- Single nearest hemi-octa view per plant (no 4-view blend): possible view
  popping in motion (not verified in a moving capture).
- Close-up (< 1 m) impostors go blobby — 128px views magnified; inherent.
- calamagrostis/elymus are community tiles baked as one "plant", so density
  reads busier than ground truth (same caveat as 005).
- Stamps write no depth (nothing draws after them in the frame).
- Enumerate-mode footprints come from visible ground only: plants whose base
  is hidden behind a close ridge but whose tops should peek over can be
  missed at close range (column mode handles the common grazing case).

## Findings

- The depth buffer as tile->world footprint map works remarkably well: sky
  tiles cost almost nothing, ridges bound the walk automatically, and the
  near-quad depth primes the sorted-occlusion early-out so pixels behind
  near plants walk ~1 entry.
- Slot exhaustion is THE failure mode of screen-tile plant lists — three
  separate mechanisms were needed (near-field ownership pass, view-angle
  card clip, per-tile overflow horizon + crossfade) before all four standard
  cameras held up.
- Impostor orientation gotchas found the hard way: atlas v-flip (plants
  rendered upside down — dense root line floating at the skyline) and uv
  clamp-vs-reject at atlas tile borders (clamping smears cropped
  community-tile edges into hollow card outlines).
- No bench JSONs yet (contended GPU). A/B vs ground truth:
  `#/ab/016-screen-stamp/000-ground-truth?stand=default&cam=grazing&seed=42`.

## Audit (structural waste review)

Found and fixed:

1. **Binning kept grinding cells after the tile list was full** (the big one).
   `try_plant()` does `atomicAdd` then drops the plant when `idx >= MAX_LIST`,
   but nothing stopped the *enumerate* path from walking its whole padded
   footprint (up to 7x7 cells x 3 entries x 128 slots) long after the 64-entry
   list had filled. With 4 m cells and density 3/m² a single cell already holds
   ~144 candidates, so a full list after cell 1-2 was the norm — and `tileView=mode`
   shows enumerate is the *dominant* mode at grazing, not just top-down.
   Column mode had a per-column budget break but nothing inside a column.
   Fix: one `atomicLoad(&wg_count) >= MAX_LIST` early-out at the top of
   `process_cell()`. Provably image-identical: `wg_count` only grows, so a
   thread can only skip once no append can succeed for anyone.
2. **Tile frustum computed 128x per tile.** All 128 threads redundantly did the
   same four `inv_view_proj` unprojections + four plane normals. Folded into the
   existing thread-0 block that fills `wg_red` and broadcast through the
   `workgroupUniformLoad` that was already there — no new barrier, same bits.
3. **160-byte uniform re-uploaded every frame** although it holds seed, entry
   count, per-species impostor meta and three params — nothing per-frame. Now
   written only when a param actually changes.
4. Dead `R_SLACK` constant removed; stale `16x16` / `256 texels` comments fixed
   (tiles are 16x8 / 128 texels).

Deliberately left alone:

- **Thread 0 insertion-sorts the tile list (O(n²), up to 64 entries, 127 lanes
  idle).** A bitonic sort over 128 threads is the textbook answer, but it is a
  rewrite of that step and needs a quiet GPU to justify — and the column walk
  already appends roughly front-to-back, so insertion sort usually runs near
  O(n). Suggestion for a later pass, not an obvious win.
- **Near pass draws 6x6x128x3 = 13824 instances**, ~63% of which collapse to
  degenerate triangles because the scatter slot is empty. Compacting would need
  a compute prepass; the count is constant and region-bounded, which is the
  property that matters.
- **`apply_fog` per stamp inside the composite loop** (up to 64x per pixel).
  Folding it to one fog application on the composite would change the image.
- Per-pixel `pixel_scale` / `tiles_x` recomputation, the 4-plane cull loops and
  the 4-tile bilinear header fetch: ALU-level, out of scope.
- Tile-list VRAM (2080 B/tile, ~16.6 MB at 1280x800) is still the honest weak
  spot; K or the 32 B packing could halve it, but that is a quality/perf
  trade-off needing measurement.

Verification: `debug=off` screenshots at grazing / topdown / inside-plant /
far-horizon with `det=1&t=3` before and after the structural fixes differ by
RMSE 0.0005-0.018, which is exactly the *same-code* run-to-run noise floor
(0.0004-0.018) — the tile lists are inherently race-ordered when they overflow.
No image change. The debug-view work did not touch the `debug=off` path.
