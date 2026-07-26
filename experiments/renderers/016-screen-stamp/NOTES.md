# 016 screen-stamp — tile-frustum binning + per-pixel stamp resolve

## Idea

Invert the loop: iterate over SCREEN TILES, not plants. Two shapes of species
take two paths — scattered plants go through the tile lists, a CARPET
(`stand_table[i].carpet_div > 0`, the bog's Sphagnum) is stamped per pixel
straight from the depth buffer. Per frame:

0. **carpet pass** (fullscreen fragment, only when the stand has carpet
   entries): the mat layer. Per pixel: unproject the terrain depth texel ->
   carpet grid node -> the wetness field picks which of the three moss states
   owns that node -> its 90-degree turn -> sample that species' own high-res
   top-view bake, with one parallax step and a height-gradient normal. No list,
   no binning, O(1), and terrain-conforming by construction. See "Moss carpet"
   below for why a mat gets its own path.
1. **near pass** (render): the constant 6x6 ring of scatter cells around the
   camera (< ~8 m), scattered species only, drawn as alpha-tested hemi-octa
   impostor cards
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

Scattered species (HUD-confirmed 16.6 / 25 MB each):
- albedo atlas 1280x1280 rgba8, 4 mips: 6.25 MiB x 1.328 = 8.3 MiB
- normal atlas same: 8.3 MiB
- => 16.6 MiB / species (calamagrostis, elymus, poa).

Carpet species — a DIFFERENT budget split, because a mat shows a different
part of itself (HUD-confirmed 10.7 / 25 MB each):
- albedo+coverage 1024x1024 rgba8, full 10-level chain: 4 MiB x 1.333 = 5.33 MiB
- normal.xz + height + coverage, same: 5.33 MiB
- => 10.7 MiB / species for all three Sphagnum states, i.e. LESS than the
  upright path while multiplying the texel density that lands on screen by 8x
  linearly (1024px across a 0.18m tile = 0.18mm/texel, against 128px across a
  0.35m sphere-fitted card = 2.7mm). 100 hemi-octa views of a cushion spend
  ~96 of them on angles a mat never shows.

Screen-scoped (not per species, scales with resolution, NOT with plants):
- tile lists: ceil(W/16) x ceil(H/8) x (8 + 64*8) u32 = 2080 B/tile;
  16.6 MB at 1280x800, ~66 MB at 4K-retina — flagged as a known cost, K or
  the packing could shrink it 2x if needed.
- 272 B uniform.

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

Carpet species take `bakeCarpet()` instead (`shaders/bake-carpet.wgsl`): ONE
straight-down orthographic capture at 1024px, cropped to the tile's own square
from the GCMESH1 header (`tileOrigin`/`tileSize`, 0.18m). The draw is INSTANCED
9x at (-1,0,+1)^2 periods, because the mesh overflows its period (0.24m of
geometry inside a 0.18m tile): the neighbours' hangover fills the parts of the
window this tile's own geometry left empty, and without it the mat shows a grid
of gaps along every tile edge. Targets are albedo+coverage and
(normal.xz, height, coverage); mips are coverage-weighted on the CPU
(`rgb = sum(rgb*a)/sum(a)`), so the stored colour is ALREADY normalised and the
runtime must not divide by alpha — verified by the per-level luma print, which
is flat (0.451/0.451/.../0.454). Measured on the wet-vigorous bake: coverage
0.985 at level 0 and 0.987 at the deepest level (the mat cannot thin out with
distance — the "distant tiles fail the alpha test" problem 001 had does not
arise), mean canopy height 7.32cm, height sd 0.73cm. One view instead of 100
also makes the 19.8M-triangle moss bake ~100x less rasterisation; all three
moss species bake inside a ~10s page load.

## Debug views

All passes (`carpet`, `near`, `stamp`) answer the global `view` selector
(`frame.debug_mode`, URL `debug=`):

- `carpet.wgsl` composites one opaque-ish layer: `debug_shade(shaded, albedo,
  n_ws, alpha, world)` with the real height-derived normal, and it writes its
  coverage OPAQUE in `coverage` mode (like the near pass) because the terrain
  already wrote 1.0 everywhere. The stamp pass then leaves carpet ground pixels
  alone instead of overwriting them with its own ~0 (`dims.y > 0 && cov < 0.5`
  -> discard), so `debug=coverage` on `bog` reads white over the whole mat and
  black only in the sky — probed, not eyeballed.


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
per-pixel in resolve). Covers all six species (three grasses + three
Sphagnum states).

The `bog` stand is verified at grazing / carpet-close / 0.3m macro / topdown /
inside-plant / far-horizon, two sloped eye-level poses across the ridges
(23-degree slope, looking down it and back up it), and
albedo / normals / lighting / coverage — no toasts, no console errors.
`default` is unchanged: before/after masked pixel diffs at
grazing / inside-plant / topdown / far-horizon are RMSE 0.0004-0.0136, against
a **same-code control** of 0.0127 at grazing (the tile lists are race-ordered
when they overflow, so this is the method's own noise floor). The carpet
pipeline is not even created on a stand without carpet entries — `default`'s HUD
has no `carpet` row, so the added cost there is exactly zero.

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
- The carpet pass costs 0.6-2.9 ms p50 at 1400x800 on `bog` across the standard
  cameras — but every timing this session was contended (several agents on one
  GPU), so treat that as an order of magnitude, not a number. It is 5 texture
  samples + 3 terrain fetches per ground pixel and it is fullscreen; a
  half-res carpet pass with a depth-aware upsample is the obvious lever if it
  ever matters.
- The carpet assumes every carpet entry of a stand shares one `carpet_div`
  (the stand contract implies it — they must partition one grid). It warns and
  uses the first entry's div otherwise.
- More than 3 scattered or 3 carpet entries in one stand would still be
  dropped (warned); no shipped stand has more.

Harness wishlist (from this round):
- `standEntrySlots()` exists in TS but there is no WGSL equivalent — a shader
  that wants "how many slots does entry i have" must read `carpet_div^2`
  itself and remember that 0 means `SCATTER_MAX_PER_CELL`. A
  `stand_entry_slots(i)` helper in `scatter.wgsl` would have removed the
  hardcoded 128 from two shaders here.
- The scatter has no *inverse*: `scatter_candidate(seed, entry, cell, i)` maps
  slot -> position, and a deferred renderer needs position -> slot. For a
  carpet that inverse is exact and short, but it has to be re-derived (and
  kept bit-identical) inside the experiment. A shared
  `carpet_node_at(seed, entry, xz)` in `scatter.wgsl` would be a better home
  for it than three copies in three experiments.
- Nothing exposes the periodic tile's *height* (`topH` / canopy relief) to a
  renderer that does not bake the mesh itself; `stand_table` has
  `footprint_m` but no vertical counterpart. Here it came out of the bake, but
  `height_scale` (0.091) is the mesh's total y extent, not the canopy mean
  (7.3cm), and using it for a parallax offset would be ~25% off.

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

## Moss carpet (bog stand)

Sphagnum palustre is a 0.18m periodic community tile, 0.07-0.09m tall, laid
out by `bog` as a grid-snapped mat: 22x22 tiles per 4m cell (484 slots, life
size, scale 1.01), 90-degree-only yaw, three states partitioning the wetness
axis. Before this pass 016 rendered it as if it were an upright plant, and it
was indefensible: `bin.wgsl` re-implemented the scatter hash INLINE without the
carpet branch, so the mat became randomly-placed, continuously-yawed confetti
at 8/m^2; the near pass did honour the carpet layout but enumerated slots
0..127 of 484, i.e. a quarter of the grid, drawn as camera-facing round discs
with bare peat between them; and only the first 3 of the stand's 5 entries were
rendered at all, so the bog had no grass.

What changed (smallest first, all inside this experiment):

1. **All five stand entries render.** Entries are split by shape into <=3
   "card" slots (scattered) and <=3 "carpet" slots, each with its own atlas
   pair; the packed stamp word now carries the CARD SLOT rather than the stand
   entry index. Before, `MAX_ENTRIES = 3` silently dropped the bog's
   calamagrostis and poa.
2. **The habitat band is honoured in binning.** `try_plant()` mirrors
   `scatter_candidate()`'s hash-slot-6 acceptance test, so the bog's fen grasses
   grow in the wet hollows instead of everywhere. (No effect on `default`, which
   declares no bands — the hash sequence is unchanged there.)
3. **Carpet species leave the tile lists entirely** and get their own deferred
   pass (below). 484 tiles per 4m cell is hopeless for a K=64 per-tile list: one
   16x8 screen tile overlaps hundreds of them at any real distance, so the list
   would overflow into "confetti plus tint" no matter how the binning is tuned.
4. **A carpet-specific bake**: one 1024px straight-down capture of the tile's
   own square, 9x periodic instancing, albedo+coverage and normal.xz+height —
   see Bake. Cheaper VRAM (10.7 vs 16.6 MiB), 8x the texel density, ~100x less
   bake rasterisation.
5. **Relief, which is the actual point.** The mat is not shaded from the mipped
   normal map (averaging leaf normals collapses them to straight up — mip 3 of
   this atlas is already a flat green sheet). The shading normal is the sum of
   two slopes: the baked leaf normal, plus the gradient of the baked HEIGHT over
   a fixed ~1.2cm (capitulum) baseline, taken two mip levels finer than that
   baseline. A cavity term darkens texels that sit below the local canopy mean
   (a second, coarser height tap), which is what stops a lit heightfield from
   reading as embossed paint.
6. **One parallax step** by the layer's baked mean height (7.3 / 5.3 / 5.7cm per
   state), so the lookup slides toward the camera the way a 7cm-thick layer
   really would.

### The deferred carpet stamp

The map from a ground point to its carpet tile is a pure function of xz, and the
terrain base pass has already handed us the visible ground point per pixel. So
`carpet.wgsl` runs FIRST, while every non-sky depth texel is still terrain, and
inverts the scatter: `floor(xz/step)` -> cell + slot index -> the same node-only
jitter hash and half-open wetness interval the scatter uses -> which state owns
the node -> `h & 3` quarter turn -> that state's atlas. Bit-exact mirror of the
carpet branch of `scatter_candidate()`, read backwards.

This is O(1) per pixel with no list, no overflow, no sorting and no
plant-count dependence — a 1.13M-tile life-size mat costs exactly what a sparse
one would. Everything drawn afterwards (near cards, far stamps) simply
composites in front of it, which is the correct layering for a ground mat.

### Terrain fitting — ladder rung 3+, for free

The sample point IS the visible ground point from the depth buffer, so the mat
follows the terrain exactly, at pixel resolution, with no fit at all: it cannot
crack between tiles (there is no per-tile plane to disagree about), cannot bury
one edge or float the other, and needs no `slope_align` plumbing —
`carpet_div > 0` already says "this is a mat". The shading basis is built from
`terrain_sample(xz)` at the final (post-parallax) sample point, one bilinear
fetch giving height and (nx, nz) together, so a mat on a slope lights as a
slope. The only thing this rung cannot do is silhouette: the mat's own 7cm of
thickness does not push the terrain outline outwards at a ridge crest.

### Measured, not guessed

- Mip-level height statistics of the bake (mean |dh01| per baseline: 0.030 at
  1.4mm, 0.066 at 11mm, 0.094 at 45mm; sd 0.078 = 0.73cm) are what set the
  1.2cm relief baseline — finer baselines report leaf-scale sandpaper, coarser
  ones wash out.
- Anisotropy: Dawn/Metal does **not** apply `maxAnisotropy` to an
  explicit-gradient sample. Verified by forcing both gradient axes to the minor
  one: the same grazing ground went from mud to crisp, i.e. the level really was
  coming from the long axis. The major axis is therefore shortened to
  `sqrt(major*minor)` — the standard geometric-mean compromise — and the code
  still degrades gracefully to true anisotropy on a driver that honours it.
- Parallax by the PER-TEXEL height combs the mat into radial streaks up close
  (obvious at 0.3m, gone with the offset disabled). The offset now uses the
  species' baked mean height, which is constant per species, continuous across
  every tile, costs no tap, and is the honest "flat layer at height h" model.

### What improved, and what is still bad

Improved: the mat is continuous with no holes and no confetti; at
`carpet-close` (1m) and at 0.3m it reads as a mass of individual capitula with
dark gaps between them, which is what the source geometry actually is; the
wetness zoning reads as ecology from `topdown`; the mat conforms to the ridged
slopes exactly; the bog's grass exists and is zoned; `debug=coverage` is white
over the whole mat.

Still bad, honestly:
- **Grazing, 2-10m.** A capitulum is 2-4 px there and the mat flattens into a
  textured ochre plain. Some of that is physics (sub-pixel structure must
  average out), but the representation makes it worse: every tile of a species
  is the SAME 0.18m image, so there is no variation between 0.18m (tile) and 12m
  (wetness field) to carry the eye. Real Sphagnum has decimetre hummocks. A
  renderer cannot invent them — the stand grid-snaps identical tiles at constant
  scale — but it is the main reason a grazing bog looks like paint.
- Tile seams are faintly visible at 0.3-1m. Rotating a periodic tile by 90
  degrees does not actually match its neighbour's content along the shared edge
  (only the grid and the statistics match), so a discontinuity is inherent to
  the lattice; the gradient taps step inward at the border so at least the
  relief does not draw the lattice as a grid of shading lines.
- No silhouette: the mat is painted at the terrain surface, so at a ridge crest
  the outline is the terrain's, not the moss's 7cm.
- Single-tile species flips (a brown tile in a green zone) appear where the
  wetness field crosses a zone boundary slowly — the stand's own interlocking
  jitter, amplified by three states with very different albedos.

### Is this representation suited to a moss carpet?

The tile-list half of 016 is not: a bounded per-tile plant list cannot hold a
life-size mat, and no amount of tuning fixes that. But the OTHER half of the
method — "the depth buffer is the tile-to-world map, so stamp per pixel" — is
almost tailor-made for one. A mat is exactly the case where a screen-space
deferred evaluation is both correct and free: no placement data structure, no
overflow, exact terrain conformance, and the whole VRAM budget spent on the one
view a mat ever shows. Within the limits above it beats the flat-quad baseline
(001) clearly at close range — same coverage, real relief shading and cavity
depth instead of printed texture — and is roughly a wash with it at grazing,
where both are texture on ground.

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
