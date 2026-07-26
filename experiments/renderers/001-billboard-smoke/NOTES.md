# billboard cards

## Idea

The classic billboard baseline, but over REAL baked imagery instead of tinted
quads. Precomputed per species (in-browser, once): a 3x3 atlas of 512px
orthographic captures of the raw GCMESH1 mesh — 8 side views at 45deg azimuth
steps plus 1 straight-down top view — as albedo+coverage (rgba8) and
oct-encoded mesh-frame normals (rg8), supersampled 2x, coverage-weighted
downsampled, color-dilated 6 texels so filtering never bleeds background.

Per frame: a compute pass evaluates the shared scatter WGSL twin over a
camera-centered cell region (clamped to the stand's exact cell range on the
CPU, so cells outside the stand are never dispatched), rejects whole cells
against the region circle and the frustum before touching the scatter, then
frustum-culls each surviving plant's bounding sphere and compacts survivors
into a 16B/instance buffer + indirect draw args. Each surviving plant draws 12
vertices: a cylindrical camera-facing side card that samples the baked view
nearest the current azimuth in the plant's yawed frame, and a horizontal top
card with the top-view bake that fades in only for steep viewing elevations
(it is what keeps the top-down cam dense; at eye level it would read as a
floating cutout, so it erodes away below ~35deg). Hard alpha test with depth
write — no dither, no blending; camera-inside and region-edge fades erode
coverage through the alpha reference (edges dissolve first). Wind = shared
wind include, tips displaced by height fraction. Lighting = baked normal
rotated by yaw into world, shared light_surface + fog, plus a small
bottom-shade gradient for grounding. All five debug views (albedo / normals /
lighting / coverage / depth) are wired through the shared `debug_shade()`.

O(1) in plant count: nothing is ever materialized for the whole stand — cost
is region area (regionRadius param, default 110m, cap 128m), identical for
557k (default) and 134.2M (scaling-100m) plants. Verified: both stands render
at the same ~5.8ms GPU p50 on this machine, same VRAM.

Carpet species (`stand_table.carpet_div > 0`, i.e. the bog's Sphagnum) take a
second shape instead: ONE ground-parallel quad per tile, sized to the species'
periodic footprint, terrain-conformed per vertex, textured with the tile's own
sub-rectangle of the top-view bake, and no camera-facing card at all. See
"Moss carpet" below.

## VRAM budget math

Per species (HUD-verified, MiB):
- albedo atlas 1536x1536 rgba8 + full mip chain: 12.0 MiB
- normal atlas 1536x1536 rg8 + mips: 6.0 MiB
- culled instance buffer (16 B/instance), sized for the smaller of
  regionRadius-max 128 and the stand's own cell span, times the slots an entry
  can actually fill (`carpet_div^2` for a mat, `density/8 * 128` otherwise):
  default stand density 3 -> 3.3 MiB, 2.5 -> 2.75 MiB; bog Sphagnum (±96m
  stand, 11x11 tiles/cell) -> 4.7 MiB
- entry info uniform (176 B) + indirect args (16 B): noise

Totals on the default stand: calamagrostis 21.3/25, elymus 20.7/25,
poa 21.3/25 MiB (unchanged by the carpet work). Bog stand: Sphagnum
22.7/25 each, calamagrostis 18.9, poa 18.3 MiB — the moss was 26.8 (over
budget, HUD-striped) until the instance buffers stopped being sized for a
128m region on a 96m stand. Worst defined stand (dense-mixed, density 5) fits
at ~23.5 MiB. Bake uses ~200-400MB transient GPU memory (vertex+index buffers
of the raw mesh, 3072^2 targets, readbacks) — allocated via ctx.res tagged
`bake-scratch`, destroyed at the end of the bake.

## Bake

`bake.ts` -> `mesh/baked/001-billboard-smoke/cards-v1-<species>.bin`,
13.5 MiB each (64B header + 1536^2 rgba8 albedo + 1536^2 rg8 oct normals).
Flow is the harness bakedArtifact/commitBake infra (OPFS cache, committed
file, auto-commit after a fresh bake). Every load is magic-validated because
the dev server answers missing /mesh/baked files with index.html at 200,
which would otherwise poison both the OPFS cache and the committed-file path;
a poisoned entry is rebaked and the OPFS entry repaired in place. Full
first-run bake of all three species (incl. fetching 374MB of raw meshes) took
~10s on this machine; subsequent loads read the committed artifacts.

Normals are flipped toward the bake camera per view (two-sided foliage), so
cards light consistently from every azimuth. Mips are generated at load time
on the GPU (coverage-weighted for albedo).

## Status

working — verified by headless screenshots at grazing, topdown, inside-plant,
far-horizon (default stand) and grazing on scaling-100m, zero console errors,
plus a t=2 vs t=5 diff confirming deterministic wind sway.

Carpet path verified on the `bog` stand at grazing / inside-plant / topdown /
far-horizon, three close top-downs (1.0m and 0.4m above ground), two sloped
eye-level views (13deg and 17deg), and albedo/normals/lighting/coverage; no
toasts, no console errors. `default` is unchanged: masked pixel diffs (HUD and
panel excluded) of before vs after at all four standard cams differ in
188-642 of 695,580 scene pixels with a mean delta of 1.6-1.9/255 — isolated
alpha-test threshold flips caused by two extra interpolants, against a
same-code control run that differs in 6 pixels. GPU Σp50 on `default` stayed
in the 5.2-6.6ms band across four headless runs (baseline 6.6ms), i.e. inside
the 9ms ceiling; no bench JSON is claimed because 4 other agents were sharing
this GPU (CLAUDE.md rule), and a before/after A/B page is impossible for one
experiment id.

NOTE for anyone reproducing the screenshots: URL `cam=` poses are ABSOLUTE
world coordinates while the standard bookmarks are terrain-relative, and the
terrain sits at y = -7.69m at the origin. `cam=0,2.2,0.01,0,-1.5533,60` is
therefore ~10m above the ground, not 2.2m — at that height a 0.36m moss tile is
~16px and everything mips into mush. Use `cam=0,-6.69,...` for a 1m top-down.

## Findings

- All three species read distinctly from the baked imagery: calamagrostis =
  pale pink fluffy panicles over green stems, elymus = darker upright spikes,
  poa = short fresh-green tufts. Ground truth (000) at grazing shows the same
  green-stems + pink-heads character for the calamagrostis patch.
- The top card is essential for the topdown cam (side cards vanish edge-on)
  but is the technique's weakest artifact: at mid elevations it pops in as
  pale discs (its imagery is brighter than the side views because straight
  down you mostly see fluff). Restricting it to elevation > ~35deg (vs the
  card's own height) removed the "floating pancake" artifacts at inside-plant
  and the pale flecks at far-horizon while keeping topdown dense.
- Camera-inside fade via alpha-reference erosion works: the inside-plant cam
  shows the surrounding stand, not a texture wall.
- Known billboard limits, left as honest baseline behavior: 8-view azimuth
  snap pops when orbiting a single plant; close-up cards go flat and soft
  (512px tiles magnified); plants beyond regionRadius (cap 128m) erode out in
  a fog-assisted band, so far-horizon coverage ends at ~110m by default.
- GPU p50 on this machine, default stand, 1280x800: cull 0.4-0.5ms + cards
  1.3ms (topdown) to 3.1ms (inside-plant); scaling-100m identical to default.
  No `results/` bench JSONs are claimed: other agents were rendering in
  parallel during this session, so bench numbers would be contaminated
  (CLAUDE.md rule). Rerun `#/bench/001-billboard-smoke?stand=default` on an
  idle GPU before quoting numbers.
- Harness wishlist: none — bakedArtifact was sufficient once magic-validated;
  a dev-server 404 (instead of SPA fallback) for missing /mesh/baked files
  would let experiments drop the validation shim.

## Moss carpet (bog stand)

Sphagnum palustre is a 0.18m periodic community tile, 0.07-0.09m tall, laid out
by the `bog` stand as a grid-snapped mat (11x11 tiles per 4m cell, constant
scale 2.02, 90deg-only yaw). Rendered as an upright plant it was indefensible:
a 0.65m-wide camera-facing card 0.18m tall plus a 0.65m top card on a 0.36m
grid, i.e. every tile drawn ~3.2x too large and overlapping its neighbours,
horizontal slabs ignoring the slope, and vertical cards slicing through the
ground and each other ("rows of bricks" at eye level, dark grey slabs in the
lighting view).

What changed (smallest first, all inside this experiment):

1. **Carpet detection + a second card shape.** `stand_table[i].carpet_div > 0`
   selects a single GROUND-PARALLEL quad and skips the side card entirely; the
   indirect draw for those entries emits 6 vertices instead of 12, and `topCard`
   does not apply to them. A mat has no silhouette worth a camera-facing card,
   and vertical cards through a mat are pure artifact.
2. **Width from `footprint_m`, not the capture radius.** The quad spans exactly
   `footprint_m * scale` (= the carpet's 0.3636m grid step), and the top-view
   texture is cropped to the tile's own square `[0, footprint_m]^2` in the mesh
   frame, computed from the baked capture box (`card_r`, `card_cx/cz`). That one
   change turns 0.65m overlapping squares into an exactly-abutting mat and
   quadruples the texel density that lands on screen.
3. **Terrain conforming — ladder rung 3 (per-vertex).** Every quad corner gets
   its own `terrain_sample(xz)`: the height conforms the mat and the (nx, nz) in
   the same four texel loads gives the shading basis, so the normal is free.
   Rung 3 rather than 1 or 2 for a specific reason: neighbouring tiles SHARE
   corner positions, so a per-vertex fit is the only rung that keeps the whole
   mat C0-continuous. Any rung that fits one plane per tile (point normal or
   `terrain_plane_fit`) leaves a wedge-shaped crack at every tile boundary,
   because two neighbours pick two different planes — the cheap rungs are not
   cheaper here, they are wrong. Cost is 4 texel loads per vertex, 6 vertices
   per tile.
4. **Baked normals lifted into the ground frame.** The carpet fragment builds
   `plant_basis_from_up(up, yaw)` (inlined — the card already carries
   cos/sin(yaw), and the harness helper wants the angle) and rotates the baked
   mesh-frame normal into it, so a mat on a slope lights as a slope. The upright
   path keeps its yaw-only rotation, bit-for-bit.
5. **A carpet-specific alpha reference (0.06, vs the 0.4 grass default).** A mat
   is a closed surface and must not dissolve with distance. Tile alpha is ~80%
   solid up close, but the mip chain pulls it toward the whole top tile's mean
   (only 27-37% of that tile is covered), so at the grass reference entire
   distant tiles failed the test and punched tile-shaped holes in the carpet. At
   0.06 the mat stays a solid depth-writing occluder while the genuinely empty
   texels (16-20% of the tile — the gaps down to the peat) still open. This is
   the opposite of dithering: more hard-edged opaque coverage, not less.
6. **No camera-inside fade for carpets.** Eroding a mat you are standing on
   opens a hole under your feet; only the region-edge fade remains, and it is
   measured from the tile centre (see the sliver bug below).
7. **Instance capacity clamped to the stand.** Capacity was sized for a 128m
   region and 128 slots/cell regardless of the stand; on the ±96m bog with
   11x11=121 carpet slots that over-allocated 4 MiB/species. Now
   `min(region, stand span)` x the slots an entry can actually fill. This is
   why moss fits the 25MB budget again (26.8 -> 22.7 MiB). On the ±128m stands
   the formula produces the identical number, so `default` VRAM is unchanged.
8. `carpetOverscale` param — see below; default 1.0.

Two bugs found and fixed on the way, both mine:

- **Stretched slivers.** The region fade was evaluated per vertex, and a card
  whose fade drops below the alpha reference is emitted behind the near plane —
  so a tile straddling the threshold got 1-2 vertices at its real position and
  the rest at the clip point, rasterizing as a bright streak metres long. Fixed
  by measuring the fade from the tile centre (the upright path always did, via
  the instance-uniform `to_cam`).
- Overscale + coplanar quads z-fight, so overlapping tiles get a 4-phase
  sub-millimetre height bias (only when `carpetOverscale > 1`); four phases,
  not a checkerboard, because with overscale a tile also overlaps its diagonal
  neighbours, which share a checkerboard's parity.

### Overscale: allowed, but it measured worse

A uniform overscale is legal (it does not touch the grid step, the yaw or the
per-tile scale) and it does shrink the placement holes below, so I tried 1.0 /
1.15 / 1.35 at `cam=grazing`. 1.15 and 1.35 both draw a visible dark LATTICE
over the whole mat: the overscaled fringe samples the part of the top view
outside the tile square, which is sparse overhanging foliage with partial
coverage and dilated colour, so every tile edge paints itself as a dark line.
A flat single-plane tile has no thickness to hide an overlap inside — the
overlap edge IS the artifact. Default 1.0 (exact abutment) is visibly the
cleanest, and at 1.4mm/texel the 90deg-rotation seams are invisible without
any overlap. The knob stays because the trade-off would flip for a
representation with real thickness.

### What improved / what is still bad

Screenshots at cam=grazing, inside-plant, topdown, three genuinely close
top-downs, two sloped eye-level views, far-horizon, and albedo/normals/
lighting/coverage (before and after, same cameras):

- Improved: the mat is closed, continuous and has real moss detail at every
  distance from 0.4m up; it follows the terrain exactly (no buried or floating
  edges on the bog's slopes); no vertical cards cutting through the ground; the
  three micro-habitat states read as coherent zones; per-fragment normals are
  real (`debug=normals` shows moss-scale variation near the camera, not a flat
  up-normal); moss VRAM is back inside the budget; and the moss draws half the
  vertices and one card instead of two.
- Still bad #1, and not mine to fix: **tile-sized bare-ground holes along every
  zone boundary.** `scatter_candidate()`'s carpet branch derives its boundary
  jitter from a hash that includes `entry_index`, so the three Sphagnum entries
  evaluate a DIFFERENT wetness at the same grid node and do not partition [0,1)
  exactly. Measured over 20,449 nodes (bog, seed 42, cells -6..6): 2.55% of
  nodes are claimed by NO entry and 2.63% by two. The old renderer hid this
  because each card covered 3.2x its own cell; a tile that covers exactly its
  cell cannot. At grazing angles a single 0.36m hole foreshortens into a large
  wedge of visible peat, which is what makes it the most noticeable remaining
  artifact. Fix belongs in `src/wgsl/scatter.wgsl` + its TS twin: derive the
  jitter from the node (cell + slot, or quantised xz) instead of `h`, and all
  entries see the same value.
- Still bad #2, inherent: **the mat is flat.** A single ground-parallel plane
  has zero silhouette, so at eye level the carpet reads as a very good moss
  TEXTURE on the ground rather than as 18cm of springy cushion. The source
  mesh has 3.3cm of capitulum relief (6.7cm at the stand's 2.02 scale) and none
  of it survives. Nothing cheap fixes this inside a card representation; it
  wants a height/relief channel and a parallax or shell step, i.e. a different
  method.
- Still bad #3, cosmetic: the wet-vigorous state is authored as a strongly
  saturated flat green, so its zone reads as poster paint next to the ochre and
  bronze states. Confirmed to be the atlas albedo, not the lighting
  (`debug=albedo` shows the same colour). Faint 1-texel dark lines are
  occasionally visible along a tile boundary in that zone; they come from the
  crop edge of the baked tile, not from the geometry (removing the height bias
  did not change them).
- Not done, but the obvious next win: for a carpet species the atlas is 96%
  dead weight — the 8 side tiles are never sampled and only ~40% of the top
  tile's area is inside the crop. A carpet-only bake (one tile, top view,
  ortho'd over exactly `[0, tile]^2`) would be ~2 MiB instead of 18 MiB and
  would raise the effective resolution 1.75x at the same size. Skipped here to
  avoid re-baking three 479MB source meshes for a change that is orthogonal to
  the geometry fix.

### Are billboard cards suited to a moss carpet?

Half. The *card* part is fine: a periodic mat viewed from above is exactly a
texture-mapped plane, and one grid-snapped, terrain-conformed, tile-sized quad
per node is arguably the correct primitive for it — it is closed, cheap
(6 verts/tile, one card), writes solid depth, and at 1.4mm/texel it holds real
moss detail down to 0.4m viewing distance. The *billboard* part is worthless
here and had to be deleted: a camera-facing card is a proxy for a silhouette,
and a 7cm mat seen from a standing eye has none. What the representation cannot
do at all is thickness — no relief, no self-occlusion parallax, no
cushion-to-cushion depth. So: an honest, fast, closed carpet with no volume.
If the bar is "reads as moss ground cover from a standing camera", this passes;
if it is "reads as a Sphagnum cushion you could press with your hand", cards
cannot get there and a relief/volume method should own the carpet species.

## Audit (structural waste + debug views)

Debug views — the renderer honoured none of them (it always returned lit+fogged
colour). Now `cards.wgsl` includes `src/wgsl/debug.wgsl`, skips fog unless
`debug_mode() == DEBUG_OFF`, and returns
`debug_shade(color, alb.rgb, n, alb.a, in.world)`. What the modes exposed:

- normals: already real per-fragment normals (oct atlas decoded, rotated into
  world by the plant yaw) — verified rather than assumed. At `grazing` (camera
  at +x+z looking at the origin) near-field normals read pink = +x+z = pointing
  back at the camera, which is what the bake's flip-toward-the-view-camera
  convention should produce; `topdown` reads green = +y from the top card.
  Both are the expected answers, so the yaw rotation sign is right.
- lighting: goes through the shared `light_surface()` exactly once. Sampled
  linear values run ~0.14 (away from the sun) to 1.0+ (toward it) — it *looks*
  blown out only because the view is displayed in sRGB; nothing is unlit,
  double-lit, or premultiplied (the albedo view shows the raw baked atlas).
  `in.shade` (the grounding gradient) is deliberately counted as part of the
  light term, not albedo: it is fake occlusion, so `albedo` stays exactly the
  baked capture and `lighting` shows sun+ambient x grounding.
- coverage: the baked alpha the fragment resolved to. With the hard alpha test
  everything below `erode` is already gone, so the view doubles as the
  alpha-test margin — thin stems show up as the near-threshold greys.
- depth/albedo: correct, region edge clearly visible in depth at ~110m.

Structural fixes (all three verified image-identical: frozen-time
`det=1&t=3` screenshots at all four standard cams, old vs new, differ by the
same handful of speckles that two *identical* runs differ by — the instance
compaction order is `atomicAdd`-nondeterministic, so cards at tied depth win
the depth test arbitrarily):

1. Fully-eroded cards were rasterized anyway. `erode = alpha_ref / fade`, so
   any card with `fade < alpha_ref` has `erode > 1` and *every* one of its
   fragments discards — yet the quad was still rasterized and every fragment
   still did two texture samples before dying. That set is exactly the
   screen-filling one: all top cards at low elevations (the `smoothstep(0.35,
   0.6, elev)` erosion, i.e. every eye-level view), plus cards the camera is
   standing inside (which cover half the screen at `inside-plant`), plus the
   region rim. The vertex shader now emits those behind the near plane.
2. The cull dispatch was sized by the full square region regardless of the
   stand. The stand cell clamp lived in the shader, so on `close-quality`
   (±24m) ~96% of the 400k threads/species existed only to return immediately,
   and on `default` the same happens whenever the camera nears the stand edge.
   The rect is now clamped to the stand's cell range on the CPU (`side_x` /
   `side_z` replace the single `side`), and a zero-area region skips the
   dispatch entirely.
3. The cull evaluated the shared scatter — 4 heightmap texel loads + 7 hash
   rounds — for every candidate in the full 360° region, and only *then*
   frustum-tested. A workgroup is 64 threads and a cell holds 128 slots, so
   every workgroup covers exactly one cell: two workgroup-uniform cell-level
   rejects now run first (cell rect vs the region circle, and a conservative
   cell box vs the 6 frustum planes), and whole workgroups exit before a single
   texel is fetched. The box is exactly conservative: cell footprint in xz,
   `1.05 * frame.terrain_height_scale` in y (the shared FBM's octave amplitudes
   sum to 1.015) plus the tallest card at `scale_max` and the same +0.35 wind
   margin the per-plant test uses. The per-plant tests are unchanged and still
   exact, so the visible set is identical.
4. Minor: `frustumPlanes()` allocated a fresh `Float32Array(24)` every frame;
   it now fills a preallocated one.

Deliberately left alone:

- The `info` uniform (176 B x 3 entries) is rewritten every frame although only
  the planes, region rect and two params change. Splitting it into a constant
  and a dynamic buffer saves three tiny `writeBuffer`s per frame — not worth
  the field-order churn.
- `uploadAtlas()` builds its own mipgen BGL + 2 pipelines per species (3x
  identical). Init-time only, ~6 pipeline compiles at startup.
- No depth prepass and no front-to-back ordering for the cards. Sorting or a
  prepass might pay off at grazing angles where overdraw is deepest, but the
  win is not obvious from the code — needs a bench on an idle GPU.
- The instance buffer is recompacted every frame even when neither camera nor
  params moved. A dirty flag would skip it, but wind does not move plants, so
  correctness would hinge on catching every input — measure first.
- Normal mips are a plain box filter over *oct-encoded* values, so at extreme
  distance a species' normals drift toward one per-tile mean instead of the
  true average direction (visible in the normals view as the far field losing
  its point-at-the-camera character). Fixing that means decoding, averaging and
  re-encoding in `mipgen.wgsl` — a bake/quality change, out of an audit's
  scope, but it is the first thing to try if distant lighting looks flat.
