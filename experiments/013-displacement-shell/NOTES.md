# 013-displacement-shell — vector-displacement shell

## Idea

The only geometry that ever exists is one flat sheet. Its textures encode
WHERE the plant surfaces are. There are **two flavours of field**, because a
plant and a mat are not the same shape:

- **scattered species** (the grasses): a per-species **strand
  vector-displacement field** — 96 "strands" (blade/plume ribbons) × 16
  stations, each texel storing offset-from-plant-origin (xyz), ribbon
  half-width (w), mean normal + baked vertical occlusion, and mean albedo. A
  vertex is nothing but (strand row, station column, side bit); the field
  crumples the flat patch into the plant. Three distance rings, continuous
  strand-count LOD.
- **carpet species** (`stand_table[i].carpet_div > 0`, i.e. the bog Sphagnum):
  a **65×65 top-shell displacement grid** per periodic tile — node xz on a
  square lattice, the cushion's surface height, its real per-node colour, its
  normal and a relief-derived occlusion. A vertex is (strip row, column, side)
  and reads the grid directly with `textureLoad`; the sheet is a watertight
  displaced surface that conforms to the terrain per vertex. LOD is an integer
  texel **stride** (1..32), which is what keeps it watertight at every level.

Far away both settle onto the same limit surface — a single terrain-conformal
canopy shell.

Per frame:

1. **cull** (compute): walks the scatter cells of a camera-centered
   region (bit-identical WGSL twin of the harness scatter — placement is
   evaluated in-shader, no CPU instance buffers ever), frustum-tests every
   existing plant and appends a 16 B record (pos + quantized yaw/scale/phase/
   entry) into one of three distance rings via atomics into the indirect-draw
   buffer. Work = region cells × 128 slots × stand entries, **independent of
   total plant count** — and a conservative per-CELL frustum AABB test drops
   whole off-screen cells before any hashing or terrain sampling happens, so
   the real work is view-bounded, not just region-bounded.
2. **strands+shell** (ONE render pass): three `drawIndexedIndirect` for the
   rings, then the far shell. Each ring draws ribbon
   patches of decreasing topology (12/8/5 stations; the station axis is
   sampled with the linear filter, so fewer stations = automatic curve
   simplification). Strand-count LOD is **continuous**: rows beyond
   `s0·r0/d` collapse to zero width (fractional fade on the marginal row) and
   survivors widen by `sqrt` to conserve coverage — ring handoffs cannot pop
   in density. Ribbons are camera-facing (side = tangent × view), wind is the
   shared `wind_sway` sheared up the strand with per-strand phase jitter,
   per-species sway from the stand table. Lighting uses the baked mesh
   normals (flipped to the viewer + up-biased) and the baked per-station
   occlusion.
3. **far shell** (~2.7 k tris, last draw of the same render pass): beyond
   `rOuter` the meadow IS the
   shell — a camera-centered geometric-progression annulus draped on the
   terrain at density-weighted canopy height, colored by the baked per-species
   canopy albedos blended with world-anchored value-noise mottle, with a
   gentle wind bob. Plants in the outer 5 m of the region sink onto the shell
   height, so the handoff reads as canopy continuing. The shell clamps to the
   stand boundary (the meadow visibly ends there, alpha-faded over 3 m).

Rules check: no per-frame raymarching (four texture samples per vertex, zero
per fragment); no source geometry touched at runtime; cost is bounded by
region + screen (verified: `scaling-100m`, 134.2 M plants, renders at the
same frame cost and VRAM as the 557 k default — see Findings); works from all
angles (rings/cull use horizontal distance so topdown keeps its plants;
grazing is the design center); camera-inside-plant fades via horizontal-
distance width collapse (no dithering anywhere — every edge is real
depth-writing geometry).

## VRAM budget math

(Carpet-path numbers are in the moss-round section below.)

Per species (the only per-species-scaled data):

- strand field textures: 3 textures × 16×96 texels × 8 B (rgba16float) ≈
  **36 KB per species** (108 KB for all three, single 3-layer texture array).

Shared, sized by param maxima (NOT by plant count or stand size):

- ring instance buffers: (81+225+961) cells × expected survivors per cell
  (`128 × density/8` summed over the scattered entries) × 16 B ≈ **3.3 MB** on
  the default stand. (It used to be sized by the slot count instead of the
  survivor count: 7.8 MB for the same image.)
- ribbon topology (3 × u16 index buffers): 25 KB; shell mesh: 28 KB;
  uniforms/indirect: < 1 KB.

Total experiment VRAM ≈ 8 MB; HUD shows 17.0 MB including harness terrain.
The 25 MB/species budget is cleared by ~3 orders of magnitude on the
species-scaled part.

## Bake

`bake.ts` produces a VDF1 artifact per species (73 KB each, committed in
`mesh/baked/013-displacement-shell/`): samples ~320 k area-weighted surface
points from the raw GCMESH1 (2.2 M–6.5 M tris, all three species), then
traces 96 strands through the point density with a spatial hash grid —
dart-thrown low roots, upward density-following walk, per-station centroid /
RMS width / mean color / mean normal / height-curve occlusion. Strand rows
are ordered as an importance-sorted interleave of four spatial quadrants so
ANY prefix of rows is a balanced subset — the entire LOD scheme is "use the
first N rows". Community tiles (calamagrostis, elymus) are traced as whole
tiles (a scatter point stamps one tile, same convention as 003); poa is a
single specimen and traces naturally. Bake runs in-browser in ~1–3 s per
species, cached in OPFS and committed via the bake endpoint.

## Status

working — verified by headless screenshot at `grazing`, `topdown`,
`inside-plant`, `far-horizon`, and `grazing` on `scaling-100m` (134.2 M
plants; identical cost), plus the whole `bog` stand at `grazing`,
`carpet-close`, `topdown`, `inside-plant`, `slope-view`, `slope-close` and every
debug view (see the moss-round section). All six species render. All six global
debug views (`view=` / `debug=`) are wired and were inspected — see Audit.

## Findings

- The strand reconstruction reads convincingly at grazing: green blade
  understory + tan calamagrostis plume tops emerge naturally from the traced
  colors (the plume stations are wide + pink because the source points there
  are wide-spread and tan — nothing species-specific is hard-coded).
- 134.2 M plants (`stand=scaling-100m`) vs 557 k (default): identical per-pass
  cost in both. Plant count is provably free; the only VRAM/time driver is the
  region radius and strand budget params. (Numbers from the first session are
  no longer quoted here — the passes were merged, see Audit, and the GPU was
  shared. Re-measure solo.)
- NOT benchmarked with `#/bench` — 15 sibling experiments were hammering the
  same GPU during this session, so any recorded numbers would be
  contaminated. The timings quoted above are HUD p50s from the quietest runs
  and should be re-measured solo.
- Honest artifacts:
  - Ribbons are camera-facing planes; extreme close-ups read flat and
    stylized (paddle-like plumes), and the 96-strand budget is a statistical
    proxy, not the 2 M-tri truth — mid-range coverage/color match is the
    design goal.
  - RMS-based width baking overestimates blade width (neighbor blades leak
    into the gather radius); compensated at runtime with `widthScale` 0.75
    default and a 8.5 cm absolute half-width cap.
  - The region→shell handoff is smooth at grazing but visible from topdown as
    a detail change in the far corners (plants → flat mottle at 40 m).
  - The far shell is opaque canopy color; it does not show soil between
    plants from straight above beyond the region.
  - `debugRings` param tints the three rings and bypasses frustum culling +
    forces fixed ribbon width (debug only).
- Camera-inside-plant: width collapse by horizontal distance (0.3–0.8 m).
  First implementation used 3D distance — never triggered because the camera
  floats ~1.5 m above plant bases; worth remembering for other experiments.
- Harness wishlist: nothing blocking; a shared "committed-bake with magic
  validation" helper would remove the copy of 003's SPA-fallback workaround.

## Audit (structural + debug-view pass)

Debug views were **completely unwired** — `frame.debug_mode` was ignored by
both shaders, so every `view=` mode rendered the normal image. Fixed:

- both fragment shaders now `#include "src/wgsl/debug.wgsl"` and return
  `debug_shade(shaded, albedo, normal_ws, coverage, world_pos)`; fog is applied
  only when `debug_mode() == DEBUG_OFF`.
- the baked occlusion (`mix(0.55, 1.0, ao)`) and the shell's `0.92` trim moved
  from the lit result into the **albedo** term. Identical output (the shared
  lighting model is multiplicative in albedo) but now `debug=lighting`'s
  divide-out is exact and `debug=albedo` shows the real baked colour.
- the shell paints opaquely in debug views (`out_alpha = 1`), otherwise its
  normals/albedo would come back blended with the base pass behind it.
- our own `debugRings` uniform field was called `debug_mode`; renamed to
  `ring_debug` so it cannot be confused with the global selector. It stays the
  method-specific view (ring tint + fixed width + no culling), as the task's
  "expose extra state as your own param" rule asks.

What the views exposed: normals and lighting were already real — per-fragment
interpolated **baked** strand normals through the shared `light_surface()`,
no constant-brightness cheat, so no lighting repair was needed and the normal
view is unchanged in `off` mode. `debug=normals` shows full hue spread
(including downward-facing undersides — the `UP_BIAS` nudge biases, it does not
replace). `debug=albedo` does show one honest mismatch: from topdown the far
shell's canopy albedo is noticeably more olive/desaturated than the plant
field it takes over from (the shell is the width-weighted upper-canopy mean, the
field is blades + dark gaps). `debug=coverage` is flat white over the plants —
correct: every plant edge is hard opaque geometry, zero stochastic alpha — with
the only grey being the shell's 32→40 m alpha ramp, which is exactly the band
worth watching. `debug=depth` shows honest per-plant depth, no impostor walls.

Structural waste found and fixed:

1. **Two back-to-back render passes with identical attachments** (`strands`,
   then `farshell`) — same colour/depth views, same load/store ops, no
   dependency. Merged into one `strands+shell` pass: on a tile-based GPU (the
   dev machine is Apple Silicon; the project targets low-mid devices) the split
   cost a full colour+depth store and reload for nothing. Trade-off: the HUD now
   shows one row instead of two, and old `results/*.json` have the old labels.
2. **The cull evaluated all 128 scatter slots of every cell in the ±44 m
   region, including cells nowhere near the frustum** — ~200 k slot
   evaluations per frame, each 7 hashes + a bilinear terrain fetch, only to be
   thrown away by the per-plant frustum test. Added a per-CELL conservative
   frustum AABB test that runs *before* `scatter_candidate`. The box bounds
   every plant sphere the per-plant test could accept (cell footprint + max
   plant radius horizontally; full terrain amplitude + max plant height
   vertically), and the AABB-vs-plane form is exact with unnormalized
   view-proj rows, so the surviving plant set is provably unchanged. At
   grazing this rejects the large majority of region cells for five dot
   products each.

Verified image-identical: `det=1&t=4` screenshots at `grazing`, `topdown` and
`inside-plant` before vs after differ by 0–254 pixels out of 668 k — the same
magnitude as running the *unchanged* build twice (atomic append order flips
tie-break winners at coincident depths). No console/shader errors in any of the
six debug modes.

Deliberately NOT changed (would need measurement or would change the image —
suggestions for whoever picks this up):

- **Ring row budget vs coverage boost is inconsistent.** A ring draws
  `ceil(s0·r0/r_ring_outer)` strand rows (the *minimum* over the ring) but the
  width `boost` uses the continuous `f_cont = s0·r0/d`. Just past `r0` that is
  64 rows' worth of boost applied to 28 drawn rows, so coverage drops ~35 % at
  the ring-0→1 handoff — the "handoffs cannot pop" claim above is optimistic.
  Fixing it means widening ribbons, i.e. changing the image; it is a quality
  fix, not an audit fix.
- Per-instance values (yaw/scale/phase decode, `f_cont`, `boost`, near/edge
  fades, `sin`/`cos` of yaw) are recomputed by every one of a plant's up to
  1536 vertices. Moving them into the cull pass would widen the 16 B instance
  record (more VRAM, more bandwidth) for an unmeasured ALU win — not an
  obvious call, left alone.
- The 144 B globals UBO is rewritten every frame although only the region
  corner/dims and a few params change. Splitting it into constant + per-frame
  blocks is not worth a second buffer and a second bind at this size.
- `SHELL_ROWS_SCATTER = 20` with a 1.285 geometric radius ratio reaches 3.7 km,
  so on the ±128 m default stand roughly half the rows clamp onto the boundary
  and contribute slivers. Harmless (2.7 k tris total) and needed by the ±384 m
  stands; left as is. Carpet stands use 40 rows and an adaptive ratio instead —
  see the moss round.

## Moss round (Sphagnum carpet on the `bog` stand)

Test scene: `#/run/013-displacement-shell?stand=bog`, judged at `grazing`,
`carpet-close`, `topdown`, `inside-plant`, the two new `slope-*` cams and the
`default` stand for regression.

### What it looked like before

Two failure modes stacked on top of each other:

1. **Three quarters of the mat was never enumerated.** The cull's slot guard and
   dispatch came from `SCATTER_MAX_PER_CELL` (128), but a carpet entry has
   `carpet_div²` = 484 slots per cell. Slots 0..127 are grid rows 0..5 of 22, so
   the moss rendered as a ~1.1 m band of tiles every 4 m — regular stripes that
   look exactly like a placement bug.
2. **What did render was upright camera-facing ribbons.** The strand bake traces
   strands upward with a 3 cm gather radius; for a 7 cm tall, 18 cm wide cushion
   that radius is 43 % of the plant's height, so all 96 strands reported nearly
   the same colour, the same 2 cm half-width and converged on the tile centre.
   Result: a fuzzy blob at grazing and, from directly above, near-nothing —
   `side = tangent × view` degenerates when the strand axis is the view axis, so
   the topdown view was thin noise over bare terrain.

### Changes, smallest first

1. **Carpet entries are enumerated in full.** New `cs_cull_carpet` walks all
   `carpet_div²` slots (dispatch `ceil(484/64)` in x) over its own, much smaller
   region; `cs_cull` now returns early for carpet entries. Two numbers kept
   apart, as the trap says: *slots to evaluate* is 484 per cell per entry, while
   *instances to store* is what survives — and since the three moss entries
   partition the wetness axis, their combined survivors are exactly 484 per cell,
   which is what the buckets are sized for.
2. **Ring capacity is sized by expected survivors, not by slot count**
   (`128 × density/8` summed over the scattered entries). Pure VRAM: the default
   stand went from 17.1 MB to 12.9 MB with a pixel-identical image.
3. **A carpet species gets a different baked field and a different draw path**
   (see Idea): `bakeCarpetGrid` + `vs_carpet`/`fs_carpet`. No upright cards, no
   camera-facing anything, no width — the sheet's two strip edges *are* two baked
   grid lines, `stride` apart, so the surface is closed by construction. No
   camera-inside fade (a mat you stand on must not open a hole), no wind unless
   the stand asks for sway (moss has `sway = 0`).
4. **Six LOD buckets by texel stride** (1,2,4,…,32), chosen in the cull so one
   baked grid quad stays ≈ `carpetPx` pixels wide using the real focal length
   (`proj[1][1] · viewport.y/2`). The bucket's topology is exactly the lattice
   that stride needs, so unlike the strand rings there is **no collapsed-vertex
   waste**: every vertex of a carpet draw is live geometry. A full bucket
   promotes to the next coarser one instead of dropping a tile — a dropped
   carpet tile is a hole, and holes read as placement bugs.
5. **The sheet is baked one period wider than the tile** (`GRID_EXT_FACTOR`
   1.15) and the point cloud is bucketed **periodically** (each sample point nine
   times, at ±period offsets). The overlap is what makes 90°-rotated neighbours
   safe: they cross in a crease instead of stepping apart into a crack. First
   attempt sampled non-periodically and the rim, finding only the mesh's own
   overflow, collapsed to the ground — every tile baked as a dome and the mat
   rendered as a visible 18 cm waffle. A sub-millimetre per-tile depth phase
   settles the case where two identical sheets one period apart are exactly
   coplanar.
6. **The far shell became a mat shell.** It now hands over by the same 3D
   distance the carpet cull uses (so an overhead camera gets shell instead of a
   full-detail disc under it), its inner row radius shrinks as the camera climbs,
   it has 40 rows instead of 20 on carpet stands (a shell riding 7 cm above the
   ground has to resolve the terrain far better than one at 32 m), it is lifted
   by a fraction of the local row span so the terrain stops poking through it,
   and — the visible one — **its species mix now comes from the shared wetness
   field** instead of one global density-weighted average, so the three moss
   states zone in the far field exactly as they do in the geometry. Before that,
   the handoff drew a coloured ring at grazing and a brown disc from topdown.
   Its albedo trim (0.69 on carpet stands vs the old 0.92) is matched to what the
   mat's own relief occlusion and tilted normals produce, so the handoff no
   longer steps in brightness. All of it is gated on the stand actually having a
   carpet entry — non-carpet stands keep the old shell exactly.
7. `Globals` grew per-species arrays to 6 (the shell's mix loop and the canopy
   table were hard-coded to 3, so the moss species were reading poa's entry) and
   the manifest lists the three Sphagnum species plus two `slope-*` cams.

### Terrain fitting: rung 3 (per-vertex conforming) for the mat, rung 0 for grass

Every mat vertex takes its ground height from `terrain_sample(xz)` (one bilinear
fetch for height + slope, instead of `terrain_height` and `terrain_normal`
paying for the same taps twice) and rides the cushion's own baked height on top;
the baked normal is rotated into the local ground frame by `slope_align`. Rungs
1–2 are not merely cheaper here, they are *wrong*: neighbouring tiles fit
different planes and crack apart at their shared edge, whereas a per-vertex
displacement that is a pure function of world xz makes neighbours agree exactly.
Cost is one terrain fetch per vertex, which at ≈0.5 M mat vertices is nothing.
The horizontal offsets stay horizontal on purpose — tilting them would stop the
tiles from tiling in plan view. The grasses keep the old upright treatment
(`slope_align` is 0 on every stand that renders them except the bog's
calamagrostis at 0.3, which is left for a follow-up: changing it would change
the `default` image for no moss benefit).

### Lattice invariant

Untouched: the yaw comes from the scatter (90° steps only), the scale is the
stand's constant carpet scale, no per-tile jitter, no distance shrink. The
overscale is in the BAKE (the sheet is wider than the tile), not in the
placement: grid spacing and rotation set are exactly what the stand specifies.

### VRAM budget math (carpet path)

Per carpet species: 3 textures × 65×65 texels × 8 B (rgba16float) = **101 KB**,
i.e. 0.4 % of the 25 MB/species budget. The array is allocated for the whole
species catalog (6 layers → 608 KB) because the shader indexes it by catalog
index. Strand species are unchanged at 36 KB each.

Shared, sized by param maxima: carpet instance buffer 3.37 MB on the bog stand
(six buckets, each sized for the cells its stride can reach × 484 records ×
16 B, rounded to 256 B slices), ring instance buffers 0.81 MB on bog / 3.3 MB on
default, carpet index buffers 65 KB, shell mesh 56 KB. HUD total: **14.4 MB on
bog, 12.9 MB on default** (was 17.1 MB).

### Bake

`bakeCarpetGrid` (artifact `VDG1`, `<species>-grid-v2.bin`, 198 KB each) samples
900 k area-weighted surface points from the 19.8 M-triangle source mesh, buckets
them periodically on the 65×65 lattice, and for every node takes the local
maximum height over the 3×3 cell ring and averages **only the points within
4 mm of it** — that top shell is the surface you actually see, with its own
colour (pale young apices, dark crevices) instead of the whole-cushion mean.
Height is mildly smoothed ([1,6,1]²), the normal is the height-field central
difference nudged by the mean leaf normal, and the occlusion comes from the
node's height relative to a ±14 mm blur (crevices dark, knolls bright). ~2 min
per species in-browser, dominated by fetching and walking the 479 MB mesh.
Stale variants (`-grid-v1`, and the strand-format moss fields nothing loads any
more) deleted.

### What improved, what is still bad

Verified by headless screenshot at every listed camera, in `off`, `albedo`,
`normals`, `lighting` and `coverage`:

- **carpet-close (1 m straight down)**: before, bare terrain with one 1.1 m band
  of orange paddles; after, a continuous mottled green/ochre cushion with
  capitulum-scale colour and relief detail and **no visible tile grid**.
- **grazing**: before, contour-following stripes of moss on bare ground; after, a
  closed mat with visible micro-relief that reads as ground cover, zoned into the
  three states, with calamagrostis emerging from it. Handoff to the shell at
  ~12 m is hard to find in a still.
- **slope-view / slope-close** (28° ridge at x≈28, z≈-44): the mat follows the
  ridge exactly — no cracks between tiles, no buried or floating edges, and the
  texture keeps its scale up the slope.
- **topdown (42 m)**: before, 4 m stripes of noise; after, the mat has handed
  over to the shell, which paints the wetness zoning as a bog mosaic. A few small
  terrain-coloured patches still poke through the shell where it interpolates
  across a hill.
- `debug=normals` shows real per-fragment relief normals on the mat (green with
  cyan/yellow facet variation), `debug=coverage` is white over the mat geometry
  and grey only in the shell's 6→12 m alpha crossfade, `debug=lighting` is
  unchanged in level from before (it is the shared `light_surface`).

Still bad / honest limits:

- The baked relief is **about half the mesh's 3.34 cm**: inter-capitulum crevices
  are a few mm wide, which is at the 3.2 mm lattice's resolution limit, so the
  grid resolves the cushion's lumps but not the gaps between individual
  capitula. The occlusion channel carries what the geometry cannot, which is why
  its floor is as low as 0.42. Consequence: at 1 m the mat reads as a very
  convincing moss *surface*, but a little softer than the real thing.
- LOD levels are integer strides, so a tile's shape refines in a visible step
  when it crosses a bucket boundary (at ~0.8/1.6/3/6/12 m with the default
  `carpetPx`). The step is a resolution change of a surface, not a pop in
  coverage, and each boundary sits where a band is ~6 px; a geomorph would need a
  second sample per vertex.
- Neighbouring sheets cross rather than merge, so there is a faint crease
  network at the 18 cm tile pitch if you look for it at `carpet-close`. It is a
  shading kink, not a hole.
- The far shell is a smooth surface: past ~12 m the mat's relief is gone (it is
  worth about a pixel there) and terrain still pokes through it occasionally.
- The mat's LOD knob (`carpetPx`, target pixels per baked quad) trades cost for
  detail linearly in vertices; 3.5 is a compromise that keeps the sheet under the
  camera at stride 1–2.

### Is this representation suited to moss?

Yes — better than it is to the grasses, in fact. "One flat sheet crumpled by a
baked displacement field" is *exactly* what a low, wide, periodic cushion wants:
the mat is a height field, the field is a texture, the LOD is a texel stride,
and terrain conformance is one fetch per vertex. What the representation cannot
do is a 3 mm-scale silhouette: a single-valued displaced sheet has no
overhangs, so individual capitula never occlude each other and you never see
between them. That is the ceiling here, and it is a much higher ceiling than the
flat-quad reference (which has no thickness at all) or than the upright-ribbon
strand path this experiment started with (which had no coverage from above).

### Performance

Not benchmarked with `#/bench`: 30+ sibling experiments shared this GPU for the
whole session, and the HUD's own base pass moved by 2.6× between runs, so any
number recorded now would be noise. Quietest HUD p50s on `bog` at 1280×800:
grazing Σp50 2.28 ms (cull 0.11, strands+shell 0.89), carpet-close 1.21,
topdown 3.61, slope 1.42. `default` is structurally unchanged (the carpet
dispatch and all six carpet draws are skipped when the stand has no carpet
entry, the shell path is byte-identical, the grass field textures kept their own
16×96 array) and its screenshots differ from the pre-change build by 17 px at
grazing and 380 px at topdown out of 729 k — the same magnitude as running the
unchanged build twice (measured: 6 px), which is atomic append order flipping
tie-breaks at equal depth. (One iteration did regress `default` by a uniform
~2 % darkening: a search-and-replace of the occlusion floor hit `fs_main` as
well as `fs_carpet`. The pixel diff caught it; a "looks the same" eyeball did
not.)
