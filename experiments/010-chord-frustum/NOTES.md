# 010-chord-frustum — chord-field frustum proxies

## Idea

Each plant (clump) is a **convex cone-frustum proxy volume** whose INTERIOR
appearance is precomputed. The proxy is real rasterized geometry (an 8-sided
tapered prism fitted per species from the raw mesh), so the silhouette is
correct from every angle — grazing, overhead, below. The fragment shader on
the proxy surface answers "what would the eye see inside this volume" with a
single precomputed lookup:

1. Intersect the true eye ray with the ANALYTIC cone frustum in closed form
   (one quadratic + a y-slab; no marching) → the ray's **chord**: entry point,
   exit point, entry/exit faces.
2. Chart both points on the proxy surface with a (azimuth φ, meridian m)
   chart, where m runs bottom-cap-center → bottom rim → side wall → top rim →
   top-cap-center (one seamless 2D chart of the whole convex surface).
3. Look up a baked **4D chord light field**: outer atlas grid = entry bin
   (φe 48 × me 32), inner tile = chord shape (Δφ = exit−entry azimuth 32+1
   gutter column for seamless wrap × exit meridian 24). Stored per chord:
   premultiplied albedo, coverage, first-hit fraction t01 along the chord,
   oct-encoded mesh normal.
4. Runtime filtering is genuine 4D interpolation at 8 texture taps: manual
   bilinear over the 4 nearest entry bins × hardware bilinear over the inner
   chord-shape tile (the gutter column makes Δφ wrap filterable).
5. The blended t01 reconstructs a real 3D hit point → written as
   `frag_depth`, so plants inter-occlude and intersect terrain correctly;
   the baked normal drives the shared lighting.

Wind is an **inverse shear on the ray** (borrowed insight from 004): sway
displaces the proxy vertices linearly in y, and the fragment shader applies
the exact inverse shear to the eye ray before all chord math — a shear maps
lines to lines, so this is exact, not an approximation. Per-species sway from
the stand table, phase per plant, all animation from `frame.time` only.

Coverage resolve is a hard alpha test against a **world-anchored organic
pattern**: the threshold is jittered by a hash of the ~2cm quantized
plant-local reconstructed hit cell. This is NOT screen-space dither — the
pattern rides on the reconstructed surface, so it neither screen-doors nor
swims; it turns mid-coverage chord bins into stable foliage clumps instead of
solid ovals. Justification (taste rule): partial coverage IS what a chord
through a translucent volume returns; a plain hard threshold makes topdown
(α≈0.25–0.5) and grazing (α≈0.1–0.9) mutually unsatisfiable — verified both
ways in screenshots before adopting this.

Fade rules: far fade at the region edge, near fade within ~1 proxy radius,
inside fade when the geometric entry goes behind the eye (camera inside the
volume) — all multiply coverage, so plants erode organically rather than pop.

**Plant-count independence:** a per-frame compute pass enumerates the scatter
slots of a camera-bounded cell region (the `scatter_candidate` WGSL twin, so
placement is bit-identical to the stand), frustum/distance-culls, and
compacts survivors into an indirect draw. Work per frame is bounded by
regionRadius, never by stand size; nothing per-plant lives on the CPU.

## VRAM budget math (per species)

- chord surf atlas: 1584×768×4 B rgba8 = 4.87 MB
- chord geom atlas: 1584×768×4 B rgba8 = 4.87 MB
- instance buffer: the region at the max radius (80 m) is 41² cells × 128
  candidate slots = 215k slots, but a slot only EXISTS with probability
  density/8, and only existing plants are ever appended. Sized at
  `slots × density/8 × 1.15 + 4096` — over 2×10⁵ Bernoulli trials the count is
  within a few hundred of its mean, so the margin is unreachable (and the cull
  shader drops overflow gracefully). density 3 → 96,888 × 32 B = 3.10 MB
  (elymus, density 2.5 → 2.61 MB).
- proxy index buffer 192 B, info/cull uniforms + indirect: ~160 B
- Total ≈ 12.8 MB (HUD: 12.2/25 MiB per species; 11.8 for elymus). Under
  budget with room to spare — was 16.6 MB before the audit right-sized the
  instance buffer.

Bake-time transients (~130 MB G-buffer + mesh buffers) are raw allocations,
destroyed at the end of the bake; they never coexist with rendering.

## Bake

Two GPU stages per species (`bake.ts`), tens of seconds total for all three
species on this machine, cached via bakedArtifact/commitBake
(`mesh/baked/010-chord-frustum/chords-v6-<species>.bin`, 9.7 MB each):

1. **View captures**: the raw GCMESH1 mesh rasterized orthographically from a
   full-sphere 20×20 octahedral grid of directions (400 views, 256px tiles →
   5.5 mm texels; albedo+mask, oct normal, depth32). Rendered in 2 row-batches
   of 200 views to cap transient memory (~130 MB); chunked submits (25 views)
   so a 6.5M-tri mesh cannot TDR.
2. **Chord gather**: one compute thread per chord texel picks the nearest
   view for its chord direction and **marches the chord in 2-texel steps
   against that view's first-hit depth image** (bake-time only — the runtime
   never marches): a hit is the first step whose pixel's surface lies within
   a 2.5-texel tube around the chord. 8 jittered parallel chords per bin
   prefilter coverage into an honest fraction. This replaced a 3-iteration
   fixed-point parallax solve that silently failed for long near-vertical
   chords (fluffy discontinuous depth + ~9° view mismatch).

The frustum fit is per-species from the vertex cloud: per-height-band max
radius about the bounds axis, least-squares line inflated to contain all
bands (+3%). Fits: calamagrostis r0=0.53→r1=0.47 h=1.15; elymus 0.63→0.40
h=1.19; poa 0.18→0.18 h=0.77.

Gotchas hit and fixed along the way (documented for reuse):
- `sign(0.0)=0` in the octahedral ENCODE collapses straight-up/down chord
  directions onto the map center → topdown sampled the wrong views and
  rendered almost nothing. Fixed with a never-zero sign.
- 128px view tiles (1.1 cm texels) fatten sub-mm fluff into solid blobs —
  coverage saturates and the fluffy heads render as solid pink ovals. 256px
  tiles + fractional 8-chord coverage + the anchored threshold pattern fixed
  it.
- The dev server answers missing /mesh/baked files with index.html (200), so
  the committed-artifact path can hand back poisoned bytes; the loader
  validates a magic header and rebakes.

## Audit (structural waste + debug views)

Second pass over the finished method: wire the global debug views, then hunt
structural waste. No frame times are quoted — the GPU was shared with other
agents during this pass, so every fix below is justified from the code alone.

**Debug views (were entirely missing).** `render.wgsl` now includes
`src/wgsl/debug.wgsl`, applies fog only when `debug_mode() == DEBUG_OFF`, and
returns `debug_shade(color, albedo·var, n, alpha, hit_w)`. What each mode
turned out to show — the method needed no repair here, only exposure:
- *normals*: real per-fragment normals. They are the baked mesh normal at the
  chord's first hit (coverage-weighted mean of the 4 entry-bin taps), oct-decoded,
  yawed into world, flipped towards the eye. At `inside-plant` they resolve
  into coherent blade-shaped patches; at `grazing` they look like coloured
  noise, which is honest — one pixel there covers many sub-mm blades.
- *lighting*: goes through the shared `light_surface()`, once. Albedo is baked
  premultiplied by COVERAGE (not by light) and divided back out at runtime, so
  nothing is double-lit. The view is bright/clipped over most of the frame, but
  so is the terrain's — that is the shared half-lambert model, not this
  renderer.
- *albedo*: the raw baked chord colour × the anchored-noise variation actually
  used for shading (so the *lighting* view's division is exact).
- *coverage*: the resolved chord coverage after the far/near/inside fades —
  exactly the number the hard alpha test judges.
- *depth*: continuous through the field with no proxy-shaped plateaus,
  confirming `frag_depth` really is the reconstructed hit, not the proxy hull.
- `debugChart` stays an own-manifest param (chord chart coords) and only
  applies in the `off` view.

**Fixed — vertex-stage/uniform work that was done per fragment.** The
fragment shader recomputed, for every fragment, quantities that are constant
over the whole instance: the wind shear `wl`, the shear factor, and the eye
position transformed into the plant-local unsheared frame — three
`rot_y(·, ±yaw)` calls, i.e. six transcendentals per fragment, on the most
fill-bound pass in the experiment. They are now computed once in the vertex
stage and passed flat (`ray_o_shear`, `shear_cs`), with `cos/sin(yaw)` passed
down so the two remaining fragment rotations (`rot_y_cs`) cost no
transcendentals at all.

**Fixed — the proxy was drawn non-indexed.** Its 32 triangles have only 18
distinct vertices (8 bottom ring, 8 top ring, 2 fan centers), so `draw` was
shading 96 vertices per plant — 5.3× redundant. Now a 192 B index buffer +
`drawIndexedIndirect`. Bonus: the ring seam used to be `cos(2π)` on one quad
and `cos(0)` on the next (a hairline crack waiting to happen); indexing makes
them literally the same vertex.

**Fixed — per-frame uploads of values that never change.** `update()` rewrote
both uniform buffers for every stand entry every frame. The render uniform
holds only the baked frustum fit + params, so it is now written at init and
from `onParamsChanged`. The cull uniform additionally carries the region's
corner cell, which changes only when the camera crosses a 4 m cell boundary —
written on that transition. Only the indirect counter reset is genuinely
per-frame now.

**Fixed — instance buffer sized for an impossible worst case** (see VRAM math
above): 6.89 → 3.10 MB per species, total VRAM 54.3 → 43.0 MB.

**Verified unchanged:** before/after screenshots at `grazing`, `topdown`,
`far-horizon` differ in 0.15–1.2% of pixels, all isolated single pixels
scattered over the field with no structure, no silhouette shifts and no
missing plants. That is the expected last-ULP jitter of the world-anchored
alpha-test hash (`floor(hit_u·48)` flips a cell when the reconstructed hit
moves by 1e-7), caused by evaluating the ray setup in the vertex stage instead
of the fragment stage. The indexed-draw change on its own is bit-identical
(0–3 pixels of 247k). Typecheck clean, console free of WebGPU errors.

**Deliberately left alone** (each needs a bench to justify, which this session
cannot provide honestly):
- No depth prepass. `frag_depth` disables early-z, so grazing overdraw is
  shaded then late-z rejected. A depth-only proxy prepass (or front-to-back
  ordering by entry distance) is the obvious next experiment, but it doubles
  the geometry pass and only pays off if overdraw dominates — measure first.
- The cull enumerates all 128 candidate slots of every cell in the region
  square. That is forced by the shared scatter twin (existence is per-slot), so
  it cannot shrink; a per-cell early-out (cell AABB vs frustum, or cell-to-camera
  horizontal distance vs max_dist) would skip whole cells before the hash +
  terrain sample, and the square's corners alone are 21% of the threads. Left
  out because the cull pass is a rounding error next to the fragment stage and
  a sloppy conservative bound would pop plants.
- The 5 frustum planes are rebuilt per cull thread rather than uploaded as a
  uniform. Uploading them would make the cull uniform per-frame again, undoing
  the fix above, for ~40 ALU next to a hash chain plus a terrain texture fetch.
- Skipping the cull dispatch entirely while the camera and params are
  unchanged. Real win for a parked camera, but it needs every frame-uniform
  input (pose, viewport/aspect) tracked or plants pop — not worth the hidden
  dependency for a case where nothing is under load anyway.
- The bake's second gather dispatch covers the whole atlas and early-outs on
  the non-resident view rows (two batches, ~half the threads wasted each). Bake
  time only, and it keeps the batching logic trivially correct.

## Moss round: making the `bog` carpet work

The `bog` stand adds three *Sphagnum palustre* states as a CARPET: a periodic
0.18 m community tile, 0.07-0.09 m tall, on a 22x22 grid per 4 m cell (484 slots,
life size, 90°-only yaw). Chord fields for all three baked without any change to
the bake — the fits are squat cylinders (wet-vigorous r0 0.156 → r1 0.141,
h 0.093; late-season 0.175 → 0.153, h 0.070; sun-exposed 0.170 → 0.149,
h 0.074) — but the renderer got the carpet wrong in four ways. Fixes, smallest
first:

1. **Cull sphere** (2 lines): the wind margin is now 0 for a rigid species
   (`sway <= 0.01`) instead of 0.5 m — a 30x over-bound on a 9 cm mat — and the
   sphere sits at `h/2` above the node rather than `rad/2`.
2. **Alpha reference per entry** (`carpetCoverage`, default 0.20 vs the grass's
   0.45). A chord through a moss cushion returns coverage ~1 in the tile's middle
   and tails off over its last centimetre; the grass reference ate that
   centimetre and left a hairline crack lattice. `debug=coverage` at grazing is
   now white over the whole mat — a solid depth-writing occluder, no dither.
   Also: **no near fade for a carpet** (a mat you stand on must not open a hole
   under the camera), gated on `carpet_div > 0`.
3. **Slots to evaluate vs instances to store.** The cull decoded `gid.x` with
   `SCATTER_MAX_PER_CELL`, so it visited 128 of the carpet's 484 slots — which,
   because the grid index is `(i % 22, i / 22)`, was 6 of 22 ROWS: the mat
   rendered as 1 m bands of moss every 4 m (see `topdown`, before). Enumeration
   now comes from `standEntrySlots(entry)` per entry (484 for moss, 128 for the
   grasses), while capacity stays sized for survivors only — a carpet node exists
   iff this entry's wetness interval claims it (~`wetWidth`, doubled for margin
   because the wetness field is smooth over 12 m and one zone can dominate a
   region), capped at 200k tiles (6600 m² of closed mat — more than any standard
   camera sees inside a 56 m region).
4. **Terrain conformance** (`slope_align`): the cull evaluates
   `terrain_sample(node)` once per surviving instance, blends the terrain normal
   towards up by `slope_align`, and stores `up.xz` in the two spare instance
   floats; the vertex stage rebuilds the frame with `plant_basis_from_up`. All
   yaw rotations (proxy vertices, the inverse-shear ray setup, the baked normal)
   now go through that basis, so `slope_align = 0` reduces to exactly the old
   `rot_y` and the grasses are unaffected. Bog calamagrostis gets its 0.3 tilt
   for free.
5. **THE BIG ONE — a lattice tile must rotate about its own centre.** The scatter
   puts the *mesh origin* at the grid node, and for these community tiles the
   mesh origin is the CORNER of the period square (`tileOrigin (0,0)`, geometry
   spanning [-0.013, 0.197] x [-0.019, 0.215] for a 0.18 m period). Rotating
   there sends each of the four quarter turns into a different quadrant, so the
   mat lost ~3/4 of its nodes to holes and double-stacked the rest: with fix 3
   alone, `carpet-close` still showed 28% bare ground in tile-sized patches and
   the raw hulls sat on a 0.36 m lattice instead of 0.18 m. Anchoring
   `footprint_m / 2` at the node makes the quarter turns map the square onto
   itself, which is the entire reason the invariant is "90° steps". Carpet
   entries only — a scattered clump has no lattice to preserve and moving its
   rotation centre would change the default stand.
6. **Carpet lift** (`carpetLift`, default 0.4 proxy heights = 3.7 cm, along the
   conformed up axis). Not cosmetic: the base pass draws the terrain as **1 m
   quads** while placement samples the **0.5 m bilinear heightmap**, so the drawn
   surface sits ABOVE the placed height by up to 6.5 cm (>1 cm over 21% of the
   area, >2 cm over 6.8%; measured by mirroring both interpolations in TS). That
   is enough to bury a 9 cm mat: at the worst spot I could find, (3.75, -4.25),
   `carpetLift=0` shows a 0.6 m patch of bare terrain punched through the mat and
   `0.4` closes it. Harness-level mismatch, mitigated here — see Interface notes.
7. **Distance-filtered normals** (`carpetNormalMix`, default 0.85, ramped over
   0.7-1.6 m of eye distance). An entry bin covers ~1.5 cm of a cushion built
   from sub-mm leaves, so the stored first-hit normal is ONE leaf standing in for
   a whole neighbourhood. Up
   close that reads as cushion detail; past ~2 m a tile is a few pixels wide, only
   its per-tile mean survives, and because neighbours differ by a quarter turn
   the mean swings — the field broke into a hard bright/dark CHECKERBOARD at
   grazing (`debug=albedo` was smooth, `debug=lighting` was not, which is how I
   localised it). The normal is now blended towards the mat's macro normal (the
   conformed up axis) with distance — the normal-map equivalent of dropping to a
   coarser mip, and the honest limit normal of a cushion surface.

### Terrain-fitting rung, and why

**Rung 1-2 (rigid tilt from the interpolated terrain normal), deliberately.** Two
reasons, and the second is specific to this method:

- At this footprint the ladder collapses. The heightmap is 0.5 m per texel and a
  moss tile is 0.18 m, so a 5-tap `terrain_plane_fit` over the footprint returns
  the same gradient as the single bilinear tap, at 5x the cost —
  `terrain_sample`'s stored (nx, nz) is itself a 1 m central difference computed
  at heightmap build time, i.e. already a plane fit, then bilinearly interpolated.
- Per-vertex conforming (rung 3) is **incompatible with a closed-form chord**.
  The whole method rests on the eye ray meeting an *analytic* cone frustum in one
  quadratic; displacing the hull's vertices individually makes the hull
  non-analytic and there is no longer a chord to look up. Rung 4 (warping the
  query) degenerates to the same rigid tilt unless the shader iterates, because
  straightening a ray in a "terrain-flattened" space needs the local gradient and
  nothing more.
- The usual objection to a rigid per-tile fit — neighbouring tiles fit different
  planes and crack apart at the shared edge — does not bite here, because the
  proxy is a VOLUME that overlaps its neighbours by design (hull radius 0.157 m
  vs a 0.091 m half-step) and the visible surface is the depth-resolved union of
  overlapping cushions. A flat plane has no thickness to hide a mismatch in; a
  cushion does. Checked on the ridged slopes at 30° (see `bog-slope`,
  `bog-slope-close`, `bog-slope-graze`): the mat follows the hillside with no
  stepping, no floating edges and no cracks.

### What the screenshots show (bog, seed 42, det=1, t=3.5)

- `grazing`: before — moss only in 1 m bands every 4 m, i.e. ploughed furrows on
  bare terrain. After — a continuous carpet to the fade ring with the wetness
  zones legible as colour (green wet hollows, ochre flanks, bronze crests) and
  emergent calamagrostis. No lattice, no checkerboard, no dither.
- `carpet-close` (1 m down): before — 59% bare ground in tile-sized patches.
  After — closed mat, individual capitulum rosettes visible, tile seams faintly
  readable as brightness steps. Softer than 001-billboard-smoke's crop-corrected
  top view (its texels are sub-mm, my entry bins are ~1.5 cm) but with real
  relief instead of a flat plane.
- `topdown`: continuous mat out to the region ring (still a hard ring — the
  method has no distant-collapse layer, unchanged from before).
- `inside-plant`: no hole under the camera; the mat reads as ground you are
  standing on.
- `debug=depth`: smooth continuous gradient, no proxy-shaped plateaus — the
  reconstructed hit is a real surface.

### Still bad

- **Per-tile "quilt" in the ramp band (~0.7-2 m).** Wherever the normal filter is
  only partly on, tiles still differ in mean brightness; the ramp is a taste
  trade (detail at 1 m vs artifact at 2 m) and I settled on 0.7-1.6 m, which
  leaves a couple of faintly darker tiles at 1.5-3 m. Root cause is bake
  filtering, not the runtime: the gather prefilters each bin with 8 sub-chords
  over a 5 mm disc inside a 13-21 mm bin, and sums raw two-sided leaf normals
  (front/back faces partially cancel, so `normalize` amplifies what is left).
  Flipping each sample into the hemisphere facing the chord before averaging and
  widening the jitter to the bin footprint is the fix; it needs a moss-only bake
  variant (~2 min/species) and was left for the next pass. Diagnosis is
  reproducible: at `carpetNormalMix=1` the `normals` view is uniform past the
  ramp and still varies per tile inside it, while `albedo` is smooth everywhere —
  so the checkerboard was never in the stored colour.
- **Resolution ceiling — the budget split is wrong for a cushion.** The atlas
  spends 48x32 = 1536 entry bins on the whole hull surface and 32x24 = 768 chord
  *shapes* (view directions) inside each. For an upright 1.15 m plant that is
  sane; for a 0.18 m cushion it starves exactly the axis you look at. On the moss
  fit the entry bins land at 1.3-2.1 cm on the top cap — about one bin per
  capitulum — which is why close-ups are soft. Re-cutting the same 1584x768 bytes
  as 96x48 entry x 16x16 exit (same VRAM, 3x the entry bins, 1.7x linear, ~7-9 mm
  bins ≈ 22x22 per tile footprint) is the obvious next step: a low cushion has
  weak view-dependence (3.3 cm of relief), so 256 view samples are plenty. It
  needs per-species atlas dims in the shaders (they are compile-time consts
  today) plus a moss-only re-bake, so it did not fit in this pass. Same re-bake
  should clamp a carpet's fit radius to `tileM/sqrt(2)` (0.131 vs 0.156 mesh
  units): it covers the cell diagonal exactly, cuts the 2.5x proxy overlap to
  1.5x, and buys another 1.2x of bin density for free.
- **Cost.** Drawing the whole mat is ~4x the fragments the striped version was
  drawing, and `frag_depth` disables early-z, so the bog stand is fill-bound at
  grazing. No numbers quoted: the GPU was shared with ~30 sibling agents for this
  entire session (the same `default`/grazing frame measured 15.4 ms and 31.4 ms
  40 minutes apart). Bench on an idle GPU before believing anything.
- The three moss entries each enumerate the same 484 nodes and evaluate the same
  wetness, differing only in which interval accepts — 3x redundant work in the
  cull (see Interface notes).

## Status

**working** — the `bog` carpet renders as a closed, terrain-conforming mat and
the `default` stand is unchanged: before/after pixel diffs at `grazing`,
`topdown` and `inside-plant` differ in 0.011-0.078% of pixels, all isolated
single pixels (worst row 11 of 1280), which is the documented last-ULP jitter of
the world-anchored alpha-test hash re-evaluated through the new basis — no
silhouette shifts, no missing plants, VRAM identical at 12.2/11.8 MB per grass
species. Bog VRAM is 15.4 MB per moss species (2 x 4.87 MB atlas + 6.4 MB
instance list). Typecheck clean, no console errors, no validation toasts. Timing
HUD numbers were contended all session, so no perf claims; bench on an idle GPU
(`#/bench/010-chord-frustum?stand=bog&spline=orbit-low`) before quoting.

### Interface feedback (harness)

- **The drawn terrain and the placement height disagree by up to 6.5 cm.**
  `terrain-draw.wgsl` uses 256 quads over 256 m (1 m triangles) while placement
  and every renderer's conforming code use the 0.5 m bilinear heightmap. For
  1 m-tall grass that is invisible; for a 7-9 cm mat it buries the mat over
  ~7% of the area. Every carpet renderer will have to invent its own lift. Either
  tessellate the base terrain to the heightmap (512 quads) or state the maximum
  disagreement in the contract so lifts can be principled.
- **A carpet's zone partition is evaluated once per competing entry.** With three
  carpet entries the cull runs 3 x 484 threads per cell and 2/3 of them do a
  wetness evaluation (hash chain + terrain tap) only to discard. A
  `scatter_carpet_owner(seed, cell, i) -> entry_index` primitive (or having
  `scatter_candidate` return the winning entry for a node) would make one
  dispatch enough and would also let a renderer batch all carpet species into one
  instance list.
- **`footprint_m` is documented as "the horizontal footprint", but for a lattice
  species it is also the ROTATION CENTRE offset**, since the mesh origin is the
  tile corner. That is the single most expensive thing I got wrong, and nothing
  in `CLAUDE.md` or the stand table says it: the lattice rule says "rotation only
  in 90° steps" without saying "about the tile centre", which is what actually
  makes quarter turns lattice-preserving. Worth one sentence in the carpet
  contract, or a `plant_tile_basis(node, yaw, footprint)` helper.
- `standEntrySlots()` is exactly the right primitive and the warnings about it
  are accurate — but the WGSL side has no twin. `SCATTER_MAX_PER_CELL` is a const
  in `scatter.wgsl` and is the natural thing to divide `gid.x` by; a
  `stand_entry_slots(entry_index)` helper next to `scatter_candidate` would close
  the trap where it is actually sprung.
- No way to ask the harness for a carpet's grid step other than recomputing
  `SCATTER_CELL_SIZE / carpet_div`. Fine, but `step` and `scale` are the two
  numbers every carpet renderer needs and both are derived rather than given.

## Findings

- The two-point (entry, exit) chord parameterization on a proxy-native chart
  WORKS as the loophole: runtime is one quadratic + 8 filtered taps, zero
  marching, and depth/normals come along for free. The gutter-column trick
  makes the wrap-around azimuth axis hardware-filterable.
- Rendering the proxy's FAR side (cull front) is strictly better than the
  near side: no near-plane clipping, and camera-inside-plant still shades,
  which makes the inside fade trivial.
- Chord fields integrate THROUGH the volume, so overlapping fluffy-head
  regions read denser/pinker than surface impostors (005) at grazing — more
  physically honest, but the bin-mean color smears fine detail; the anchored
  noise buys back texture but not structure. Entry bins are ~5cm — close-ups
  (<1.5 m) stay soft/blobby by construction.
- Coverage semantics are the crux of the method: what "a ray hits" means for
  a 5.5 mm-sampled proxy of sub-mm geometry decides the whole look. The
  effective ray thickness (view texel + tube radius) inflates optical density
  relative to ground truth; a principled transmittance correction (bake true
  per-texel alpha at higher supersampling and composite T = Πα along the
  march) is the clear next step.
- Known artifacts, honest list: (a) mid-alpha cap chords can survive while
  side chords erode → occasional detached "cap flakes" against the sky in
  upward views; (b) plants pop in/out at the regionRadius ring (far-horizon
  shows bare hills beyond ~56 m — no distant-collapse layer); (c) overdraw at
  grazing is real (frag_depth disables early-z; late-z still culls after
  shading) — a depth-only proxy prepass or rough front-to-back entry ordering
  would help; (d) topdown reads sparser than intuition — measured top-down
  pixel coverage of these meshes at 5.5 mm really is ~0.25, but the hard
  threshold thins it further.
- No bench JSONs recorded: frame times were contaminated by parallel agent
  GPU load throughout this session (see Status).
- Species coverage: all six species baked and rendered — the three grasses
  (calamagrostis-canescens, elymus-repens, poa-pratensis) and the three Sphagnum
  states. A grass tile mesh is still treated as one clump per instance (same
  convention as the other impostor experiments); a CARPET tile is anchored by its
  tile centre so the lattice survives quarter turns (see the moss section).
- **Is a chord-field frustum proxy suited to a low cushion?** Partly, and for one
  specific reason it is better than a card: the proxy is a real volume with real
  reconstructed depth, so the mat has thickness — a silhouette at grazing, tiles
  that inter-occlude, capitula that read as bumps rather than as texture painted
  on the ground. That is exactly what 001-billboard-smoke's flat quad cannot do.
  What the method does badly here is DETAIL DENSITY: its resolution lives in the
  entry-bin chart over the hull surface (1536 bins for the whole cushion), not in
  a texture, so it cannot approach an atlas's sub-mm texels — a 0.18 m tile gets
  ~8-13 distinguishable bins across, and closer than ~1 m it is visibly soft.
  A cushion is also the worst case for its per-bin normal (one leaf standing in
  for 1.5 cm of a surface you view nearly along its normal), which is why the
  normals have to be filtered with distance. Verdict: it renders moss *decently*
  and honestly — better silhouette and depth than a card, worse texture — and the
  headroom that is left is a budget re-cut, not a rewrite.
