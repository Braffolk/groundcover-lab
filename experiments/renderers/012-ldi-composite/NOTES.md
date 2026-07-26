# 012 LDI composite

## Idea

Layered depth images as a groundcover representation. Per species, the source
mesh is baked ONCE into 5 key capture directions (4 side views at azimuth
0/90/180/270 deg, elevation 20 deg, plus one straight-down top view), each
depth-peeled into 4 layers. A layer stores albedo+coverage and oct-normal +
16-bit linear capture depth. At runtime a compute pass evaluates the scatter
WGSL twin over a bounded camera-centred cell window (regionRadius), frustum
culls, and compacts survivors into one instance array consumed by two indirect
draws:

- NEAR (< layerCullDist): 12 cards per plant — the 2 side stacks nearest the
  camera azimuth + the top stack, x4 layers. Cards are FIXED to the capture
  basis at each layer's mean baked depth; fragments write true per-texel baked
  depth via frag_depth, so the union of stacks composites through the plain
  depth test. No sorting, no blending, no cross-fade weights: an off-axis
  stack foreshortens toward invisibility, which IS the view weighting (side
  cards vanish from above exactly as top cards turn face-on). A one-tap
  parallax reprojection (sample own depth once, shift UV, resample — a
  precomputed lookup, not marching) tightens texels within 30 m.
- FAR: 2 cards (front peel layer of nearest side + top stack), rasterizer
  depth only — no frag_depth, so early-z eats the overdraw where the field is
  deepest. Near draws first to prime the depth buffer. The cull pass compacts
  far survivors downward from the top of the shared instance array while near
  fills upward from 0; the descending mapping is applied in `vs_far`, so both
  indirect draws keep `firstInstance = 0` (see Audit).

Per-frame cost is bounded by regionRadius (cells around the camera) and screen
coverage — the stand's total plant count never enters any loop: 557k (default)
and 134.2M (scaling-100m) issue byte-identical work (verified, see Findings).
No source geometry is touched after the bake. Wind = shared `wind_sway` on
card corners, roots pinned, per-species stand sway. Lighting = baked normals
yaw-rotated, two-sided flip, shared `light_surface` + fog. Camera inside a
plant: coverage-threshold fade over 0.4–0.95 of the plant's bounding radius.

Carpet species (`stand_table[i].carpet_div > 0`, i.e. the bog's Sphagnum) take a
different shape out of the SAME bake: one ground-parallel tile-sized quad per
grid node, out of the top capture alone, with the baked depth written per texel
so the mat has real cushion relief. See "Moss carpet" at the end.

### Where LDIs break for thin foliage (and the fixes that landed)

1. **Naive depth peeling wastes layers.** The first 3–4 surfaces under a pixel
   of grass are often the same tuft 2 mm apart; the crown interior never gets
   captured and other views show a hollow shell. Fix: peeling with a MINIMUM
   SEPARATION (~3 cm, `0.04 * boundR`): layer L keeps the nearest surface at
   least that far behind layer L-1, forcing successive layers through the
   crown.
2. **Bilinear taps at silhouettes blend into nothing.** Empty texels hold
   black albedo and FAR depth, so edge samples smear dark fringes and scatter
   "beads" of geometry behind the plant (dramatic close up). Fix: bake-time
   dilation — flood albedo/normal/depth (frontmost neighbour wins) 6 texels
   into empty space, alpha stays 0. This one change took close-range quality
   from broken to usable.
3. **Layer gaps off-axis.** Between two capture azimuths both stacks are ~45
   deg foreshortened and inter-layer space opens up. Fix: draw the stack UNION
   with per-texel depth (they interleave correctly in the depth buffer), plus
   the one-tap parallax reprojection near the camera.

## VRAM budget math

### Specimen species (grasses) — unchanged

Per species (HUD-confirmed 17.1 MB for density-3 entries, 15.9 MB for 2.5):

- atlas0 (albedo+coverage) 1280x1024 rgba8: 5.24 MB
- atlas1 (oct normal + depth16) 1280x1024 rgba8: 5.24 MB
- instance array, worst-case regionRadius 128: 65^2 cells x 128 slots x
  density/8 x 1.15 margin x 32 B = 6.9 MB at density 3
- 2 indirect + uniform: < 1 KB

Total 17.4 MB at density 3 — under the 25 MB budget (would exceed only at
density 8 + radius 128, which no stand uses). Bake-time transients (mesh
buffers up to 214 MB for poa, strips, readbacks) live outside `ctx.res` and
are destroyed after the bake — they trip the expected "untracked" dev warning
once per fresh bake.

### Carpet species (Sphagnum) — 29.0 MB before, 5.8 MB now

The budget was in completely the wrong place for a mat. HUD, bog stand, per
moss species:

| | before | now |
|---|---|---|
| 5x4 LDI atlas pair (never sampled by a mat) | 10.5 MB | 0 |
| tile textures: 2 layers x (albedo + aux) 256^2 rgba8 + mips | 0 | 1.4 MB |
| instances | 19.9 MB (622k x 32 B) | 4.65 MB (1.16M x 4 B) |
| **total** | **29.0 MB (over budget, HUD-striped)** | **5.8 MB** |

- The atlas is dropped for a species this stand only uses as a carpet: 18 of
  its 20 tiles are side views of a 7 cm cushion, and the carpet path samples
  none of them.
- The instance record is 4 bytes (9+9+9+2 bits: window cell dx, dz, grid slot,
  yaw quadrant) because a life-size mat has 484 slots per 4 m cell, so the
  instance array — not the imagery — is what a carpet spends VRAM on. At 4 B the
  worst case is affordable *exactly*: `min(regionWindow, standSpan)^2 x div^2`
  = 49^2 x 484 tiles on the bog, so capacity overflow is impossible and the mat
  can never show capacity holes that look like a placement bug.
- Grasses on the bog stand are untouched (13.3 MB calamagrostis, 11.2 MB poa).

## Bake

`bake.ts`: 4 render passes (one per peel layer), each drawing the full GCMESH1
mesh 5x (one viewport per capture direction) with the previous layer's depth
strip as peel input; min-separation discard in the fragment shader; ortho
framing from bbox projection per direction. Readback, CPU dilation (see
above), per-(dir,layer) mean-depth stats into the header, packed as
`<species>-ldi-v2.bin` (10.5 MB) and committed via `commitBake` — in
`mesh/baked/012-ldi-composite/`. Fresh bake ~2–4 s per species in-browser;
`bakedArtifact()` was bypassed because the dev server 200-serves index.html
for missing files (validated magic instead).

## Status

working — verified by headless screenshots at grazing, topdown, inside-plant,
far-horizon (scaling-100m), plus A/B vs 000-ground-truth (reference patch).
Carpet path (bog stand) verified separately — see "Moss carpet" below.
Audited since (see Audit): debug views wired, dead far-draw path repaired,
per-frame uniform traffic and a retained host-memory copy of the atlases
removed. Re-verified by screenshot at grazing/topdown/inside-plant/far-horizon
in every debug mode, console clean.

## Findings

- **All-angle behaviour**: grazing and topdown both read as a dense meadow
  with per-species color character; the same bake serves both because the top
  stack takes over exactly as the side stacks foreshorten out. Inside-plant is
  legible grass (blades, not billboards) with the documented fade.
- **Plant-count independence**: default (557k) vs scaling-100m (134.2M) show
  the same ldi-layers cost envelope — identical dispatch + indirect work by
  construction. Screenshots confirm no per-plant setup exists anywhere.
- **Timings are contended** (15 sibling agents on this GPU while developing)
  — ldi-layers read 30–45 ms p50 with the base pass itself inflated 10x, so I
  do NOT quote absolute numbers; no bench JSON is claimed. Structure-level
  perf work done regardless: near/far split (frag_depth only near, early-z
  far), foreshortening cull (edge-on cards degenerate beyond 12 m), parallax
  gated to 30 m, compute-side compaction so vertex work only runs on real
  plants. Re-bench solo before quoting.
- **Known artifacts, honestly**: close-up (<1.5 m) calamagrostis heads go
  "melted candy" — 6 mm texels + dilation smear; region-edge plants erode
  per-card rather than dissolving smoothly (binary alpha + threshold fade);
  the field ends at regionRadius (96 m default) with fog only partly covering
  the cutoff on long sightlines; frag_depth in the near pass disables early-z
  there (the price of true layer interleaving).
- **Not stochastic**: hard alpha test everywhere, opaque depth writes; fades
  are coverage-threshold based, no dither.
- A/B: `#/ab/000-ground-truth/012-ldi-composite?stand=default&cam=grazing&seed=42`
  (A is a reference patch and ignores the stand — compare plant character at
  the patch, not field density).

## Debug views

Both fragment stages route through `debug_shade()` (`src/wgsl/debug.wgsl`) and
apply fog only when `debug_mode() == DEBUG_OFF`, so `view=` /
`debug=albedo|normals|lighting|coverage|depth` work here:

- **normals** — the baked oct normal decoded from atlas1.rg, yaw-rotated into
  world space and flipped toward the viewer (thin foliage is two-sided). Real
  per-texel foliage normals, not a card normal: grazing reads as dense
  per-blade noise, topdown as mostly +y green with blade speckle, which is
  what a 6 mm-texel capture of a 2M-tri tuft should look like.
- **lighting** — `shaded / albedo`, i.e. the shared `light_surface()` term
  alone. It saturates to white over most of the frame, but so does the
  harness terrain at this sun/ambient (checked side by side against
  `000-ground-truth`'s terrain in the same view), so this is the shared
  model's range, not a skipped or doubled lighting term; the dark speckle is
  foliage whose normal faces away from the sun.
- **coverage** — `baked alpha * fade`, exactly the value the alpha test uses:
  binary white inside foliage with a grey bilinear rim at silhouettes and the
  smooth ramps of the inside-plant / region-edge fades. No dither anywhere.
- **depth** — reconstructed from `world_texel`, i.e. the parallax-corrected
  per-texel position that also drives `frag_depth`, so this view is a direct
  read of what the depth buffer receives.

Method-specific state is a param of THIS experiment (`inspect`), applied only
when the global view is off: `dir` colours by which of the 5 baked capture
directions won the fragment (top capture = purple, essentially absent at
grazing — the foreshortening weighting working), `layer` by peel layer, and
`path` by draw path (orange = near 12-card frag_depth, blue = far 2-card
early-z). `path` is the fastest way to see what `layerCullDist` is buying.

## Audit

Structural/debug pass over the first-generation code. Found:

1. **The FAR draw never executed at all.** `far_args` was seeded with
   `firstInstance = capacity` and the cull pass walked it down with
   `atomicSub`. WebGPU only allows a non-zero `firstInstance` in an indirect
   draw with the optional `indirect-first-instance` feature (the harness does
   not request it), so the driver silently no-ops the whole draw. Everything
   past `layerCullDist` was therefore simply missing: with
   `p.layerCullDist=10` the field ended 10 m from the camera and the rest was
   bare terrain. **Fixed** by keeping the descending compaction but expressing
   it on the shader side (`slot = capacity - 1 - instance_index` in `vs_far`,
   `slot = capacity - 1 - atomicAdd(far_count)` in the cull), leaving
   `firstInstance = 0`. This deletes the `farTemplate` buffer and its
   per-frame 16-byte `copyBufferToBuffer` as a side effect (the far args are
   now init-written once and only the instance count is cleared per encode,
   exactly like the near args). **This one legitimately changes the image** —
   the 70–96 m band of the field now renders instead of being empty.
2. **The whole 400-byte uniform block was rewritten every frame, per entry**,
   including 80 floats of bake-time capture bases and the stand/atlas
   constants, and it rebuilt them through `[...d.right, d.halfW, ...]` array
   spreads (15 throwaway arrays per frame). **Fixed**: the block is now split
   into a static half written once at create time and a contiguous dynamic
   pair (`region`, `tune` — camera cell window + the four params), so the
   per-frame traffic is one 32-byte `writeBuffer` per entry from a preallocated
   scratch and zero allocation. Same values reach the GPU; no image change.
3. **~31 MB of host memory pinned for nothing.** `SpeciesGpu` held the whole
   unpacked `LdiBaked`, whose `atlas0`/`atlas1` are views into the 10.5 MB
   packed artifact, for the life of the session — although the bytes are dead
   the moment `writeTexture` has run. **Fixed**: only the header (`LdiMeta`)
   is retained. Not VRAM, so the HUD never showed it.
4. Debug views were not wired at all (the plants ignored `view=`). **Fixed**,
   see above. The existing shading was otherwise sound — real decoded baked
   normals, one pass through the shared `light_surface()`, no premultiplied
   light in the atlas — so the normal view is unchanged by the debug work.

Deliberately left alone:

- **Cell-level frustum/circle rejection in the cull pass.** The dispatch is a
  square cell window culled to a circle, so ~21% of the cells cannot produce a
  survivor, and off-frustum cells are the majority at grazing. But `ldi-cull`
  is two orders of magnitude cheaper than `ldi-layers` on every camera I
  looked at, so this optimises the wrong pass and would need a conservative
  per-cell height bound to stay exact. Not worth the risk.
- **Instance capacity sized for `regionRadius = 128` regardless of the current
  value** (6.9 MB of the 17.4 MB budget at density 3, ~55% unused at the
  default radius 96). `regionRadius` is a live param the harness mutates in
  place without recreating the experiment, so the worst case has to be
  allocated up front. Correct as it stands.
- **Per-fragment `uni.dirs[dir_i]` loads and the `uni.atlas` divisions in
  `atlas_uv_for`.** Uniform across the primitive and foldable, but that is
  ALU-level, not structural.

Suggestions for a later (measured) pass, not done here:

- With the far path alive again, `layerCullDist = 70` looks far too generous:
  `inspect=path` at grazing shows ~95% of the screen on the 12-card
  frag_depth path, which is the pass that has no early-z. Re-tune it solo.
- `inspect=layer` shows peel layer 0 winning almost every visible pixel in a
  dense stand; layers 1–3 survive only in the gaps. A distance-graded layer
  count (4 layers up close, 2 mid, 1 far) instead of today's binary near/far
  switch is the obvious next structural win, but it changes the image and
  needs a bench + a look in motion.

## Moss carpet (bog stand)

*Sphagnum palustre* is a periodic **0.18 m community tile, 0.07-0.09 m tall**,
laid out by the `bog` stand as a grid-snapped mat: 22x22 tiles per 4 m scatter
cell (484 slots), constant scale 1.0101 (life size), 90-degree-only yaw. Two of
the LDI's assumptions are wrong for that shape, and one is exactly right:

- **wrong**: the plant is a specimen centred on its bbox centre, and the four
  20-degree-elevation side captures carry the silhouette. For a 7 cm cushion the
  side captures are slivers, and the bbox centre is *not* the tile centre.
- **right**: the straight-down capture stores albedo + coverage + per-texel
  normal + **per-texel capture depth** at 0.84 mm/texel. For a mat that is not
  an impostor view, it is a **displacement map of the cushion surface** — and
  displacement is the one thing 001-billboard-smoke's flat card cannot express.

So carpet entries get their own cull + render path (`shaders/carpet_cull.wgsl`,
`shaders/carpet.wgsl`, `carpet.ts`) and the specimen path is left bit-for-bit
alone. What changed, smallest first:

1. **Evaluate all 484 slots, not 128.** `cull.wgsl` is one 128-thread workgroup
   per cell, so on a carpet it visited slots 0..127 of 484: the mat rendered as
   **1.1 m stripes of moss with 2.9 m of bare peat between them** (see the
   before screenshots — at grazing it was corduroy). The carpet cull walks the
   remaining slots in the dispatch's z dimension (`ceil(div^2 / 128)` workgroups
   per cell) and drives everything from `carpet_div^2` / `standEntrySlots()`.
   Biggest single win in this whole task, and it is four lines.
2. **One ground-parallel quad per tile, centred on the TILE centre.** Instead of
   12 cards fixed to the capture bases, a mat tile is one quad exactly
   `footprint_m * scale` across (= the grid step), so tiles abut exactly. The
   rotation centre matters: the mesh's tile spans `[0, tileM]^2`, so the tile
   centre is `(tileM/2, tileM/2)`, while the specimen path rotates about the
   *bbox* centre — which for this mesh is 0.13 m from the node, so every tile
   used to orbit its own grid node by 0.13 m under the 90-degree yaw. That alone
   guarantees gaps and double-cover on a 0.18 m grid.
3. **Terrain fitting: rung 3, per vertex.** Each corner gets its own
   `terrain_sample(xz)` — height and (nx, nz) from one bilinear fetch, so the
   shading basis is free. Rung 3 rather than 1 or 2 deliberately: the quad is
   exactly one grid step, so neighbouring tiles *share* corner positions and a
   per-vertex fit is the only rung that keeps the mat C0-continuous. A per-tile
   plane fit (point normal or `terrain_plane_fit`) is not cheaper here, it is
   wrong — two neighbours pick two different planes and crack at their shared
   edge. The baked normal is then lifted into the ground basis
   (`plant_basis_from_up(up, yaw)`), so a mat on a slope lights as a slope.
   `debug=depth` across the ridged slopes shows one continuous surface with no
   per-tile steps.
4. **A load-time carpet tile texture** (`carpet.ts`, no re-bake, no bake-version
   bump — the moss meshes are 479 MB each):
   - cropped to the tile's own square in the mesh frame;
   - **periodic wrap-around composite**: each output texel takes the frontmost
     (min capture depth) of the 9 samples at `(x + dx*tileM, z + dz*tileM)`, so
     the overhangs that a single-tile capture cuts off come back and the result
     is exactly periodic. `repeat` addressing then makes it seamless;
   - **a mip chain**, which the 5x4 atlas cannot have. Without it a 0.18 m tile
     at 40 m sampled one texel of 65k and the mat was shimmering speckle. `rgb`
     coverage-weighted (already normalised — the shader must not divide by alpha
     again), `a` the plain mean (true fractional coverage), depth
     coverage-weighted, normals averaged as decoded unit vectors;
   - sampled **anisotropically**. A mat is the textbook anisotropic case: with a
     max-axis explicit LOD the whole near field at `cam=grazing` went one flat
     mustard colour, because a ground-parallel quad at grazing has a uv
     derivative tens of times larger along the view direction than across it.
5. **Normals flipped into the capture hemisphere before mip averaging.** The
   bake stores raw two-sided mesh normals, so a leaf's front and back faces
   store near-opposite vectors; averaging them cancels and `normalize()` then
   amplifies the sideways residue. Symptom (caught in `debug=normals`, screenshot
   `after1-bog-normals`): whole tiles with horizontal normals in four groups
   matching the four yaw quadrants, and the mat lit as a patchwork of bright and
   dark squares. Averaging decoded unit vectors is not enough on its own — this
   is the octahedral-mip trap in CLAUDE.md with one extra step.
6. **Thickness: per-texel baked depth into `frag_depth`,** plus the one-tap
   parallax reprojection and, inside `carpetNear` (10 m), the second peel layer
   under the first so the gaps between capitula show interior moss instead of
   peat. This is the part that is *not* a billboard.
7. **Budget re-allocated** (see VRAM section): 29.0 -> 5.8 MB per moss species.
8. New params, all render-only: `carpetNear`, `carpetAlpha` (0.06, a mat must
   not dissolve — coverage is 93-98% at level 0, so the mat is a solid
   depth-writing occluder; `debug=coverage` at grazing is white with sparse
   specks, no dither anywhere), `carpetOverscale`, `carpetSharpen`.

No camera-inside fade for a mat (verified at `inside-plant`: the mat is
continuous under the camera, no hole). No wind (the stand gives Sphagnum
sway = 0, and a carpet must not sway).

### Measurements

- **Overscale**: 1.0 by default. At 1.15 the mat shows *no* dark lattice here —
  unlike 001's flat plane, the tile texture is periodic (so the overscaled
  fringe samples valid neighbour content) and `frag_depth` resolves the overlap
  by height instead of painting an edge. But at 1.0 there are no gaps to fix, so
  overscale only buys ~32% more fragment work. Left at 1.0.
- **Mip bias** (`carpetSharpen`, default -1): a flat card asks for the mip level
  a *flat* surface would need, which over-blurs a relief surface at oblique
  angles (the bumps facing the camera are not foreshortened). 0 is visibly
  over-blurred beyond ~8 m, -2 recovers grain out to ~15 m but is noisy and
  measurably heavier on the texture cache. -1 is the compromise. **Caveat: this
  is a static-screenshot judgement — a negative bias undersamples and can
  shimmer in motion, which headless screenshots cannot show. 0 is the safe
  setting if it does.**
- **Does the thickness show?** Yes, but only up close. `p.carpetNear=0`
  (flat cards, no parallax, no second layer) vs the default at 0.3 m eye height:
  the flat version is an even fine-grained speckle, the relief version has
  blotchy light/dark structure that reads as cushion bumps. By ~5 m the two are
  indistinguishable — the cushion's relief is 1.5-3.3 cm, which is a couple of
  pixels at that range. Honest conclusion: per-texel depth is worth its
  frag_depth cost only inside `carpetNear`, which is why the far path drops it.
- **Timings are contended** (a dozen sibling agents on this GPU): bog Σp50 read
  3.6-13.5 ms across runs where the base terrain pass alone varied 0.5-2.1 ms,
  so no numbers are claimed. Structurally the mat now draws 4x as many tiles as
  the striped version did and costs *less*, because a tile is 1 quad (6 verts)
  instead of 12 cards (72 verts): `ldi-carpet` sat at 1.8-4.4 ms p50 vs
  `ldi-layers` 5.0-6.5 ms for the striped moss before. Re-bench solo before
  quoting.

### What improved / what is still bad

Screenshots (before and after, same cameras): `bog` at grazing / carpet-close /
topdown / inside-plant / a 0.3 m "moss-eye" view / two views across the 23-degree
ridge at (32, -52) / `debug=normals|lighting|coverage|depth|albedo`, plus A/B vs
001-billboard-smoke at carpet-close, moss-eye and grazing.

- **Improved, massively**: the mat is *closed*. Before: 1.1 m stripes of moss on
  bare peat at every camera (the single worst artifact in the experiment). After:
  continuous cover at every distance, correct 0.18 m grid, no seams at
  carpet-close, and at topdown the three micro-habitat states read as smooth
  interlocking zones instead of striped confetti.
- **Improved**: no per-tile lighting patchwork (normal hemisphere fix); the mat
  follows the ridged slopes exactly with no buried or floating edges and no
  cracks between tiles; real per-texel normals in `debug=normals` (up-dominant
  green with moss-scale noise, not a flat up-normal); the moss is inside its VRAM
  budget with room to spare; cushion relief is visible within ~2 m.
- **Still bad #1 — the mat flattens with distance.** Beyond ~10 m it is a
  smooth coloured surface with the terrain's shading and no moss structure. Part
  of this is correct (a 0.18 m tile is 5 px at 25 m, and a diffuse rough surface
  really does average out), but part is the flat card: it cannot show the near
  faces of bumps un-foreshortened at grazing, which is what would keep a real
  cushion legible. `carpetSharpen` mitigates, it does not fix. A proper fix is
  parallax-occlusion marching along the view ray, which the project's O(1)/no-
  raymarching rule excludes, or a shell/prism representation.
- **Still bad #2 — the moss reads as khaki/mustard at range.** Confirmed to be
  the baked albedo, not the lighting: `debug=albedo` A/B against
  001-billboard-smoke matches almost exactly, and the source tile's mean colour
  really is (79, 65, 14) for the sun-exposed state. Up close the lit result is
  darker/browner than 001, because gaps in the cushion show the dark interior
  peel layer instead of bare terrain, and because the normals are real per-texel
  normals rather than an up-normal. I believe both differences are in my favour
  physically; they do not look prettier.
- **Still bad #3 — zone boundaries are blocky at distance.** The wetness
  partition is resolved per grid node, so at 20-40 m the boundary between two
  moss states is a 0.18 m staircase. Nothing here can fix that without breaking
  the placement contract.
- **Not done**: distance aggregation. The mat draws one quad per grid node out to
  `regionRadius` (~292k tiles per moss entry at 96 m), where a K x K aggregate
  card with wrapped uv would be exact for a periodic tile (all four rotations of
  a periodic tile are identical at 2 px). The three carpet entries also each
  evaluate the same wetness field at the same node, so `carpet-cull` does 3x the
  necessary work. Both are structural wins left on the table; neither changes
  the image.

### The `default` stand is unchanged

The carpet path only exists for entries with `carpet_div > 0`, so on `default`
neither carpet pass is even encoded and the specimen code is byte-identical
(`ldi.wgsl` and `cull.wgsl` were not touched). Verified rather than argued:
masked pixel diffs of before vs after, same seed and `det=1&t=3`, HUD/panel/
toolbar excluded — `cam=inside-plant` differs in **0 of 733,700** scene pixels,
`cam=grazing` in **7** (isolated alpha-test coin flips at silhouettes; 001
measured 6 for a same-code control run). VRAM identical: 17.1 / 15.9 / 17.1 MB.
GPU timings on `default` are not quotable this session — the same code read
27 ms p50 for `ldi-layers` at the start of the session and 62 ms an hour later
with a dozen agents sharing the GPU; the image identity is the real evidence.

### Is an LDI the right representation for a mat?

Partly. Honest verdict: **the top capture's per-texel depth is genuinely the
right asset for a cushion — the rest of the LDI is dead weight for it.** The
method's own machinery (multi-view stacks, azimuth selection, per-layer mean
planes, foreshortening weights) exists to give a specimen a silhouette from all
angles, and a mat has no silhouette worth capturing: 18 of 20 baked tiles are
never sampled. What survives is exactly the part that makes an LDI more than a
billboard — depth per texel — and it buys real cushion structure within a couple
of metres, plus correct interpenetration with the grass and the ground. Beyond
that the mat degenerates to what 001 already is: a very good moss texture on a
terrain-conforming surface. So: better than the billboard baseline up close, tied
with it at range, at a fifth of its VRAM. Not a wrong representation for moss —
just a mostly-unused one, and a carpet-only bake (one top-view tile, no side
views, at 2x the resolution) would be the honest form of this method for a mat.
