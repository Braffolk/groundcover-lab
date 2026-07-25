# 006 fin cards — trig-weighted crossed fins

## Idea

The ancient crossed-cards tree trick, pushed until its two classic artifacts
(edge-on fins vanishing into lines; the star-shaped crossing seen from above)
disappear. Every plant is **5 static quads in world space**:

- 3 vertical fins crossed at 60° through the plant axis (rotated by the
  per-plant yaw), and
- 2 horizontal slab cards at ~20% and ~62% of plant height.

Nothing billboards. Because the geometry is static, parallax, intersections,
inter-plant occlusion and motion stability are real 3D — no card rotation, no
frame-to-frame swimming, depth-write on everywhere (no sorting).

Three tricks carry the technique:

1. **Per-card content specialization.** The bake produces 8 orthographic
   views: 6 azimuthal captures every 60° plus 2 top-down captures of the
   lower/upper *vertical slab* of the plant. Fin `k` shows azimuth view `k`
   from its front side and view `k+3` (U-mirrored) from its back side — 6
   distinct views live on 3 static quads, so orbiting a plant changes real
   content instead of sliding one poster around. The two horizontal cards show
   different geometry (below/above the 42% height split), so looking down
   yields genuine inter-card parallax instead of a flat decal.
2. **Trigonometric view weighting.** Per card, opacity is
   `smoothstep(0.03, edgeFade, |cos(view, card normal)|)` — a fin dissolves
   exactly as it approaches edge-on, when the other two fins are near their
   best angle, so coverage holds through the transition. Additionally all
   vertical fins are killed by a `smoothstep(topBlend, 0.98, viewDir.y)` as
   the view goes overhead — that removes the star crossing — while the slab
   cards fade in with elevation (`smoothstep(0.05, 0.28, |viewDir.y|)`).
3. **Hashed-alpha dissolve.** All partial opacities go through an interleaved
   gradient-noise alpha test (decorrelated per card via a phase/yaw seed), so
   the crossfades are order-independent with full depth writes — no OIT, no
   sorting, stable in stills.

Per-frame cost is plant-count independent: a compute pass
(`shaders/cull.wgsl`) evaluates the shared `scatter.wgsl` twin once per
candidate slot of a bounded cell region around the camera (`regionRadius`),
drops slots with no plant / outside the stand region / outside the fade
envelope, and compacts the survivors into an indirect draw — 100 plants and
134M plants issue the same region-bounded dispatch + draw. No source geometry
is touched per frame; the 8 captures are baked once per species, ever
(harness bake cache + `mesh/baked/006-fin-cards/`).

A **carpet** species (`stand_table[i].carpet_div > 0` — the bog's Sphagnum) gets
a different card set out of the same machinery: no fins at all, and a stack of
ground-parallel cross-sections through the cushion instead. See "Moss carpet"
below.

Wind: shared `wind_sway` per species (stand `sway`), applied in the vertex
shader weighted by normalized height — fin tops lean, roots stay pinned, and
the slab cards translate by the sway at their own height so they stay attached
to the fins.

Lighting: atlases store mesh-local normals (baked from the authored octahedral
normals); at runtime they are yaw-rotated to world space, flipped toward the
viewer (thin foliage is lit from both sides — the same rule
`000-ground-truth` uses) and fed through the shared `light_surface` + fog, so
the cards sit in the same light as terrain.

Camera-inside-plant: cards collapse via `smoothstep(0.35R, 0.95R, dist)`
(hashed-alpha faded), per the rules.

## VRAM budget math

Per species: one 2048x1024 rgba8 albedo atlas + one 2048x1024 rgba8 normal
atlas (4x2 tiles of 512px: 6 azimuth + 2 top slabs), each with a full 5-level
mip chain (premultiplied, built on CPU at bake time):

- 2048*1024*4 B * ~1.333 (mips) = 11.2 MB per atlas
- x2 atlases = **22.3 MB / species** (HUD: 21.3 MiB/25 MiB) + 80 B cfg
  uniform + 16 B indirect args per stand entry.

Plus the compacted-plant buffer of the cull pass, which scales with the
*region*, never with the plant count: `capacity = cells x slots x fill x 0.78 x
1.15 + 8192` records of 24 B, where `slots` is `standEntrySlots(entry)` and
`fill` the fraction of slots that hold a plant. The 0.78 is the region CIRCLE
over the dispatched cell SQUARE (pi R^2 / (2R+4)^2 = 0.73 at the default radius,
plus margin) — both fades are exactly zero past `regionRadius`, so the square's
corners can never contribute. At the default `regionRadius` 56 that is ~44k
records = **1.1 MB / grass species** (HUD: 22.3 MiB/25 MiB, was 22.6 before the
circle factor; the image is unchanged — see the regression check in Status). It
is re-allocated on `regionRadius` change, so a large region costs more: at the
param maximum (160 m) it is ~7 MB, which does push the species row over 25 MB —
a deliberate, documented exception at a non-default setting, and the reason the
record is 6 scalars rather than two vec4s.

Per carpet species (Sphagnum), a completely different split — see "Moss carpet":
- albedo, 6 cross-sections of 512px rgba8 + 5 mip levels: 8.0 MiB
- ONE macro-surface map, 256px rgba8 + 4 levels: 0.33 MiB
- compacted tiles: 484 slots/cell x 0.667 fill -> ~237k records = 5.4 MiB,
  plus ~10k for the near list (0.3 MiB)
- **14.3 MiB / species** total (HUD-verified). The whole point of the split is
  that it is NOT the upright plant's budget: 6 azimuth views of a 7 cm mat would
  spend 74% of every tile on empty space, and a life-size mat needs ~4x the
  instance records a 2x-oversized one does.

Transient bake-only allocations (source vertex/index buffers up to ~135 MB for
poa, render targets, readbacks) are created outside `ctx.res` and destroyed
right after the bake — they trip the dev "untracked" console warning, but now
only on the ONE session that actually bakes a species.

## Bake

`bake.ts` renders the GCMESH1 source mesh 8 times (ortho, MRT
albedo+coverage / local-frame normal, depth-tested) into the 4x2 tile atlas —
one viewport+scissor+dynamic-offset draw per tile. Top-down tiles clip to
their vertical slab in the fragment shader. Framing: half-width = half the XZ
bounds diagonal (valid for every azimuth), full Y extent, 5% transparent
margin so mip filtering never clips content. Atlases are read back and a
premultiplied mip chain (5 levels, alpha kept honest — the `alphaSharp`
runtime param compensates coverage thinning) is built on CPU and uploaded.

The result goes through the harness `bakedArtifact()`/`commitBake()` flow as
`fins-v1-<species>.bin` (64 B header: magic + version + the 8 framing floats;
then the albedo mip chain, then the normal mip chain — 22,347,840 B total).
`unpackFins()` rejects anything whose length or magic does not match and
rebakes, which is what the earlier hand-rolled bypass was working around (the
dev server used to answer a missing `.bin` with a 200 `index.html`;
`bakedArtifact` now filters that itself).

`carpet.ts` bakes the Sphagnum states instead, under its own artifact key
(`moss-v3-<species>.bin`, 8.7 MB each) so the grass artifacts stay valid and are
never re-baked. It renders 6 top-down cross-sections framed to the tile's own
0.18 m square, reads back albedo AND the depth buffer, and derives the
macro-surface normal + occlusion from the depth (see "Moss carpet"). All three
moss species bake in ~20 s total on this machine, not the ~2 min each the brief
warned about — the 6 ortho passes over 19.8 M triangles are cheap next to the
479 MB mesh fetch. Baked serially: three concurrent Sphagnum meshes would hold
~1.3 GB of transient GPU buffers.

## Status

working — verified by headless screenshots at `cam=grazing`, `topdown`,
`inside-plant`, `far-horizon` (1280x800, det=1): no star from above, no
disappearing plants at grazing, inside-plant fades correctly, wind coherent.
Re-verified after the audit below, including all six `view` debug modes, the
`cardTint` param and `regionRadius=96` (which exercises the buffer
re-allocation path): console clean, no WebGPU or shader errors.

Carpet path verified on the `bog` stand at `grazing`, `carpet-close`, `topdown`,
`inside-plant`, `far-horizon`, two poses across the ridged slopes (a 32 deg flank
at (24,-40) and looking up the same hill from (10,-30)), a 20 cm-high oblique and
a 12 cm-high near-level pose for silhouette, plus `albedo` / `normals` /
`lighting` / `coverage` and the `cardTint`, `mossLayers` and `mossRelief` params.
No console errors, no toasts.

**`default` is unchanged.** Masked pixel diffs (HUD, param panel and bottom bar
excluded) of before vs after at 1280x800, det=1, t=3:
`grazing` 5 / 730,400 px differ, `topdown` 17, `inside-plant` 0 — isolated
depth-tie flips in the compaction order, the same 0.0x% the previous audit
measured for a change it proved image-neutral. Timings could not be trusted this
session (four other agents on the GPU: repeats of the same URL swung 7.2 -> 17.9
ms), but on the sample where the harness base pass read 0.24 ms (i.e. an idle
GPU) `default` grazing was Sigma p50 **7.23 ms** with fin-cards 3.61, against
7.68 / 3.58 before — unchanged, inside the 9 ms ceiling. The only code the grass
path shares with this work is the cull pass's new whole-cell region reject (a
saving) and a smaller instance buffer.

Honesty / known issues:

- calamagrostis and elymus source meshes are periodic ~0.52 m *community
  tiles*, not single specimens — each scatter point renders a baked clump, so
  density reads busier than a true per-specimen render. poa is a real single
  specimen and is the cleanest case. (Same caveat as every baked method here.)
- Beyond `regionRadius` (default 56 m) there is no grass; the far band fades
  out but distant bare terrain is visible from high cameras. Raising the param
  to 96 costs ~1.4x (5.4 -> 6.8 ms p50 grazing on Apple M-series).
- The hashed-alpha crossfades read as slight pixel speckle up close in stills;
  with TAA they would average out, the harness has none.
- Calamagrostis flower heads read stronger/pinker at distance than ground
  truth: mip-averaged head fluff plus `alphaSharp` coverage boost solidifies
  the fluffy tips. Tunable (lower `alphaSharp`), at the cost of overall
  thinning.
- At mid elevations (~20-45°) fins and slab cards are all on; the union
  silhouette reads slightly denser than the source plant. The trig weights
  only normalize the transitions, not the union.

## Audit (structural + debug views, 2026-07-24)

**Debug views.** `shaders/fins.wgsl` now includes `src/wgsl/debug.wgsl`,
applies fog only when `debug_mode() == DEBUG_OFF`, and returns
`debug_shade(color, albedo, n_world, coverage, world)`. What the modes showed:

- *albedo* — clean unlit atlas colour (pink heads, green blades). No light was
  baked into it; the bake writes raw vertex colour. Good as-is.
- *normals* — real per-fragment normals from the baked normal atlas (it WAS
  decoded and used before this audit), but they were **not flipped toward the
  viewer**. The atlas stores the authored mesh normal of whatever surface the
  capture ray hit, and on thin single-sided blades roughly half of those point
  away from the camera, so those fragments were lit from behind and came out
  much too dark. `000-ground-truth` explicitly flips ("thin foliage is lit
  from both sides"); fin-cards now does the same — one line in `fs_main`.
  **This is the one change that alters the image**: ~28% of grass pixels at
  `cam=grazing`, mean delta ~21/255. It removes some dark patches that read as
  contact shadow but were really wrong-side lighting; revert the single
  `dot(n_world, camera - world) < 0` line to get the old look back.
- *lighting* — high-variance speckle in both variants, which is honest: the
  source is a 2M-triangle clump of thin blades, so neighbouring atlas texels
  genuinely hold very different normals. Lighting goes through the shared
  `light_surface()` exactly once (no double-application, no premultiplied
  light in the atlas — verified by the albedo view).
- *coverage* — now reports the value actually tested against the dither
  (`min(atlasAlpha * alphaSharp, 1) * cardAlpha`), so the near-binary core and
  the dithered transition bands are both visible.
- *depth* — smooth, silhouettes and the `regionRadius` falloff both read.

Method-specific state got its own param instead of a competing global mode:
`cardTint` hues each fragment by which of the 8 baked views covers it.

**Structural waste found and fixed.**

1. *The big one — 86 placement evaluations per drawn plant.* The draw was
   `draw(30, side*side*128)` per stand entry with the scatter hash evaluated
   in the VERTEX shader: every one of the 30 vertices re-ran
   `scatter_candidate` (hash4 + 6x hash2 + a terrain heightmap tap), including
   for the ~62% of candidate slots that hold no plant at all (`density 3 / max
   8`) and for the square's corners, which are past the fade radius and
   therefore invisible. At the default region that is 3 x 107,648 instances =
   **9.7M vertex invocations per frame to place ~113k plants**. Replaced with
   the house pattern (same as 010-chord-frustum): a `@workgroup_size(64)`
   compute pass evaluates each candidate ONCE, rejects non-existent /
   out-of-region / fully-faded plants, and compacts survivors into an instance
   list + `drawIndirect`. The vertex shader now just reads a 24-byte record.
   Placement is still the shared WGSL twin, so plants land in identical spots.
2. *Bake re-run on every single page load.* The experiment deliberately
   bypassed `bakedArtifact()`/`commitBake()`, so every runner/AB/bench load
   re-uploaded up to 135 MB of source mesh, rasterized 2M triangles 8 times,
   read back 16 MB and built two mip chains on the CPU — per species. The
   reason given (dev-server SPA fallback poisoning the cache) is handled
   inside `bakedArtifact` now, so the atlas is packed into a versioned
   artifact, cached in OPFS and committed to `mesh/baked/006-fin-cards/`.
3. *Plants outside the stand region were being drawn.* The region window was
   never clamped to `Scatter.standRegion()` (±`stand.radius`), so with the
   camera near the stand edge the renderer invented plants the stand does not
   contain. The cull pass now clamps cells to `[-radius/4, radius/4]`, like
   the CPU scatter and 010 do. (No visible difference at the standard cameras,
   which sit well inside ±128 m.)

**Verified image-neutral.** With the two-sided normal flip temporarily
disabled, deterministic screenshots (`det=1&t=3`, HUD cropped out) at
`grazing` / `topdown` / `far-horizon` differ from the pre-audit build by
0.03% / 0.02% / 0.05% of pixels — isolated pixels where two coincident cards
tie on depth and the compaction order decides which wins. Everything else is
byte-identical.

**Deliberately left alone (suggestions, not done — each would need a bench to
justify, and the brief forbids measuring on a contended GPU):**

- *Frustum culling in the cull pass.* 010-chord-frustum rejects plants outside
  the view frustum there; fin-cards does not, so at `grazing` well over half
  the region is shaded and then clipped. Skipped because the bounding radius
  is genuinely fiddly here — `radius` is the fin quad's half-diagonal, but the
  slab cards reach `hypot(1.414*halfW, |y_card - yc|)` from the same centre,
  plus wind sway — and a radius that is slightly too small pops plants at the
  screen edge, exactly the defect a still screenshot hides.
- *Cell mask for the region circle.* Cards are fully faded past `0.97 *
  regionRadius`, so ~30% of the dispatched candidate slots (the square's
  corners) are known-dead. After fix 1 each of those is one cheap compute
  thread rather than 30 vertex invocations, so the remaining win is small
  relative to the complexity of a precomputed cell-offset table.
- *Splitting the cfg uniform.* 17 of its 20 floats never change; only the
  region origin does. It is one 80-byte `writeBuffer` per entry per frame —
  not worth two buffers and two bind-group entries.
- *Deferring the normal-atlas tap until after the alpha test.* Would need
  explicit-LOD sampling (derivatives are invalid after `discard`), which is
  precision/ALU territory and out of this audit's scope.

## Findings

All numbers below predate the audit (they were measured on the pre-cull
vertex-shader-placement build) and have NOT been re-run — the audit session
shared the GPU with other agents, so any new timing would be garbage. Re-bench
before quoting fin-cards against anything.

- Bench (Apple metal-3, 1600x900, orbit-low, stand `default`, ~557k plants):
  `results/006-fin-cards__default__p-e787a1a6__apple-metal-3__2026-07-24T16-08-02-171Z.json`
  — fin-cards pass **5.42 ms p50 / 7.59 ms p95**, VRAM 21.3 MiB/species.
  For reference, 005-octa-impostors' HUD showed ~14 ms p50 on the same machine
  at grazing: 1 texture tap x 2 atlases (vs 8 taps) and near-binary alpha with
  early-z-friendly depth writes make the crossed-fin fragment path ~2.5x
  cheaper, at the price of 5x the vertex work (30 verts/plant candidate).
- Scaling stand (134M plants, ±2048 m):
  `results/006-fin-cards__scaling-100m__p-e787a1a6__apple-metal-3__2026-07-24T16-08-42-414Z.json`
  — fin-cards 13.9 ms p50. The increase over `default` is view-driven (the
  hillier large terrain fills the orbit view with far more visible cards; the
  harness-owned base pass also went 0.27 -> 5.1 ms and composite 3.2 -> 7.4 ms
  on the same run), NOT plant-count-driven: the draw is the same bounded
  region at both 557k and 134M plants.
- A/B: `#/ab/006-fin-cards/005-octa-impostors?stand=default&cam=grazing&seed=42`
  and `#/ab/006-fin-cards/000-ground-truth?stand=close-quality&cam=grazing`.
  Compared to 005, the static fins hold color character (pink calamagrostis
  heads survive; 005's 4-view blend washes them toward green mush) and are
  rock-stable under camera motion; 005 wins slightly directly overhead where
  its hemisphere sampling is denser than my 2 slab cards.

## Moss carpet (bog stand, 2026-07-25)

Sphagnum palustre is a periodic community tile 0.18 m across and 0.07-0.09 m
tall, laid out by the `bog` stand as a life-size grid-snapped mat: 22x22 tiles
per 4 m scatter cell (**484 slots**, deliberately over the 128 scatter budget),
constant scale 1.01, 90 deg-only yaw, three states partitioning the wetness axis.
Rendered as an upright plant it was indefensible, and in a specific, measurable
way — every one of the failure modes below was visible in the before shots.

### What changed, smallest first

1. **Enumerate `carpet_div^2` slots, not `SCATTER_MAX_PER_CELL`.** The cull
   dispatch was `side^2 * 128 / 64` workgroups with `gid / 128` as the cell
   index, so it only ever visited slots 0..127 of 484 — the first 5.8 of each
   cell's 22 grid rows. The bog rendered as **corduroy stripes**: 26% of the mat,
   in bands, which looks exactly like a placement bug and is not one. Slots now
   come from `standEntrySlots(entry)` per entry (`fin.slots` in the cfg), and the
   dispatch is sized per entry. One line each in `main.ts` and `cull.wgsl`; the
   single biggest visual fix here.
2. **No fins for a carpet, no elevation fade.** The 3 crossed vertical fins on a
   7 cm mat sliced through the ground and each other (they read as rows of dark
   boxes with the classic star crossing visible from above), and the two
   horizontal slab cards faded OUT below 16 deg of view elevation
   (`smoothstep(0.05, 0.28, |vd.y|)`) — i.e. the mat disappeared exactly at
   grazing. A carpet entry now runs a separate pipeline (`shaders/moss.wgsl`)
   with ground-parallel cards only and no elevation term.
3. **Width from `footprint_m`, framing from the tile square.** Cards were sized
   from the bake's `halfW` = half the mesh's XZ bounds diagonal, i.e. 0.36 m
   quads on a 0.18 m grid: every tile 2x too large, 4x the overdraw, and the
   texel density quartered. A card is now exactly `footprint_m * scale` across.
4. **No camera-inside fade.** `fin_fade`'s near term opened a ~0.2 m bare crater
   around the camera at `inside-plant`. A carpet gets `carpet_fade`: region edge
   only, measured from the tile centre, eroded through the alpha reference
   (never through a per-vertex clip, which stretches slivers).
5. **Hard alpha test, no dither.** The fins' hashed-alpha dissolve turned the
   distant mat into speckled mush (see the before `topdown`). Carpet cards test
   hard: 0.06 for the closure card (a mat must stay a solid depth-writing
   occluder — the mip chain pulls tile alpha toward the tile mean, and a
   grass-like reference punches tile-shaped holes through to bare peat) and the
   `mossRelief` param (0.5) for the cross-sections above it, whose silhouettes
   should stay honest. `debug=coverage` on the bog is now near-binary white.
6. **A stack of cross-sections instead of a poster** — the part that is actually
   novel, and the reason to bother: see below.
7. **Macro-surface normals from the capture's own depth buffer** — the single
   biggest change to how the moss LOOKS: see below.
8. **Conform to the terrain the harness DRAWS, not to `terrain_height()`** — the
   bug that cost the most time: see below.

### The card set: 6 horizontal cross-sections

Card k is the top-down capture of everything **above** height `y_k`, drawn at
`y_k`. For a ray from above, the highest card that is opaque at a given xz is
where the cushion surface actually is, so the stack is a terraced height field of
the real capitulum relief (3.3 cm on the wet-vigorous state) rather than a decal.
`y_k` fractions are 0.12 / 0.50 / 0.62 / 0.72 / 0.82 / 0.91 of tile height: the
manifests put all three states' capitulum apices between 0.55 and 0.92 (mean
0.74), so five levels sit in the relief band and card 0 sits just above the peat,
below every capitulum, where its mask is the full top view — that is what closes
the mat.

Levels live in a `texture_2d_array` with **repeat** addressing rather than an
atlas. The tile is periodic (bounds run -0.013..0.197 for a 0.18 m period: the
overhang past one edge is the same geometry that re-enters at the other), so
wrapping is the CORRECT filter at the tile boundary, cutting at the square loses
nothing, and per-layer mip chains cannot bleed one cross-section into another.
Framing to the tile square instead of the bounds diagonal also quadruples the
on-screen texel density: 512 px over 0.18 m is 0.35 mm/texel.

`cardTint` shows the stack directly: the near field is a mix of five hues (the
visible surface genuinely varies in height at capitulum scale), the far field is
one (the closure card). LOD collapses the stack by distance — 6 cards under
1.5 m, 3 under 4 m, 2 under 9 m, 1 beyond — which is an LOD on how many cards,
never on tile size, spacing or rotation, so the lattice is identical at every
distance. The closure card's height lerps from its cross-section height to the
mean capitulum apex between 4 and 10 m: with one card the best flat stand-in for
the surface is its mean height, and because the lerp is a pure function of xz,
tiles that share a corner still share its height exactly (no ledge at the LOD
ring), while the far height already equals the card the 2-card LOD adds, so that
transition is invisible by construction.

Structurally the stack is TWO draws per carpet entry off one cull pass: the main
draw is 6 vertices (card 0) for every surviving tile, and a second draw with
`first_vertex = 6` adds cards 1..5 for the ~8k tiles inside 9 m. Without the
split, all ~300k tiles emitted 36 vertices and 84% of them returned degenerate —
10.8 M vertex invocations to draw 2.0 M useful ones, which measured ~2x the
fin-cards pass time at grazing. `first_instance` stays 0 (a non-zero one is
silently dropped without `indirect-first-instance`).

### Lighting: the mesh's own normals are unusable here

The first working version was dark and blotchy, and `debug=normals` showed why:
two populations of camera-facing noise, not a mat. Sampling 200k vertices of the
wet-vigorous mesh, the authored normals are **isotropic** — mean n.y = -0.001,
mean |n.y| = 0.494 (a uniform sphere gives 0.5), 35% pointing DOWN — because the
geometry is a mass of tiny leaves facing every direction. A coverage-weighted
average of that cancels to near zero and normalising the remainder is noise; the
two-sided flip then aimed the noise at the camera. This is the CLAUDE.md
mip-averaging trap arriving through a different door: nothing to do with
octahedral encoding, just genuinely opposite normals inside one texel.

What a texel of aggregate cushion needs is the normal of the cushion's own macro
surface, and the bake already contains it: the depth buffer of a top-down capture
IS the height field of the visible surface. So `carpet.ts` reads the depth back,
smooths it with a coverage-weighted 5x5 box (a leaf is 1-3 texels, a capitulum
~28, so the dome survives and the leaf roughness does not), takes central
differences at +-2 texels with the slope clamped to tan(75 deg), and stores
`(nx, nz)` plus a crevice-occlusion term from the difference between the smoothed
height and a ~1.1 cm local mean. `n.y` is recovered as `sqrt(1 - nx^2 - nz^2)`,
which is exact for a height field and — unlike an octahedral pair — mip-filters
honestly. All stencils wrap, so the normal field is seamless across tiles.

There is ONE such map for the whole stack, and that is correctness rather than
thrift: wherever cross-section k is opaque there IS geometry above `y_k`, so the
topmost surface there is the same surface card 0 sees. Shading each card from its
own level measurably flattened the close-up (a high level's height field is
mostly crowns, and its occlusion term averages toward 1). Occlusion multiplies
the light, not the stored colour, so `debug=albedo` stays the raw baked colour
and `debug=lighting` shows the crevice shading.

### Terrain fitting: rung 3, and it has to be the DRAWN surface

Per-vertex conforming (`terrain_sample` at every card corner: height and
`(nx, nz)` from one bilinear fetch, so the shading basis is free). Rung 3 and not
1 or 2 for the reason CLAUDE.md gives — neighbouring tiles share corner
positions, so only per-vertex conforming keeps the mat C0-continuous; any rung
that fits one plane per tile cracks at every seam. `slope_align` (1 for carpets)
blends both the height (`mix(tile centre, this vertex, align)`) and the shading
basis, so a hypothetical partly-conforming carpet degrades sensibly.

The trap: **`terrain_height()` is not the ground the harness draws.** The base
pass renders a 256x256 quad grid over the 256 m terrain, evaluating
`terrain_height()` only at the 1 m quad corners and interpolating linearly across
each triangle, while the heightmap is 0.5 m. Measured over the bog region, that
drawn surface sits up to **7.2 cm above** the sampled height (mean |deviation|
1.06 cm, 14% of the area over 2 cm) — and a Sphagnum tile is only 7-9 cm TALL. A
mat conformed to `terrain_height()` is therefore buried by the terrain it is
lying on: with the stack collapsed to its closure card, roughly half the tiles
vanished and the terrain's own facets showed through in patches. `drawn_ground()`
in `moss.wgsl` reproduces the drawn triangle from 3 corner taps instead. Tall
grass never notices this; nothing shorter than ~10 cm can ignore it. (This wants
to be a harness primitive — see interface feedback.)

Overscale stays at 1.0 and no depth bias is needed: cards abut exactly, share
their corner positions and their heights, so there is no overlap to z-fight and
no gap to hide. Given that the tile square already contains one complete period
(the overhang is the periodic image of the opposite edge), an overscale would
only double-draw.

### What is better, what is still bad

Better, from screenshots at the cameras listed in Status:
- the mat is continuous everywhere the stand places it — no stripes, no holes, no
  crater under the camera, no bare peat on the 32 deg flank;
- `topdown` reads as a bog mosaic: the three states form interlocking zones with
  distinct colour instead of dark speckled mush;
- close-up (`carpet-close`, and a 15 deg fov magnifier 0.7 m out) it reads as
  cushiony moss with capitulum-scale dome shading and dark crevices. Against
  001-billboard-smoke at the same pose, 001 is a flat high-frequency confetti of
  the same colours with no shading structure at all;
- `debug=normals` is a green (up) field with real slope and relief variation;
  `debug=lighting` is smooth and matches the terrain's own light term;
  `debug=coverage` is near-binary;
- 14.3 MiB/species against 001's 25.8 (over budget) for the same stand.

Still bad, honestly:
- **The stack buys less than the machinery suggests.** `cardTint` proves the
  visible surface really is quantised into six heights, and the geometric depth
  is real (it self-occludes and parallaxes correctly), but at 20 cm eye height
  the terracing does not read as cushions breaking a silhouette — six flat slabs
  6-8 mm apart, mip-filtered anisotropically at grazing, mostly read as one
  textured surface. `mossLayers=1` versus `6` at the same pose is a visible but
  modest difference. Most of what makes the moss look good is the macro-normal
  map and the occlusion, which a single card also gets.
- The mat ends at `regionRadius` (56 m) and the hill beyond is bare terrain.
  001 covers to 110 m. Raising the region is not affordable here without
  distance aggregation, because a life-size carpet is one quad per 0.18 m tile
  all the way out: ~300k tiles per entry at 56 m already, quadrupling at 112 m.
  The fix is to draw one quad per NxN block of tiles beyond ~25 m (repeat
  addressing is already on, so the texture tiles for free) at the cost of a
  single yaw and a single state per block — invisible at that distance, but a new
  mechanism and a real risk of blocky zone edges, so it is not in this change.
- Per-tile 90 deg rotation is visible as a faint patchwork grid at close range,
  because a periodic tile only matches a neighbour that shares its rotation. That
  is the stand's placement contract, not something a renderer may fix; 001 shows
  it too.
- The per-level albedo is redundant (the levels differ only in their masks). A
  mask-only array plus one albedo would cost ~4 MiB/species instead of 8, at the
  price of a third texture tap. Not done: the budget has room and the tap does
  not.

### Is this representation suited to moss?

Partly. "Fin cards" as such is not: crossed vertical fins are the wrong shape for
anything that lies along the ground, and the honest answer was to drop them
entirely for carpets rather than dress them up. What the method's actual engine —
static world-space quads, each carrying a different baked view, hard alpha, real
depth — does transfer well, and stacking cross-sections is a legitimate way to
give a mat thickness that a single quad cannot have. But the thickness it buys is
quantised to six steps and horizontal-only, so it reads mostly as shading and
self-occlusion rather than as intricate 3D. For a 3 cm relief on a 0.18 m tile,
the macro-normal map plus baked occlusion is doing more of the work than the
geometry is, and I would expect a proper relief/parallax step (marching the
height field in the fragment shader) or true prism geometry to beat it. As a
low-cost, budget-light, artifact-free carpet it is good; as a claim to render the
intricate 3D of a Sphagnum cushion it is honest-but-limited.
