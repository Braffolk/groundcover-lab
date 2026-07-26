# 009 canopy BTF field

## Idea

Give up per-plant identity: beyond arm's length a stand is a statistical
texture. Each species' periodic community tile is baked into a **canopy BTF**
— a 5x5 hemi-octahedral grid of ORTHOGRAPHIC captures parameterised by the
view ray's **ground-plane intersection**. Texel (u,v) of bin `d` stores the
aggregate of what an observer looking along `-d` through ground point
`(u*T, 0, v*T)` sees after the ray has traversed the (periodically wrapped)
canopy above and upstream of that point: mean albedo, coverage (fraction of
supersampled rays that hit), mean hit height, mean normal, luminance sigma.
Because the tile is periodic, one 0.52-0.9 m capture wraps over infinite
ground.

Per frame there is exactly ONE draw whose cost is screen/region-bounded and
completely independent of plant count: a camera-centred terrain-following grid
(128x128 quads, uniform 2 m spacing inner +-64 m, exponential to ~2.1 km,
quads entirely outside the stand region culled in the vertex stage)
rasterised twice — a ground layer and a canopy-top shell (for crest/sky
silhouettes and camera-below-canopy views). Every fragment:

1. finds its ray's ground hit G (two fixed heightfield refinement steps — a
   lookup, not a march; bottom-layer fragments ARE the ground),
2. hemi-oct encodes the view direction and picks a view bin **stochastically
   with probability = the bilinear weight** (blending bins ghosts the periodic
   tile into moiré fringes; dithering resolves the blend noise-free-of-ghosts),
3. pre-taps the height plane at the ground anchor and re-anchors the parallax
   reprojection at the RECONSTRUCTED content height (this killed the
   fingerprint-moiré between neighbouring bins),
4. samples albedo/coverage + height/normal/sigma with explicit gradients
   (manual per-bin-cell wrap, mips clamped to the 4-level chain),
5. reconstructs the hit point along the actual view ray -> frag_depth, fog,
   shared lighting on the aggregate normal, sigma re-injected as distance-faded
   sparkle, macro tint to break periodicity (per-2-tile constant for foliage, a
   smooth 1.4 m field for mats),
6. composites the stand's entries (up to 6, compacted into up to 4 layers —
   the zoned states of one carpet grid are mutually exclusive per tile and share
   a single coverage-weighted layer) front-to-back by reconstructed hit distance
   and resolves total coverage by dithered discard (opaque pass, order-free,
   depth-writing).
7. for a CARPET species (`carpetDiv > 0`), evaluates the stand's habitat zoning
   per fragment from the shared wetness field — see "Moss carpet" below.

Wind: the shared `wind_sway` field advects each species' texel column at
mid-canopy weight (per-species stand `sway`), plus a subtle gust brightness
ripple — the aggregate of blades leaning, not per-blade motion.

## VRAM budget math

Per species: one rgba8 texture array 1360x1360x2 (layer A = albedo+coverage,
layer B = height/oct-normal/sigma), 4 mips:
(1360^2 + 680^2 + 340^2 + 170^2) x 4 B x 2 layers = **19.65 MB / species**
(HUD: 18.7/25 MB). Plus one 864 B uniform. No per-plant buffers of any kind
exist at any plant count. Under the 25 MB budget for every species.

The split is unchanged for the moss and that is deliberate: 25 view bins x 256^2
puts a 0.18 m Sphagnum tile at **0.70 mm per texel**, and since the cushion's
whole appearance is view-dependent self-occlusion of its pits, the bins are not
the wasteful half. What IS mis-provisioned for a small tile is the *mip* end: 4
levels stop at 5.6 mm/texel, which is where a grazing view already needs to be,
so the mat fades to its (correct, bin-independent) mean colour earlier than the
grasses do. See "Moss carpet" below.

Whole-stand totals, for honesty: the bog stand holds five species = **98.2 MB**
of BTF (5 x 19.65). The per-species budget is what CLAUDE.md sets and each
species is inside it, but a five-species stand is a lot of texture; halving the
moss to 128^2 per bin (1.4 mm/texel) would cost 4.9 MB/species instead of 19.65
and lose only the sub-2 mm detail that no camera except `carpet-close` resolves.

## Bake

`bake.ts`, in-browser, per species (~seconds to ~a minute for poa, one-time,
committed): 25 orthographic captures at 1024^2 (4x4 supersample per stored
texel). Tile instances are replicated along each bin's upstream reach
(elevation clamped to >=10 deg; up to a few hundred instances of the 2.2-6.5M
tri source mesh, chunked to ~250M tris per submit). CPU reduce: coverage,
alpha-weighted means, normal averaging in vector space, luminance sigma,
dilation into empty texels, per-bin 4-level mips, periodic 8 px borders per
bin cell (so linear+mip filtering wraps correctly inside the atlas).
Artifacts: `mesh/baked/009-canopy-btf/btf-v1-<species>.bin` (19.65 MB each,
GCBT magic validated on load — the dev server answers missing bakes with 200
index.html, so magic is checked and the bake auto-commits).

- calamagrostis-canescens / elymus-repens: native periodic tiles (0.52 / 0.62 m).
- poa-pratensis is a finite specimen: a synthetic 0.9 m periodic tile is
  composed from 3 hash-placed rotated/scaled copies at bake time (fixed bake
  seed — deliberately independent of the runtime placement seed).
- the three Sphagnum states: native 0.18 m periodic tiles, 19.8M tris each,
  ~1 min per species to bake (81 tile instances per bin at the 10 deg ring).
  **Surface-normal reconstruction applies to these** (`surfaceNormals()`, gated
  on `topH < tileT`), see "Moss carpet". The grass artifacts are byte-identical
  to before that change and were not re-baked.

## Moss carpet (bog stand)

Sphagnum palustre arrives as a 0.18 m periodic community tile, 0.07-0.09 m tall,
laid out by the `bog` stand as a carpet: 22x22 grid nodes per 4 m scatter cell
(484 slots, over the 128 scatter budget) at a constant life-size scale, 90 deg
rotations only, with the three micro-habitat states partitioning a wetness
field. A BTF field is an unusually good fit for that — a periodic tile wrapped
over infinite ground is exactly what it stores, and it needs no instances at all
for the 1.13M tiles — but almost nothing about it worked, for several
independent reasons (a mis-keyed texture slot, a normal channel that was pure
noise at moss scale, no zoning, and a shell layer fighting the ground layer).
Fixes, smallest first:

1. **`manifest.species` now declares the three Sphagnum ids**, so the A/B view
   stops warning that this renderer does not draw them.
2. **`MAX_ENTRIES` 4 -> 6.** The bog stand has FIVE entries; the 4-entry uniform
   silently dropped the poa. (`entry_count`, the entry loop, the layer sort and
   the uniform layout all follow the constant; the header is padded to 16 floats
   so `entries` stays 16 B-aligned.)
3. **Texture slots are keyed by the stand's unique species, not by the catalog
   index.** This was the headline bug. `slotTex` was `[0,1,2]` matched against
   `speciesById(sid).index`, and the moss is catalog index 3-5, so *every*
   Sphagnum entry fell through to slot 0 and sampled the **calamagrostis**
   atlas — at 0.18 m period. The bog rendered as a green/pink corduroy of
   18 cm-repeated grass streaks, which is what the "featureless blob" looked
   like from close up. Slots are now the index within the stand's unique species
   list, with six bindings (the sixth aliases slot 0 and is never sampled).
4. **The tile period comes from the carpet grid step, not from `ctx.stand`'s
   scale range.** For a carpet entry `scaleMin`/`scaleMax` in `ctx.stand` are
   *placeholders* (1.7-2.5 for the bog moss) that the harness overrides in
   `carpetScale()` — but only into the GPU `stand_table`, not into the object an
   experiment sees. The real invariant is "div tiles per 4 m cell, each filling
   its step", so `tile = SCATTER_CELL_SIZE / carpetDiv` (0.1818 m) and the
   implied uniform scale is `step / tileSize` (1.0101). Using the raw mean scale
   would have stretched the wallpaper 2.1x.
5. **Coverage is not density for a mat.** A carpet's `density` is 8 (meaningless
   — the grid sets the cover), so `alpha_scale` is exactly 1 for carpet entries.
   The old `clamp(density/ref, 0.3, 1.3)` also had a 0.3 FLOOR that doubled the
   bog's trace poa; the floor is gone (all default-stand entries are 1.0, so
   `default` is unaffected).
6. **No camera-inside fade for a mat.** `smoothstep(0.12, 0.55, tH)` is skipped
   for carpet entries: a mat you are standing on must not open a hole under you,
   and at 7 cm tall the camera is never inside it. Verified at `inside-plant`.
7. **Mats are not drawn by the canopy-top shell layer.** The shell exists for
   crest/sky silhouettes of tall foliage; a 9 cm mat has none. Worse, the shell
   reaches its ground point through two refinement steps instead of exactly, so
   it landed in a *different* carpet node than the ground layer and the two
   disagreed about which zoned state lives there — per-pixel species speckle
   over every mid-field tile (visible in `inspect=layer`, which showed the
   mid-field being drawn mostly by the shell). One `continue`; the mid-field
   went from mushy dithered patches to crisp tiles, and it removes work.
8. **The aggregate normal is rotated into the terrain frame** by the entry's
   `slope_align` (1 for carpets, 0.3 for the bog's calamagrostis, 0 for the
   default stand's grasses — hence bit-identical there). The captures are taken
   over a flat tile, so their normals live in the tile frame; without this a
   whole hillside of moss lights as if it were level. The ground `up` is free:
   the last ground tap became `terrain_sample()`, which returns height AND
   (nx, nz) from the one bilinear fetch `terrain_height()` was already paying
   for.
9. **Smooth macro variation for mats.** The per-2-tile constant tint is
   invisible under grass texture but paints its own cell grid onto a
   near-uniform carpet: at 0.36 m it was a plainly visible checkerboard of hard
   squares over the whole near field. Mats now use a C1-interpolated value
   noise over 1.4 m (hummock mottling); foliage keeps the exact historical
   expression.
10. **Habitat zoning is evaluated per fragment** — this renderer has no
    instances to zone, so it asks the same question the scatter asks, with the
    same seed: `scatter_wetness()` at the carpet node centre plus the
    node-keyed interlock jitter (`carpet_node_wetness()`, an exact twin of the
    carpet branch of `scatter_candidate()`), and the linear band acceptance for
    scattered entries (whose expected local density IS that acceptance
    fraction, which is precisely what an aggregate renderer represents). Before
    this, all three moss states were composited on top of each other everywhere
    and the front-most won arbitrarily, so the bog had no zoning at all and the
    grasses covered the dry crests they have no business on.
11. **The zoned states share ONE layer, mixed by coverage.** They are mutually
    exclusive per node, not surfaces to composite: over-compositing two
    0.5-coverage boundary states gives 0.75 total coverage on a mat that is in
    fact closed, and the missing 25% comes out as dither holes along every zone
    boundary. Layers are now compacted and accumulated coverage-weighted, then
    normalised — which also means the bog's five entries never need more than
    three layers.
12. **Hard up close, statistical at distance.** The per-node test is a point
    sample of an 18 cm-period map; once a pixel covers several tiles it aliases
    into crawling speckle. It crossfades into this entry's expected *share* of
    the nodes in the pixel (the fraction of the uniform jitter window inside
    `[wet_a, wet_b)`), keyed on an **analytic** footprint (distance x pixel
    angle / incidence) rather than `dpdx(G)`: the proxy grid is 2 m quads, so a
    derivative-based footprint is constant within a quad and steps at its
    edges, which made whole quads flip between crisp and blended.
13. **The bake: the stored normal of a surface-like tile is now the normal of
    its own height field.** This is the fix that made the moss look like moss.
    At 0.18 m / 256 texels the capture is 0.70 mm per texel — about ONE
    Sphagnum branch leaf — so the 16 supersampled first-hit normals inside a
    texel belong to 16 differently-oriented leaves and their mean is
    ill-conditioned. Measured on the top bin of the old wet-vigorous artifact:
    `normal.y` **mean 0.002** (a ground-hugging cushion averaging to no up
    component at all), sd 0.344, and neighbour differences at **82%** of the
    white-noise level, i.e. the shading normal was static. `light_surface()`
    therefore returned per-pixel noise and the mat read as flat dark grain with
    no relief whatsoever — the featureless-blob look, from a bake that was
    otherwise fine.
    The HEIGHT channel is the opposite: sd 8.2 mm, neighbour difference only
    48% of white noise, and 71% of that sd survives an 8x8 (5.6 mm) box — real
    capitulum-scale structure. So `surfaceNormals()` smooths the height field
    with two [1,2,1] passes (sigma ~0.7 mm, far below the 5-10 mm capitulum
    scale), takes a +-2 texel central difference, un-shears it (a texel's world
    position is `uv*tileT + (h/d.y)*d.xz`, so the (u,v) grid is sheared by the
    bin direction) and stores the resulting surface normal plus the smoothed
    height. Result on the same bin: `normal.y` mean 0.60, sd 0.24, neighbour
    difference 33-38% with structure out to 8 texels.
    Where the surface is steeper than the view elevation the (u,v)->world map
    folds (that is a silhouette) and the Jacobian determinant, which reduces to
    `1 + grad(h).d.xz/d.y`, flips sign; restoring the orientation with
    `sign(det)` took the grazing bins from "mean 0.53, sd 0.60, a fifth of
    texels facing downward" to "mean 0.75, sd 0.27, none below zero".
    Gated on `topH < tileT` — a physical test, cushion vs foliage: Sphagnum is
    0.5, the grasses are 2.3. A blade canopy has no aggregate surface (its
    height field IS noise: calamagrostis sd 285 mm at 28% lag-1) and its
    first-hit leaf normal is the right answer there, so the grass artifacts are
    untouched and were not re-baked. `BAKE_VERSION` deliberately stays 1 for
    that reason; the three moss `.bin` files were deleted and re-baked in place.
14. **Committed-artifact loading retries** (3x) and validates the FULL length.
    A stand can want five 19.7 MB artifacts, and one flaky read used to fall
    straight through to a fresh bake — minutes of GPU, silently. (This happened
    repeatedly during this session's screenshot runs.)

### Fitting to the terrain

Rung 3-4 of the CLAUDE.md ladder, and per **pixel** rather than per vertex —
which for this representation is the natural place, since it has no per-plant
geometry at all:

- the ground height under a fragment comes from the heightfield (`terrain_sample`
  at the fragment's own ground hit), so the mat follows every bump continuously;
  there is no per-tile plane fit to crack at a shared edge, which CLAUDE.md
  warns is *wrong* for a tiled species, and no rigid tilt to bury one edge;
- the parallax reprojection is re-anchored at that height, and the reconstructed
  hit that becomes `frag_depth` is measured from it, so the mat's silhouette and
  depth follow the terrain too;
- the aggregate normal is rotated into the local ground frame by `slope_align`
  (fix 8);
- the lookup itself stays parameterised in world XZ *on purpose*. The carpet grid
  is snapped in world XZ, so a horizontal-projection parameterisation is exactly
  what the placement does — foreshortening the wallpaper along the slope would
  disagree with the grid it is supposed to represent.

Verified across the ridged slopes (a 28 deg hillside at x=18, viewed from its
foot and from 7 m up): continuous cover over crest and flanks, shading that
tracks the slope, no floating or buried edges, no cracks.

### What is still bad

- **At grazing incidence the mat loses its detail from ~3 m out.** Nothing about
  the mat: the atlas packs 25 bins into one texture with an 8 px border, which
  rules out anisotropic filtering (aniso taps would spill into the neighbouring
  bin cell at any mip above 1), so `textureSampleGrad` picks its LOD from the
  MAX gradient — the along-view one, which at 8-30 deg incidence is 3-10x the
  across-view one. The tile is then filtered to mip 3 (5.6 mm/texel) and
  `lodX` fades it the rest of the way to the (bin-independent, verified
  colour-neutral) mean. 001-billboard-smoke is equally flat at the same crop, so
  this is the viewing geometry more than the method — but the fix is known and
  is the single biggest remaining win: store the 25 bins as 25 *array layers* of
  a 256^2 texture instead of a 5x5 atlas. Same bytes, `addressMode: repeat`
  replaces the manual border and the wrap arithmetic, per-bin mips stop bleeding
  across cells, and `maxAnisotropy: 16` becomes legal. That is a bake-format
  change (version bump, re-bake all five species) and was out of scope here.
- The zone mosaic is tile-quantised by construction, so at 4-10 m each 0.18 m
  tile is a flat rectangle of one state's mean colour and the boundary reads as
  a coarse green/bronze checkerboard. That is what the placement does (a
  geometric renderer shows the same mosaic) and the three states genuinely are
  very different colours, but it is the least natural-looking part of the bog.
- No ambient occlusion in the pits, so the relief is read purely from the
  normal; a cushion is darker between capitula than the half-lambert term
  admits.
- `debug=coverage` on the bog shows dark bands in the mid-field. That is the
  shell layer's own (grass-only) coverage winning the depth test in a view where
  the coverage mode deliberately skips the dither discard; in the normal view
  those fragments stipple out and the mat below shows through.
- Wind is still zero for moss (correct — `sway = 0`).

### Is a canopy BTF suited to a moss carpet?

Yes — better than to the grasses, and better than cards. Everything the method
is bad at (per-plant identity, close-up blades, 1.13M individually placed
instances) is irrelevant for a periodic mat, and the two things it does that a
card cannot — view-dependent parallax between 25 captures, and a per-pixel
reconstructed hit height that becomes real depth — are exactly what a cushion
needs. At the `carpet-close` bookmark it renders visible capitulum-scale relief
with coherent bumpy shading, zone colour mixed in, and no tile seams, where
001-billboard-smoke at the same crop is flat mottled noise with a visible tile
grid. The honest ceiling is the grazing case above, which is a texture-filtering
and mip-depth problem, not a representation problem.

## Status

working — verified by headless screenshots at grazing / topdown /
inside-plant / far-horizon, plus scaling-100m stand, A/B vs 000-ground-truth;
moss on the `bog` stand verified at grazing / topdown / inside-plant /
carpet-close / far-horizon / two sloped views / normals / lighting / coverage /
albedo, and against 001-billboard-smoke at matched crops.

All three species render. **Honest coverage statement**: this method
deliberately does NOT reproduce the stand's exact per-plant placements — that
is the premise of the exploration direction. It renders the stand's species
mix, region, per-entry mean scale/height, per-entry sway, and density
(coverage modulated by stand density vs a per-species reference), but
individual plants have no identity. A/B vs per-plant renderers matches in
aggregate (palette, height, coverage), never plant-for-plant.

## Debug views

Wired to the global `frame.debug_mode` selector (runner/AB `view` dropdown,
URL `debug=`). Because this renderer composites several species layers in one
fragment, `debug_shade()` is called **per entry** — albedo, the aggregate
normal and the light term are per-entry quantities, and the front-to-back
composite then blends the debug colours with exactly the same weights as the
shaded colour, so `albedo`/`normals`/`lighting` are per-layer exact instead of
an ad-hoc average. Coverage and depth are composite quantities, so they are
resolved once after the composite.

- `normals` — the baked oct-encoded aggregate normal, decoded per fragment and
  fed to the shared `light_surface()` (this was already true before the audit;
  nothing here was flat-shaded). Mostly up-facing with per-pixel speckle: that
  speckle is honest — it is the stochastic view-bin pick plus texels where the
  16 supersampled normals nearly cancel (leaf front/back faces), so the
  normalised mean is ill-conditioned.
- `lighting` — near-white over the whole ground. Not a bug in this renderer:
  `SUN_COLOR` is 1.15 and the half-lambert term plus hemisphere ambient exceeds
  1.0 for up-facing normals, so the terrain base pass blows out identically.
- `coverage` — reports the RAW reconstructed coverage (before the
  `1-(1-A)^1.6` presentation lift) and **skips the dither discard in this mode
  only**, so the field reads as a continuous coverage map instead of its own
  stipple. A dense 3-species stand genuinely resolves to ~1 over most of the
  frame; the structure is at crests, the region edge and the near-camera fade.
- Fog is applied only when `debug_mode() == DEBUG_OFF`.
- The normal view divides by the lifted A (unchanged historical behaviour); the
  debug views divide by the true weight sum so they are not scaled by the lift.

Method-local state the global modes cannot express is a param in this
manifest, `inspect` (`off`/`bin`/`lod`/`layer`), never a competing global mode:

- `bin` — colour-codes which of the 25 hemi-oct view bins the pixel resolved
  to. This is the single most useful view for this method: sector boundaries,
  the stochastic dither between bins and bin popping are all directly visible.
- `lod` — the distance detail fade (footprint-derived). Shows visible per-quad
  faceting at grazing angles, because the footprint comes from `dpdx/dpdy` of
  the ground hit across a coarse proxy quad. Pre-existing, worth a look.
- `layer` — ground grid (green) vs canopy-top shell (orange); confirms the
  shell only contributes in the mid-distance band and on crests.

## Audit

Structural pass over pipeline + shader (no ALU micro-optimisation, no
retuning). Frame times were NOT used — the GPU was shared with other agents.

Found and fixed:

1. **The proxy grid rasterised the whole world out to ~2.1 km even though a
   stand is a bounded region** (default: ±128 m). Every fragment beyond the
   region ran the ground-hit refinement, the hemi-oct encode and the 4-entry
   loop only to hit `edge == 0` and `discard`. `vs_main` now culls a quad
   (both its triangles, all 6 vertices, so it is crack-free) when the quad's
   whole footprint is outside the region: ~60% of the 16 384 quads on the
   default stand, plus every fragment of the distant ground band — at
   `far-horizon` that band is most of the visible ground. Exactness: for the
   ground layer `G.xz == world.xz`, so the quad bound is the fragment test; for
   the top-shell layer `G` only walks FORWARD along the view ray, and for a
   camera inside the region `‖(1+s)W − sC‖∞ ≥ ‖W‖∞`, so it stays outside too
   (hence the extra `cam_in` guard — outside-the-region cameras cull the ground
   layer only). Verified: cull vs no-cull is **byte-identical** on
   grazing/topdown/far-horizon/inside-plant, on both `default` and
   `close-quality` (the stand where the cull is most aggressive).
2. **View-bin selection was recomputed once per species entry.** `i0`, the
   smoothstepped bin fraction, both `ign()` dither taps, the bilinear weights,
   the tap count and the nearest-bin index depend only on the view direction
   and the pixel — never on the entry — but sat inside the 4-iteration entry
   loop. Hoisted above it; bit-identical output (the diff vs the pre-hoist
   build was exactly zero pixels).
3. **The uniform buffer was fully rewritten every frame** including the entry
   block (tile size, canopy top, sway, density modulation, species slot, mean
   colour), the shell height and the 25 bin directions — all pure functions of
   stand + baked meta. Those are now computed and uploaded ONCE in `create()`;
   `update()` writes only the 12-float header (camera snap + params).
4. Dead code: `nidx` was computed per entry per fragment and never used.
5. `loadOrBakeSpecies()` fetched a root-absolute `/mesh/baked/...` URL, which
   404s under the production base path — now goes through `assetUrl()`
   (CLAUDE.md rule). Dev behaviour unchanged.

Deliberately left alone:

- **Three separate BTF textures with a 3-way `switch` around every
  `textureSampleGrad`.** Merging them into one `2d_array` with a per-entry
  layer base would delete the switch, but it would also collapse the
  per-species VRAM accounting the HUD budget bar depends on (one texture
  cannot be attributed to three species). Not obviously a win; left as a
  suggestion.
- **Hoisting `wind_sway` out of the entry loop.** It is exactly linear in
  `sway` (every component scales by it), so one evaluation times `ent.sway`
  would replace 3–4 — but that is ALU-level, explicitly out of scope, and the
  multiplication reorder would flip dither-threshold pixels.
- **Earlier rejection of redundant top-shell fragments.** The `< 2.5 m`
  redundancy discard already exists but only after two heightfield refinement
  steps; a conservative per-quad version in the vertex stage is possible but
  needs care not to punch holes in the sky silhouette, and it cannot be
  justified without measuring. Suggestion, not a change.
- **`frag_depth` writes disable early-z**, and the dithered coverage punches
  depth holes (the CLAUDE.md taste rule). Both are load-bearing for the
  technique — reconstructed hit depth IS the method, and the dither IS the
  bin-blend/coverage resolve (see Findings). Changing either is a redesign.
- The entry loop runs to the constant MAX_ENTRIES with a `continue` guard rather
  than to `entry_count`; the guard skips the work and the default stand uses 3
  of 6 (the bog uses 5).

Image equivalence: with `windAmount=0` (which removes every `frame.time` term,
making the render deterministic) the `off` view before and after the whole
audit differs by **11 pixels out of 836 400** across grazing/topdown/
far-horizon — isolated single pixels flipping across the stochastic-alpha
threshold from float re-association, no structural change. The cull itself is
byte-exact (proved separately above).

## Findings

- **Perf, moss round:** not re-benched — dozens of renderers were being worked
  on in parallel, and four identical repeats of `stand=default&cam=grazing`
  measured canopy-btf p50 at 4.14 / 4.60 / 1.72 / 1.82 ms (gpu Σp50 8.50 / 9.41
  / 3.53 / 3.73), so contention swamps any before/after difference. What can be
  said structurally: the `default` path gained only a handful of ALU ops (an
  analytic footprint, a ground-up vector, two loop iterations that `continue`
  immediately) and no extra samples; the bog path now takes FEWER samples than
  before (one zone-selected mat layer instead of three moss layers at 3 samples
  each, and the shell layer no longer samples mats at all) in exchange for two
  `scatter_wetness()` evaluations per fragment.
- **`default` is unchanged by the moss work.** Before/after screenshots at
  `cam=grazing` and `cam=topdown` differ in 891 and 14 subpixels out of 1.85M
  (0.048% / 0.001%) — isolated pixels flipping across the stochastic-alpha
  threshold, because the layer compaction normalises `(c*a)/a`, which is not
  bit-identical to `c`. Same class as the audit's 11-pixel delta below; no
  structural change.
- Bench numbers below are from BEFORE the audit and were not re-run (the GPU
  was shared during the audit session); the structural fixes only remove work.
- Bench (apple-metal-3, 1280x800, orbit-low):
  - `results/009-canopy-btf__default__p-8bf7dbf3__apple-metal-3__2026-07-24T16-28-06-648Z.json`
    — canopy-btf p50 **4.82 ms**, p95 5.81 ms (~557k plants).
  - `results/009-canopy-btf__scaling-100m__p-8bf7dbf3__apple-metal-3__2026-07-24T16-30-14-242Z.json`
    — p50 **5.01 ms**, p95 6.12 ms (**134.2M plants — plant count is free**;
    the +4% is the larger grass-covered pixel area of the +-2048 m region).
  - Standard cams: grazing ~3.5 ms, topdown ~2.7 ms after the top-layer
    redundancy discard (a top-shell fragment whose ray lands <2.5 m from its
    own foot point discards — the ground layer produces the identical pixel).
- Bilinear view-bin blending of a PERIODIC capture produces moiré/fingerprint
  fringes (shifted copies of the same quasi-periodic content beat against each
  other). Two fixes that worked: (a) anchor the parallax reprojection at the
  reconstructed content height instead of a fixed mid-canopy plane, (b)
  resolve the bin blend stochastically per pixel (`taps=dither`, the default);
  `taps=4` shows the fringes, `taps=1` shows sector popping.
- Known limitations / artifacts (honest list):
  - No per-plant identity; close-up (< ~2 m) reads as streaky aggregate mush,
    not blades. `close-quality` stand comparisons are aggregate-only.
  - Grass on terrain crests can be "shaved" against far background: the lookup
    evaluates content at the ray's ground hit, not at the crest it skims.
    The canopy-top shell restores the sky silhouette but not crest-vs-far-field
    silhouettes.
  - Looking UP from inside the canopy fades to nothing (sanctioned
    inside-plant breakdown; the fade is a hard-ish v.y >= 0 cut).
  - Stochastic bin + coverage dither = static per-pixel noise; under camera
    motion it crawls. A TAA-style resolve would eat it; the harness has none.
  - View elevations below 10 deg clamp to the lowest capture ring.
  - Tile periodicity is visible at some angles despite the macro tint
    (rotation variants would break the periodic wrap the captures rely on).
  - Wind advects whole texel columns; no per-height differential inside one
    texel. Sway direction is shared with the true wind field, so A/B flicker
    against geometric renderers shows coherent motion.
- Harness wish (noted per CLAUDE.md): a 404 (not 200 index.html) for missing
  /mesh/baked files would let bakedArtifact()/OPFS be used safely.

## Harness notes from the moss round

Where the harness made this harder than it needed to be:

- **`carpetScale()` is not exported from `@harness`, and `ctx.stand` carries the
  un-resolved placeholder scales.** `createStandBuffer()` resolves them into
  `stand_table.scale_min`, so a WGSL renderer that reads the stand table gets the
  right answer while a TS renderer reading `ctx.stand.species[i].scaleMin` gets
  1.7-2.5 for a tile whose real scale is 1.0101 — a 2x wallpaper stretch that
  looks plausible enough to ship. Either export `carpetScale()` or (better)
  resolve the scale range in the `Stand` object the experiment receives, so
  `scaleMin === scaleMax === the constant` and there is nothing to get wrong.
- **The placement seed is not in the frame UBO.** `scatter_wetness()` and every
  other scatter twin take a seed, but `frame` has no `seed` field, so each
  renderer plumbs `ctx.seed` through its own uniform (as 001 does too). One f32
  in the frame UBO would remove that from every experiment.
- **`ctx.scene.terrain` mirrors `terrain_plane_fit`, but there is no shared
  primitive for "rotate this tile-frame normal into the ground frame"** —
  `plant_basis_from_up(up, 0.0) * n` works, but every renderer that lays a mat
  into the terrain has to know that trick. A `tilt_normal_to_ground(n, xz,
  align)` next to `plant_basis()` would be the obvious companion.
- **A per-fragment footprint is a recurring need and always hand-rolled.** Any
  renderer that fades detail with distance wants "metres of ground per pixel",
  and `dpdx/dpdy` of a proxy position is wrong wherever the proxy is coarse.
  `frame` already has `viewport` and `proj`; a `pixel_footprint_m(dist, normal)`
  helper (or just the pixel angle precomputed in the UBO) would stop everyone
  deriving it from the projection matrix.
- Not a complaint, an endorsement: `stand_table[i].carpet_div` /
  `footprint_m` / `slope_align` plus `scatter_wetness()` being callable from a
  fragment shader is what made a *placer-free* renderer able to reproduce the
  stand's zoning exactly. Without the shared wetness function this method could
  not have honoured the bog's zones at all.
