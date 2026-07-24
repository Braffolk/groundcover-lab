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
   sparkle, per-2-tile macro tint to break periodicity,
6. composites the (up to 4) stand entries front-to-back by reconstructed hit
   distance and resolves total coverage by dithered discard (opaque pass,
   order-free, depth-writing).

Wind: the shared `wind_sway` field advects each species' texel column at
mid-canopy weight (per-species stand `sway`), plus a subtle gust brightness
ripple — the aggregate of blades leaning, not per-blade motion.

## VRAM budget math

Per species: one rgba8 texture array 1360x1360x2 (layer A = albedo+coverage,
layer B = height/oct-normal/sigma), 4 mips:
(1360^2 + 680^2 + 340^2 + 170^2) x 4 B x 2 layers = **19.65 MB / species**
(HUD: 18.7/25 MB). Plus one 752 B uniform. No per-plant buffers of any kind
exist at any plant count. Under the 25 MB budget for all three species.

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

## Status

working — verified by headless screenshots at grazing / topdown /
inside-plant / far-horizon, plus scaling-100m stand, A/B vs 000-ground-truth.

All three species render. **Honest coverage statement**: this method
deliberately does NOT reproduce the stand's exact per-plant placements — that
is the premise of the exploration direction. It renders the stand's species
mix, region, per-entry mean scale/height, per-entry sway, and density
(coverage modulated by stand density vs a per-species reference), but
individual plants have no identity. A/B vs per-plant renderers matches in
aggregate (palette, height, coverage), never plant-for-plant.

## Debug views

Wired to the global `frame.debug_mode` selector (runner/AB `view` dropdown,
URL `debug=`). Because this renderer composites up to 4 species layers in one
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
- The 4-iteration entry loop runs to 4 with a `continue` guard rather than to
  `entry_count`; the guard skips the work and the default stand uses 3 of 4.

Image equivalence: with `windAmount=0` (which removes every `frame.time` term,
making the render deterministic) the `off` view before and after the whole
audit differs by **11 pixels out of 836 400** across grazing/topdown/
far-horizon — isolated single pixels flipping across the stochastic-alpha
threshold from float re-association, no structural change. The cull itself is
byte-exact (proved separately above).

## Findings

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
