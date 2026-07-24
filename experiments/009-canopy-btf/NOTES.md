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
(128x128 quads, uniform 2 m spacing inner +-64 m, exponential to ~1.3 km)
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

## Findings

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
