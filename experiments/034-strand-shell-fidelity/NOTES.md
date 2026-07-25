# 034-strand-shell-fidelity — blade library shell

Pushes **013-displacement-shell** further. 013's verdict from the owner was
"it's fast, it has no visual errors, but the FIDELITY, especially close to
camera, is very low — it simplifies the geometry WAY too much." So this
experiment keeps 013's structure and cost and replaces the thing that was
actually missing: the geometry.

## Idea

Bake a library of **real individual blades** out of the source mesh, and draw
each one as a ribbon **in the blade's own plane** instead of facing the camera.

Why 013 lost its detail, concretely (from its `bake.ts`): it dart-threw 96
roots among the lowest 30 % of surface points and walked *upward*, gathering
with radius `topH * 0.032` (~3.8 cm) and setting half-width to `1.15 * RMS` of
that gather. A 3.8 cm gather spans roughly ten neighbouring blades, so every
strand is the **average of a bundle** and its width is the bundle's spread —
a 4–8 cm paddle. It then flattened all of them into camera-facing planes and
pulled every shading normal 0.9 toward +Y. Four independent detail-destroying
steps, each individually reasonable.

What replaces them:

1. **Tip seeding.** A leaf tip is the one point on a blade that is
   unambiguous — its whole neighbourhood lies on one side of it. Every point is
   scored by "extremality" (distance from its own local centroid),
   non-maximum-suppressed on a fine grid, and stratified over 4 xz quadrants ×
   3 height bands.
2. **Tip → base walk carrying a real frame.** From each tip the walk goes
   *down* the blade with a blade-scale gather (5–16 mm, adapted to the measured
   width). At each station a weighted 3×3 covariance is decomposed by power
   iteration + deflation into `T` (tangent), `B` (width axis) and `N` (blade
   plane). Every station therefore carries the blade's **own twist and
   curvature**, including blades that arc over and point back down — the thing
   013's `ndy = max(ndy, 0.15)` forbade outright.
3. **Gap-detected width, not RMS.** Perpendicular offsets inside a thin slab
   are sorted and walked outward until a 3 mm hole appears: that is the real
   blade edge, and a neighbour 1 cm away cannot inflate it. Then clamped to a
   physical maximum — 6 mm half-width for a leaf, 1.6 mm inside a panicle.
4. **Oriented keeled ribbons.** The ribbon is built from `B` and `N`, so the
   silhouette genuinely changes with view direction and blades occlude each
   other in 3D. Ring 0 is **3 vertices across with a geometric keel** (the mid
   vertex lifts along `N`), so a near blade is a channelled surface with real
   thickness. Every ring interpolates the shading normal across the width
   (`curl`), which is what puts a highlight along a blade's crease.
   Past `orientFar` the width axis rotates to face the viewer, because there
   the blade is sub-pixel and orientation buys nothing but shimmer.
5. **Real occlusion.** Neighbour-point density in a 4–5 cm ball, periodically
   wrapped for the community tiles, instead of 013's `0.35 + 0.65*h^0.75`
   height curve. The inside of a tuft is genuinely darker than an exposed tip.
6. **Silhouette calibration.** 014-ribbon-skeleton's headline failure was
   coverage ~4× short of ground truth. The bake rasterizes the source point
   cloud and the reconstructed ribbon set from 4 azimuths and bisects one global
   width multiplier until the areas match (see Bake for the numbers).
7. **`upBias` 0.25, not 0.9.** The shared lighting model is half-lambert, so
   nothing goes black without a big +Y nudge; 013's 0.9 collapsed every blade
   normal onto one direction, which is a large part of why it read as a
   stand-in. `debug=lighting` now shows real per-blade light and shade.

Runtime per frame is 013's shape: one **cull** compute pass walking the scatter
cells of a camera-centred region (bit-identical WGSL twin of the harness
scatter, per-cell frustum AABB reject before any hashing), appending 16 B
records into five distance rings via atomics into the indirect-draw buffer;
then ONE render pass with five `drawIndexedIndirect` calls near-to-far plus the
far canopy shell.

**Cost does not scale with plants in view.** Zero texture taps per fragment
(four per *vertex*), no marching, no per-pixel loops, no `discard`, no
`frag_depth`. Everything is hard opaque geometry that writes depth and is drawn
front-to-back, so early-z rejects the layers behind it. No dithering anywhere:
the camera-inside-plant and region-edge fades collapse ribbon **width**, never
alpha (`debug=coverage` is flat white over every plant).

Two LOD mechanisms pay for the near field:

- **Continuous blade count.** `rows = c_rows / d` with a fractional marginal
  row, `c_rows = blades × rFull`, capped by what the ring draws. Coverage is
  conserved by widening survivors as `(blades_baked / rows)^coverPow`, so ring
  handoffs do not pop in density.
- **Per-ring `ROW_GAIN`** `[1, 1, 0.85, 0.7, 0.5]`. Because coverage is
  conserved by the width boost, shedding rows far away trades many thin strands
  for fewer wide ones at distances where a whole plant is a 27-pixel smudge.
  This is literally what buys ring 0 its 320 blades.

## VRAM budget math

Per species (the only species-scaled data):

- blade library: 3 × rgba16float textures, 16 stations × 384 blades, one array
  layer per species → 3 × 6144 texels × 8 B = **144 KB per species**
  (432 KB for all three). Clears the 25 MB/species budget by ~170×.

The library is one 3-layer array texture per channel — a single binding indexed
by species — so the HUD attributes it to the experiment rather than to a
species row. That sharing is exactly why the number is so small.

Shared, sized by param maxima (NOT by plant count or stand size):

- ring instance buffer, one allocation partitioned into five sub-ranges:
  (25+49+81+225+625) cells × 128 slots × 3 stand entries × 16 B = **6.17 MB**.
  Sized from a strict upper bound (`RING_R_MAX`, the largest each ring's radius
  can be anywhere in the param space) so a plant can never be dropped for lack
  of room; the default stand fills ~40 % of it.
- ribbon topology (5 × u16 index buffers): 263 KB. Shell mesh: 28 KB.
  Globals/ring-info/indirect: < 1 KB.

Total experiment ≈ **6.9 MB**; HUD reads 16.0 MB including the harness terrain
and depth. Identical on `default`, `dense-mixed` and `scaling-100m`.

## Bake

`bake.ts` → `mesh/baked/034-strand-shell-fidelity/<species>-v10.bin`, a "BLD1"
artifact, **288 KB each**, all three committed. In-browser, ~1.7–3 s per
species (cached in OPFS, committed through the bake endpoint; every load
magic-checks the bytes because the dev server answers missing baked files with
`index.html` at HTTP 200).

Flow: ~400 k area-weighted surface points from the raw GCMESH1 (2.2 M–6.5 M
tris) → spatial hash (periodic in xz for the community tiles, so a blade
crossing the tile seam keeps its points and edge AO is not artificially
bright) → tip detection → 576 traced blades → 1-2-1 smoothing along each blade
→ AO → silhouette calibration → the best 384 kept, emitted round-robin over 8
spatial buckets so **any window of consecutive rows is spatially balanced**
(the runtime LOD is "use `rows` consecutive rows from a per-plant offset").

Calibration result — projected silhouette area, mean of 4 azimuths, ribbon set
vs the source point cloud:

| species | mesh | ribbons | ratio |
|---|---|---|---|
| calamagrostis-canescens | 17.8 % | 13.9 % | 0.78 |
| elymus-repens | 19.6 % | 16.4 % | 0.84 |
| poa-pratensis | 15.5 % | 14.0 % | 0.90 |

For reference, 014-ribbon-skeleton measured itself at ~0.25 of the mesh on
calamagrostis and poa at its *finest* level. The raw bisection wanted 14× on
all three species and is deliberately clamped to **1.80×**: on calamagrostis
the target is unreachable by construction (a 2.2 M-tri panicle's silhouette is
near-solid at any raster resolution, while its honest look is see-through), and
letting the multiplier run just rebuilds 013's paddles. So calibration is a
nudge toward the mesh's density, not a solver.

Two bake-side rejections that were found by looking at screenshots, not by
theory:

- **Rhizome reject.** Both community meshes model explicit rhizomes (elymus 12,
  poa 4) — horizontal creepers that in the real plant lie in the soil. Traced
  as blades they rendered as a tangle of brown noodles lying on the terrain. A
  run that stays inside the litter band and barely climbs is now dropped.
- **Panicle rebalance.** Area-weighted sampling puts most points (and most
  extremal "tips") inside the panicle, so an even stratification spent a third
  of the library on plume strands and the meadow read as a pink crust with the
  green understory buried. The two lower height bands are now visited twice as
  often, and fluffy blades get extra position smoothing (in isotropic fluff the
  PCA tangent is near-arbitrary, so the raw walk wanders and the strand comes
  out lumpy instead of a clean radiating hair).

## Status

**working** — all three species render. Verified by headless screenshot at
`grazing`, `topdown`, `inside-plant`, `far-horizon`, a 1.3 m close-up and a
0.75 m macro, on `default`, `calamagrostis-pure`, `dense-mixed` and
`scaling-100m`, plus all six global debug views and both A/B pages. No console,
shader or validation errors in any of them.

## Findings

### A/B ratios (the only trustworthy timings)

16 sibling agents shared this GPU all session, so **no absolute milliseconds
are quoted**. These are same-frame A/B page ratios of (cull + draw) p50, taken
in the quietest window available (harness `base` p50 at 0.30–0.44 ms, i.e. near
idle):

| camera | vs 001-billboard-smoke | vs 013-displacement-shell |
|---|---|---|
| grazing | **1.00×** | **0.96×** |
| far-horizon | 1.09× | — |
| topdown | 1.22× | — |
| inside-plant | 1.31× | 1.45× |

At `grazing` — the design centre and the impostor killer — it is at **parity
with the champion billboard renderer and slightly cheaper than 013**, while
drawing 320 real oriented blades per near plant against 013's 64 flat
camera-facing strands. Worst case is 1.31× vs billboards at `inside-plant`,
inside the ≤1.5× bar. The cull pass is consistently 2–4× cheaper than the
billboard cull.

Two LOD bugs were found by measuring rather than reasoning, and both were
worth a lot:

1. **Ring selection by horizontal distance** (inherited from 013, which used it
   so overhead views keep their plants) meant a plant 42 m straight below the
   `topdown` camera had `d_xz ≈ 0` and landed in the *richest* ring: it drew the
   full 320-row, 16-station, keeled near-field topology for a 20-pixel smudge.
   `topdown` cost **4.6×** the billboards. Region membership still uses
   horizontal distance (so nothing is lost from overhead) but ring selection now
   uses true 3D distance with ring 4 as the catch-all: **4.6× → 1.22×**.
2. **The far rings dominated vertex cost.** Adding `ROW_GAIN` and dropping the
   keel outside ring 0 took `inside-plant` from 2.12× → 1.31× and
   `far-horizon` from 1.81× → 1.09×, with no visible difference: coverage is
   conserved by the width boost, so it is the same silhouette drawn with fewer,
   wider strands.

### Plant count is free

`default` (557 k plants), `dense-mixed` (7.7 M) and `scaling-100m` (134.2 M)
all report **identical VRAM (16.0 MB)** and identical cull p50 (0.05–0.07 ms).
Nothing is ever materialized for the whole stand. `dense-mixed` costs ~1.45×
`default` in the draw pass, which is correct and is not a plant-count effect:
its *density* is 13/m² against 8.5/m², so the bounded region genuinely holds
more plants. A solo bench on an idle GPU is still owed here — the cross-stand
comparison could not be made contention-free in this session, so it rests on
identical VRAM, identical cull time and the structural argument (region-bounded
cull, no global instance array) rather than on a timing.

### What the close-ups actually show

- At 1.3 m the wipe against 013 is unambiguous: 013 is broad flat pale-green
  paddles with a torn-paper silhouette and no interior depth; this is a dense
  thicket of thin blades of visibly different widths, with real curvature, a
  long dark leaf arcing across the foreground in front of the ones behind it,
  and dark recesses where the tuft closes up. Blades occlude each other in 3D.
- `inside-plant` is the strongest view: tall channelled blades with sky between
  them, pale panicles behind, and the shading gradient running *across* each
  blade rather than along it — the keel plus `curl` reading exactly as intended.
- `debug=normals` shows the full hue spread with per-blade variation and the
  cross-blade ramp clearly visible on individual leaves. Nothing is flattened.
- `debug=lighting` shows real per-blade light and shade (bright lit faces, dark
  turned-away ones), not a near-constant term.
- `debug=coverage` is flat white over every plant. Zero stochastic alpha; the
  only grey is the far shell's alpha ramp near the horizon, which is the one
  band worth watching.
- Moving the camera around a plant changes which blades are broadside and which
  are edge-on, so the silhouette really does change with view direction. This is
  the thing a billboard cannot do at any texture resolution.

### Honest artifacts

- **The calamagrostis panicle is the weak spot.** A real plume is airy; mine is
  ~240 opaque strands of 3.2 mm packed into an 8 cm volume, so near the camera
  it merges into pale pink lumps instead of feathering. 001-billboard-smoke
  genuinely looks better *at grazing distance* on this species, because
  per-texel alpha in a 512 px capture can do sub-pixel fluff that opaque
  geometry cannot. Making the plume sparser was tried (bake v9) and is worse,
  not better: isolating the strands exposes that each one is an irregular
  wandering shape. The honest fix is more, finer strands — i.e. more rows — and
  that is a cost decision, not a bake decision.
- Overall the meadow reads **pinker** than the billboard baseline at eye level.
  The colours are the mesh's own, but the plume/understory ratio is still
  slightly plume-heavy.
- Blade positions are f16, so a station quantizes to ~1.2 mm at plant scale.
  Static and sub-pixel, but it is why very close silhouettes have a faint
  facet to them.
- The `rOuter` (38 m) → canopy-shell handoff is smooth at grazing but visible
  from `topdown` as a ring where plants become flat mottle. The shell albedo was
  darkened to 0.72 to close the gap 013 documented; it is closer, not gone.
- Some brown arcing material remains at ground level after the rhizome reject.
  It reads as dry litter, which a real meadow has, but it is unintended.
- `debugRings` (our param, distinct from the global `debug=` selector) tints the
  five rings, bypasses frustum culling and forces a fixed ribbon width.

### Harness wishlist

- A `VramAttr` that can attribute a *fraction* of one allocation to a species
  would let a shared per-species array texture show up on the budget bars
  instead of only in the owner total.
- A dev-server 404 (instead of the SPA fallback) for missing `/mesh/baked`
  files would let every experiment drop its magic-validation shim. Third
  experiment to ask for this.
