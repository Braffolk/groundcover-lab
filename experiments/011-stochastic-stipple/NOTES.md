# 011 — Stochastic thinning cascade ("opacity as statistics")

## Idea

Distant overlapping plants are structured noise; render the statistics, not
the plants. Three hierarchical stochastic levels, all fed by one baked
"coverage card" per species (9-view atlas: 8 azimuths + top; PREMULTIPLIED
albedo·coverage + plant-local normals, full GPU-generated mip chain — so every
mip texel is exactly the fractional-coverage / mean-radiance statistic of its
footprint):

1. **Plant level — stochastic thinning with conserved ensemble coverage**
   (Cook-style aggregate simplification, made continuous). A plant survives at
   distance d iff its scatter slot rank r = (slot+0.5)/128 < keep(d) =
   min(1, (R0/d)^2); survivors widen by 1/sqrt(keep) (capped), so the expected
   total frontal area of the ensemble is preserved while the drawn count falls
   as 1/d². Because survival is keyed to the slot INDEX, enumeration can be
   hierarchical: concentric cell annuli around the camera where ring k doubles
   the radius (quadrupling area) and quarters the slot cap — constant work per
   ring, O(rings) total. Defaults: 4 rings to 256m, ~163k candidate instances
   per stand entry (~0.98M lightweight VS invocations, most early-out) — fully
   procedural via the shared scatter WGSL twin, zero instance buffers, cost
   INDEPENDENT of stand plant count (134M-plant scaling stand renders the
   same work as 20k).
2. **Pixel level — hashed coverage realization.** The mip-averaged alpha is a
   coverage probability; the fragment shader realizes it with a hash threshold
   anchored to (plant id, card texel). Near the camera the threshold is a hard
   0.5 (crisp silhouettes, solid early-z occluders); it morphs to fully
   stochastic between stochStart(25m)..stochFull(70m). Amplified clump cards
   additionally "densify": a_eff = 1-(1-a)^γ with γ grown by the amplification
   — the coverage a card standing in for 1/keep overlapping plants would have.
3. **Field level — mean-field carpet.** Beyond the radius where width
   amplification saturates (widthCap·R0), a terrain-following canopy sheet at
   mean canopy height dissolves in underneath the cards (1m hashed patches):
   species-mottled (2.5m patches weighted by stand density × baked top-view
   coverage), colored by the baked top-view premultiplied mean (the exact
   color the ensemble converges to from above), lit with terrain normals +
   hash perturbation, with a statistical gust-wave shimmer standing in for
   per-plant sway. One draw, ~12k tris, camera-bounded.

Wind: shared `wind_sway` sheared over card height per plant (species sway from
the stand table, per-plant phase from the scatter hash). Lighting: baked
normals rotated by plant yaw, blended toward the statistical canopy normal at
distance, shared light_surface + fog. Camera-inside-plant: cards collapse
within ~0.75m. All three species rendered, exact stand placement via
`scatter_candidate` (positions/scale/yaw/phase bit-identical to the CPU twin).

**Why stochastic alpha is justified here (taste rule):** it is not a soft-edge
dither — it IS the distance-collapse mechanism (the seed direction of this
experiment). The near field (< 25m), where early-z and silhouette crispness
matter, uses a hard 0.5 alpha test with full depth writes; the hashed test
only takes over where cards are small and the coverage statistics are the
signal. The hash is anchored in card-texel space (stable per plant), not
screen space, so it does not swim under camera translation.

## VRAM budget math

Per species: two rgba8 768×768 atlases with 7 mip levels ≈ 768²·4·(4/3) ≈
3.1MB each → **~6.3MB per species** (HUD: 6.0/25MB), plus 160B uniform per
stand entry. No instance buffers at any plant count. Carpet: 112B uniform.
Well inside the 25MB budget.

## Bake

`bake.ts` renders the raw GCMESH1 mesh (any complexity; poa's 6.5M tris
included) once per species into the 9-view ortho atlas pair on the GPU
(~1-2s), reads back level 0, and commits `mesh/baked/011-stochastic-stipple/
<species>-v1.bin` (4.7MB each, validated magic GCSTIP1 on load — the dev
server's index.html fallback for missing files is detected and ignored).
Mips are regenerated on the GPU at load. Committed for all three species.

## Status

working — verified by headless screenshots: grazing, topdown, inside-plant
(default stand), far-horizon (scaling-100m, 134M plants), debugRings ring
placement. No console errors; typecheck clean.

## Findings

- Renders a full closed meadow from grazing to horizon; the thinning cascade
  transition is invisible in stills (debugRings shows the bands land at the
  intended distances with no seams — survival is distance-keyed, rings are
  only an enumeration structure).
- scaling-100m (±2048m, ~134M plants) renders with identical per-frame work
  to the default stand; the carpet carries everything past ~256m under fog.
- Carpet color MUST come from the baked top-view premultiplied mean, not the
  vertex-color mean — the vertex mean is biased by the (pink/straw) flower
  heads and produced orange speckle artifacts during dissolve; fixed.
- Bench NOT run / no numbers claimed: 16 experiment agents were hammering the
  same GPU all session, so any timing would be contaminated (observed ~120fps
  at 1280×800 solo-ish early in the session; stipple pass p50 fluctuated
  6.7→16ms purely with sibling load). TODO bench on a quiet machine:
  `#/bench/011-stochastic-stipple?stand=default&spline=orbit-low`.
- Known limitations / open questions:
  - Close-up (< ~3m) cards read flat, as any single-card impostor does; the
    9-view blend smears slightly at mid elevations (classic impostor smear).
  - Temporal behavior of the hashed test under camera MOTION only partially
    assessed (static deterministic captures + reasoning about the card-texel
    anchoring); plant-scale dissolve pop-in is visible by design near the
    survival boundary — smoothed by the 35%-of-keep fade band, but a video
    pass would be the honest check.
  - VS enumeration wastes ~50-60% of invocations on nonexistent/rank-killed
    slots (early-out after ~2 hashes); a compute-compaction prepass would
    roughly halve vertex work if the pass ever shows up hot on target GPUs.
  - Top-view uv de-rotation sign was validated only visually (yaw-symmetric
    clumps make the error subtle either way).
