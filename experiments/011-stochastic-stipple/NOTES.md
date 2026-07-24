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
stand entry. No instance buffers at any plant count. Carpet: 112B uniform,
card quad index buffer: 12B. Well inside the 25MB budget.

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
placement, and all six `debug=` views. No console errors; typecheck clean.

## Debug views

Both fragment shaders (`cards.wgsl`, `carpet.wgsl`) route their final colour
through the shared `debug_shade()` and apply fog only when
`debug_mode() == DEBUG_OFF`. What each view carries here:

- **albedo** — the un-premultiplied mean radiance of the covered area
  (`alb.rgb / alb.a`) times the vertical card AO ramp; carpet: the mottled
  species mean colour. No lighting anywhere in it.
- **normals** — the real decoded per-fragment card normal: the normal atlas is
  premultiplied and mip-averaged exactly like the albedo, un-premultiplied
  (`nrm.rgb / nrm.a`), rotated by plant yaw into world space, then blended
  toward the statistical canopy normal by `sigma`. It shows the expected
  camera-facing hue gradient across the screen (the bake faceforwards each
  view's normals toward its capture direction) with per-leaf variation inside
  it — the same rainbow speckle 000-ground-truth's real geometry produces.
- **lighting** — the shared `light_surface()` term alone. It reads mostly
  blown-out white; that is the harness model (SUN_COLOR 1.15 + AMBIENT 0.21
  clips >1), and 000-ground-truth's lighting view looks identical, so it is
  not a double-applied or skipped lighting term.
- **coverage** — `a_eff`, the densified fractional-coverage statistic the
  hashed test actually realizes (carpet: its dissolve ramp). Near field ~1,
  falling with distance — the technique's core quantity, visible directly.
- **depth** — standard ramp; shows the stipple holes punched through the
  depth buffer past `stochStart`.

`debugRings` (own manifest param) tints cards by cascade ring; it is applied
only in the normal view so it can't pollute the debug views.

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
  - Seen from straight above the card normals collapse toward +Y (the bake
    faceforwards the top view's normals toward its capture direction), so the
    topdown normals view is near-uniform green with only fine speckle. It is
    a property of single-sided card capture, not a missing decode — the
    grazing views carry the full normal variation.

## Audit (structural-waste + debug-view pass)

Debug views did not exist here at all: neither shader included `debug.wgsl`,
so both cards and carpet ignored the global `view` selector and always drew
lit+fogged pixels. Both are now wired (see **Debug views** above). Nothing
was broken behind them — normals were already real, per-fragment and decoded
from the mip-averaged normal atlas, lighting already went through the shared
`light_surface()` exactly once, and the bake stores authored albedo (not
pre-lit), so the debug wiring changed no pixel in the normal view.

Structural waste found and fixed (all four verified image-identical; see the
verification note below):

1. **Card quads drawn non-indexed.** `draw(6, N)` shaded 6 vertices per card
   where the two triangles share two corners. Now a 12-byte index buffer +
   `drawIndexed(6, N)` with a 4-corner array: the post-transform cache shades
   **4 instead of 6** vertices per instance — a third off the heaviest stage
   in this method (~163k instances × 3 entries, each doing hashes, terrain
   fetch, billboard basis and wind), for identical triangles.
2. **Per-fragment work that is constant over the card.** The FS recomputed
   `cos(yaw)`/`sin(yaw)` *twice*, plus `smoothstep(stochStart, stochFull, d)`
   and the densify exponent `gamma`, all from flat varyings. They are now
   computed once in the VS and carried flat (`yaw_gs`, `top_vf`); `amp` and
   `dist` no longer need to reach the FS at all. Same varying slot count.
3. **Normal atlas sampled before the alpha tests.** Both atlases were tapped
   up front, then ~half the fragments were discarded by the coverage/hash
   test. Coverage lives entirely in the albedo alpha, so the 2–3
   `textureSampleGrad`s of the normal atlas now happen *after* both discards —
   discarded quads never touch it. (Safe: gradients are taken in uniform
   control flow and the taps use `textureSampleGrad`.)
4. **Carpet drawn before the cards.** The near card layer is a solid, hard-
   alpha-tested, depth-writing occluder and the carpet is a large 54m→800m
   sheet largely hidden behind it. Order can't affect the image (both are
   depth-tested opaque, no blending), so cards now draw first and early-z
   rejects the covered carpet fragments instead of shading them.
5. **Per-frame CPU/uniform churn that depends on nothing per-frame.**
   `carpetStats()` (a pure function of stand + baked meta) was recomputed and
   re-uploaded every frame — now computed once at create. The ring table was
   rebuilt (allocating row objects) every frame and all three 160-byte card
   uniforms were rewritten every frame, although their only time-varying
   field is the camera's 4m cell. Both are now change-driven: a still camera
   with untouched params uploads zero bytes.

Deliberately left alone:

- **The ~50-60% of vertex invocations that early-out** on nonexistent or
  rank-killed slots. Fixing it means a compute compaction prepass — a
  redesign, and its win depends on measurement this session cannot provide.
- **`scatter_candidate()` recomputing the `hash4` + offset hashes** the VS
  already evaluated for the cheap existence/rank rejection. Inlining them
  would duplicate the shared placement twin inside experiment code and risk
  silent divergence from `src/wgsl/scatter.wgsl`; the redundancy is a handful
  of ALU ops on the surviving path only.
- **The ≤6-iteration ring lookup loop** and the 4-entry species-pick loop in
  the carpet: bounded, tiny, and collapsing them is pure ALU shuffling.
- **The carpet's non-indexed 11.5k-vertex draw** (indexing saves ~4k vertex
  invocations — noise next to the base terrain pass).
- **Mip level 6 is generated but `lodMaxClamp: 5` never samples it** (~576
  bytes and one load-time blit). Left for the owner: dropping `MIP_LEVELS` to
  6 would also change the committed bake's shape for no measurable gain.

Verification: `?det=1&t=3` (paused, deterministic) canvas captures at
grazing/topdown/far-horizon, before vs after, diffed with ImageMagick. Two
runs of the same build are bit-exact (0 differing pixels), and before-vs-after
differs by **11 / 3 / 3 pixels** out of 422k in the HUD-free crops — 1-ULP
differences from evaluating `cos`/`sin`/`smoothstep` in the vertex stage
flipping a few fragments across the alpha threshold. Visually identical.
