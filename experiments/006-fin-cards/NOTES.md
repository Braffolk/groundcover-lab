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
*region*, never with the plant count: `capacity = cells x 128 x density/8 x
1.15 + 8192` records of 24 B. At the default `regionRadius` 56 that is ~53k
records = **1.3 MB / species** (HUD: 22.6 MiB/25 MiB). It is re-allocated on
`regionRadius` change, so a large region costs more: at the param maximum
(160 m) it is ~8.9 MB, which does push the species row over 25 MB — a
deliberate, documented exception at a non-default setting, and the reason the
record is 6 scalars rather than two vec4s.

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

## Status

working — verified by headless screenshots at `cam=grazing`, `topdown`,
`inside-plant`, `far-horizon` (1280x800, det=1): no star from above, no
disappearing plants at grazing, inside-plant fades correctly, wind coherent.
Re-verified after the audit below, including all six `view` debug modes, the
`cardTint` param and `regionRadius=96` (which exercises the buffer
re-allocation path): console clean, no WebGPU or shader errors.

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
