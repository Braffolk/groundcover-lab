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

Per-frame cost is plant-count independent: placement is the shared
`scatter.wgsl` twin evaluated in the vertex shader over a bounded cell region
around the camera (`regionRadius`), identically to the stand's scatter — 100
plants and 134M plants issue the same bounded draw (30 verts x cells x 128
slots per stand entry). No source geometry is touched per frame; the 8
captures are baked once per species per session.

Wind: shared `wind_sway` per species (stand `sway`), applied in the vertex
shader weighted by normalized height — fin tops lean, roots stay pinned, and
the slab cards translate by the sway at their own height so they stay attached
to the fins.

Lighting: atlases store mesh-local normals (baked from the authored octahedral
normals); at runtime they are yaw-rotated to world space and fed through the
shared `light_surface` + fog, so the cards sit in the same light as terrain.

Camera-inside-plant: cards collapse via `smoothstep(0.35R, 0.95R, dist)`
(hashed-alpha faded), per the rules.

## VRAM budget math

Per species: one 2048x1024 rgba8 albedo atlas + one 2048x1024 rgba8 normal
atlas (4x2 tiles of 512px: 6 azimuth + 2 top slabs), each with a full 5-level
mip chain (premultiplied, built on CPU at bake time):

- 2048*1024*4 B * ~1.333 (mips) = 11.2 MB per atlas
- x2 atlases = **22.3 MB / species** (HUD: 21.3 MiB/25 MiB) + 80 B meta
  uniform per stand entry.

No per-plant buffers exist at any plant count (fully procedural placement).
Transient bake-only allocations (source vertex/index buffers up to ~135 MB for
poa, render targets, readbacks) are created outside `ctx.res` and destroyed
right after the bake — they trip the dev "untracked" console warning once per
species per session, expected.

## Bake

`bake.ts` renders the GCMESH1 source mesh 8 times (ortho, MRT
albedo+coverage / local-frame normal, depth-tested) into the 4x2 tile atlas —
one viewport+scissor+dynamic-offset draw per tile. Top-down tiles clip to
their vertical slab in the fragment shader. Framing: half-width = half the XZ
bounds diagonal (valid for every azimuth), full Y extent, 5% transparent
margin so mip filtering never clips content. Atlases are read back and a
premultiplied mip chain (5 levels, alpha kept honest — the `alphaSharp`
runtime param compensates coverage thinning) is built on CPU and uploaded.

The harness `bakedArtifact()`/`commitBake()` cache is intentionally bypassed
(same reason as 005: the dev server answers missing `/mesh/baked/...bin` with
a 200 `index.html`, which would poison the cache). All three species bake
fresh in ~2-4 s per session.

## Status

working — verified by headless screenshots at `cam=grazing`, `topdown`,
`inside-plant`, `far-horizon` (1280x800, det=1): no star from above, no
disappearing plants at grazing, inside-plant fades correctly, wind coherent.

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

## Findings

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
