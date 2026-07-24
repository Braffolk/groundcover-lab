# 013-displacement-shell — strand vector-displacement shell

## Idea

The only geometry that ever exists is one flat sheet. Its textures encode
WHERE the plant surfaces are: a per-species **strand vector-displacement
field** — 96 "strands" (blade/plume ribbons) × 16 stations, each texel
storing offset-from-plant-origin (xyz), ribbon half-width (w), mean normal +
baked vertical occlusion, and mean albedo. At runtime a vertex is nothing but
(strand row, station column, side bit); the field textures crumple the flat
patch into the plant. Near the camera the sheet is subdivided per plant slot;
far away it settles onto its limit surface — a single terrain-conformal
canopy shell.

Per frame:

1. **cull** (compute, ~0.07 ms): walks the scatter cells of a camera-centered
   region (bit-identical WGSL twin of the harness scatter — placement is
   evaluated in-shader, no CPU instance buffers ever), frustum-tests every
   existing plant and appends a 16 B record (pos + quantized yaw/scale/phase/
   entry) into one of three distance rings via atomics into the indirect-draw
   buffer. Work = region cells × 128 slots × stand entries, **independent of
   total plant count**.
2. **strands** (render, three `drawIndexedIndirect`): each ring draws ribbon
   patches of decreasing topology (12/8/5 stations; the station axis is
   sampled with the linear filter, so fewer stations = automatic curve
   simplification). Strand-count LOD is **continuous**: rows beyond
   `s0·r0/d` collapse to zero width (fractional fade on the marginal row) and
   survivors widen by `sqrt` to conserve coverage — ring handoffs cannot pop
   in density. Ribbons are camera-facing (side = tangent × view), wind is the
   shared `wind_sway` sheared up the strand with per-strand phase jitter,
   per-species sway from the stand table. Lighting uses the baked mesh
   normals (flipped to the viewer + up-biased) and the baked per-station
   occlusion.
3. **farshell** (render, ~2.7 k tris): beyond `rOuter` the meadow IS the
   shell — a camera-centered geometric-progression annulus draped on the
   terrain at density-weighted canopy height, colored by the baked per-species
   canopy albedos blended with world-anchored value-noise mottle, with a
   gentle wind bob. Plants in the outer 5 m of the region sink onto the shell
   height, so the handoff reads as canopy continuing. The shell clamps to the
   stand boundary (the meadow visibly ends there, alpha-faded over 3 m).

Rules check: no per-frame raymarching (four texture samples per vertex, zero
per fragment); no source geometry touched at runtime; cost is bounded by
region + screen (verified: `scaling-100m`, 134.2 M plants, renders at the
same frame cost and VRAM as the 557 k default — see Findings); works from all
angles (rings/cull use horizontal distance so topdown keeps its plants;
grazing is the design center); camera-inside-plant fades via horizontal-
distance width collapse (no dithering anywhere — every edge is real
depth-writing geometry).

## VRAM budget math

Per species (the only per-species-scaled data):

- strand field textures: 3 textures × 16×96 texels × 8 B (rgba16float) ≈
  **36 KB per species** (108 KB for all three, single 3-layer texture array).

Shared, sized by param maxima (NOT by plant count or stand size):

- ring instance buffers: (81+225+961) cells × 128 slots × 3 entries × 16 B ≈
  **7.8 MB** (worst-case capacity at max density 8/m²; default stand fills
  ~40 %).
- ribbon topology (3 × u16 index buffers): 25 KB; shell mesh: 28 KB;
  uniforms/indirect: < 1 KB.

Total experiment VRAM ≈ 8 MB; HUD shows 17.0 MB including harness terrain.
The 25 MB/species budget is cleared by ~3 orders of magnitude on the
species-scaled part.

## Bake

`bake.ts` produces a VDF1 artifact per species (73 KB each, committed in
`mesh/baked/013-displacement-shell/`): samples ~320 k area-weighted surface
points from the raw GCMESH1 (2.2 M–6.5 M tris, all three species), then
traces 96 strands through the point density with a spatial hash grid —
dart-thrown low roots, upward density-following walk, per-station centroid /
RMS width / mean color / mean normal / height-curve occlusion. Strand rows
are ordered as an importance-sorted interleave of four spatial quadrants so
ANY prefix of rows is a balanced subset — the entire LOD scheme is "use the
first N rows". Community tiles (calamagrostis, elymus) are traced as whole
tiles (a scatter point stamps one tile, same convention as 003); poa is a
single specimen and traces naturally. Bake runs in-browser in ~1–3 s per
species, cached in OPFS and committed via the bake endpoint.

## Status

working — verified by headless screenshot at `grazing`, `topdown`,
`inside-plant`, `far-horizon`, and `grazing` on `scaling-100m` (134.2 M
plants; identical cost). All three species render.

## Findings

- The strand reconstruction reads convincingly at grazing: green blade
  understory + tan calamagrostis plume tops emerge naturally from the traced
  colors (the plume stations are wide + pink because the source points there
  are wide-spread and tan — nothing species-specific is hard-coded).
- 134.2 M plants (`stand=scaling-100m`) vs 557 k (default): cull 0.07 ms /
  strands ~2.1 ms / farshell ~1.9 ms in both. Plant count is provably free;
  the only VRAM/time driver is the region radius and strand budget params.
- NOT benchmarked with `#/bench` — 15 sibling experiments were hammering the
  same GPU during this session, so any recorded numbers would be
  contaminated. The timings quoted above are HUD p50s from the quietest runs
  and should be re-measured solo.
- Honest artifacts:
  - Ribbons are camera-facing planes; extreme close-ups read flat and
    stylized (paddle-like plumes), and the 96-strand budget is a statistical
    proxy, not the 2 M-tri truth — mid-range coverage/color match is the
    design goal.
  - RMS-based width baking overestimates blade width (neighbor blades leak
    into the gather radius); compensated at runtime with `widthScale` 0.75
    default and a 8.5 cm absolute half-width cap.
  - The region→shell handoff is smooth at grazing but visible from topdown as
    a detail change in the far corners (plants → flat mottle at 40 m).
  - The far shell is opaque canopy color; it does not show soil between
    plants from straight above beyond the region.
  - `debugRings` param tints the three rings and bypasses frustum culling +
    forces fixed ribbon width (debug only).
- Camera-inside-plant: width collapse by horizontal distance (0.3–0.8 m).
  First implementation used 3D distance — never triggered because the camera
  floats ~1.5 m above plant bases; worth remembering for other experiments.
- Harness wishlist: nothing blocking; a shared "committed-bake with magic
  validation" helper would remove the copy of 003's SPA-fallback workaround.
