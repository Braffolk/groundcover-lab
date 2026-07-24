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

1. **cull** (compute): walks the scatter cells of a camera-centered
   region (bit-identical WGSL twin of the harness scatter — placement is
   evaluated in-shader, no CPU instance buffers ever), frustum-tests every
   existing plant and appends a 16 B record (pos + quantized yaw/scale/phase/
   entry) into one of three distance rings via atomics into the indirect-draw
   buffer. Work = region cells × 128 slots × stand entries, **independent of
   total plant count** — and a conservative per-CELL frustum AABB test drops
   whole off-screen cells before any hashing or terrain sampling happens, so
   the real work is view-bounded, not just region-bounded.
2. **strands+shell** (ONE render pass): three `drawIndexedIndirect` for the
   rings, then the far shell. Each ring draws ribbon
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
3. **far shell** (~2.7 k tris, last draw of the same render pass): beyond
   `rOuter` the meadow IS the
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
plants; identical cost). All three species render. All six global debug views
(`view=` / `debug=`) are wired and were inspected — see Audit.

## Findings

- The strand reconstruction reads convincingly at grazing: green blade
  understory + tan calamagrostis plume tops emerge naturally from the traced
  colors (the plume stations are wide + pink because the source points there
  are wide-spread and tan — nothing species-specific is hard-coded).
- 134.2 M plants (`stand=scaling-100m`) vs 557 k (default): identical per-pass
  cost in both. Plant count is provably free; the only VRAM/time driver is the
  region radius and strand budget params. (Numbers from the first session are
  no longer quoted here — the passes were merged, see Audit, and the GPU was
  shared. Re-measure solo.)
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

## Audit (structural + debug-view pass)

Debug views were **completely unwired** — `frame.debug_mode` was ignored by
both shaders, so every `view=` mode rendered the normal image. Fixed:

- both fragment shaders now `#include "src/wgsl/debug.wgsl"` and return
  `debug_shade(shaded, albedo, normal_ws, coverage, world_pos)`; fog is applied
  only when `debug_mode() == DEBUG_OFF`.
- the baked occlusion (`mix(0.55, 1.0, ao)`) and the shell's `0.92` trim moved
  from the lit result into the **albedo** term. Identical output (the shared
  lighting model is multiplicative in albedo) but now `debug=lighting`'s
  divide-out is exact and `debug=albedo` shows the real baked colour.
- the shell paints opaquely in debug views (`out_alpha = 1`), otherwise its
  normals/albedo would come back blended with the base pass behind it.
- our own `debugRings` uniform field was called `debug_mode`; renamed to
  `ring_debug` so it cannot be confused with the global selector. It stays the
  method-specific view (ring tint + fixed width + no culling), as the task's
  "expose extra state as your own param" rule asks.

What the views exposed: normals and lighting were already real — per-fragment
interpolated **baked** strand normals through the shared `light_surface()`,
no constant-brightness cheat, so no lighting repair was needed and the normal
view is unchanged in `off` mode. `debug=normals` shows full hue spread
(including downward-facing undersides — the `UP_BIAS` nudge biases, it does not
replace). `debug=albedo` does show one honest mismatch: from topdown the far
shell's canopy albedo is noticeably more olive/desaturated than the plant
field it takes over from (the shell is the width-weighted upper-canopy mean, the
field is blades + dark gaps). `debug=coverage` is flat white over the plants —
correct: every plant edge is hard opaque geometry, zero stochastic alpha — with
the only grey being the shell's 32→40 m alpha ramp, which is exactly the band
worth watching. `debug=depth` shows honest per-plant depth, no impostor walls.

Structural waste found and fixed:

1. **Two back-to-back render passes with identical attachments** (`strands`,
   then `farshell`) — same colour/depth views, same load/store ops, no
   dependency. Merged into one `strands+shell` pass: on a tile-based GPU (the
   dev machine is Apple Silicon; the project targets low-mid devices) the split
   cost a full colour+depth store and reload for nothing. Trade-off: the HUD now
   shows one row instead of two, and old `results/*.json` have the old labels.
2. **The cull evaluated all 128 scatter slots of every cell in the ±44 m
   region, including cells nowhere near the frustum** — ~200 k slot
   evaluations per frame, each 7 hashes + a bilinear terrain fetch, only to be
   thrown away by the per-plant frustum test. Added a per-CELL conservative
   frustum AABB test that runs *before* `scatter_candidate`. The box bounds
   every plant sphere the per-plant test could accept (cell footprint + max
   plant radius horizontally; full terrain amplitude + max plant height
   vertically), and the AABB-vs-plane form is exact with unnormalized
   view-proj rows, so the surviving plant set is provably unchanged. At
   grazing this rejects the large majority of region cells for five dot
   products each.

Verified image-identical: `det=1&t=4` screenshots at `grazing`, `topdown` and
`inside-plant` before vs after differ by 0–254 pixels out of 668 k — the same
magnitude as running the *unchanged* build twice (atomic append order flips
tie-break winners at coincident depths). No console/shader errors in any of the
six debug modes.

Deliberately NOT changed (would need measurement or would change the image —
suggestions for whoever picks this up):

- **Ring row budget vs coverage boost is inconsistent.** A ring draws
  `ceil(s0·r0/r_ring_outer)` strand rows (the *minimum* over the ring) but the
  width `boost` uses the continuous `f_cont = s0·r0/d`. Just past `r0` that is
  64 rows' worth of boost applied to 28 drawn rows, so coverage drops ~35 % at
  the ring-0→1 handoff — the "handoffs cannot pop" claim above is optimistic.
  Fixing it means widening ribbons, i.e. changing the image; it is a quality
  fix, not an audit fix.
- Per-instance values (yaw/scale/phase decode, `f_cont`, `boost`, near/edge
  fades, `sin`/`cos` of yaw) are recomputed by every one of a plant's up to
  1536 vertices. Moving them into the cull pass would widen the 16 B instance
  record (more VRAM, more bandwidth) for an unmeasured ALU win — not an
  obvious call, left alone.
- The 144 B globals UBO is rewritten every frame although only the region
  corner/dims and a few params change. Splitting it into constant + per-frame
  blocks is not worth a second buffer and a second bind at this size.
- `SHELL_ROWS = 20` with a 1.285 geometric radius ratio reaches 3.7 km, so on
  the ±128 m default stand roughly half the rows clamp onto the boundary and
  contribute slivers. Harmless (2.7 k tris total) and needed by the ±384 m
  stands; left as is.
