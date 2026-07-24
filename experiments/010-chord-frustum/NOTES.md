# 010-chord-frustum — chord-field frustum proxies

## Idea

Each plant (clump) is a **convex cone-frustum proxy volume** whose INTERIOR
appearance is precomputed. The proxy is real rasterized geometry (an 8-sided
tapered prism fitted per species from the raw mesh), so the silhouette is
correct from every angle — grazing, overhead, below. The fragment shader on
the proxy surface answers "what would the eye see inside this volume" with a
single precomputed lookup:

1. Intersect the true eye ray with the ANALYTIC cone frustum in closed form
   (one quadratic + a y-slab; no marching) → the ray's **chord**: entry point,
   exit point, entry/exit faces.
2. Chart both points on the proxy surface with a (azimuth φ, meridian m)
   chart, where m runs bottom-cap-center → bottom rim → side wall → top rim →
   top-cap-center (one seamless 2D chart of the whole convex surface).
3. Look up a baked **4D chord light field**: outer atlas grid = entry bin
   (φe 48 × me 32), inner tile = chord shape (Δφ = exit−entry azimuth 32+1
   gutter column for seamless wrap × exit meridian 24). Stored per chord:
   premultiplied albedo, coverage, first-hit fraction t01 along the chord,
   oct-encoded mesh normal.
4. Runtime filtering is genuine 4D interpolation at 8 texture taps: manual
   bilinear over the 4 nearest entry bins × hardware bilinear over the inner
   chord-shape tile (the gutter column makes Δφ wrap filterable).
5. The blended t01 reconstructs a real 3D hit point → written as
   `frag_depth`, so plants inter-occlude and intersect terrain correctly;
   the baked normal drives the shared lighting.

Wind is an **inverse shear on the ray** (borrowed insight from 004): sway
displaces the proxy vertices linearly in y, and the fragment shader applies
the exact inverse shear to the eye ray before all chord math — a shear maps
lines to lines, so this is exact, not an approximation. Per-species sway from
the stand table, phase per plant, all animation from `frame.time` only.

Coverage resolve is a hard alpha test against a **world-anchored organic
pattern**: the threshold is jittered by a hash of the ~2cm quantized
plant-local reconstructed hit cell. This is NOT screen-space dither — the
pattern rides on the reconstructed surface, so it neither screen-doors nor
swims; it turns mid-coverage chord bins into stable foliage clumps instead of
solid ovals. Justification (taste rule): partial coverage IS what a chord
through a translucent volume returns; a plain hard threshold makes topdown
(α≈0.25–0.5) and grazing (α≈0.1–0.9) mutually unsatisfiable — verified both
ways in screenshots before adopting this.

Fade rules: far fade at the region edge, near fade within ~1 proxy radius,
inside fade when the geometric entry goes behind the eye (camera inside the
volume) — all multiply coverage, so plants erode organically rather than pop.

**Plant-count independence:** a per-frame compute pass enumerates the scatter
slots of a camera-bounded cell region (the `scatter_candidate` WGSL twin, so
placement is bit-identical to the stand), frustum/distance-culls, and
compacts survivors into an indirect draw. Work per frame is bounded by
regionRadius, never by stand size; nothing per-plant lives on the CPU.

## VRAM budget math (per species)

- chord surf atlas: 1584×768×4 B rgba8 = 4.87 MB
- chord geom atlas: 1584×768×4 B rgba8 = 4.87 MB
- instance buffer: the region at the max radius (80 m) is 41² cells × 128
  candidate slots = 215k slots, but a slot only EXISTS with probability
  density/8, and only existing plants are ever appended. Sized at
  `slots × density/8 × 1.15 + 4096` — over 2×10⁵ Bernoulli trials the count is
  within a few hundred of its mean, so the margin is unreachable (and the cull
  shader drops overflow gracefully). density 3 → 96,888 × 32 B = 3.10 MB
  (elymus, density 2.5 → 2.61 MB).
- proxy index buffer 192 B, info/cull uniforms + indirect: ~160 B
- Total ≈ 12.8 MB (HUD: 12.2/25 MiB per species; 11.8 for elymus). Under
  budget with room to spare — was 16.6 MB before the audit right-sized the
  instance buffer.

Bake-time transients (~130 MB G-buffer + mesh buffers) are raw allocations,
destroyed at the end of the bake; they never coexist with rendering.

## Bake

Two GPU stages per species (`bake.ts`), tens of seconds total for all three
species on this machine, cached via bakedArtifact/commitBake
(`mesh/baked/010-chord-frustum/chords-v6-<species>.bin`, 9.7 MB each):

1. **View captures**: the raw GCMESH1 mesh rasterized orthographically from a
   full-sphere 20×20 octahedral grid of directions (400 views, 256px tiles →
   5.5 mm texels; albedo+mask, oct normal, depth32). Rendered in 2 row-batches
   of 200 views to cap transient memory (~130 MB); chunked submits (25 views)
   so a 6.5M-tri mesh cannot TDR.
2. **Chord gather**: one compute thread per chord texel picks the nearest
   view for its chord direction and **marches the chord in 2-texel steps
   against that view's first-hit depth image** (bake-time only — the runtime
   never marches): a hit is the first step whose pixel's surface lies within
   a 2.5-texel tube around the chord. 8 jittered parallel chords per bin
   prefilter coverage into an honest fraction. This replaced a 3-iteration
   fixed-point parallax solve that silently failed for long near-vertical
   chords (fluffy discontinuous depth + ~9° view mismatch).

The frustum fit is per-species from the vertex cloud: per-height-band max
radius about the bounds axis, least-squares line inflated to contain all
bands (+3%). Fits: calamagrostis r0=0.53→r1=0.47 h=1.15; elymus 0.63→0.40
h=1.19; poa 0.18→0.18 h=0.77.

Gotchas hit and fixed along the way (documented for reuse):
- `sign(0.0)=0` in the octahedral ENCODE collapses straight-up/down chord
  directions onto the map center → topdown sampled the wrong views and
  rendered almost nothing. Fixed with a never-zero sign.
- 128px view tiles (1.1 cm texels) fatten sub-mm fluff into solid blobs —
  coverage saturates and the fluffy heads render as solid pink ovals. 256px
  tiles + fractional 8-chord coverage + the anchored threshold pattern fixed
  it.
- The dev server answers missing /mesh/baked files with index.html (200), so
  the committed-artifact path can hand back poisoned bytes; the loader
  validates a magic header and rebakes.

## Audit (structural waste + debug views)

Second pass over the finished method: wire the global debug views, then hunt
structural waste. No frame times are quoted — the GPU was shared with other
agents during this pass, so every fix below is justified from the code alone.

**Debug views (were entirely missing).** `render.wgsl` now includes
`src/wgsl/debug.wgsl`, applies fog only when `debug_mode() == DEBUG_OFF`, and
returns `debug_shade(color, albedo·var, n, alpha, hit_w)`. What each mode
turned out to show — the method needed no repair here, only exposure:
- *normals*: real per-fragment normals. They are the baked mesh normal at the
  chord's first hit (coverage-weighted mean of the 4 entry-bin taps), oct-decoded,
  yawed into world, flipped towards the eye. At `inside-plant` they resolve
  into coherent blade-shaped patches; at `grazing` they look like coloured
  noise, which is honest — one pixel there covers many sub-mm blades.
- *lighting*: goes through the shared `light_surface()`, once. Albedo is baked
  premultiplied by COVERAGE (not by light) and divided back out at runtime, so
  nothing is double-lit. The view is bright/clipped over most of the frame, but
  so is the terrain's — that is the shared half-lambert model, not this
  renderer.
- *albedo*: the raw baked chord colour × the anchored-noise variation actually
  used for shading (so the *lighting* view's division is exact).
- *coverage*: the resolved chord coverage after the far/near/inside fades —
  exactly the number the hard alpha test judges.
- *depth*: continuous through the field with no proxy-shaped plateaus,
  confirming `frag_depth` really is the reconstructed hit, not the proxy hull.
- `debugChart` stays an own-manifest param (chord chart coords) and only
  applies in the `off` view.

**Fixed — vertex-stage/uniform work that was done per fragment.** The
fragment shader recomputed, for every fragment, quantities that are constant
over the whole instance: the wind shear `wl`, the shear factor, and the eye
position transformed into the plant-local unsheared frame — three
`rot_y(·, ±yaw)` calls, i.e. six transcendentals per fragment, on the most
fill-bound pass in the experiment. They are now computed once in the vertex
stage and passed flat (`ray_o_shear`, `shear_cs`), with `cos/sin(yaw)` passed
down so the two remaining fragment rotations (`rot_y_cs`) cost no
transcendentals at all.

**Fixed — the proxy was drawn non-indexed.** Its 32 triangles have only 18
distinct vertices (8 bottom ring, 8 top ring, 2 fan centers), so `draw` was
shading 96 vertices per plant — 5.3× redundant. Now a 192 B index buffer +
`drawIndexedIndirect`. Bonus: the ring seam used to be `cos(2π)` on one quad
and `cos(0)` on the next (a hairline crack waiting to happen); indexing makes
them literally the same vertex.

**Fixed — per-frame uploads of values that never change.** `update()` rewrote
both uniform buffers for every stand entry every frame. The render uniform
holds only the baked frustum fit + params, so it is now written at init and
from `onParamsChanged`. The cull uniform additionally carries the region's
corner cell, which changes only when the camera crosses a 4 m cell boundary —
written on that transition. Only the indirect counter reset is genuinely
per-frame now.

**Fixed — instance buffer sized for an impossible worst case** (see VRAM math
above): 6.89 → 3.10 MB per species, total VRAM 54.3 → 43.0 MB.

**Verified unchanged:** before/after screenshots at `grazing`, `topdown`,
`far-horizon` differ in 0.15–1.2% of pixels, all isolated single pixels
scattered over the field with no structure, no silhouette shifts and no
missing plants. That is the expected last-ULP jitter of the world-anchored
alpha-test hash (`floor(hit_u·48)` flips a cell when the reconstructed hit
moves by 1e-7), caused by evaluating the ray setup in the vertex stage instead
of the fragment stage. The indexed-draw change on its own is bit-identical
(0–3 pixels of 247k). Typecheck clean, console free of WebGPU errors.

**Deliberately left alone** (each needs a bench to justify, which this session
cannot provide honestly):
- No depth prepass. `frag_depth` disables early-z, so grazing overdraw is
  shaded then late-z rejected. A depth-only proxy prepass (or front-to-back
  ordering by entry distance) is the obvious next experiment, but it doubles
  the geometry pass and only pays off if overdraw dominates — measure first.
- The cull enumerates all 128 candidate slots of every cell in the region
  square. That is forced by the shared scatter twin (existence is per-slot), so
  it cannot shrink; a per-cell early-out (cell AABB vs frustum, or cell-to-camera
  horizontal distance vs max_dist) would skip whole cells before the hash +
  terrain sample, and the square's corners alone are 21% of the threads. Left
  out because the cull pass is a rounding error next to the fragment stage and
  a sloppy conservative bound would pop plants.
- The 5 frustum planes are rebuilt per cull thread rather than uploaded as a
  uniform. Uploading them would make the cull uniform per-frame again, undoing
  the fix above, for ~40 ALU next to a hash chain plus a terrain texture fetch.
- Skipping the cull dispatch entirely while the camera and params are
  unchanged. Real win for a parked camera, but it needs every frame-uniform
  input (pose, viewport/aspect) tracked or plants pop — not worth the hidden
  dependency for a case where nothing is under load anyway.
- The bake's second gather dispatch covers the whole atlas and early-outs on
  the non-resident view rows (two batches, ~half the threads wasted each). Bake
  time only, and it keeps the batching logic trivially correct.

## Status

**working** — verified by headless screenshots at `grazing`, `topdown`,
`inside-plant`, `far-horizon` (default stand, det=1, t=2), all three species
rendered from the stand at their exact placements. Typecheck clean. Timing
HUD numbers during development were contended (many parallel agents on one
GPU), so no perf claims here; bench properly on an idle GPU
(`#/bench/010-chord-frustum?stand=default&spline=orbit-low`) before quoting.

## Findings

- The two-point (entry, exit) chord parameterization on a proxy-native chart
  WORKS as the loophole: runtime is one quadratic + 8 filtered taps, zero
  marching, and depth/normals come along for free. The gutter-column trick
  makes the wrap-around azimuth axis hardware-filterable.
- Rendering the proxy's FAR side (cull front) is strictly better than the
  near side: no near-plane clipping, and camera-inside-plant still shades,
  which makes the inside fade trivial.
- Chord fields integrate THROUGH the volume, so overlapping fluffy-head
  regions read denser/pinker than surface impostors (005) at grazing — more
  physically honest, but the bin-mean color smears fine detail; the anchored
  noise buys back texture but not structure. Entry bins are ~5cm — close-ups
  (<1.5 m) stay soft/blobby by construction.
- Coverage semantics are the crux of the method: what "a ray hits" means for
  a 5.5 mm-sampled proxy of sub-mm geometry decides the whole look. The
  effective ray thickness (view texel + tube radius) inflates optical density
  relative to ground truth; a principled transmittance correction (bake true
  per-texel alpha at higher supersampling and composite T = Πα along the
  march) is the clear next step.
- Known artifacts, honest list: (a) mid-alpha cap chords can survive while
  side chords erode → occasional detached "cap flakes" against the sky in
  upward views; (b) plants pop in/out at the regionRadius ring (far-horizon
  shows bare hills beyond ~56 m — no distant-collapse layer); (c) overdraw at
  grazing is real (frag_depth disables early-z; late-z still culls after
  shading) — a depth-only proxy prepass or rough front-to-back entry ordering
  would help; (d) topdown reads sparser than intuition — measured top-down
  pixel coverage of these meshes at 5.5 mm really is ~0.25, but the hard
  threshold thins it further.
- No bench JSONs recorded: frame times were contaminated by parallel agent
  GPU load throughout this session (see Status).
- Species coverage: all three stand species baked and rendered
  (calamagrostis-canescens, elymus-repens, poa-pratensis). Tile meshes are
  treated as one clump per plant instance (same convention as the other
  impostor experiments); no periodic wrapping.
