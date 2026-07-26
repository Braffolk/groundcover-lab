# 031 — part-cloud cards + amortised canopy cache

Seed: *"amortise detail across frames."*

## Idea

Two halves, both aimed at the one thing a billboard cannot do: read as a real
3D object.

**1. The card cloud (geometry).** A billboard fails because a plant is ONE plane:
nothing inside the silhouette can move against anything else. So at bake time the
plant is split into FOUR spatial sub-clumps — the mesh's triangles sorted into
quadrants by their CENTROID (per triangle, never per vertex, so a blade always
lands whole on one card instead of being cut in half and torn apart by parallax),
against the median x/z of the vertex cloud so the four parts carry comparable
mass. Each part is baked as its own little impostor (8 azimuths + a crown view)
against its OWN tight box, so a 0.22m sub-clump gets ~1.8mm per texel — slightly
finer than the baseline's whole-plant card.

At runtime a near plant is that cloud: four camera-facing cards standing at four
different world positions, each choosing the baked azimuth nearest ITS OWN view
direction. Because they are separated in space and write real depth:

- near parts parallax against far parts (0.2–0.35m of depth spread inside one
  plant; at 3m that is ~25px of differential shift for a 1m sideways step),
- the union silhouette genuinely changes with view direction — cards slide across
  each other and occlude each other instead of a flat shape rotating,
- self-occlusion, depth layering and ground contact come out of the depth buffer,
  not out of a gradient. No frag_depth: real geometry writes depth, so early-z
  stays alive.

The cost of overlapping cards is paid back with per-(part, view) **tight coverage
boxes** measured from the baked image: the quad spans exactly the box, typically
~0.7 of the naive card, so four cards cover ~1.5–2x one card's area rather than
4x. Beyond `nearRadius` (16m) the cloud collapses to a single whole-plant card
(part 4) with a per-plant jittered threshold, so the handoff is a staggered band
rather than a sweeping ring — and the far field then IS a billboard, which is why
it costs what the baseline costs.

**2. The amortised canopy cache (the seed).** What is genuinely expensive and
genuinely low-frequency here is not geometry, it is VOLUMETRIC LIGHT: how much
canopy sits above a point and how much sun reaches it. That lives in a
world-anchored 128x8x128 grid (48m window, y = height above the plant base plane)
addressed TOROIDALLY — cell index = world cell & 127 — so following the camera
moves no data at all. Per frame:

- `cs_clear` + `cs_splat` rebuild the density field from the shared scatter
  (bounded by window area, never by plant count) — but only when the window
  actually SCROLLED, because density is a pure function of (stand, seed, window).
- `cs_light` marches each cell toward the sun (12 steps, terrain-gradient-sheared
  because the grid's y axis is terrain-relative) and straight up for sky
  visibility. This is the expensive part: ~131k cells x ~20 taps. Only ONE
  QUARTER of the cells run per frame, round robin. Because the grid is
  world-anchored and plants do not move, a cell's cached value stays CORRECT
  while it waits its turn — amortization costs freshness only in the column that
  just wrapped in at the rim, which is exactly where the renderer fades the
  cache's influence out. A camera jump relights the whole grid in one frame; once
  every slab has had its turn the pass has nothing left to do and is skipped
  entirely.

The renderer buys all of that for ONE trilinear tap: sky visibility darkens the
canopy interior and the plant bases (contact), sun transmittance shadows plants
against their neighbours. Crown cards are sampled at 0.92 of the plant height
because a crown card stands in for the TOP surface of the plant — sampling it at
the height the quad happens to sit at is what made an early build render a dark
disc exactly the size of the cache window.

Per fragment: 1 albedo tap decides a hard alpha test (no dither anywhere); only
survivors pay the normal tap (explicit LOD, computed before the discard so the
tap can live after it) and the cache tap. No loops, no blending, no marching.

O(1) in plant count: nothing is materialized for the stand. Cost is region area +
cache window. `scaling-100m` (134.2M plants) renders at the same per-pass cost and
the same VRAM as `default` (557k) — verified below.

## VRAM budget math

Per species (HUD-verified): **calamagrostis 22.1/25, elymus 21.5/25, poa
22.1/25 MiB**.

- albedo atlas 1792x1792 rgba8, 6 mip levels: 12.25 + 4.1 = **16.4 MiB**
  (45 of 49 tiles used: 4 sub-clumps x (8 azimuths + crown) + whole plant x 9)
- normal atlas 896x896 rg8 (half res — lighting is low frequency), 5 levels:
  **2.0 MiB**
- culled instances, one buffer holding both LOD lists, 16 B each: near list sized
  for `nearRadius` max 40 + far list sized for `regionRadius` max 128, at the
  entry's density: **2.7–3.2 MiB**
- entry info (176 B) + atlas info (880 B) + indirect args (32 B): noise

Shared, not attributable to a species (tagged `canopy-*`, ~1.1 MiB total): density
accumulator 128x8x128 u32 = 0.5 MiB, cache texture 128x8x128 rgba8 = 0.5 MiB,
info uniform 192 B (sized for up to 8 stand entries). Mip levels are capped (6 / 5) on purpose: deeper levels of an
atlas start blending neighbouring tiles into each other.

Bake uses ~450–500 MB transient GPU memory (raw vertex + reordered index buffers,
3584^2 targets, readbacks) — all through `ctx.res` tagged `bake-scratch` and
destroyed at the end of the bake.

## Bake

`bake.ts` -> `mesh/baked/031-temporal-detail/parts-v3-<species>.bin`, **13.8 MiB
each** (1 KiB header + 1792^2 rgba8 albedo + 896^2 rg8 oct normals).

- CPU: median split point from a 512-bin histogram of vertex x/z; one pass to
  classify triangles by centroid and accumulate per-part boxes; one pass to get
  exact per-part radii and counting-sort the index buffer so each part is a
  contiguous index range (~1.5s for poa's 6.5M triangles).
- GPU: 45 orthographic captures in ONE render pass (viewport + scissor per tile,
  dynamic-offset uniform per tile, 2x supersampled), so the vertex cost is only
  2x the baseline's bake for 5x the tiles. Normals are flipped toward the bake
  camera per view (two-sided foliage).
- CPU post: coverage-weighted downsample, 6 dilation passes (tile-clamped, so
  parts never bleed into each other), tight coverage box per tile, oct encode.
- Mips at load time on the GPU: coverage-weighted for albedo; normals are
  decoded, averaged as directions and re-encoded (a box filter over oct pairs
  drifts, which the baseline's notes flag as a known flaw).
- Every load is magic + geometry validated because the dev server answers missing
  /mesh/baked files with index.html at 200, which would otherwise poison both the
  OPFS cache and the committed-file path.

Full first-run bake of all three species (including fetching the raw meshes):
~60s on this machine; later loads read the committed artifacts. Only v3 artifacts
exist — v1/v2 were deleted, not left lying around.

## Status

**working** — verified by headless screenshots at all four standard cams, all five
debug views, `stand=scaling-100m`, and under keyboard-driven motion; zero console
errors in every run.

## Findings

### Speed (A/B, same frame, same GPU — contended, 15 sibling agents rendering)

`#/ab/001-billboard-smoke/031-temporal-detail?stand=default&cam=<cam>&seed=42`,
B/(cache+cull+cards) over A/(cull+cards), 5 samples each, 1400x900:

| cam | ratio (min–median) |
| --- | --- |
| grazing | **1.36–1.41x** |
| topdown | **1.45x** |
| inside-plant | **1.15–1.17x** |
| far-horizon | **1.17–1.18x** |

So ~1.3x the baseline overall, worst case 1.45x — inside the "≤1.5x" bar but
honestly *not* parity. Where it goes: with `nearRadius=0` (pure far LOD) the ratio
at grazing is 1.11x, so the card cloud itself is worth ~0.3x and everything else
(cache pass, the extra cache tap, the tight-box quads) ~0.1x. `nearRadius=14`
gives 1.35x and `nearRadius=20` 1.45x, which is why the default is 16 — plants
past ~16m are under 30px and their internal parallax is invisible anyway.

**No bench numbers are claimed.** Absolute milliseconds this session are
worthless: other agents were hammering the same GPU. Solo HUD numbers on the
quietest runs, default stand, 1400x900, for scale only: total GPU Σp50 4.9
(topdown) – 6.9 (grazing) ms, of which the harness base pass + composite are
~2.0–3.5. One spline bench did get recorded mid-session —
`results/031-temporal-detail__default__p-cef69647__apple-metal-3__2026-07-25T08-16-53-769Z.json`
(orbit-low, 600 frames, 1600x900: cards p50 3.01ms, cache 0.13, cull 0.20, VRAM
23.2/22.6/23.2 MiB) — but it was taken under that same contention and its pass
labels predate the cache+cull merge, so treat it as provenance, not as a result.
Rerun `#/bench/031-temporal-detail?stand=default&spline=orbit-low` on an idle GPU
before quoting anything.

Two structural notes that came out of measuring rather than guessing:
- The cache and the cull are ONE compute pass. Splitting them cost ~0.2ms of pure
  pass overhead on this machine (an *empty* timed compute pass measures 0.24ms
  here) for nothing.
- The canopy build is skipped when its inputs have not changed (window did not
  scroll / every slab already relit), so a parked camera pays only the pass
  overhead. That means the standard static cams UNDERSTATE the cache's cost;
  under motion it is ~0.2ms, and the bench spline will show that.

### Looks — does it beat the baseline?

**Yes at grazing, inside-plant and far-horizon; parity at topdown.** Honest
readings from the wipe/flicker A/B:

- **grazing** (the money shot): the baseline is a uniform sheet of identical pink
  plumes — sharp, but flat, and you cannot see into it. The cloud version has
  visible depth: green understory showing between plumes, dark gaps where the
  canopy is deep, sunlit tops, and per-plant tint variation so the field reads as
  many individuals instead of one stamp repeated ~200k times. Under the flicker
  toggle the baseline's plumes stay rigid while the cloud's internal features
  shift against each other as the camera moves.
- **inside-plant**: clearly better, and it is where the technique is most obvious
  — blades and stems at many depths, sky between them, real mutual occlusion, a
  dark understory brightening toward the tips. The baseline shows a few big flat
  cutouts here.
- **far-horizon**: near-identical by construction (the far LOD *is* a whole-plant
  billboard); the cloud version differs only in the near band and in the canopy
  shading.
- **topdown**: parity. Coverage, brightness and character match the baseline;
  mine has slightly finer rosette detail inside `nearRadius` (four crown cards at
  four heights instead of one disc) and this is its worst cost ratio. If topdown
  cost ever mattered more than that detail, dropping the near crowns to the single
  whole-plant crown is a 3-quads-per-plant saving with no visual loss.
- **parallax check**: a pure lateral camera translation (same yaw/pitch, 0.4m
  steps, `cam=<pose>` in the URL so both methods get bit-identical cameras) moves
  the cloud's near cards visibly against its far cards and re-orders which parts
  occlude which; the baseline's card can only slide as one piece. The
  `debug=depth` view is the objective version of the same claim: inside a single
  near plant's silhouette there are several distinct depth tones, where a
  billboard has one.

### Honest artifacts and limits

- **Under ~1m the cloud gets mushier than the baseline.** Four cards at ~3x
  texture magnification, mutually alpha-tested, read as a mottled patchwork; the
  baseline's single card at the same magnification is soft but coherent. The
  camera-inside fade (per plant, on the whole-plant radius, eroding through the
  alpha reference) takes over below ~0.5m.
- Azimuth snapping still exists (8 views), but it is much less visible than the
  baseline's: the four cards of one plant sit at different positions, so they
  switch views at different moments and the plant's macro structure stays
  continuous.
- Crown cards remain the weakest element, as they are for the baseline. They are
  dropped below ~25° elevation, and below ~38° when seen from underneath (from
  below a crown shows the straight-down capture, i.e. the wrong face — that fix
  removed the pale floating cutouts at inside-plant).
- The canopy cache's influence fades to a neutral constant over the outer ~15% of
  its 48m window, which is also where a freshly scrolled column may be a frame or
  two stale. Nothing was visible in the motion test (fly forward, capture while
  moving, `cacheSlabs=1` vs `4`): no seam, no band, no popping.
- Sun shadowing is floored at 0.45 rather than going dark, and sky occlusion
  multiplies the whole light term rather than the ambient only. Both are honest
  approximations chosen to keep `light_surface()` as the single lighting call;
  they are documented in the shader where they happen.
- Wind does not move the cache (deliberate: the cache is a low-frequency light
  field, and re-splatting swaying plants would break the world-anchored
  correctness that makes the amortization free). Foliage still sways; only its
  self-shadowing is computed against the un-swayed canopy.

### Plant-count independence

`stand=scaling-100m` (134.2M plants, ±2048m) at grazing: cache+cull 0.43ms,
cards 2.31ms, VRAM 22.1/21.5/21.5 MiB — the same as `default` (557k plants).
Cost is set by `regionRadius`, `nearRadius` and the cache window, never by the
stand.

### An idea that did NOT work (kept here so nobody repeats it)

The near LOD looked sparse from above in the first build, so I baked per-tile
**coverage-matched alpha scales**: for each tile, find the threshold whose
surviving texel fraction equals the tile's true covered area (mean alpha) and hand
the runtime `0.4/a*`. Two lessons:

1. It was unnecessary. Every tile came out wanting a scale ≤ 1, i.e. a hard test
   at 0.4 already keeps MORE area than the plant really covers. The sparse-looking
   near region was the crown-card cache-height bug described above, not coverage.
2. Where it did apply it fought the mip chain: boosted alpha keeps semi-covered
   texels, and at coarse mips (a plant is ~17px from the topdown cam) the whole
   tile drifts toward its mean and turns into a muddy mat instead of collapsing
   into crisp specks. Distant coverage collapse is the baseline's accidental LOD
   and it looks better than filling it in.

The machinery was removed (it also overran the 1 KiB header by 5 floats, silently
zeroing the last five tiles' scales — caught by dumping the artifact header, worth
doing for any new format).

### Harness wishlist

- A dev-server 404 (instead of the SPA fallback) for missing `/mesh/baked` files
  would let every experiment drop the magic-validation shim.
- The runner does not write the live camera pose into the URL, so scripted
  verification cannot capture "the pose I am looking at". `cam=<x,y,z,yaw,pitch,
  fov>` works for playback, but building a pose needs the terrain height, which is
  only reachable by importing `src/scene/terrain.ts` into the page and
  re-generating the heightfield with a stub scope. A "copy pose" button, or
  writing the pose to the URL on camera change, would make repeatable A-vs-B
  camera studies trivial.
