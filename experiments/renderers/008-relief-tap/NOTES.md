# 008-relief-tap — secant relief cards

## Idea (carpets)

A ground carpet is the case this method was waiting for. A moss mat is a lumpy,
mostly-single-valued surface seen from above, which is exactly the shape relief
mapping is good at — unlike a tall grass clump, where the heightfield is
multi-valued in every direction. So a carpet species (`stand_table[e].carpet_div
> 0`) gets its own entry-point pair, `vs_carpet`/`fs_carpet`:

- **one ground-parallel, tile-sized quad**, terrain-conformed per vertex, using
  the stand's 90-degree yaw and constant scale as given. No camera-facing card:
  it would break the lattice and a 7 cm cushion seen edge-on shows nothing.
- **one baked view instead of 25**, cropped to the periodic tile square and
  mipmapped, so the whole budget goes into the detail you actually look at.
- **the same fixed 3-tap secant solve**, run against that view's heightfield
  with the capture axis pointing along the local ground normal, writing real
  `frag_depth`. This is where it beats a flat billboard: 001-billboard-smoke's
  quad has no thickness, so the mesh's ~22 mm of cushion relief is entirely
  lost, while here the cushions have parallax, crevices and depth.

## Idea (scattered plants)

The relief/parallax-mapping family, pushed to a FIXED tiny tap count: what can
depth-augmented cards buy without any per-frame marching.

Each species is baked once into a small 5x5 hemi-octahedral fan of
**depth-augmented orthographic captures** (25 views, horizon ring to zenith):
albedo+coverage, plus a signed **heightfield** h(u,v) — the distance of the
foliage surface from the plant's mid-plane along the capture axis — plus baked
local-frame normals. At runtime every plant is one camera-facing card whose
fragments intersect the TRUE eye ray with the selected view's heightfield
using exactly three texture taps and one division (secant / false position):

1. tap h0 at the card point; error e(0) = f·o − h0,
2. analytic reprojection: slide along the ray to the plane f·p = h0
   (t1 = (h0 − f·o)/(f·d) — this alone is the classic one-step parallax),
   tap h1 there; e(t1) = f·(o + t1 d) − h1,
3. secant solve t2 = t1 − e1·t1/(e1 − e0) — the root of the linear model
   through two EXACT samples of the ray-vs-heightfield error function — and a
   final surface tap (albedo/normal/height) at t2.

No loop, no march: the tap count is a compile-time constant (param
`reliefMode` exposes 1/2/3-tap variants for A/B). The solved hit is snapped
onto the sampled height sheet and written as real `@builtin(frag_depth)`, so
plants inter-occlude by their reconstructed 3D surfaces (no billboard
cardboard), and the interior of a clump genuinely opens up with parallax as
the camera moves — within one view cell the motion is continuous, not a
crossfade.

Two more tricks:

- **Stochastic view selection.** Instead of blending the 4 nearest view cells
  (4x the taps, ghosting), each pixel picks ONE view with probability equal
  to its bilinear weight, from the shared deterministic PCG hash of its pixel
  coords. Expectation equals the bilinear blend; at a node direction all
  pixels agree; in between, view mismatch appears as foliage-friendly grain
  rather than double edges. `viewSelect=nearest` shows the popping variant.
- **Wind as inverse ray shear** (borrowed from 004, composes perfectly with
  relief): sway is a linear-in-height shear, and lines stay lines under
  shear, so the eye ray is inverse-sheared into the un-swayed baked frame —
  exact for the model, zero cost in the bake.

Plant-count independence: a GPU cull pass walks only the scatter cells within
`maxDist` of the camera (fixed-size dispatch of the shared `scatter_candidate`
WGSL twin, stand-region clamped exactly like `Scatter.region()`, plus
sphere-vs-frustum test), appends survivors via atomics, renders with
`drawIndirect`. 557k (default) and 134.2M (scaling-100m) plants issue
identical work — verified visually, same VRAM, same frame shape. Cards are
sized to the species' PROJECTED tight box, not the bounding-sphere square
(~2x overdraw saving on these tall-narrow species).

Camera-inside-plant: dithered dissolve when the camera enters the bounding
sphere (near fade), same dither for the far fade at `maxDist`.

## VRAM budget math (per species)

Scattered plant (5x5 view fan):

| item | bytes |
|---|---|
| albedo atlas 1280x1280 rgba8 | 6.55 MB (6.25 MiB) |
| geom atlas 1280x1280 rgba16float (h, oct normal) | 13.11 MB (12.5 MiB) |
| culled instances `ceil(pi*88^2*density*1.15)*32B` (88 m = maxDist cap) | 2.69 MB @ d3 / 2.24 MB @ d2.5 |
| ubos + indirect args share | < 300 B |
| **total (default stand)** | **21.3 / 20.9 / 21.3 MiB** ✓ (HUD confirms) |

Carpet species (one cropped zenith tile + mips) — the split is completely
different, and this is the point of the carpet bake rather than a side effect:

| item | bytes |
|---|---|
| albedo 512x512 rgba8 + 9 mips | 1.40 MB (1.33 MiB) |
| geom 512x512 rgba16float + 9 mips | 2.80 MB (2.67 MiB) |
| culled tiles `ceil(pi*88^2 * (484/16) * accept)*16B`, accept = wetWidth*1.25+0.35 = 0.77 | 9.03 MB (8.61 MiB) |
| **total (bog stand, per moss state)** | **12.6/25 MB** ✓ (HUD confirms) |

Notes on the carpet numbers:
- 512 texels across a 0.18 m tile is **0.35 mm/texel**, about 1.8x the linear
  density 001-billboard-smoke gets from its cropped 512 top view and 3.7x what
  this experiment's own 5x5 atlas managed (~140 texels per tile, 1.29 mm). It
  cost 4.2 MB instead of 18.75 MB, because 24 of the 25 views were grazing
  captures of a 7 cm cushion and stored essentially nothing.
- The instance buffer is now the **dominant** cost, so the stride went 32 B → 16 B
  for carpets: a mat has one constant scale and no wind phase, so position+yaw is
  all there is. Capacity is sized for expected survivors (~`wetWidth` of the
  484 slots, plus headroom because wetness is damped on slopes and the driest
  zone can then claim well over its nominal third), not for all 484 — sizing all
  three moss entries for every slot would waste 3x, since only one entry can own
  a grid node. Verified no clamping at `maxDist=88` (the cap): the mat has no
  holes at either 64 or 88.

Bake-time resources (source mesh upload, up to ~238 MB index buffer for the
moss meshes, render targets, readbacks) are transient, created outside
`ctx.res` and destroyed right after the bake — they trip the dev "untracked"
console warnings once per session, expected.

## Bake

Two artifacts, picked by whether the active stand lays the species out as a mat.

**Carpet** (`carpet-v2-t512-<species>.bin`, 3.15 MB) — ONE zenith ortho capture
cropped to the tile square: projection centre `(tileM/2, ., tileM/2)`, half
extent `tileM/2`, which also normalises the height channel (h lands in ~±0.52).
Two details that took a measurement to get right:

- The mesh holds **one** tile's geometry, and that geometry overflows its period
  (0.24 m of cushion inside a 0.18 m tile). The periodic image inside the crop
  is therefore the tile *plus the overflow of its eight neighbours*, and the
  first version rendered a single copy — leaving a band along every edge
  under-covered (measured 0.73–0.87 alpha against 0.95 interior), which showed
  up as a faint dark grid over the entire mat in `debug=coverage`. v2 composites
  all nine copies into the same window (offsetting the projection centre is
  equivalent to translating the mesh; the depth test resolves overlaps). Border
  coverage is now 0.96–0.99, flat against the interior, and the tile is
  genuinely periodic — which is what makes address mode `repeat` correct, so the
  relief march walking off an edge needs no clipping at all.
- One view has no neighbouring view cells to bleed into, so unlike the 25-view
  atlas it **can** carry mips. The chain is built on the CPU at load: colour and
  height coverage-weighted (a half-empty texel must not drag the canopy height
  toward the −1 background), normals decoded to unit vectors, averaged,
  renormalised and re-encoded — oct codes are not mip-averageable. The stored
  rgb is therefore already normalised by coverage at every level and the shader
  does **not** divide by alpha. The load pass also measures the coverage-weighted
  mean and std-dev of h, which set the card plane height and the crevice ramp.

**Scattered** (`views-v1-g5t256-<species>.bin`, unchanged) — renders the mesh 25
times (one ortho draw per hemi-oct node, viewport + dynamic-offset uniform) into
the two atlases; a
depth32 buffer resolves self-occlusion, normals are flipped toward the
capture axis (two-sided foliage). After readback, covered texels are
**dilated** 4 rings into empty neighbours per tile (raw byte copies, coverage
kept 0) so parallax reprojections that land just past a silhouette read a
plausible height/color instead of background; empty background height is −1
(far side) so rays travel THROUGH gaps to whatever is behind.

Both go through the standard harness bake flow (`bakedArtifact()`: OPFS cache →
committed `mesh/baked/008-relief-tap/<key>.bin` → in-browser bake, then
`commitBake()`), so a normal load never fetches a raw source mesh at all. Header
is the same 64 B for both (magic `'R8TP'`/`'R8CP'`, version, px, grid, centre /
radius / halfExt / topH), magic+size validated on load — the SPA-fallback HTML
that poisoned other experiments' caches can only cause a rebake here, never a
bad atlas. The stale moss view-fan artifacts (3 x 18.75 MB) were deleted; the
carpet artifacts replacing them are 3 x 3.15 MB.

## Status

**working** — verified by headless screenshots on `default` at `grazing`,
`topdown`, `inside-plant` (dissolve works), `far-horizon`, plus
`scaling-100m` at grazing (134.2M plants, identical cost/VRAM) and
`close-quality` for the 1/2/3-tap comparison. All three species render at the
stand's exact placement via the scatter twin. Typecheck clean, no console
errors.

On `bog`: `grazing`, `topdown`, `inside-plant`, `carpet-close`, a 0.5 m oblique
(`cam=0,-7.43,0,0,-0.35,60`), a 3 m oblique, a ridge view
(`cam=24,0,24,-0.785,-0.32,60`), `debug=normals|lighting|coverage|albedo`, and
`maxDist=88` (capacity headroom). `default` at `grazing` and `inside-plant` is
**pixel-identical** to before this change (RMSE 0.000/255, 0 pixels differing by
more than 4/255 over a 920x350 crop) — the scattered-plant path was not touched.

## Debug views

The fragment shader routes its final colour through the shared
`debug_shade()` (`src/wgsl/debug.wgsl`), so the global `view` selector
(`debug=albedo|normals|lighting|coverage|depth`) works here exactly as it does
on the terrain; fog is applied only in `DEBUG_OFF`. What each mode reads:

- **albedo** — `surf.rgb` of the solved hit times the depth-in-clump AO ramp
  (the same value handed to `light_surface`, so **lighting** — which divides
  it back out — is exact).
- **normals** — the baked per-texel local normal at the hit, oct-decoded and
  yaw-rotated. It is genuinely per fragment and genuinely high frequency:
  one texel of a 256px view covers several blades of a 2M-tri plant, and
  stochastic view selection means neighbouring pixels can read different
  captures. The lighting term measured over a grazing crop spans 0.13 → 1.0
  (mean 0.76), i.e. real half-Lambert variation, not a constant.
- **coverage** — the bilinear alpha that was alpha-tested, so it is bounded
  below by `covThresh` (0.38) by construction: white = solid interior, grey =
  silhouette texels sitting on the threshold.
- **depth** — from the reconstructed hit written to `frag_depth`, so it shows
  the relief displacement, not the card plane.

Method-specific state lives in the experiment's own `inspect` param (`off` /
`height` = the solved hit's relief height along the capture axis / `view-cell`
= which of the 25 baked views each pixel sampled — the stochastic dither shows
as two-colour grain, and the cell boundaries sweeping across the meadow are
the view quantization made visible).

The carpet path routes through the same `debug_shade()` and honours the same
`inspect` modes, with two differences: `coverage` is bounded by the
carpet-specific reference (0.10) rather than `covThresh`, and `view-cell` shows
the isotropic mip level instead of a view index, since a mat has exactly one
baked view and the texture footprint is what actually varies. `normals` is the
blended cushion-scale + per-leaf normal lifted into the local ground frame.

## Moss round — what changed, smallest first

1. **`viewSelect` was wired backwards.** `VIEW_MODES` in `main.ts` listed
   `['nearest', 'stochastic']` while the shader reads index 0 as *stochastic*, so
   the option labelled `nearest` selected stochastic and vice versa. Fixed the
   order and moved the manifest default to `stochastic`, which is what the
   experiment has actually always rendered — the pictures are unchanged, the
   labels are now true.
2. **The cull evaluated 128 of a carpet's 484 slots.** `cs_cull` was one
   workgroup of 128 invocations per cell using `local_invocation_index` as the
   slot, which for the bog's `carpet_div = 22` rendered grid rows 0..5 of 22 in
   every 4 m cell: the mat came out as ~1 m stripes of moss with 3 m of bare peat
   between them, on every camera. Now one thread per (cell, slot) with the slot
   count from `standEntrySlots(entry)`, and the dispatch simply takes more
   workgroups per cell for a carpet. **This was the single biggest visual fix.**
3. **Instance capacity and stride** derived from carpet slots and zone
   acceptance rather than from `density`, at 16 B per tile (see the budget table).
4. **A carpet path in the renderer** (`vs_carpet`/`fs_carpet`, its own pipeline so
   the grass path is byte-identical): one ground-parallel tile-sized quad,
   footprint from `stand_table[e].footprint_m`, yaw and scale taken from the
   stand as given, no camera-inside fade, region fade eroded through the alpha
   reference rather than dithered, and a carpet-specific alpha reference (0.10 vs
   the grass 0.38) so mip-reduced distant coverage cannot punch holes in a closed
   mat.
5. **A carpet bake**: one zenith capture cropped to the tile square, mipmapped,
   nine-fold periodic (see Bake). This is the budget-split fix — 4.2 MB of tile
   detail instead of 18.75 MB of views a cushion never shows.
6. **Anisotropic gradient sampling.** The first working version collapsed the
   footprint to one mip level with `max(|ddx|, |ddy|)`, and since a mat is seen
   at grazing angles nearly always, that level is chosen for the long axis: the
   entire field became a flat khaki wash of per-tile mean colours, with the zone
   boundaries reading as hard blocks. Feeding the gradient pair to
   `textureSampleGrad` against a `maxAnisotropy: 16` sampler keeps the
   across-view axis sharp. Gradients come from the smooth card-plane uv, never
   from the solved uv, whose derivatives blow up across a relief discontinuity.
7. **Card plane at the mean canopy height, relief solved on a smoothed sheet.**
   Measured on the baked tile, this heightfield has ~10 mm of variation within any
   3x3 texel neighbourhood at *every* mip level — i.e. ~13:1 slopes at 0.35 mm
   texels, p05..p95 spread 22–30 mm. Textbook relief mapping puts the plane at the
   maximum so nothing is unreachable, but from there every pixel must march ~20 mm
   through that spike field and three taps resolve it to noise (this is exactly
   what the first version looked like: correct coverage, zero structure). Plane at
   the coverage-weighted mean + height taps at mip >= 4 (~5.6 mm texels, where the
   slopes are about 1:1) turned it into coherent cushion parallax.
8. **A cushion-scale normal from the smoothed height gradient**, blended 55/45
   with the baked per-leaf normal. `debug=normals` showed why: filtered to the
   pixel footprint the baked normals average to straight up plus high-frequency
   grain — honest for a horizontal cushion, but it left the mat with no macro
   shading, so all that intricacy read as flat noise. The centre height for the
   gradient is free (the secant's root already *is* h(p2)), so this costs two
   taps. Crevice darkening was also re-scaled by the tile's own height spread:
   the cushion sits in the top quarter of the mesh box, so the old ramp over
   `[0, top_h]` only spanned 0.65..0.90 and did nothing.

### Terrain fitting: rung 3 (per-vertex conforming)

Every quad vertex takes the ground height under its own xz from
`terrain_sample` (height and the (nx, nz) of the normal in one bilinear fetch).
Neighbouring tiles share corner positions, so the mat is C0 continuous across the
whole field. Rungs 1–2 are not merely cheaper here, they are wrong: a per-tile
plane fit makes adjacent tiles fit different planes, and they crack apart along
their shared edge. `slope_align` is 1 for a carpet and is honoured implicitly —
the quad *is* the ground surface.

The relief ray frame needs a single linear basis over the primitive, so it uses
the ground normal at the tile **centre** (`plant_basis_from_up` inlined, since the
card already carries cos/sin of the yaw). That is the one place a cheap rung is
used, and it cannot crack the surface: positions still come from the per-vertex
conforming, and the relief hit is reconstructed *relative to the interpolated
conformed position*, not to the tangent plane — otherwise a tile straddling a
heightmap texel boundary would disagree with its neighbour by up to a few cm,
which is the same order as the whole cushion's thickness.

## Findings — moss carpet

- **The tap ladder matters far more on moss than on grass.** At the 0.5 m oblique
  camera: `flat-1tap` is a blurry flat mottle with no cushions at all (it is
  essentially 001-billboard-smoke); `linear-2tap` finds the cushions but drags
  visible vertical smears along the view direction where the single reprojection
  overshoots; `secant-3tap` resolves them into clean mounds with lit tops,
  shaded flanks and dark crevices. On grass the three modes differ subtly; here
  they differ completely.
- **vs 001-billboard-smoke at 1 m** (`#/ab/001-billboard-smoke/008-relief-tap?stand=bog&cam=0,-7.43,0,0,-0.35,60`):
  001 is uniform fine grain on a flat plane — good moss *texture*, no cushions.
  This is a field of distinct mounds with real occlusion between them. At grazing
  the two are near-indistinguishable (see below), and 001 is slightly brighter.
- **The light term saturates on a mat, so the 3D reads through AO, not NdotL.**
  Over a 0.5 m oblique crop `debug=lighting` is mean 0.95, p05 0.72, p50 1.00:
  a horizontal surface under a high sun with half-Lambert is simply at full
  light, and a 45-degree cushion facet still clips. That is physically right and
  not fixable from inside an experiment (`light_surface` is harness-owned), but it
  means the cushion shape is carried by the crevice ramp and the albedo, with the
  macro normal only contributing at the flanks. It is also why the mat looks a
  touch darker than 001 — `ao` is the knob.
- **At `cam=grazing` the mat is inherently a far-field view here** and both this
  and 001 read as a filtered wash. The terrain falls away from the grazing
  bookmark, so the nearest visible ground is ~30 m out, where a 0.18 m tile is
  under 2 px. Judge tile-level quality at `carpet-close` or a 0.5–3 m oblique.
- **The tile-sized blocky steps along the wetness zone boundaries are placement,
  not rendering** — 001 shows the identical pattern at the identical cameras. Each
  grid node is claimed by exactly one moss state, so a boundary is a staircase of
  whole tiles.
- **No holes, no double-stacking, no seams** after the harness carpet-jitter fix:
  `debug=coverage` at `carpet-close` is a solid white sheet with speckles (the
  genuine 2–6% gaps down to the peat) and **no tile grid** once the bake became
  nine-fold periodic. Overscale stayed at 1.0 — nothing asked for it.
- **Honest limits of relief mapping on this shape.** (1) The card plane sits at
  the mean, so peaks above it are shaded and depth-written correctly but cannot
  extend the **silhouette**: at true grazing the mat's skyline is a plane, not a
  lumpy horizon. Fixing that needs a shell/prism per tile (30 verts instead of 6
  at ~400k tiles) or real marching, both out of scope for "smallest fix". (2) The
  solve runs on a 5.6 mm-smoothed sheet, so sub-capitulum relief is normal-mapped,
  not displaced. (3) Above about 3 m the cushion relief is sub-pixel and the mat
  correctly collapses to filtered texture — the method has nothing extra to offer
  there over a flat quad, which is fine, because neither does reality.
- **Verdict: yes, this representation suits moss — better than it suits grass.**
  A depth-augmented card is a bad fit for a tall clump (multi-valued heightfield,
  25 views, dis-occlusion skirts) and a good fit for a mat (single-valued, one
  view, and the relief is precisely the thing a billboard cannot express). The
  qualifier is scale: it wins clearly from ~0.3 m to ~3 m, ties a flat card beyond
  that, and cannot produce a correct grazing silhouette at any distance.
- Timings are **not quoted**: 15+ sibling agents shared this GPU throughout. For
  shape only, `bog` Σp50 ran 2.0–8.5 ms across the cameras above at 1280x800 with
  `maxDist` 64–88, and `default` at `grazing` produced a pixel-identical image, so
  its cost is unchanged by construction (same pipeline, same instances, same
  atlases). Run `#/bench/008-relief-tap?stand=bog&spline=orbit-low` on an idle
  machine before quoting anything.

## Interface notes (things the harness did not give me)

- `carpetScale(entry)` is not exported from `@harness`, so TypeScript cannot see
  the constant tile scale the stand computed; only the WGSL side gets it via
  `stand_table[i].scale_min`. It worked out (the shader is where it was needed),
  but `ctx.stand.species[i].scaleMin` being a stale placeholder for carpet
  entries is a trap.
- Nothing tells an experiment the **expected acceptance fraction** of a carpet
  entry's wetness band. `wetWidth` is the nominal share, but the wetness field is
  damped on slopes, so the driest zone can claim far more than its width and a
  capacity sized at `wetWidth` would silently drop tiles. I guessed
  `wetWidth*1.25 + 0.35`. A `standEntryExpectedShare(entry)` helper, or a
  documented worst case, would remove the guess.
- `light_surface()` saturates for an upward-facing surface under the default sun,
  which flattens any mat's macro shading regardless of how good its normals are.
  An experiment cannot do anything about it from the outside.
- The standard bookmarks have no **low oblique** camera, which is the single most
  informative view of a carpet: `carpet-close` is nadir (where relief parallax is
  zero by definition) and `grazing` is, on this terrain, a 30 m+ view. I had to
  hand-roll `cam=0,-7.43,0,0,-0.35,60` and could only find the absolute y by
  loading a bookmark and nudging the camera to make the runner serialise the pose
  into the URL. A terrain-relative `carpet-oblique` bookmark would help every
  renderer in this round.
- `slope_align` for a *scattered* species (the bog's calamagrostis at 0.3) is
  still ignored by this renderer's camera-facing card path: honouring it means
  tilting the card basis, which breaks the affine `to_mesh()` inverse the relief
  solve depends on. Left alone deliberately — it is a change to the grass path,
  which this round is not allowed to regress.

## Findings — scattered plants (unchanged)

- The tap ladder is clearly visible up close (`close-quality`, inside-plant
  cam): `flat-1tap` reads crisp but poster-flat (it IS a plain single-view
  impostor); `linear-2tap` recovers most of the interior parallax;
  `secant-3tap` adds depth layering on stems/heads. The cost of the secant on
  a *discontinuous* foliage heightfield is silhouette fray/chatter where the
  linear error model straddles a depth gap — foliage tolerates it, a smooth
  surface would not.
- Stochastic view selection kills the 4-view crossfade for one tap's price;
  the residual inter-view mismatch shows as stable grain, worst midway
  between horizon nodes (22.5° apart on the outer ring).
- frag_depth from the reconstructed hit makes plant-vs-plant intersections
  read correctly (no sorting), but disables early-z: grazing overdraw is the
  perf ceiling, exactly as in 004.
- Honest artifacts: (1) 25 views can't store what a view never saw —
  dis-occluded interior regions stretch (dilated skirts); (2) alpha-test +
  white-noise dither = shimmer in motion, no blue-noise/TAA in the harness;
  (3) no atlas mips (view cells would bleed) → distant sparkle; (4) beyond
  `maxDist` (default 64 m, cap 88) groundcover ends — visible as a meadow
  edge at far-horizon; a far carpet layer is the natural complement;
  (5) calamagrostis/elymus source meshes are ~0.52 m periodic community
  tiles, so their card is a baked clump repeated per scatter point — density
  reads busier than ground truth; poa (true specimen) is the cleanest.
- Perf (HONESTLY CONTENDED — 15 sibling agents hammered the same GPU during
  every measurement, so these are shapes, not claims): relief-cards pass p50
  at 1280x800 ~13–45 ms depending on cam; the tight-box cards + frustum cull
  took topdown from 62 → 13 ms p50 under the same load. `flat-1tap` was
  roughly half the cost of `secant-3tap` in the close-up test. No bench JSON
  recorded for this reason — run
  `#/bench/008-relief-tap?stand=default&spline=orbit-low` on an idle machine
  before quoting numbers.
- A/B vs ground truth:
  `#/ab/008-relief-tap/000-ground-truth?stand=default&cam=grazing&seed=42`.

## Audit

Structural pass (no frame times used — the GPU was shared with other agents;
everything below is argued from the code and checked by screenshot).

**Fixed**

1. **The committed baked artifacts were dead weight.** `mesh/baked/008-relief-tap/`
   already held all three view sets, but `bake.ts` never looked at them: every
   page load downloaded the raw GCMESH1 source mesh (poa-pratensis alone is
   229 MB), re-rendered 25 full-mesh ortho captures per species, read two
   1280² atlases back, and ran the 4-ring CPU dilation over 1.6 M texels ×3 —
   all to reproduce bytes that were sitting in the repo. It is now on the
   normal harness flow (OPFS → committed `.bin` → bake → `commitBake`), with
   the mesh load moved *inside* the bake closure so a cache hit fetches no
   triangles at all. Measured effect on load: with the old code a fresh
   `far-horizon` load was still rendering black 8 s in; with the artifact it
   is up in ~2 s. The reason the first version gave for skipping the cache
   (Vite's SPA fallback returning `index.html` at HTTP 200) is now handled in
   `src/bake/io.ts`, and the loader magic+size validates anyway.
2. **Per-fragment recomputation of per-card constants.** The fragment shader
   ran `to_mesh()` twice per pixel — each one a `rot_y` with its own
   `cos`/`sin` — plus `hemioct_encode` + normalizes to pick the view cell,
   for quantities that are identical over the whole card. All of it moved to
   the vertex stage: the eye position in mesh frame, the view-cell coords
   (`g0`, frac), and `cos(yaw)`/`sin(yaw)` are `@interpolate(flat)` varyings
   now. The card point itself is an *affine* function of world position, so
   passing `o_u` as an ordinary perspective-correct varying reproduces it
   exactly instead of re-deriving it — the fragment stage no longer contains
   a single trig call. Also drops `yaw`/`scale`/`fade`/`root`/`sway` from
   perspective interpolation to flat, which is what they always were.
3. **`drawUbo` was re-uploaded every frame for every species** although not
   one of its 24 floats depends on the camera or on time (bake constants +
   params). Now uploaded only when its contents actually change.
4. **Cull dispatch was not clipped to the stand region.** The dispatch is
   sized by the camera's `maxDist` box; cells outside the stand aabb were
   launched only for the shader to reject them with the same `lo/hi` test.
   The CPU now intersects the box with that region (identical predicate, so
   the surviving cell set is unchanged) and skips the dispatch entirely when
   the window is empty.

**Verified unchanged.** Deterministic (`det=1&t=3.5`) crops at `grazing`,
`topdown`, `inside-plant` and `far-horizon`, old code vs new: RMSE 0.5 %–1.2 %,
visually indistinguishable — i.e. the committed artifact is equivalent to a
fresh bake, and the vertex-stage hoisting only moved float rounding around.

**Deliberately left alone**

- `node_basis()` still runs per fragment. It has only 25 possible results and
  could be a small uniform table, but the choice is per-pixel (stochastic view
  selection), the replacement trades ~20 ALU for a dynamically indexed uniform
  read, and which wins is hardware-dependent — that needs a bench, which this
  session cannot honestly run. Same verdict for folding the three per-pixel
  PCG draws into fewer rounds.
- `frag_depth` + `discard` defeats early-z, which is the method's real perf
  ceiling at grazing angles. Fixing that means a depth prepass or giving up
  reconstructed depth — a redesign, not an audit item.
- The dithered near/far dissolve stays: it *is* the fade mechanism (CLAUDE.md's
  dither rule allows exactly that case), and it only runs inside the fade
  bands.
- The instance buffers stay sized for `MAX_DIST_CAP` rather than the current
  `maxDist`, so the `maxDist` slider never reallocates GPU memory mid-session;
  the 25 MB/species budget still holds (21.3 MiB worst case on `default`).

**Open question for a future pass.** The normals debug view shows just how
high-frequency the baked normal field is — one texel spans several blades, and
bilinear filtering across an oct-encoded discontinuity can point anywhere. A
bake-time normal filter (or storing a filtered plus a detail normal) would calm
the lighting noise, but it changes the visual character, so it is a design
change rather than an audit fix.
