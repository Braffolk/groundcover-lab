# 019 true-depth impostors

## Idea

**An impostor texel is not paint on a card — it is a 3D point.** The bake stores,
per baked view `k`, an orthographic image with its own orthonormal basis
`(R_k, U_k, F_k)` and its own half-extents `(eu, ev, ew)`, plus a **signed depth**
channel. A texel at image coordinate `(u, v)` carrying depth `d` therefore *is* the
surface point

```
Q = C + u·R_k + v·U_k + d·F_k                                            (1)
```

in the plant's own frame — a real sample of the source mesh. Everything the
technique does follows from having (1) at runtime.

**1. The card is screen-aligned; the image maps to it linearly.** Under
local-orthographic projection (a plant subtends a fraction of a degree) the screen
offset of `Q` in the plant's screen basis `(rs, us)` — built from the view ray to
*this* plant — is

```
s = M·(u, v) + d·g,   M = [[R_k·rs, U_k·rs], [R_k·us, U_k·us]],
                      g = (F_k·rs, F_k·us)                               (2)
```

`M`, `M⁻¹`, `g` and the whole card rectangle are constant per plant, so the vertex
shader computes them once.

**2. Parallax = inverting (2) with one Newton step.** The fragment does

```
t0 = M'⁻¹·card + 0.5        naive texcoord, assumes d = 0
d0 = depth(t0)              tap 1
t  = t0 − d0·gw             first-order exact
```

then shades at `t` (tap 2 = albedo, tap 3 = normal/depth/AO, issued only if the
coverage test passed). **Three taps, no loop, no marching, no per-fragment
iteration.** When the camera sits on a baked view `g = 0` and `M = I`, so the warp
is exact and free; between views it slides near parts of the plant against far
parts and hands the alpha test the *warped true silhouette* instead of a rotating
cutout. `warp=0` in the params turns the whole mechanism off, which is the built-in
A/B for what it buys.

**3. True per-pixel depth.** The nearest shells rebuild `Q` from equation (1) and
write its projected depth as `frag_depth`. That is the thing a billboard
fundamentally cannot do: blades of neighbouring plants interleave per pixel, the
plant base meets the terrain along a ragged per-blade line instead of a straight
card/ground intersection, and the canopy self-occludes correctly. It costs
early-z, so **only the nearest shell pays for it** (see LOD).

**4. Baked canopy occlusion.** The bake splats the source vertices into a
24×40×24 density grid and traces 12 short cosine-weighted hemisphere rays per
voxel; the transmittance is sampled at vertex rate into the atlas' alpha channel.
Grass is a chaotic volume, so this is a far better occlusion cue than anything a
single view can carry — it is what makes the canopy interior read as *interior*
(see `debug=lighting`).

**LOD is the architecture, not an afterthought.** The cull pass sorts survivors
into **seven fixed distance shells** (6 / 11 / 18 / 28 / 45 / 75 / 128 m) inside
one instance buffer, each with its own indirect draw, and they are issued near to
far so every shell's hard-alpha-tested depth writes reject the shells behind it
before they shade. Per shell:

| shell | pipeline | per fragment |
|---|---|---|
| < 6 m | `fs_near` | warp + **frag_depth** (3 taps) |
| 6–30 m | `fs_mid` | warp, planar depth, full early-z (3 taps) |
| > 30 m | `fs_far` | no warp, planar depth (2 taps) |

The warp fades smoothly to zero across the last warped shell, so crossing into the
no-warp shells is continuous rather than a silhouette pop, and it also fades out
below ~1.5 m — for a plant that fills the screen the warp displacement spans
hundreds of pixels and its unavoidable disocclusion error reads as bent blades,
which is worse than the parallax it buys (verified at `cam=inside-plant`). Distant plants also
collapse through the mip chain: the mip level is computed **in the vertex shader**
from the plant's projected size and is therefore primitive-uniform, so every
fragment of a quad hits the same level of the same layer.

**Per-frame cost is O(visible region), never O(plants).** Verified: the
`scaling-100m` stand (**134.2 M plants**) renders *faster* than `default`
(557 k plants) — 3.21 ms vs 3.7–3.9 ms in the impostor pass, because the far
shells there are all tiny.

### What is NOT here, deliberately

- **No dithering / stochastic alpha anywhere.** Hard alpha test with depth write
  throughout; the near-fade and region-rim fades erode the alpha reference, and a
  quad whose fade drops below the reference is emitted behind the near plane
  instead of rasterizing a guaranteed-empty quad.
- **No per-fragment loop, no marching, no view-set blending.** Max 3 taps.
- **No `frag_depth` outside the nearest shell.** Writing it disables early-z, so
  it is spent exactly where it changes pixels and nowhere else (measured: it costs
  ~0.07× of the billboard baseline at `depthDist=6`, but ~0.6× at `depthDist=14`).

## VRAM budget math

Per species: 17 layers (10 azimuths at 0° elevation, 6 at 40°, 1 straight-down
top view) of 352 px, 9 mip levels, two rgba8 planes.

Mip chain per layer = 352² + 176² + … + 1² = 165 195 texels.

| item | bytes |
|---|---|
| albedo + coverage, rgba8 array, mipped | 17 × 165 195 × 4 B = **10.71 MiB** |
| geo (oct normal ×2, signed depth, AO), rgba8 array, mipped | 17 × 165 195 × 4 B = **10.71 MiB** |
| per-view table (basis + extents + coverage rect, 64 B) | 17 × 64 B = 1.06 KiB |
| culled instances, 7 shells (default stand, density 3, R = 128 m) | 97 024 × 16 B = **1.48 MiB** |
| entry uniform + indirect args | 512 B |
| **total** | **22.90 MiB / 25 MB** |

Confirmed against the HUD budget bar: 22.9 MB (calamagrostis, poa), 22.7 MB
(elymus — lower density, so smaller shells), 23.8 MB on the `dense-mixed` stand
(density 5). **Within budget on every stand.**

Why 352 px and 17 layers, and not e.g. 256 px and 25: at 256 px the plants were
visibly softer than the billboard baseline (which spends 512 px per side view),
and with the depth warp fixing the *geometry* between views, the remaining cost of
a wider azimuth gap is disocclusion only — which on chaotic grass is far less
objectionable than blur. 352 px × 17 layers is the sharpest configuration that
fits the budget.

## Bake

`bake.ts` + `shaders/bake.wgsl`, in-browser, once per species; committed to
`mesh/baked/019-depth-writing-impostors/views-v2-<species>.bin` (21.4 MiB each,
one file per species, no stale variants left behind).

1. **Local frame.** Capture centre = AABB centre with `y0 = min(0, boundsMin.y)`;
   every view's half-extents are that AABB *projected onto the view's own basis*,
   so no layer wastes resolution on margin.
2. **Render.** 17 orthographic passes over the raw GCMESH1 mesh at 2× (704 px)
   into two rgba8 array textures: `(albedo, coverage)` and
   `(octU, octV, depth01, ao)`. Depth is `0.5 − 0.5·d/ew` in clip z so a `less`
   test keeps the surface actually visible from that view; normals are flipped
   toward the bake camera so both faces of a blade light like the front.
3. **AO grid.** Vertex-splatted 24×40×24 density grid → transmittance trace
   (12 dirs × 6 steps) → 3×3×3 smoothing → r8 3D texture, sampled at vertex rate
   during the bake.
4. **Downsample 2×**, coverage-weighted for colour/normal/AO — but **depth takes
   the nearest subsample, not the average**, so silhouette edges do not grow
   ghost half-depth surfaces.
5. **Coverage rectangle** per layer: the tight bounding box of texels with ≥12%
   coverage, written into the view table. The runtime builds the card from *that*
   rectangle mapped through `M`, not from the AABB, so no quad rasterizes empty
   margin.
6. **Dilate** colour/normal/depth/AO 5 texels into empty space (alpha stays 0) so
   bilinear and mip taps never pull in background.
7. **Full mip chain with Castaño alpha rescaling**: per level, bisect the alpha
   scale that preserves the fraction of texels above the alpha reference. Without
   it a hard alpha test dissolves distant grass into specks; with it the far
   shells stay solid (visible in the `far-horizon` A/B, where the baseline thins
   out and this does not).

## Status

**working.** Verified by screenshot on the default stand at `grazing`, `topdown`,
`inside-plant`, `far-horizon`, plus `debug=normals`, `debug=lighting`,
`debug=depth`, `debug=albedo`, `debug=coverage`, plus the `scaling-100m` and
`dense-mixed` stands. No console errors in any run.
`npx tsc --noEmit | grep 019-depth-writing-impostors` is silent.

## Findings

### Speed — A/B in-frame ratios vs `001-billboard-smoke` (CONTENDED GPU)

15 sibling agents shared this GPU, so **absolute milliseconds are worthless** and
none are quoted as a claim. The numbers below are the ratio of the two technique
passes *rendered in the same frame*, median of 5–8 HUD p50 samples per row
(`#/ab/001-billboard-smoke/019-depth-writing-impostors?stand=default&seed=42&cam=…`):

| camera | B/A (impostors / cards) |
|---|---|
| `grazing` | **1.39–1.43×** |
| `topdown` | **1.02×** |
| `far-horizon` | **1.20×** |
| `inside-plant` | **1.27×** |
| close orbit (~2.5 m) | **1.45×** |

Bench (`#/bench/019-depth-writing-impostors?stand=default&spline=orbit-low&seed=42`,
also contended — do not read the absolutes as a clean claim):
`results/019-depth-writing-impostors__default__p-31153a36__apple-metal-3__2026-07-25T01-02-16-265Z.json`
(plus `…T00-49-58-396Z.json`, `…T00-49-07-779Z.json`). Total GPU Σp50 came out
**7.7–8.8 ms at 1600×900**, i.e. under the 9 ms ceiling even with the contention
and at a higher resolution than the bar's ~1280×800 (where a solo run of the run
page reported Σp50 7.98 ms); `scaling-100m`
at `far-horizon` measured Σp50 6.6 ms with 134.2 M plants in the stand, and
`dense-mixed` (7.7 M plants, density 5/4/4) measured Σp50 9.1 ms.

**Where the cost went, and what fixed it.** The first working version was **3.2×**
the baseline. Four structural fixes, each measured by A/B ratio:

1. **Interstage data was the dominant per-fragment cost** (35 flat components; a
   flat varying is a per-fragment load on this GPU class, and grass overdraw
   multiplies it). Folding the atlas extents, the plant scale and the `[0,1]`
   texture remap into `inv_m` / `gw` / three pre-scaled reconstruction axes, and
   applying the wind shear in the vertex stage only, cut it to 26 components — 16
   of which the mid/far shaders actually touch. **1.93× → 1.62×.**
2. **Explicit, vertex-computed mip level** instead of screen-space derivatives of
   a *warped* coordinate (which spike to mip 0 at every depth discontinuity and
   thrash the cache), which also lets the coverage test `discard` *before* the
   second tap is issued. Part of the same step.
3. **Seven distance shells drawn near to far** instead of three unordered
   buckets — hard-edged depth from the near shells rejects the far ones.
   **1.62× → 1.54×.**
4. **`frag_depth` confined to the nearest shell** (`depthDist` 14 → 6 m).
   **1.54× → 1.43×**, and the visual difference is negligible because beyond ~6 m
   the interleaving it buys is a couple of pixels.
5. Reading the warp's depth estimate at (nearly) full resolution rather than a
   blurred mip turned out to be both prettier *and* slightly cheaper, and the
   sub-1.5 m warp fade shaved the last bit: **1.43× → 1.39×** at `grazing`.

An instructive negative result: raising `alphaRef` to 0.65 made the pass **2×
slower**, because fewer fragments pass the alpha test, so less depth is written, so
early-z stops rejecting the shells behind it — exactly the overdraw trap CLAUDE.md
warns about. It is also why the coverage-preserving mip chain is a *performance*
feature as well as a looks feature.

### Looks — honest assessment

**Yes, this beats the billboard baseline, and it is decisive at close and medium
range.** Evidence, all from
`#/ab/001-billboard-smoke/019-depth-writing-impostors?stand=default&seed=42`:

- **Close orbit (~2.5 m, `cam=7.61,-4.30,6.62,-1.1354,-0.5490,45.0`)** — the
  clearest result. The baseline shows unmistakable flat-card artefacts: hard
  straight cut edges, the same silhouette mirror-repeated across neighbouring
  plants, and dark bare soil between cards. This side has a continuous canopy with
  no card seams, blades of different plants threading through each other, and the
  ground correctly occluded.
- **`inside-plant`** — the baseline is scattered flat blades with sky between them;
  this reads as a canopy you are standing inside, with a dark occluded interior
  below and lit tips above.
- **`far-horizon`** — the baseline thins toward the horizon (mip-eroded alpha);
  this keeps its coverage all the way out.
- **`topdown`** — this covers the ground more completely; the baseline shows more
  soil. Roughly a draw aesthetically, at 1.02× the cost.
- **Parallax check** —
  `#/ab/019-depth-writing-impostors/019-depth-writing-impostors?cam=grazing&seed=42&a.warp=0&b.warp=1&mode=diff`
  reports *54.4 % of pixels differ, mean 26.9/255*. The warp is not cosmetic: it
  re-resolves which surface each pixel sees, concentrated on plant interiors and
  silhouette edges. Orbiting through the azimuth gap, the silhouette deforms
  continuously instead of snapping at view boundaries.
- **Silhouette vs view direction** — 17 views (10 azimuths + a 40° elevation ring
  + a top view) with the warp covering the gaps, versus the baseline's 8 azimuths
  with a hard switch plus a separate horizontal top card that has to be faded in.
  The elevation behaviour in particular is a genuine improvement: there is one
  card, and it *becomes* the top view as the camera rises.

**Where it is worse or only equal:**

- **A single-step depth warp is destructive to fine detail when its depth estimate
  is coarse.** My first attempt read the warp depth from a deliberately blurred mip
  (~3.5), reasoning that grass depth is discontinuous and a smooth field would warp
  gently. That was wrong: it chewed flower heads into mush and made blades wavy,
  and at 2–3 m it looked clearly *worse* than no warp at all. Reading the depth at
  (nearly) full resolution — `warpBlur=0.5` — is much better, because a blade's own
  depth then displaces that blade rigidly. `warpBlur` is left exposed as a param so
  the failure mode stays reproducible. The same error, in its other form, is why
  the warp is faded out below 1.5 m: no depth resolution rescues a single-layer
  reprojection when the plant is 800 px tall and the displacement is hundreds of
  pixels. Inside that radius the 3D read comes from the per-pixel depth writes and
  the baked occlusion instead.
- Right at the sky silhouette the baseline's individual blades are crisper; this
  method's warped edges are slightly softer there.
- The baked AO makes this render a touch darker/greener overall than the baseline.
  I judge that a gain (it is what sells canopy depth — see `debug=lighting`), but
  it is a taste call; `aoStrength` exposes it.
- **`000-ground-truth` is not a usable A/B partner** for this: it is a
  stand-independent 3-tile reference patch at the origin, so at any standard camera
  it covers a small part of the frame and the rest is bare terrain.

### Things a reader should check next

- The warp's disocclusion error grows with the azimuth gap. 10 azimuths (±18°) is a
  budget choice; if the 25 MB soft budget were relaxed, more azimuths at 352 px
  would shrink the residual smearing further at no runtime cost.
- The mid/far shells use the plant centre as the fog / `debug=depth` position (they
  write planar depth, so that *is* their depth). `debug=depth` therefore shows
  per-plant flat depth beyond 6 m and per-texel depth inside it — honest, not a bug.
- Wind: the quad is sheared by height fraction in the vertex stage and only the
  card-centre sway value is folded into the reconstruction centre, so the
  reconstructed depth lags the shear by at most the sway amplitude (a few cm).
  Invisible in depth, and it keeps interstage components out of the fragment shader.
