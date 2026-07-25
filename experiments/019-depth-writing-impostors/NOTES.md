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

A **carpet** species spends its budget completely differently — one layer, not
17, because a mat never samples an azimuth view — and lands at 6.9 MB. See
"Moss carpet" at the end.

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

## Carpet species (the `bog` stand's Sphagnum) — see the section at the end

A carpet entry (`stand_table[i].carpet_div > 0`) is a periodic 0.18 m community
tile, not an upright plant, and it draws a completely different shape:
`shaders/carpet.wgsl`, one ground-parallel terrain-conformed quad per tile, with
the top view's depth channel driving a relief step and the near band's
`frag_depth`. The card path above is untouched and still describes every
scattered species.

## Status

**working.** Verified by screenshot on the default stand at `grazing`, `topdown`,
`inside-plant`, `far-horizon`, plus `debug=normals`, `debug=lighting`,
`debug=depth`, `debug=albedo`, `debug=coverage`, plus the `scaling-100m` and
`dense-mixed` stands. No console errors in any run.
`npx tsc --noEmit | grep 019-depth-writing-impostors` is silent.

Carpet path verified on the `bog` stand at `grazing`, `carpet-close`, `topdown`,
`inside-plant`, an oblique eye-level view, two sloped views across the ridges,
and `debug=albedo / normals / lighting / coverage`; no toasts, no console errors.
`default` is unchanged — see "Do the grasses still look the same" below.

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

## Moss carpet (`bog` stand)

Sphagnum palustre is a 0.18 m periodic community tile, 0.07-0.09 m tall with
~3.3 cm of capitulum relief, laid out by the `bog` stand as a grid-snapped mat:
22x22 tiles per 4 m cell (**484 slots**, life size, constant scale 1.0101,
90-degree-only yaw), zoned into three micro-habitat states across the wetness
field. Rendered as an upright plant it was indefensible, and the before shots
show exactly that: rows of tilted, screen-aligned "bricks" standing on edge in
stripes, with three quarters of the mat missing entirely and bare peat between.

### What changed, smallest first

1. **Evaluate every slot.** `main.ts` sized the cull dispatch from
   `SCATTER_MAX_PER_CELL` and `cull.wgsl` indexed slots with the same constant.
   A carpet entry has `carpet_div^2 = 484` slots per cell, so **356 of 484 grid
   nodes per cell were never visited — 74% of the mat did not exist.** Slot count
   now comes from `standEntrySlots(entry)`, rounded up to the 64-wide workgroup
   (484 -> 512) so the cell-level frustum/region rejects stay workgroup-uniform;
   the 28 padding slots fall out of `scatter_candidate()` as `exists = false`.
   This is the single biggest visual fix and it is two lines.
2. **Instance capacity from the grid, not from `density`.** A carpet ignores
   `density` entirely (the stand sets 8 as a placeholder); its cover is
   `carpet_div^2 / 16 = 30.25` tiles/m^2, of which each state's wetness interval
   claims about `wetWidth`. The near shells are sized for the FULL grid rate
   because a 6 m shell can sit wholly inside one zone, the outer annuli for the
   band rate x1.15 because they average over many 12 m zones. Non-carpet entries
   take the identical old code path, so `default` capacities are unchanged.
3. **Carpet render path** (`shaders/carpet.wgsl`, its own pipelines, used only
   for entries with `carpet_div > 0`; `impostor.wgsl` is untouched):
   - ONE ground-parallel quad per tile, **axis-aligned in xz and exactly one
     grid step across** (`stand_table.footprint_m` x the carpet scale). The 90°
     yaw rotates the TEXTURE inside the tile square, never the quad, so corners
     land on the lattice and are shared with the four neighbours. No
     billboarding, no per-tile scale, no distance shrink, no overscale.
   - the tile's own square of the baked **top view only** — a mat has no
     silhouette worth an azimuth view.
   - **Terrain fitting: rung 3, per-vertex conforming.** Every corner gets its
     own `terrain_sample(xz)` — height and (nx, nz) from one bilinear fetch, so
     the shading basis is free. Rung 3 rather than 1 or 2 for a specific reason:
     neighbouring tiles share corner positions, so per-vertex is the only rung
     that keeps the mat C0-continuous. Any per-tile plane fit leaves a wedge
     crack at every tile boundary, because two neighbours fit two different
     planes — the cheap rungs are not cheaper here, they are wrong.
   - baked normals lifted into the ground frame (`plant_basis_from_up(up, yaw)`,
     inlined), so a mat on a slope lights as a slope.
   - **carpet alpha reference 0.06** (param `carpetAlphaRef`) instead of the 0.4
     grass one, so the mip chain cannot dissolve distant tiles into holes.
   - **no camera-inside fade** — a mat you stand on must not open a hole — and
     the region-rim fade is measured from the tile centre, never per vertex.
   - mip level from `dpdx/dpdy` of the *unwarped* texcoord. A mat is a ground
     plane: its minification is strongly anisotropic and the distance-derived
     level the card path uses (correct for an upright plant) would either blur
     it or alias it.
4. **The relief — this is what the method has that a card does not.** The top
   view's signed depth makes a texel a real 3D point `Q = P + N*(2*ew*s)*(d-0.5)`:
   - **parallax.** The ray landing at `P` actually sees the surface at `P + W`,
     `W = d*(V/(V.N) - N)` — the in-plane slide of a point lifted by `d`. One
     dependent tap, one first-order step, no loop and no marching. Offset-limited
     (`1/(V.N)` clamped at 0.30) and faded out below ~25 degrees, where a single
     depth layer has no right to an answer. Because the tile image is periodic, a
     step that leaves the tile square **wraps** instead of clamping, and the wrap
     is the exact geometric continuation.
   - **true per-pixel depth** in the near shell: `Q` is written as `frag_depth`,
     so grass stems growing through the mat are occluded at the real moss surface
     rather than at a plane 4.5 cm above the peat, and the mat meets the terrain
     at its true height. Same shell rule as the cards (`depthDist`), so only the
     nearest band pays for the lost early-z.
   `warp` scales the relief step (0 = flat card, the built-in A/B); `depthDist`
   owns the depth write.
5. **Top layer only in VRAM for a mat species.** A carpet never samples an
   azimuth view, so uploading 17 layers wasted 20 MiB per species. Moss went
   **25.1/25 MB (over budget, HUD-striped) -> 6.9/25 MB**; the grasses are
   untouched at 22.2-22.9.
6. **The baked AO grid is wrong for a mat, and had to go.** This one was found by
   screenshot, not by reasoning. With `aoStrength` at its 0.75 default the whole
   carpet drew a hard **basketweave**: broad soft dark bands, 90 degrees apart
   between neighbouring tiles, obvious at `carpet-close` and dominant across the
   entire near field at `grazing`. `debug=albedo` and `debug=normals` were clean,
   so it was the geometry-atlas alpha channel; `aoStrength=0` removed it
   completely and `0.35` still showed it. The cause: the AO grid is 24x40x24 over
   the mesh AABB, which for a 0.21 x 0.09 x 0.23 m mat is 9.2 mm voxels
   horizontally and 2.4 mm vertically, and the trace steps in VOXELS — the
   up-rays leave the canopy after 22 mm while the sideways rays run 83 mm, so the
   result is dominated by horizontal transmittance and paints the source tile's
   ramet rows as bands. For a 1.2 m grass the same grid is roughly isotropic,
   which is why nobody noticed. Carpets now take their occlusion from the DEPTH
   channel instead — `cavity = 0.55 + 0.45*depth`, per texel, already sampled,
   full atlas resolution: capitulum tops stay lit, the gaps down toward the peat
   go dark. The contrast is deliberately gentle, because the depth channel also
   carries the tile's own low-frequency hummock and a steep ramp would repeat
   THAT in every tile — the same basketweave by another route.

### VRAM (bog stand, HUD-verified)

| item | bytes |
|---|---|
| top-view albedo layer, mipped rgba8 (165 195 texels) | 0.63 MiB |
| top-view geo layer (oct normal, depth, unused AO) | 0.63 MiB |
| culled instances, 7 shells at the carpet's grid rate | 5.6 MiB |
| view table + entry uniform + indirect args | ~1 KiB |
| **total per moss species** | **6.9 / 25 MB** |

Grasses on `bog`: calamagrostis 22.2, poa 21.8. `default` unchanged at
22.9 / 22.7 / 22.9 MiB.

### What improved, what is still bad

From before/after screenshots at the same cameras (`grazing`, `carpet-close`,
`topdown`, `inside-plant`, an oblique 35-degree eye-level view, two sloped views
across the bog's ridges, and albedo/normals/lighting/coverage):

- **Improved, decisively.** Before: stripes of tilted brick-like cards standing
  on edge over mostly bare peat, ~26% of the mat present, moss VRAM over budget.
  After: a closed, continuous, gapless mat at every distance from 1 m to the
  horizon, conforming exactly to the ridged terrain with no buried or floating
  edges and no cracks between tiles, the three states reading as coherent
  interlocking zones, real per-fragment normals (`debug=normals` shows moss-scale
  variation over a terrain-following gradient, not a flat up-normal), ~85%
  coverage with the genuine peat gaps open (`debug=coverage`), and no hole under
  the camera at `inside-plant`.
- **The relief is real but modest.** At an oblique eye-level view, `warp=1` vs
  `warp=0` differs by RMSE 3.1% and the difference is the right kind: without it
  the surface shows sharp linear cracks and reads flat, with it the cushions
  clump and slide against the hollows. At `carpet-close` (straight down) it is
  nearly nothing, because `V ~ N` there and the correct parallax IS zero.
- **Still bad #1, and inherent to a flat quad: no silhouette thickness at
  grazing.** `frag_depth` re-orders what a pixel shows, but it cannot create
  coverage: two abutting coplanar quads project to abutting screen regions, so a
  raised capitulum on a near tile can never overlap the tile behind it. Getting
  that would need the quad extruded into the view direction plus a real march
  through the depth layer, and a march is out of scope for this project. So this
  renders a **relief-shaded, parallaxed, depth-writing ground surface**, not a
  cushion with an edge you could see against the sky.
- **Still bad #2: the parallax fades out at grazing.** Below ~25 degrees a single
  Newton step off one depth layer would displace by more than half a tile and
  smear; it is faded to zero instead. So the most important camera gets the least
  of the method's headline feature. Honest, and visible in the `warp` A/B.
- **Still bad #3, minor: faint 1-texel dark lines along tile boundaries**,
  visible in `debug=coverage`. Mechanism: at mip level L a sample near the tile
  square's edge blends up to 2^L texels of the surrounding capture margin instead
  of wrapping, so coverage dips slightly. 001-billboard-smoke sees the same lines
  from the same cause. The fix is the carpet-only bake below, where the texture
  IS the tile and hardware `repeat` addressing does the wrapping.
- **Not done, and the obvious next win: a carpet-only bake.** For a mat, 16 of 17
  baked layers are dead weight and only ~62% of the top layer's area is inside
  the tile crop, so the species spends 6.9 of its 25 MB and could instead spend
  ~12 MB on ONE 1024 px layer covering exactly `[0, tileM]^2` — 5.7x the linear
  texel density (0.18 mm/texel vs 0.61 mm), correct `repeat` addressing (no seam
  lines, exact parallax wrap) and an alpha rescale computed against the carpet's
  own 0.06 reference instead of the grass 0.4. Skipped deliberately: at
  `carpet-close` (1 m, 1.44 mm/px) the existing crop is already 2.4x finer than a
  pixel, so it buys nothing above ~0.4 m viewing distance, and it costs a ~2
  minute re-bake per species of a 479 MB source mesh.

### Do the grasses still look the same?

Yes, verified rather than reasoned. Masked scene-only crops (HUD and param panel
excluded) of before vs after on the `default` stand:

| camera | RMSE | pixels differing > 3% (of 258 000) |
|---|---|---|
| `grazing` | 6.3e-6 | **0** |
| `inside-plant` | 0 (bit-identical) | **0** |
| `topdown` | 1.2e-3 | 52 |

The 52 are the usual `atomicAdd` compaction-order flips at tied depth, the same
noise two identical runs produce. `default` VRAM is unchanged (22.9 / 22.7 /
22.9 MiB) and its GPU Sigma-p50 stayed in the 3.8-5.4 ms band across the run,
inside the 9 ms ceiling — no bench JSON is claimed, because a dozen sibling
agents were sharing this GPU for the whole session.

### Is this representation suited to a moss carpet?

**Mostly, and better than a plain card — but not all the way.** The impostor's
central claim ("a texel is a 3D point, not paint") survives the change of shape
completely: on a mat the same equation gives per-texel cushion depth, a
first-order parallax step and a true per-pixel depth write, all with the same 3
taps and no loop, and it is a strictly better ground surface than a textured
quad — sharper occlusion cues, parallax that responds to the camera, correct
depth against the grass. What does NOT survive is the *impostor* part: a
screen-aligned card is a proxy for a silhouette, and a 9 cm mat seen from a
standing eye has none, so the entire azimuth view set had to be deleted for this
species (and with it 73% of its VRAM). And the one thing the source geometry
most wants — 3.3 cm of capitulum relief you can see *against* something — needs
coverage that a single flat quad cannot produce at grazing, no matter what depth
it writes. Verdict: an honest, cheap, closed, relief-shaded carpet with real
parallax from above and no silhouette from the side. If the bar is "reads as a
Sphagnum lawn you are walking on", this passes comfortably; if it is "reads as a
springy cushion with an edge", a shell/volume method should own the mat.
