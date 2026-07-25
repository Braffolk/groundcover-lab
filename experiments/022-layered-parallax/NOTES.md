# layered parallax

## Idea

One quad per plant, like the billboard baseline — but its pixels are not flat.

Per species and per azimuth, the source mesh is split into three **depth slabs**
along that azimuth's own view axis (front / middle / back). Each slab is baked as
its own orthographic image *plus one number*: the **mean depth of the geometry
inside it**, i.e. the plane the slab lives on inside the plant. The top view is
split the same way, into three height bands.

At runtime the rasterised quad is only a proxy. The fragment shader takes the
real eye ray, intersects it with those three planes **in the plant's own frame**,
and probes the slabs front-to-back until one is opaque. The first opaque hit wins,
which is simultaneously the hard alpha test and the correct occlusion decision.

Because the planes are real 3D planes at real depths, the three images slide
against each other exactly as geometry would:

- **interior parallax** — near parts shift against far parts as the camera moves,
  including the correct differential magnification as you walk up to a plant;
- **a silhouette that reshapes** — the union of three differently-shifted
  coverages, not one flat shape that rotates;
- **self-occlusion** — front-to-back probing is a real visibility test, and slabs
  behind the front one are shaded darker because they *are* deeper in the canopy;
- **ground contact** — the proxy stands on the terrain and every reconstructed hit
  point stays inside the plant's own bounding cylinder.

Why it is cheap. The whole reprojection is a **homography** — a central
projection from the eye between two planes — so both its numerator and its
denominator are affine in the proxy's own surface coordinates. That means the
rasteriser's perspective-correct interpolator can evaluate them for free: the
fragment shader receives `(Bu, Bv, beta)` interpolated, does **one reciprocal**,
and each slab is then **two multiply-adds**. All three slabs share the
denominator because the planes are parallel.

    tu(L) = cu + k(L)·Bu/beta      tv(L) = cv + k(L)·Bv/beta
    k(L)  = plane_depth(L) − dot(eye, plane_normal)

No marching, no `frag_depth`, no blending, no dithering: hard alpha test with
depth write, so early-z still rejects. Per fragment: 1–3 albedo taps + 1 normal
tap (a slab whose reprojected uv left its tile is skipped without a fetch).

LOD is two rings, split in the cull pass and drawn by **two specialised pipelines
so neither ring pays for the other**:

| ring | condition | what it draws |
|---|---|---|
| near | `dist < lodDistance × scale` | 3-slab reprojection, near atlas |
| far  | everything else | one merged tile, one tap, implicit-derivative sampling off a small fully-mipped atlas |

The far ring is *structurally the billboard baseline* and benchmarks at parity
with it (see Findings), so a distant plant costs exactly what a billboard costs.
Cost is O(visible region): 134M plants (`scaling-100m`) measured 4.62 ms in the
draw pass against 4.4–4.7 ms for the 557k default stand.

Wind: the sway is applied as an **affine shear of space**, `X + S·vfrac(X.y)`,
using only its horizontal component. That is exactly invertible, so the eye is
un-sheared into the plant frame and the reprojected ray stays a straight line.
(`wind_sway`'s small vertical dip is dropped on purpose — it would make the shear
non-invertible for the price of a 3 cm bob.)

A **carpet** species (`stand_table[i].carpet_div > 0` — the bog's *Sphagnum
palustre*) runs the same idea turned 90°. It is a periodic tile lying on the
ground, not an upright plant, so the depth slabs are replaced by **height
bands** baked straight down over exactly the tile square, and the proxy is one
terrain-conforming ground quad whose fragment shader walks the eye ray *down*
through those planes. See "Moss carpet (bog stand)" at the end.

## VRAM budget math

Two atlases per species, because near and far want opposite things.

- **near** — the 27 slab tiles, arranged as 9 view-group **blocks** of
  `(TW + TW/2) × TH`: the front slab takes the full `TW × TH` tile and the two
  slabs behind it stack beside it at quarter area, so a fragment that falls
  through to a deeper slab stays in the same texture pages. 5 mip levels (near
  cards never minify far). Albedo `rgba8` (rgb + coverage), normals oct `rg8` at
  half the atlas resolution.
- **far** — the 9 merged tiles (8 azimuths + top) at quarter area with a **full**
  mip chain, in its own small texture. Distant plants minify into it, and small +
  fully mipped is what keeps that path cache friendly.

Tiles are sized per species from the plant's aspect ratio at a fixed texel target
(200k texels per full tile), so `calamagrostis` gets 400 × 496 and the short,
narrow `poa` gets 240 × 560.

Measured by the harness tracker (includes the culled-instance buffers):

| species | total | budget |
|---|---|---|
| calamagrostis-canescens | 21.3 MB | 25 MB |
| elymus-repens | 20.7 MB | 25 MB |
| poa-pratensis | 17.7 MB | 25 MB |

A **carpet** species does not get either of those atlases — see "Moss carpet"
below. Its budget is one array texture of `N_BAND + 1 = 5` layers plus the
instance buffer:

| | bytes |
|---|---|
| band albedo, 512² rgba8 × 5 layers, full mip chain | 6.99 MB |
| band normals, 256² rgba8 × 5 layers, full mip chain | 1.75 MB |
| culled instances, 16 B × measured worst-case tile count | 2.9 / 9.8 / 6.7 MB |

HUD totals on the `bog` stand: **11.6 / 18.5 / 15.4 MB** for wet-vigorous /
late-season / sun-exposed against the 25 MB budget (they were 33.0 / 34.1 /
30.9 MB — over budget — while the moss was still being drawn as an upright
plant off the slab atlas). The instance buffers dominate and differ per species
because the three states partition the wetness axis and that axis is nowhere
near uniformly distributed; see the capacity note below.

Worked example for calamagrostis: near atlas 1800 × 1488 rgba8 with 5 mips =
10.7 MB, near normals 900 × 744 rg8 with 4 mips = 1.4 MB, far atlas 600 × 744
rgba8 with 10 mips = 2.4 MB, far normals 300 × 372 rg8 = 0.3 MB → 14.8 MB of
texture, plus ~6.5 MB of near/far instance buffers and per-entry uniforms.
Inside budget without an exception.

Committed artifacts: `mesh/baked/022-layered-parallax/slabs-v3-<species>.bin`,
10.8–13.7 MB each for the three grasses, and
`carpet-v1-<species>.bin`, 6.55 MB each for the three Sphagnum states. The moss
`slabs-v3-*` files (42.9 MB) were deleted — nothing loads them any more, a
carpet species never touches the slab path.

## Bake

`bake.ts` + `shaders/bake.wgsl`, in-browser, once per **upright** species;
carpet species take `carpet.ts` + `shaders/carpet_bake.wgsl` instead (described
under "Moss carpet").

The full mesh is rendered orthographically into one supersampled (2×) canvas that
holds both atlases. Nine view groups (8 azimuths + straight down); each group is
**one `drawIndexed` with `instanceCount = 4`** — slabs 0/1/2 plus the merged
tile — where the per-instance record supplies the view axes, the slab range and
the NDC sub-rectangle of the tile to render into. Tiles never overlap, so one
shared depth buffer serves the whole canvas.

The slab test is per **fragment** (the interpolated depth along the view axis,
which is exact under an orthographic projection), so no triangle sorting or index
reshuffling is needed and slab boundaries are exact. Slab boundaries sit at
±R/3 of the capture radius; the plane each slab records is the **mean depth of
the vertices inside it**, sampled at bake time.

Post: coverage-weighted 2× downsample, 5 dilation passes clamped to each tile
(so filtering never pulls background black across a tile border), a 2 % inset
guard band inside every tile, oct-encoded normals at half resolution, then the
canvas is split into the near and far images. One submit per view group keeps any
single command buffer small enough for the 6.5M-triangle poa mesh, and the source
vertex/index buffers are destroyed before the readback staging is allocated
(peak stays around 400 MB).

## Status

working — verified on screen at `grazing`, `topdown`, `inside-plant`,
`far-horizon`, plus `debug=normals|lighting|albedo|coverage`, and A/B against
`001-billboard-smoke`. No console errors.

Carpet path verified on the `bog` stand at `grazing`, `topdown`,
`inside-plant`, `carpet-close`, a 30°-slope eye-level view
(`cam=23.71,2.40,-49.06,2.0239,-0.2845,60`), three close wet-zone views
(0.42 m / 3 m / 8 m), a shrunk-region rim check, and
`debug=albedo|normals|lighting|coverage` — no toasts, no console errors.
`default` is unchanged: masked pixel diffs of before vs after (HUD and param
panel excluded) differ in **0–123 of 614,404 scene pixels**; `inside-plant` and
`far-horizon` are bit-identical, and the `slab-cards` p50 on `default@grazing`
is 4.66 ms before / 4.70 ms after in back-to-back runs. No bench JSON is
claimed — a dozen sibling agents shared this GPU, and the same camera measured
Σp50 between 6.1 and 9.7 ms across runs of *identical* code.

## Findings

Bench (`#/bench/022-layered-parallax?stand=default&spline=orbit-low&seed=42`,
1600 × 900, 120 + 600 frames, apple-metal-3). **Contended** — 15 sibling agents
shared this GPU, so treat the absolute values as soft and the ratios as the
result; every pair below was run back-to-back and interleaved.

- `results/022-layered-parallax__default__p-958597b3__apple-metal-3__2026-07-25T04-37-51-535Z.json`
  — base 0.30, cull 0.22, **slab-cards 4.68**, composite 4.37 → Σp50 9.57 ms
- `results/001-billboard-smoke__default__p-86e4907a__apple-metal-3__2026-07-25T04-34-13-871Z.json`
  — base 0.31, cull 0.18, **cards 2.53**, composite 2.53 → Σp50 5.55 ms

**Measured ratio: ~1.7–1.85× the billboard baseline on the draw pass, ~1.7× on
the total** (best of six pairs 4.36/2.52 = 1.73, worst 4.68/2.52 = 1.86). That is
**above the "≈1.5×" guidance** — I am not going to dress it up: it is modestly
slower than the champion, not at parity.

On the Bar-1 setup (`#/run/...?cam=grazing`, 1280 × 800) the runner reported
Σp50 **5.7 ms in a quiet window** and up to 9.6 ms while the neighbours were at
full tilt; scaling the bench's fragment-bound passes to 0.711× the pixels puts the
honest figure at **≈7 ms**, i.e. inside the 9 ms ceiling but not by a wide margin
on this machine. (The harness `composite` pass reports ~= the draw pass in both
experiments even though it does identical work in each, so Σp50 looks inflated for
both by roughly the same amount; the ratio is the trustworthy part.)

Where the extra cost is (isolated with single-variable bench runs, same session):

| variant | slab-cards p50 | vs billboards |
|---|---|---|
| `lodDistance=0` (far ring only) | 2.16 / 2.72 | **parity** — the far path really is the baseline |
| near ring, 1 slab probe only | 3.93 | 1.56× |
| near ring, no proxy margins | 3.79 | 1.50× |
| full method (`lodDistance=6/10/16`) | 4.36–4.68 | 1.73–1.86× |

So the near ring costs ~1.9 ms over the far ring, split roughly: 0.5 ms for the
2 extra slab probes, 0.6 ms for the enlarged proxy quad (the reprojection shift
makes it ~1.2× wider and ~1.15× taller), and the rest in the extra interpolants
plus the near atlas being sampled at mip 0. Things I tried that did **not** move
the needle, listed so nobody repeats them: hoisting the whole reprojection into
the interpolator (the homography trick — 4.29 → 4.29, so it is not ALU bound);
fetching all three slabs in parallel instead of chaining them (4.29 → 4.29, so it
is not latency bound); raising `alphaRef` to 0.6 (slightly *slower* — fewer
occluders behind); and shrinking `lodDistance` from 16 to 6 (no change at all —
the cost is entirely inside the nearest few metres, where a handful of plants own
most of the screen). Cutting to 2 slabs, or excluding plants closer than ~1.5 m
from the near ring, are the two levers I did not spend: both would land near
1.4×, the first at the price of interior detail, the second at the price of a pop
right where the viewer is looking.

**Looks — honest verdict: yes, it clearly beats the baseline.** At the same pose
(`cam=8.00,-4.97,8.00,-0.7854,-0.0600,60.0`, `det=1&t=0`) the billboard field
reads as a uniform pink wash: every card shows the outside of a flower panicle,
so the meadow has one brightness and no interior. The slab version resolves into
individual spikes standing in front of dark voids, with green foliage visible
between them — the depth ordering and the per-slab occlusion do that work, and the
`albedo` debug view confirms the base colours are the *same* (both bake from the
same mesh), so the difference is honest shading, not a tint. Compared as full
frames at `grazing`, as 380 × 280 foreground crops, and as 0.6 m lateral camera
pairs with time frozen.

A/B wipe/flicker at `#/ab/001-billboard-smoke/022-layered-parallax?cam=grazing`:

- **parallax** — flickering the two sides, the billboard side's plant interiors
  are rigid while the slab side's near parts sit visibly in front of the far
  parts; translating the camera 0.6 m sideways with wind frozen, the interiors of
  near plants shift against their own silhouettes and the outlines change shape,
  which the billboard side cannot do at all (its cards only rotate).
- **silhouette** — the slab side's outlines gain and lose lobes as the azimuth
  changes, because the three coverages shift by different amounts; the baseline's
  outline is the same shape everywhere inside a 45° sector, then pops.
- **azimuth popping** — *reduced* versus the baseline, which is the pleasant
  surprise: at a sector boundary both neighbouring bakes reproject to
  approximately the same true view, so the switch has much less left to change.
- `topdown` — near parity (5.6 vs 4.2 ms total); the layered top card adds
  contrast and definition, but this is not the view the method wins in.
- `far-horizon` — indistinguishable from the baseline by construction: past
  `lodDistance` it *is* the baseline.

Known weaknesses, all seen on screen:

- Plants closer than ~1 m are soft on both methods (a 400 × 496 tile magnified),
  and mine adds a little reprojection stretch on top.
- A vertical slab seen steeply from above degenerates (one pixel spans a wide
  swath of the plane). `MIN_SLANT` floors the warp and the side card erodes away
  over `sin(elev) = 0.62…0.88` while the top card is already fully in — the
  handover is the same coverage-erosion mechanism the baseline uses, no dither.
  Before that fade existed, near-steep plants showed a blocky mush; fixed.
- The proxy grows on one edge per slab and *not* downward, so the front slab's
  bottom few centimetres can be clipped by the terrain when looking down. That is
  the darkest, most occluded part of the plant; not visible in practice.
- The far ring's tiles are quarter area, so the distant field is very slightly
  denser (fatter alpha-tested blades) than the baseline's. It reads as a greener,
  less uniformly pink horizon — arguably closer to the real plant, but it *is* a
  difference, not a match.

Plant-count independence: `stand=scaling-100m` (134.2M plants) → slab-cards
4.62 ms, the same as the 557k default stand. Cost is bounded by the visible
region, never by plant count.

## Moss carpet (bog stand)

*Sphagnum palustre* arrives as a 0.18 m **periodic community tile**, 0.07–0.09 m
tall, laid out by the `bog` stand as a grid-snapped mat — 22×22 tiles per 4 m
cell (484 slots, life size, scale 1.0101), 90°-only rotations, three states
partitioning the wetness axis. Nothing about that is an upright plant, and the
renderer treated it as one. Before this pass the moss was effectively **not
rendered at all**: at `grazing` and on the sloped view the bog was bare terrain
with a few Calamagrostis tufts, and at `carpet-close` it was a featureless dark
blur. Three separate reasons, all in the same direction.

### What changed, smallest first

1. **Evaluate `carpet_div²` slots, not `SCATTER_MAX_PER_CELL`.** The cull
   dispatch and its `idx % slots` split were hard-wired to 128, so 128 of the
   484 carpet slots per cell were ever visited — three quarters of the mat did
   not exist. Slots per cell now come from `standEntrySlots(entry)` through a
   new `EntryInfo.slots_per_cell`. One line each in `main.ts` and `cull.wgsl`,
   and by itself the single biggest fix here.
2. **Instance capacity measured, not assumed.** Capacity was
   `slotsMax × density/8`, i.e. sized from the 128-slot budget — far too small
   for 484 slots/cell, and the *shape* of the estimate is wrong too. Carpet
   entries partition the wetness axis, but that axis is heavily skewed (the
   field is damped on slopes, so the dry band claims about half the nodes and
   the wet band about a tenth); sizing all three for a third would punch holes
   in one and waste megabytes on another. `carpetCapacity()` samples the actual
   `scatter.wetness` field over the stand on a 96×96 grid and takes the worst
   camera position out of 13×13, ×1.2. That is 181 k / 612 k / 419 k tiles for
   the three states instead of 574 k each.
3. **One ground-parallel tile quad, no camera-facing card.** A carpet entry
   emits 6 vertices instead of 12 and never enters the near/far card pipelines.
   A 0.24 m-wide vertical card 0.09 m tall standing on the ground is nothing
   seen edge-on and slices through both the terrain and its neighbours; a mat
   has no silhouette worth a camera-facing card.
4. **Width from `footprint_m` and scale from the stand, not from the bake or
   the packed instance.** The quad is exactly `footprint_m × scale_min` = the
   0.18182 m grid step. The 12-bit packed scale would have been 1.00989 instead
   of 1.01010 — a 0.04 mm gap per tile, which is nothing on screen but is free
   to get exactly right, and the invariant ("every tile of a carpet is the same
   size") is worth stating in the code.
5. **Terrain fitting: ladder rung 3, per vertex.** Each quad corner takes its
   own `terrain_sample(xz)` — height and (nx, nz) out of one bilinear fetch, so
   the shading basis is free. Rung 3 rather than 1 or 2 for a specific reason:
   neighbouring tiles **share** corner positions, so per-vertex conforming is
   the only rung that keeps the mat C0-continuous. A point normal or a
   `terrain_plane_fit` gives two neighbours two different planes and cracks the
   mat open along every tile boundary — for a tiled species the cheap rungs are
   not cheaper, they are wrong. Cost: 4 texel loads × 6 vertices per tile.
   `slope_align` is not consulted: `carpet_div > 0` already says "this is a
   mat", and a mat conforms fully.
6. **A carpet-specific alpha reference (0.06) and no camera-inside fade.** The
   grass reference (0.4) dissolves a mat that the mip chain has pulled toward
   its tile mean, and eroding the mat you are standing on opens a hole under
   your feet. Only the region-rim fade survives, measured from the tile centre.
7. **A carpet bake, replacing the slab atlas for these species.** See below.
8. **Height-band parallax + depth-driven canopy occlusion.** See below.

### The carpet bake — the same idea, turned on its side

The slab atlas is 96 % dead weight for a mat: the 24 side-slab tiles are never
sampled, and the one tile that is — the top view — was captured over the mesh's
full support radius at the *side* view's aspect ratio, which for the moss is
3.3:1. Only 34 % of that tile is inside the tile square, so the moss was
rendering off roughly **450 × 130 texels of a 10.4 MB atlas**, anisotropically
squashed.

`carpet.ts` + `shaders/carpet_bake.wgsl` capture the tile properly:

- straight down over **exactly `[0, tileM]²`** in the mesh frame (tile origin is
  (0,0) for every current source mesh), square and isotropic, 512² per band —
  0.35 mm/texel at life size, ~3× the effective resolution of the cropped slab
  tile and ~1.3× that of `001-billboard-smoke`;
- **wrapped**: the mesh is drawn once per 3×3 neighbour offset, so the geometry
  that hangs 1.3–2.6 cm past the period re-enters on the opposite edge. The
  image is then exactly one period, which is what makes `addressMode: repeat`
  legal — and *that* is what removes the tile seams entirely, at every mip
  level, without a guard band or an inset. `001-billboard-smoke` crops the same
  view without wrapping and shows a visible dark grid line at every tile
  boundary (clearly so at 0.42 m and 1 m); this bake has none;
- split into **4 height bands at quartiles of the geometry's own height
  distribution**, plus one merged tile. Quartiles rather than equal slices of
  the bounding box because a Sphagnum cushion carries nearly all of its
  structure in the top 4 cm — equal slices would put three of four planes in
  empty air. Each band records the **mean height of the geometry inside it**;
  for wet-vigorous that is 76.5 / 68.6 / 57.6 / 24.6 mm with the merged tile at
  56.8 mm, and coverages 0.51 / 0.67 / 0.67 / 0.77 with the merged tile at
  **0.985** — the mat really is closed;
- all five tiles are layers of one array texture, mip-chained on the GPU. The
  2×2 box halving of a power-of-two periodic image needs no wrapping of its own,
  so the chain stays exactly periodic to 1×1.

Normals are stored as **plain unit vectors in the +Y hemisphere (rgb8 + coverage)**,
not octahedral: they are mipped, and octahedral pairs are not mip-averageable.

Bake cost is 9 wrap offsets × 5 tiles × 19.8 M triangles ≈ the same work the
slab bake already does; all three species baked in well under a minute here.

### The parallax, for a cushion instead of a canopy

This is the part that is *this* experiment rather than a better billboard. The
quad is rasterized on the topmost band's plane; the fragment shader takes the
real eye ray, expresses it as metres of horizontal travel per metre of descent,
rotates that into the tile's own frame and walks **down** through the other
planes, taking the first band whose coverage claims the fragment. Because the
image is one period under `repeat`, the ray may walk out of the tile and simply
continue into the next copy — no clamp, no in-tile test, no edge case.

Three things had to be right for it to read as relief rather than as noise:

- **The band probes need their own, much higher coverage reference**
  (`carpetBandRef`, 0.35) than the mat's alpha reference (0.06). "Does this band
  have geometry here" and "is there moss here at all" are different questions,
  and running the first at 0.06 let the top band win on 6 %-covered fringe
  texels — which flattened the cushion to nothing. This was the single biggest
  look fix after the mat existed at all.
- **Occlusion is by depth, not by band index.** `exp(-k · depth)` with `k` fixed
  by "the bottom band keeps `1 - carpetDepthShade` of the light" (0.5). Being
  continuous in depth is not cosmetic: the merged tile's plane is the mean
  height of everything, so it receives exactly the average darkening the band
  probes produce, and the dissolve into it has no brightness step. The old
  per-index `layerShade` (0.17) gave the second band a 5 % darkening — invisible.
- **The ray slope is capped** (`carpetMaxSlope`, 3). A 5 cm drop at three
  degrees above the mat lands two tiles away, and the bands stop being
  correlated with each other, which reads as speckle. In practice the cap
  rarely binds where it would hurt: past the distance where a tile stops
  resolving the band probes are skipped entirely and the merged tile takes over
  (`carpetMergeLod` 5, over 2.5 mip levels), which is both cheaper and the right
  answer.

Band 0 and the merged tile are sampled with `textureSampleGrad` (hardware
anisotropy, and legal inside the non-uniform band branch because the gradients
are explicit); the deeper bands use an explicit level. Normals also use
`textureSampleGrad`, which both keeps capitulum-scale shading at grazing and
picks the right level for the half-resolution texture without a hand-applied −1.

Measured contribution of the parallax, same camera, `carpetMaxSlope` 0.5 vs 3:
**51 % of pixels differ (mean Δ 29/255)** at 0.42 m and 45°, **38 % (mean Δ 17)**
at 3 m and 25°. It is doing real work, but be honest about the ceiling: the
cushion is 5 cm of relief on an 18 cm tile, so the largest offset available is
about 0.3 of a tile at ordinary viewing angles.

### What improved, what is still bad

Before / after, from screenshots at the same cameras:

- **`grazing`, `bog`** — before: bare olive terrain with a handful of grass
  tufts, no moss anywhere. After: a closed mat to the horizon with the three
  micro-habitat states reading as coherent interlocking zones.
- **30° slope view** — before: no moss at all. After: the mat follows the ridge
  and the flank exactly, no buried or floating edges, no cracks at tile
  boundaries, no bare peat.
- **`carpet-close` (1 m straight down)** — before: a uniform dark blur with
  faint tile-sized blocks. After: bronze-olive Sphagnum with visible cushion
  mottling at 4–9 cm and leaf-scale texture on top, **no tile seams at all**
  (the reference shows a grid of dark seam lines at the same camera).
- **0.42 m at 45°, wet zone** — reads as a springy green cushion: brighter
  raised crests, darker crevices between them. The reference at the same camera
  is a flat mat with a visible tile grid.
- **`inside-plant`** — the mat runs continuously under the camera; no hole.
- `debug=normals` shows dense real per-fragment variation around up (not a flat
  up-normal); `debug=albedo` is the raw baked colour with no shading folded in;
  `debug=coverage` is a closed mat with fine genuine gaps.

Still bad:

- **The 0.18 m period is visible once the mat has contrast.** At 2–5 m the
  tile's own hummock-crest pattern repeats into a faint diagonal weave. It is
  not a seam and not a bug — every tile really is the same 18 cm of moss, and
  the 90° rotation set only gives four variants — but it is the price of giving
  the mat the relief it was missing. `carpetDepthShade` is the knob; 0.3 is
  visibly flatter and visibly less latticed. Measured at `carpetMaxSlope` 0.5
  and 3 to confirm the parallax is not what causes it — it is the albedo's own
  periodicity.
- **Zone boundaries staircase on the tile grid**, most obvious at grazing where
  a 0.18 m step foreshortens into a wedge. Inherent to a hard wetness partition
  on a tiled grid; every carpet renderer will show it.
- **A 2 cm lip where two states meet**: the three bakes put their top band at
  76.5 / 56.6 / 60.5 mm, so the mat steps at a zone boundary. It is
  botanically right (vigorous wet moss builds a taller cushion) but it is a
  visible edge inside a metre.
- **No distance aggregation.** Life size means ~1.1 M tiles inside the default
  110 m region, all drawn as individual 6-vertex quads. The obvious next win is
  to batch 8×8 tiles into one quad with the texture repeated 8× past ~20 m —
  legal, because the grid spacing and the tile size are untouched and only the
  rotation *variety* drops — but it needs care at the LOD boundary (two
  neighbouring quads conforming to the terrain at different vertex densities
  crack) and it was not needed to hit the frame budget here.
- Overscale was not used and is not wanted: the tiles abut exactly, there is no
  z-fighting to bias away, and the wrapped bake means there is nothing for an
  overlap to hide.

### Is this representation suited to moss?

Partly, and more than I expected. The *depth-slab* half of the method — front
slabs of a plant reprojected against an eye ray — is useless for a mat and is
simply not used. What transfers is the underlying idea: **a flat proxy whose
pixels resolve against real depth planes**. Turned 90° into height bands over a
periodic tile it is a genuinely good fit for a low cushion: it gives the
thickness a card cannot (`001-billboard-smoke`'s own verdict on itself is "the
mat is flat, it wants a height/relief channel and a parallax step, i.e. a
different method"), it stays one quad and 1–4 taps, and the periodicity that
makes the tile awkward for a plant renderer is exactly what makes the ray walk
free.

What it still cannot do is silhouette. There is no geometry standing off the
ground, so a Sphagnum hummock seen against the sky at eye level is a flat
horizon line, and no amount of parallax inside the surface fixes that. Within a
few metres of the camera it reads as a cushion; at the horizon it reads as
textured ground. For groundcover that is 0.07 m tall, that trade seems like the
right one.

### Timings

Contended GPU (a dozen sibling agents), so ratios only, and no bench JSON is
claimed. `bog`, 1280 × 800, GPU Σp50 across the final pass: `grazing` 3.6–4.2,
`topdown` 4.9–5.1, `inside-plant` 4.7–6.1, 30° slope 5.0–5.6, `carpet-close`
2.2–4.6 ms — all inside the 9 ms ceiling with the moss drawn at life size and
the cull evaluating 3.35 M slots per frame (0.5–0.8 ms). `default` is unchanged
in both pixels and cost.

### Harness notes

- `standEntrySlots()` is exported and is the right primitive; what is missing is
  anything that stops you from *not* using it. The trap is not the carpet path —
  it is the shared `idx % SCATTER_MAX_PER_CELL` idiom that every renderer copied
  from the template, which silently renders a quarter of a mat.
- `stand_table[i].footprint_m` is documented as "horizontal footprint at
  scale 1", but it is populated from `species.tileM` and is therefore **0 for
  every non-carpet species**. A renderer that sizes anything from it
  unconditionally gets a zero-area plant. Either populate it for all species or
  name it `tile_m`.
- There is no `slots_per_cell` on the GPU-side `StandEntry` — `carpet_div` is
  there and squaring it works, but the TS side has a named helper and the WGSL
  side does not, which is exactly the asymmetry that makes the 128 mistake easy.
- Nothing in the harness reports an instance-buffer overflow. A carpet that
  overflows its capacity loses tiles and looks like a placement bug; a counter
  read back into the HUD would turn a whole class of silent failure into a
  visible one.
