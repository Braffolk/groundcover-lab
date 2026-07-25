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

Worked example for calamagrostis: near atlas 1800 × 1488 rgba8 with 5 mips =
10.7 MB, near normals 900 × 744 rg8 with 4 mips = 1.4 MB, far atlas 600 × 744
rgba8 with 10 mips = 2.4 MB, far normals 300 × 372 rg8 = 0.3 MB → 14.8 MB of
texture, plus ~6.5 MB of near/far instance buffers and per-entry uniforms.
Inside budget without an exception.

Committed artifacts: `mesh/baked/022-layered-parallax/slabs-v3-<species>.bin`,
10.8–13.7 MB each (one file per species, nothing stale left behind).

## Bake

`bake.ts` + `shaders/bake.wgsl`, in-browser, once per species.

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
