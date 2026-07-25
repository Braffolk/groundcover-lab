# card cloud near, impostor far

## Idea

**Near: a three-plane billboard cloud that is real 3D. Far: the billboard. Both
cut from the same capture box, so the handover has nothing left to change.**

The source mesh is split by the azimuth of each triangle's centroid around the
clump axis into `N_CARDS = 3` wedges, and wedge *k* is rendered orthographically
onto the vertical plane **through the axis that contains its own direction**
(`a_k = k·π/3`). At draw time those three planes sit in the plant's own frame —
they do **not** face the camera. That single decision is where all the 3D comes
from:

* **Interior parallax.** The three planes cross along the plant axis at 60° to
  each other, so a card at 60° to the view spreads its content over ±0.87·r of
  depth. Translate the camera and the blades on it slide across the blades on the
  fronto-parallel card. It is not an approximation of parallax — the cards are
  world-space geometry, so the parallax is exact.
* **A silhouette that changes shape, not just orientation.** The outline is the
  union of three *differently foreshortened* images: each card's projected width
  goes as `|sin(a_k − φ)|`, so orbiting morphs the outline. With three cards 60°
  apart the widest card never drops below `cos 30° = 0.87`, so it morphs without
  the silhouette "breathing" (a 2-card cross would swing 30%). A camera-facing
  card can only ever rotate a fixed shape.
* **Blades at oblique screen angles.** A camera-facing card always presents its
  baked side view upright, so every blade in a billboard field runs the same way
  on screen. A world-fixed card seen obliquely produces the long diagonal,
  foreshortened blades visible crossing the frame in the close-ups — the single
  most obvious tell between the two methods.
* **Self-occlusion and contact.** The cards write depth and alpha-test hard, so
  where they cross, the near half of one occludes the far half of another. Card
  bottoms are pushed `sink` (5% of plant height) *below* the capture box, so the
  terrain's own depth clips the cut edge and blades end in soil instead of on a
  visible seam.

**Why wedges are assigned by position and not by something cleverer.** A card
carries the geometry that *lies in it*. When card *k* turns edge-on it holds
exactly the blades pointing at/away from the camera — the ones that are
maximally foreshortened and cover the fewest pixels anyway. The decomposition
loses its mass where the loss costs least, and plane displacement is bounded by
`r·sin(30°) = 0.5r`.

**The far LOD is deliberately just the billboard**: one camera-facing card from
an 8-azimuth impostor sheet plus a top card, i.e. exactly what
`001-billboard-smoke` draws. Distant plants must not cost more than the champion
charges for them, and past ~16m there is nothing left for 3D structure to buy.

**The invisible handover** rests on three things, in order of importance:

1. Both LODs are cut from **one capture box** (`rXZ, y0, y1, cx, cz`) and share
   the same wind, fades, lighting, sink and grounding gradient. A plant cannot
   move, resize, tilt or re-light when it flips.
2. The switch is on the **3D** distance to the plant's mid-height, scaled by the
   plant's own scale — a constant apparent size. (Horizontal distance was the
   worst bug of the first version: from the top-down camera it drew card clouds
   for plants 40m *below* the camera that were 20px tall, and paid 1.67× the
   baseline for them. On 3D distance the same view is 1.12×.)
3. The radius is **jittered ±15% per plant** out of the plant's own scatter
   values, so no coherent ring of pops sweeps through the field: plants flip one
   at a time, at their own distance, while fog and ~50px silhouettes cover them.

Everything else is straight out of the champion's playbook, because it is right:
hard alpha test + depth write (no blending, no dithering, no `frag_depth`), 2
texture taps per fragment, one compute pass that evaluates the shared scatter
over a camera-centered region and compacts survivors into indirect draws — here
into **two** lists, near and far, so the LOD split itself costs nothing.

Two extras that are cheap and honestly do some of the visible work: **two-sided
normals** (a blade is a thin sheet; the baked normal faces the bake camera, so
flip it toward the viewer on a card seen from behind) and a small **sun
transmission** term, plus a subtle per-plant **albedo tint** from the plant's own
hash so a meadow stops looking like one cutout stamped 200k times. Both stay
multiplicative in albedo, so `DEBUG_LIGHTING` still divides out exactly.

### Making a 4-10% ink card cheap

Measured on the bake: a near card is only **4-10% covered**. Rasterizing three
full card rects means ~90% of fragments are alpha-test rejects, and that — not
shading — is the cost. Two structural fixes:

* **Per-height-band hulls.** Each card is drawn as `BANDS = 6` stacked quads,
  each only as wide as the ink actually reaches in its own height band. Because
  the vertex mapping is linear in height (height lerp, sink and sway all are),
  the bands are pixel-identical to one tall quad — pure area removal. Measured
  band-hull area vs the full rect: **0.46-0.66**.
* **Tight content rects** per atlas layer for everything else (top cards, all 8
  impostor azimuths), from the bake's own coverage scan.

Cost collapses with distance in this order: 18 band quads + 1 top card (near) →
1 camera-facing quad + 1 top card (far, = billboard) → vertex-culled behind the
near plane once the region/inside fade erodes the card entirely → not emitted at
all outside `regionRadius`.

## Carpet species: the same idea rotated 90 degrees

**Near: a stack of ground-parallel relief shells. Far: the one flat quad. Both
cut from ONE top-down capture of the tile's own square that carries a HEIGHT
channel.**

Sphagnum palustre is not a plant with a silhouette — it is a 0.18m periodic
community tile, 0.07-0.09m tall, laid out by the `bog` stand as a life-size mat
(22x22 tiles per 4m cell, constant scale 1.0101, 90-degree yaw only). Rendered
through the card path it was indefensible: three vertical planes 0.3m across
slicing through the ground and through each other, every tile drawn at the card
radius instead of its footprint, and — the worst of it — only 128 of its 484
slots enumerated, so the mat came out as *stripes*: six tile rows of moss then
bare peat, over and over, all the way to the horizon.

The near/far split survives, but the axis of the decomposition changes. For an
upright plant the interesting variation is around the vertical axis, so the
cards are vertical. For a cushion the entire variation is *height above the
ground*, so the "cards" are horizontal:

* shell k is a ground-parallel quad over the tile's own square at height
  `mix(hLo, hHi, k/(n-1))`, and it keeps only the texels whose baked surface
  height reaches its own plane. The stack is therefore a terraced
  reconstruction of the real 3.3cm of capitulum relief, and — because coverage
  is nested — seen from straight above it reproduces the true ortho top view
  exactly, just at the right depths.
* shell 0 carries no height gate at all. It is the closed base of the mat: a
  carpet that dissolves is worse than a carpet with no relief.
* past `shellRadius` (8m) a tile is one quad at the surface's mean height —
  precisely what `001-billboard-smoke` draws, which is all a 17px tile can show.

One capture serves both LODs, so nothing can move, resize or re-light at the
handover; the only thing that changes is how many planes it is sliced into.

### What changed, smallest first

1. **Enumerate `carpet_div²` slots, not `SCATTER_MAX_PER_CELL`.** The cull now
   takes its slots-per-cell from the stand entry (484 for the bog moss, 128 for
   everything else). This alone is the difference between a quarter of the mat
   and all of it. Capacity is kept separate from enumeration: every slot is
   *visited*, but the instance buffer is sized for the entry's wetness share
   (measured worst local share on the bog is 0.58 of a nominal 1/3 band, so
   `wetWidth * 1.9`).
2. **Instance capacity clamped to the stand.** `min(region, stand span)` cells,
   as 001 does. Identical on the ±128m stands, ~3.5x smaller on the ±96m bog.
3. **A second draw shape, on its own pipelines** (`vs_carpet` / `vs_carpet_far`
   / `fs_carpet`). The card path is untouched — same interpolants, same
   instruction count, bit-identical output on `default`.
4. **Footprint from `stand_table.footprint_m`**, never from the capture radius
   or `height_scale`, and the exact carpet scale from `stand_table.scale_min`
   rather than the 12-bit instance value (0.08% is enough to stop tiles
   abutting). Yaw is decoded as an exact quarter turn.
5. **Per-vertex terrain conforming (ladder rung 3)** — see below.
6. **A carpet alpha reference of 0.06** instead of the grass 0.4, so distant
   tiles cannot fail the test wholesale and punch holes in the mat, and **no
   camera-inside fade** — a mat you are standing on must not open under you.
7. **A dedicated carpet bake** (see Bake) and the height channel that makes the
   shells possible.
8. **LOD brightness matched by measurement.** The shells darken with depth
   (`cushionShade`, cavity occlusion). The far quad shades itself with the
   *area-weighted mean* of the shells it replaces, computed at load from the
   tile's own height histogram (0.50 for all three states), so no ring sweeps
   the field at the handover radius.

### Terrain fitting: rung 3 (per-vertex), and it is not optional here

Every shell corner gets its own `terrain_sample(xz)` — height and the ground
normal's (nx, nz) from one bilinear fetch, so the shading basis is free.

Rung 3 rather than 1 or 2 for a structural reason, not a quality preference:
neighbouring tiles **share their corner positions exactly** (the grid step *is*
the tile size), so a per-vertex fit is the only rung that keeps the whole mat
C0-continuous. Any rung that fits one plane per tile — point normal or
`terrain_plane_fit` — lets two neighbours choose two different planes and leaves
a wedge-shaped crack along every tile boundary. The cheap rungs are not cheaper
here, they are wrong. Verified on a 31-degree bog ridge at 0.45m eye height: the
mat follows the slope with no buried or floating edges and no visible tile grid.
`slope_align` is not read by the carpet path at all — `carpet_div > 0` already
means "this is a mat", and a mat conforms fully by definition.

### The lattice invariant

Grid step, 90-degree yaw and constant scale come from the stand and are used as
given. No billboarding, no per-tile jitter, no distance shrink, **no overscale**
(1.0, exact abutment — the pilot measured overscale as a net loss on a flat
tile, and the shells give an overlap even less room to hide). Every shell of
every tile is at the same height above its own ground, so neighbours agree
plane-for-plane as well as edge-for-edge.

## VRAM budget math

Per species, as the HUD reads it: **15.6-16.3 MB of 25 MB.**

| item | size | bytes |
|---|---|---|
| near albedo, `texture_2d_array` rgba8 | 512² × 4 layers, 10 mips | 5.59 MB |
| near normal, rg8 (oct, mesh frame) | 512² × 4 layers, 10 mips | 2.80 MB |
| far albedo, rgba8 | 256² × 9 layers, 9 mips | 3.14 MB |
| far normal, rg8 | 256² × 9 layers, 9 mips | 1.57 MB |
| far instance list (region worst case) | ~214k × 16 B | 3.43 MB |
| near instance list (`cloudRadius` max disc) | ~33k × 16 B | 0.53 MB |
| indirect args + entry uniform | 32 B + 544 B | ~0 |
| **total** | | **~17.1 MB** |

(The HUD reads 15.6-16.3 MB because the two instance lists scale with that
species' stand density; on the `dense-mixed` stand, densities 4-5, it reads
17.5-18.8 MB — still inside budget.) Texture arrays rather than one atlas: layers mip
independently, so no tile bleeds into its neighbour at coarse mips and no UV
inset is needed. 256px far tiles are deliberate — the handover is at ≥16m, where
a 1.2m plant is ~50px, so 256 is already ~5× oversampled and the smaller working
set filters better.

**A carpet species: 14.8 MB of 25 (HUD-verified, bog stand).**

| item | size | bytes |
|---|---|---|
| tile albedo+coverage, rgba8 512², 10 mips | one layer | 1.40 MB |
| tile normal+height, rgba8 512², 10 mips | one layer | 1.40 MB |
| far instance list (±96m stand × 484 slots × share 0.63) | ~735k × 16 B | 11.8 MB |
| shell instance list (8m disc… 24m at the param's max) | ~35k × 16 B | 0.56 MB |
| **total** | | **~15.2 MB** |

The allocation is the point. Through the card path the same species cost
**26.0 MB — over budget** — and spent 9.8 MB of it on three wedge cards and
eight azimuth views of a thing that has no silhouette, at a texel density
diluted by a capture box 1.55x wider than the tile. Here the imagery is 2.8 MB
of *only* what a mat shows (the top surface, at 0.35 mm/texel over the tile's
own square, 1.55x denser than cropping the tile out of a whole-mesh capture),
and the memory instead goes where a life-size mat actually needs it: one
instance per tile over a ±96m stand. Both far slots of the bind group point at
the same two textures — a mat's far LOD is the same picture, drawn once.

## Bake

`bake.ts`, one artifact per species,
`mesh/baked/021-geometry-near-cards-far/cloud-v3-<species>.bin`, **9.83 MB each**
(29.5 MB for the three — less than the billboard baseline's 42 MB). Bake key is
`cloud-v3-<species>`; the earlier `v1`/`v2` variants were deleted, nothing stale
is left behind.

1. Parse GCMESH1. Exact horizontal support radius from the vertices (bounds
   corners overestimate it badly on the community tiles),
   `y0 = min(0, boundsMin.y)`.
2. **Partition.** Per triangle, centroid azimuth `ang = atan2(-dz, dx)` (chosen
   so `dot(off, right_k) = ρ·cos(ang − a_k)` with `right_k` the bake's card U
   axis), folded onto `[-π/2, π/2]` mod π, assigned to the card within
   `WEDGE_DEG = 30`. Output is ONE concatenated index buffer plus a per-card
   `[offset, count]` range — poa is 6.5M triangles, so this avoids ever holding
   two full index sets.
3. **Two ortho passes, one encoder.** Near: 2×2 grid of 1024px tiles — three
   sector cards (each drawing only its own index range) + a straight-down top
   view of the whole mesh. Far: 3×3 grid of 512px tiles — 8 azimuths + top, whole
   mesh. Both write albedo+coverage and mesh-frame oct normals flipped toward the
   bake camera. The near top view reuses the concatenated range: duplicate
   triangles are harmless under a depth test.
4. **Post.** Coverage-weighted 2× downsample, 6 dilation passes into empty texels
   (per layer, so nothing bleeds), then per-layer content rects and per-card band
   hulls. Side tiles keep the full vertical range on purpose: both LODs must
   agree on where the sunk bottom edge is.

Mip chains are generated on the GPU at load, per layer per level.

### Carpet bake (`carpet-v1-<species>.bin`, 2.10 MB each)

A separate, much smaller artifact for `carpet_div > 0` species — `bake_tile.wgsl`
plus `bakeSpeciesTile()`. The three `cloud-v3-spaghnum-*` artifacts are deleted;
nothing stale is left behind.

1. One orthographic straight-down pass over exactly `[0, tileM]²` in the mesh
   frame (tile origin is (0,0) for every current source mesh), 1024² supersampled
   2x into 512².
2. **The mesh is drawn 3x3 times, offset by its period.** The mesh overflows its
   own tile (0.24m of geometry inside a 0.18m period), so a neighbour's overhang
   has to be inside our square or the mat cannot close. Measured coverage after
   this: **97-99.5%** of the tile. It also makes opposite edges identical by
   periodicity, which is why no tile lattice is visible even at 0.3m.
3. Targets: albedo+coverage, and (normal, height). The normal is stored as a
   **plain unit vector flipped into the upper hemisphere, not octahedral**,
   precisely so that box-filtered mips stay a real average (CLAUDE.md's
   octahedral trap); the height is normalized over the capture box `[y0, y1]`.
4. Post: coverage-weighted 2x downsample, 6 dilation passes, and the height
   percentiles the shell stack spans — `hLo` = p5 and `hHi` = p97 of the visible
   surface, `hMid` = its mean. Measured shell coverage for the wet state:
   95 / 84 / 55 / 20 / 2.6% of the tile's ink at the 5 default planes.

The bake is ~2 min per species, dominated by parsing and uploading a 479 MB
19.8M-triangle mesh; the rasterization itself is one pass. Transient GPU memory
is the vertex + index buffers (~420 MB), tagged `bake-scratch` and destroyed.

## Status

**working** — seen working at all four standard cameras, all six debug views, on
`default` and on `scaling-100m`. No console errors.
`npx tsc --noEmit | grep 021-geometry-near-cards-far` is clean.

Carpet path verified on `bog` at grazing / topdown / inside-plant /
carpet-close, a 31-degree ridge at 0.45m eye height, a wide sloped view, two
macro views (0.3m and 0.12m above ground) and albedo / normals / lighting /
coverage. No toasts, no console errors. `default` is **unchanged**: the mean RGB
of eight horizontal scene bands is byte-identical before vs after at
`cam=grazing`, `cam=topdown` and `cam=inside-plant`, and VRAM reads the same
16.3 / 15.6 / 16.3 MB.

## Findings

### Speed — A/B ratio, contended

15 sibling agents shared this GPU throughout, so **no absolute millisecond below
is worth anything** and none is quoted as a claim. What is robust is the ratio
inside `#/ab/001-billboard-smoke/021-geometry-near-cards-far?cam=<cam>`, where
both sides render the same stand in the same frame. Method: sample the HUD's
rolling p50s ~40× per camera and take the minimum of `A/cull + A/cards` and of
`B/cull + B/plants` independently (each dips to its uncontended cost when the
sibling load lulls).

| camera | A (billboards) | B (this) | ratio |
|---|---|---|---|
| grazing | 2.40 ms | 2.86 ms | **1.19×** |
| inside-plant | 1.88 ms | 2.20 ms | **1.17×** |
| far-horizon | 3.75 ms | 4.42 ms | **1.18×** |
| topdown | 1.74 ms | 1.95 ms | **1.12×** |

A second independent sampling round of the same four cameras read 1.25×, 1.24×,
1.28×, 1.19× (grazing / inside-plant / far-horizon / topdown), so call it
**~1.2×, range 1.12-1.28× over two rounds** — inside the 1.5× ceiling with room
to spare. The whole gap is the card
cloud in the first ~16m: forcing `p.cloudRadius=0` (impostor path only) measures
1.0-1.2×, i.e. parity with the champion, and raising `cloudRadius` to 28-36
pushes far-horizon past 2× (which is why the default is 16 and why the LOD metric
had to be apparent size).

Total GPU Σp50 on `default` at 1280×800 read 4.4-5.3 ms in quiet moments and
12-14 ms while siblings hammered the GPU. The quiet readings are the ones
consistent with the ratio table and are comfortably under the 9 ms bar, but treat
both as indicative only.

Bench recorded for the archive (**contended, do not quote**):
`results/021-geometry-near-cards-far__default__p-e13386e8__apple-metal-3__2026-07-25T00-31-25-898Z.json`
(`stand=default`, `spline=orbit-low`). Its `base` pass alone read 2.08 ms against
0.3-0.45 ms in quiet conditions, which is the size of the contamination.

### Plant-count independence

`stand=scaling-100m` (±2048m, **134.2M plants**) at `cam=grazing`: `plants` p50
2.18 ms, `cull` 0.19 ms, VRAM unchanged — the same numbers as `default` (~557k
plants). Cost is O(visible region), and the near bucket is additionally bounded
by a `cloudRadius` disc, so the expensive representation cannot grow with field
depth either.

### Looks — does it beat the baseline?

**Yes on 3D structure and on close-range sharpness; the margin in a static wide
shot is real but modest, and part of it comes from the tint/translucency rather
than from geometry. Honest verdict: better, not a landslide.**

What the A/B wipe and same-camera solo captures actually showed:

* **Close range, 1:1 pixels** (`cam=7.00,-6.20,7.00,-0.7854,0.10,26`, identical
  260×240 crop of both): this method is *sharper*. Each sector card spends 512px
  on one third of the clump — two plumes instead of four — so per-plume texel
  density is roughly double the baseline's, and spikelet structure and serrated
  blade edges resolve where the baseline's plumes go mushy. The baseline also
  shows more dashed alpha-test breakup along thin blades.
* **Blades at oblique screen angles.** In the same shots this method has long
  diagonal foreshortened blades crossing the frame; the baseline's blades are all
  near-vertical arcs, because a camera-facing card cannot present its baked view
  any other way. This is the clearest single visual tell of real 3D, and it is
  structural, not tuned.
* **Grazing wide shot** (identical 880×330 crop, both solo): this method reads
  more organic — plume tone varies plant to plant, the green blade layer between
  the plumes is visible, and dark gaps between plumes give depth. The baseline
  reads flatter and more repetitive, like one cutout stamped everywhere. It is
  also very slightly cleaner; a viewer who values tidiness over variety could
  prefer it.
* **Handover, tested directly**: same camera, `p.cloudRadius=4` vs `36` (so the
  whole visible band is impostors in one and card clouds in the other). Density,
  colour, silhouette envelope and canopy height are indistinguishable; only fine
  plume detail differs. If a full-field swap is invisible then a per-plant swap
  mid-field certainly is — which is the entire point of cutting both LODs from
  one box.
* **Top-down is identical to the baseline by construction** (at 42m every plant
  is beyond `cloudRadius` on 3D distance, so it is all impostors). Parity, not a
  win.
* **Debug views** all read correctly: normals show genuine per-blade variation in
  the viewer-facing hemisphere (not one flat card normal), lighting divides out
  exactly, coverage is solid white with thin hard edges (no dither, no soft
  fringe), depth is a continuous ramp with no LOD seam.

### Honest weaknesses

* **The missing middle.** When a card turns edge-on, the blades it carries —
  those pointing at/away from the camera — disappear instead of showing up as
  small foreshortened blobs near the clump centre. The clump therefore reads as
  slightly fewer, bolder plumes than the true ortho photo does. This is inherent
  to a through-axis billboard cloud. An overlapping wedge (`WEDGE_DEG = 40`,
  1.33× duplication) was tried and made no visible difference, because covering
  the exactly-edge-on blades needs `WEDGE_DEG ≥ 60`, which then draws
  camera-facing blades at nearly full width — wrong in the other direction. The
  exact 30° partition is kept: sharper and cheaper.
* **Depth flattening within a wedge**, bounded at `0.5r`. Invisible in practice,
  but it is a real approximation.
* **The horizontal top card is a flat plate** at moderate elevation, most visible
  from `inside-plant`. Inherited verbatim from the baseline, which shows the same
  artifact in the same shot — not a regression, but not fixed either.
* `maxAnisotropy` was raised to 16 so a strongly foreshortened card cannot
  mip-blur vertically. It made no measurable visual difference on this hardware;
  it is kept because it costs only where the projected area is small, but do not
  believe it matters.
* Tried and rejected: sampling the normal *after* the alpha-test discard with
  `textureSampleGrad`. It cannot help under Metal's discard semantics (execution
  continues) and only adds gradient work, so both taps stay in front of the
  discard, exactly as the baseline does it.

### Moss — what improved, what is still bad

All from screenshots at the same cameras before and after (1400x800, `det=1&t=3`,
seed 42), plus band-mean measurements of the scene.

**Before** (the card path applied to a mat): the bog read as *corduroy* — six
rows of dark moss slabs per 4m cell then bare peat, repeated to the horizon,
because only 128 of the 484 carpet slots were enumerated. Each surviving tile
was three vertical 0.3m planes crossing through the ground; from 1m straight
down the mat was a field of dark X-shaped crosses on bare peat with metre-wide
gaps between them. VRAM was 26.0 / 25.3 / 24.3 MB — over budget.

**After**: a closed, continuous mat at every distance from 0.3m to the horizon,
with the three micro-habitat states reading as coherent interlocking zones.
Specifically, and each of these was looked at:

* **carpet-close (1m straight down)**: fine mottled moss, no seams, no holes, no
  tile lattice anywhere — better than `001-billboard-smoke` at the same pose,
  which shows thin dark lines along its tile boundaries (its crop misses the
  neighbours' overhang; the 3x3 bake fixes that here).
* **Macro, 0.3m at 34 degrees**: this is where the shells earn their keep. Same
  camera at `shellCount=2` vs `5` vs `8`: 2 is visibly a flat photograph with a
  few bright specks, 5 reads as a dense springy cushion with depth between the
  capitula, 8 begins to smear vertically (each surface texel is drawn on every
  shell below it, so more shells means more repeats of the same texel along the
  view ray). 5 is the default for that reason.
* **31-degree ridge at 0.45m**: the mat follows the slope continuously, no
  buried or floating edges, no cracks between tiles, no stair-stepping of the
  grid. This is rung 3 doing its job.
* **Grazing / inside-plant**: the moss reads as ground *cover* rather than
  ground *texture*, but honestly the shells contribute almost nothing at eye
  level — see the verdict below.
* **Debug views**: `coverage` is solid white with hard speckled edges (a
  depth-writing occluder, no dither anywhere); `normals` shows real per-texel
  variation at close range (the tile's mean n.y is 0.61 with 29% of texels at
  45-60 degrees from up — measured from the artifact) and flattens toward up
  with distance exactly as `001` does, which is the shared normal-mip
  limitation, not a bug in this path; `albedo` and `lighting` divide out.
* Band means at `cam=grazing` (near→far): this renderer 74-99 vs `001`'s 89-103,
  i.e. **~15% darker overall** at the same hue. Roughly half of that is the
  deliberate cushion occlusion (`cushionShade`, folded into albedo the way the
  card path folds its grounding gradient), the rest is that real per-texel
  normals near the camera are genuinely darker than `001`'s oct-mip normals,
  which drift toward straight up.
* In `debug=albedo` the field brightens **~23% from 2m to 14m** (`001`: ~6% over
  the same range). ~9% of that is the shells-vs-quad occlusion difference and
  the rest is coverage convergence — near, the alpha test opens the genuine gaps
  down to the peat and you see dark terrain through them; far, the mip closes
  them. It is the same sign of drift `001` has, larger because this path has
  more contrast to lose.

**Still bad:**

* **No distance aggregation.** Life size means ~1.13M tiles on the bog and every
  one of them past 8m is still its own 6-vertex quad with four terrain taps.
  That is where the bog's frame time goes (`plants` 2.5-3.8 ms vs `001`'s
  3.0-3.5 ms on a contended GPU). The obvious fix — one quad per 2x2 or 4x4
  block of tiles with a repeat sampler past ~40m — is a real change to how zone
  boundaries resolve at distance and was out of scope for this pass.
* **The shells are invisible at eye level**, and there is no way around that: a
  3.3cm relief seen from 1.7m at 3m distance is under a pixel of vertical
  extent. They pay off between roughly 0.2m and 1.5m.
* **Shell smear.** Because coverage is nested, a tall texel is painted on every
  shell beneath it, so at oblique angles the surface streaks along the vertical.
  It reads as plush/velvet at 5 shells and as combing at 8. The alternative —
  giving each shell only its own height band — removes the smear but opens the
  risers as see-through gaps, which is worse.
* The wet-vigorous state is a strongly saturated flat green in the source
  authoring (`debug=albedo` confirms it is the capture, not the lighting), so
  its zone still reads as poster paint next to the ochre states.
* 512px per tile is 4x oversampled at the `carpet-close` bookmark but becomes
  the limit below ~0.3m. 768 or 1024 would fit the budget (the imagery is only
  2.8 MB); not done because nothing in the standard camera set asks for it.

### Is this representation suited to moss?

**Partly, and more than a flat card — but the win is narrow and honest.**

What works: a mat is a heightfield, and a heightfield decomposes into
ground-parallel shells cleanly. Nothing about the technique fights the lattice
(the shells are all horizontal, so tiles agree plane-for-plane), nothing needs
camera-facing geometry, the closed base shell keeps the mat a solid occluder,
and one capture with a height channel serves both LODs — so it is *cheaper* in
memory than the flat-card baseline while carrying strictly more information.
Between 0.2m and 1.5m the cushion genuinely has thickness: layers occlude each
other, the parallax when the camera moves is real geometric parallax, and the
depth between capitula is visible.

What does not work: past ~2m the relief is sub-pixel and this collapses to
exactly the baseline's flat quad — correctly, but it means the improvement is
confined to the range where you are almost lying in the moss. And the shells are
an approximation with their own signature (terracing, and the vertical smear
from nested coverage) that a real displaced surface would not have. The *card
cloud* half of this experiment — the three vertical wedge planes that make it
what it is — contributes nothing at all to a carpet and had to be switched off
entirely; what renders the moss here is a new shape that shares only the
experiment's near/far structure. So: this renderer can render moss decently, but
it does it by becoming a different renderer for that species, and the honest
ceiling of a shell stack on a 7cm cushion is "convincing ground cover you can
crouch over", not "a cushion you could press with your hand".

### Harness notes (what was missing or awkward)

* `standEntrySlots()` exists in TS but there is **no WGSL twin**: a shader has to
  write `u32(entry.carpet_div) * u32(entry.carpet_div)` with the
  `carpet_div == 0 → SCATTER_MAX_PER_CELL` fallback by hand. A
  `scatter_entry_slots(entry_index)` in `scatter.wgsl` would put the one number
  that silently deletes three quarters of a mat in exactly one place.
* Nothing exposes an entry's **expected occupancy**. Every renderer that sizes
  an instance buffer for a zoned carpet has to guess how much of `[0,1)` its
  wetness interval actually claims; I had to measure the field offline (worst
  local share 0.58 of a nominal 1/3) and hardcode `wetWidth * 1.9`. A
  `standEntryShare(entry)` — even a coarse precomputed bound — would remove a
  guess that is invisible when wrong until tiles start vanishing.
* The mesh header's `tileSize` / `tileOrigin` are exactly what a carpet bake
  needs and were enough; `stand_table.footprint_m` matched them. Good.
* `terrain_sample()` returning (h, nx, nz) in one fetch is the right primitive
  for per-vertex conforming — the shading basis came out free. Nothing missing.
* `slope_align` had no use here: for a mat, `carpet_div > 0` already implies
  full conformance, and for the bog's calamagrostis (0.3) the card path would
  have to tilt its whole card basis, which is a change to the grass path that
  the "do not regress `default`" bar makes unattractive for a 13k-plant entry.
  Not a complaint — just a report that the field went unread.
* The dev server still answers a missing `/mesh/baked/...` with `index.html` at
  200, so every artifact load needs the magic-validation shim. A 404 would let
  experiments delete that code.

### Links

* run: `#/run/021-geometry-near-cards-far?stand=default&cam=grazing`
* moss: `#/run/021-geometry-near-cards-far?stand=bog&cam=carpet-close`
  (also `cam=grazing`, `cam=topdown`, `cam=inside-plant`)
* the shells, on and off: the bog run URL with `p.shellCount=2` vs `p.shellCount=5`
  vs `p.shellCount=8` at `cam=0.28,-7.51,0.28,-0.7854,-0.60,50`
* handover: same URL with `p.shellRadius=0` (all flat quads) vs `p.shellRadius=24`
* A/B vs the champion on the mat:
  `#/ab/001-billboard-smoke/021-geometry-near-cards-far?stand=bog&cam=carpet-close&seed=42`
* A/B vs the champion:
  `#/ab/001-billboard-smoke/021-geometry-near-cards-far?stand=default&cam=grazing&seed=42`
  (also `cam=topdown`, `cam=far-horizon`, `cam=inside-plant`)
* A/B vs ground truth:
  `#/ab/000-ground-truth/021-geometry-near-cards-far?stand=default&cam=grazing&seed=42`
* handover check: the run URL twice, with `p.cloudRadius=4` then `p.cloudRadius=36`
* scaling: `#/run/021-geometry-near-cards-far?stand=scaling-100m&cam=grazing`
