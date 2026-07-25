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

## Status

**working** — seen working at all four standard cameras, all six debug views, on
`default` and on `scaling-100m`. No console errors.
`npx tsc --noEmit | grep 021-geometry-near-cards-far` is clean.

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

### Links

* run: `#/run/021-geometry-near-cards-far?stand=default&cam=grazing`
* A/B vs the champion:
  `#/ab/001-billboard-smoke/021-geometry-near-cards-far?stand=default&cam=grazing&seed=42`
  (also `cam=topdown`, `cam=far-horizon`, `cam=inside-plant`)
* A/B vs ground truth:
  `#/ab/000-ground-truth/021-geometry-near-cards-far?stand=default&cam=grazing&seed=42`
* handover check: the run URL twice, with `p.cloudRadius=4` then `p.cloudRadius=36`
* scaling: `#/run/021-geometry-near-cards-far?stand=scaling-100m&cam=grazing`
