# 020-curved-cards — cards that are not flat

## Idea

A billboard card is a flat quad that spins to face you. Every texel of a plant
therefore sits at **one depth**, and that single fact is what makes billboard
meadows read as stacked paper: no parallax inside a silhouette, an outline that
only rotates, and neighbouring plants that interleave per-card instead of
per-pixel.

This method changes exactly two things about the baseline and nothing else:

1. **The card is a lattice pushed out to the plant's real front surface.** The
   bake keeps, per texel, the *front-surface distance* along the view axis — how
   far the frontmost visible geometry stands off the card plane. A small vertex
   lattice (5x7 near, 3x5 mid) samples a smoothed, hole-free version of that
   field and displaces each vertex along the card normal. The card becomes a
   curved patch of the clump's actual frontal relief: ±0.25 m of stand-off on a
   1.15 m plant for calamagrostis.
2. **The card's orientation is SNAPPED to the nearest of the 8 baked azimuths
   instead of tracking the camera.** This is the half that makes the relief
   *mean* something. A camera-facing curved card would rotate its bulge along
   with you and show no parallax at all; a snapped card is a **fixed 3-D object
   inside its 45° sector**, so orbiting reveals genuine intra-silhouette
   parallax (near lattice cells slide against far ones), a genuinely changing
   outline, and per-pixel depth interleaving with its neighbours.

Two properties fall out of that geometry for free:

- At the exact bake azimuth the displacement is *parallel* to the view ray, so
  the render collapses to the pixel-exact baked image. The method is never worse
  than the billboard it replaces — only more correct in between. That is also
  why the 45° view switch pops far less than a flat billboard's: both
  neighbouring surfaces reproject to nearly the same picture, which is the whole
  point of depth-based view interpolation.
- Depth is written by real rasterized geometry. No `frag_depth`, so early-z and
  hi-z keep working; hard alpha test, no dithering anywhere.

A third card carries the view from above: the **canopy patch**, a 4x4 lattice
displaced in Y by the baked top-view *height field*, so from above you get a
bumpy canopy that depth-sorts against its neighbours instead of a flat lid. It
only exists when seen from above (signed elevation), because a height field
describes nothing when you are underneath it.

Occlusion is baked as a **column integral of coverage** — for each side-view
texel, how much canopy stands between it and the sky — and applied *normalized
by the field's own mean*, so it redistributes light (deep texels darker, exposed
tips brighter) at the same average exposure as an unshaded card. That detail
mattered: the first version dimmed the whole meadow by ~16% and read as plainly
worse than the baseline despite being more correct.

Per frame: one compute pass walks the scatter over a camera-centered cell
region (bit-identical WGSL twin of the harness scatter — no CPU instance
buffers, no per-plant CPU work), frustum-culls, and appends survivors into one
of **three distance rings** of a single instance buffer, each ring feeding its
own `drawIndexedIndirect`. Work is O(region area), never O(plants in stand):
the `scaling-100m` stand (134 M plants) renders at the same cost as `default`
(557 k).

## VRAM budget math

Per species, three atlases at three deliberately different rates:

| artifact | format | size | mips | bytes |
| --- | --- | --- | --- | --- |
| ALBEDO — rgb colour, a = coverage | rgba8 | 1536² (3x3 tiles of 512) | 11 | 12.58 MB |
| SURF — rg oct normal, b front distance, a sky occlusion | rgba8 | 768² (tiles of 256) | 10 | 3.15 MB |
| DISP — smoothed hole-free front distance (vertex stage) | r8 | 96² (tiles of 32) | 1 | 0.01 MB |

= **15.74 MB** of texture. Buffers: the three-ring instance buffer, sized for the
*param maxima* because ring radii move at runtime, is 272 k × 16 B ≈ 4.3 MB for
a density-3 entry, plus a 768 B uniform (three 256 B dynamic-offset blocks) and
a 64 B indirect block.

Measured in the HUD on the default stand: **19.2 / 18.4 / 19.3 MB** for
calamagrostis / elymus / poa — inside the 25 MB budget, and slightly *under*
the billboard baseline (21.3 / 20.7 / 21.3 MB), because halving the SURF
resolution pays for the two extra channels.

Only ALBEDO keeps full resolution: coverage defines the silhouette and is the
one thing that must stay sharp. Normals, occlusion and front distance are low
frequency by nature, and a 5x7 vertex lattice can only express a handful of
samples per tile anyway, so DISP is tiny on purpose.

## Bake

`shaders/bake.wgsl` renders the raw GCMESH1 mesh orthographically at 2x
supersample into a 3x3 atlas: tiles 0..7 are side views at 45° azimuth steps
(camera on the horizon), tile 8 is the straight-down top view. Because the
projection is orthographic (w = 1), the interpolated `position.z` **is** the
normalized distance along the view axis, so `1 - z` gives the front-surface
distance exactly — for both the side and the top parameterization — with the
depth test picking the frontmost fragment automatically.

CPU post-process (one walk over the 3072² render feeds all three rates):

- coverage-weighted reduction 2x2 → ALBEDO, 4x4 → SURF, 32x32 → DISP;
- dilation of colour / normals / distance into empty texels so bilinear and mip
  filtering never reach into unwritten background;
- **flood fill + double blur of DISP** until every texel of every tile has a
  value. The vertex lattice samples DISP *outside* the silhouette too (the
  card's rim), and a hole there would fold the card backwards; a smooth
  extrapolation of the nearest real surface is exactly the curved rim we want.
  Keeping this field deliberately coarse and smooth is also what makes the
  technique robust: a smooth surface cannot crack or rubber-sheet across a depth
  discontinuity, which is the classic failure of depth-displaced impostors.
- sky occlusion by per-column coverage integral inside each side tile (texture
  row 0 is the plant top, so walking rows downward accumulates precisely the
  canopy standing between a texel and the sky). The top tile stays fully lit.

Artifacts (`CRV1`, v2): `mesh/baked/020-curved-cards/relief-v2-<species>.bin`,
11.8 MB each, 35 MB total. v1 was deleted — its AO constant (`AO_K = 0.55`)
saturated the whole canopy interior at the floor, which made the term a flat
dimmer instead of a shading cue; v2 uses 0.12, tuned so a typical column
integral (~6 units of coverage) lands mid-ramp.

## LOD

Mandatory, and it is the ring index the cull pass assigns from
`distance / plant scale` (inverse projected size, so a big plant keeps its relief
further out than a small one):

| ring | radius (m per unit scale) | side lattice | canopy lattice | triangles |
| --- | --- | --- | --- | --- |
| 0 | < 7 | 5x7, displaced | 4x4, displaced | 66 |
| 1 | < 20 | 3x5, displaced | 3x3, displaced | 24 |
| 2 | rest | 2x2, flat | 2x2, flat | 4 |

Ring 2 is **the billboard baseline's exact geometry** and skips the vertex
texture fetch entirely (the `curvature > 0` test is draw-uniform), so a distant
plant costs precisely what a billboard costs. Rings are drawn near-to-far so the
curved cards write solid depth and early-z rejects the deep overdraw behind
them. On the default stand at the grazing camera the split is roughly
900 / 7 k / 70 k plants, i.e. ~430 k triangles against the baseline's ~310 k —
and identical fragment coverage, which is why the pass times match.

## Status

working — verified by headless screenshot at `grazing`, `topdown`,
`inside-plant`, `far-horizon`, at `debug=normals|lighting|albedo|coverage|depth`,
and on the `default`, `dense-mixed` and `scaling-100m` stands. No console errors.

## Findings

**Speed — parity.** Two solo benches taken ~2 minutes apart on a heavily
contended GPU (15 sibling agents; treat the absolute milliseconds as garbage and
the pairing as the signal), `stand=default`, `spline=orbit-low`, seed 42,
1600x900:

- `results/020-curved-cards__default__p-49fb9b80__apple-metal-3__2026-07-25T00-44-39-784Z.json`
  — base 0.49, cull 0.26, **cards 2.24**, composite 1.96 → Σp50 **4.95 ms**
- `results/001-billboard-smoke__default__p-86e4907a__apple-metal-3__2026-07-25T00-47-01-325Z.json`
  — base 0.39, cull 0.23, **cards 2.24**, composite 2.24 → Σp50 **5.10 ms**

Same-frame A/B (`#/ab/001-billboard-smoke/020-curved-cards`) is the only ratio
that survives contention. Medians over 4–5 samples per camera, across several
sessions: **grazing 1.04–1.20x, far-horizon 1.00–1.15x, topdown 0.96–1.14x,
inside-plant 1.06x** on the cards pass. Call it **~1.1x**, and note that the
spread is contention, not the method. Every number on this page was measured
under contention; none of the absolute values should be quoted.

Ratios early in development were much worse (1.6–2.6x) and two of the three
causes turned out to be measurement artifacts of a busy GPU — the third was
real (see below). Anyone re-measuring this should use the A/B page and expect
±0.3 of ratio noise while other agents are running.

**Looks — yes, better, and better at the thing that was asked.** The strongest
objective evidence is `debug=depth` at close range: the baseline's plants are
near-uniform vertical bands (one plane per card, every texel of a plant at the
same depth), while curved cards show continuous depth gradients inside every
plant and dense per-pixel interleaving between plants. Subjectively, in wipe and
flicker:

- close range (0.7–3 m): clearly better. Elymus spikes read as rows of lumpy
  spikelets with light on the near side; the baseline's are uniformly lit
  stamps. Front blades separate from the clump behind them.
- mid range (5–25 m): better, mostly through the normalized occlusion — spikes
  get bright tips and dark bases and stop reading as flat pink strokes.
- far / topdown at the standard bookmarks: indistinguishable, by construction
  (ring 2 *is* the baseline, and at 42 m altitude a plant is ~8 px wide, so
  there is no relief to show).
- parallax: checked by orbiting within one 45° sector; internal features slide
  against each other and the outline reshapes instead of rotating. The azimuth
  switch is noticeably softer than the baseline's, as predicted.

**Three things tried and thrown out, all worth recording.**

1. *Per-pixel parallax residual.* Offsetting the albedo uv by the distance the
   lattice's smooth field missed is textbook single-tap parallax mapping, and it
   is a trap here. It makes the fragment shader's second tap depend on its
   first (measurably ~20–30% of the pass), and at close range it smeared
   silhouettes into rubber and tore dark holes through the flower heads.
   Deleted — the lattice alone carries the relief, and the fragment shader is now
   identical in shape to the baseline's: two independent taps at one coordinate,
   atlas uv finished in the vertex stage.
2. *Wider relief rings.* Pushing ring 1 out to 46 m looked no better and
   slightly *worse*: at ~20 px per plant the parallax is sub-pixel, while the
   curved, oblique card raises the sampler footprint and picks a coarser mip, so
   the mid-field goes soft for nothing. Small rings are both cheaper and sharper.
3. *A displaced canopy patch seen edge-on.* The height-field lid, viewed from
   near the canopy plane, becomes a long tilted sheet that smears across the
   lower screen — the single ugliest artifact this experiment produced. Fixed by
   folding the displacement away on the same elevation ramp that erodes the
   patch's coverage, and by making that elevation *signed* so the lid does not
   exist at all when you are underneath it.

**Known limitations.**

- 8 azimuths means the snapped card sits up to 22.5° off-normal, so a plant's
  projected width breathes by up to 7.6% and pops at sector boundaries. Random
  per-plant yaw decorrelates it across the meadow, and it is the same staleness
  the baseline already has in its texture — but on a single tracked plant it is
  findable.
- Very near plants (< 1 m) magnify a 512 px tile past its resolution and go
  blocky. So does the baseline; the near-fade erodes anything closer than ~0.6 m.
- The canopy patch vanishes when seen from below (correct for a height field),
  so the plant's top is carried entirely by the side patch there.
- `dense-mixed` and `scaling-100m` were verified visually, not benched. The
  plant-count-independence claim rests on the region-bounded cull, which the
  `scaling-100m` screenshot supports: 134 M plants, cards 1.94 ms, Σp50 4.66 ms,
  i.e. the same cost as the 557 k default stand.
