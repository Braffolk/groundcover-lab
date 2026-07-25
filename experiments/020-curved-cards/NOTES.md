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

**Carpet species (Sphagnum) turn the same idea on its side.** A mat has no
silhouette to snap an azimuth to, so the card becomes a ground-parallel lattice
over the species' periodic tile and the displacement runs along the ground
normal instead of the view axis — driven by the *top* view's front-surface
distance, which for a straight-down capture is exactly the cushion's height
above the peat. Everything else is the same machinery: one baked field, one
lattice, hard alpha test, three distance rings. See the carpet section below.

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

**A carpet species spends its budget completely differently** — this is the
allocation question, not an overrun. 8 of the 9 baked tiles are side views, and
a mat never shows one, so a carpet uploads the top tile's own *tile square*
alone, resampled at load time:

| artifact | format | size | mips | bytes |
| --- | --- | --- | --- | --- |
| ALBEDO — tile square only | rgba8 | 320² | 9 | 0.55 MB |
| SURF — oct normal, height, AO | rgba8 | 160² | 8 | 0.14 MB |
| carpet height field (vertex stage) | r8 | 16² | 1 | 256 B |

= **0.69 MB** of texture instead of 15.74 MB, at *higher* effective texel
density than the uncropped tile (the 0.18 m tile only occupies ~271 of 512
texels there). That saving is what pays for the instance buffer a life-size
carpet actually needs: 484 slots per 4 m cell, of which one wetness zone can
locally claim 0.575 (measured against the scatter's own field on the bog), over
2401 cells → 822 k instances × 16 B ≈ **12.9 MB**.

Measured in the HUD on `bog`: **13.6 MB** for each of the three Sphagnum states
— against 25 MB of budget, and against the **31.1 / 30.8 / 30.5 MB** they cost
before this pass, when they were carrying eight useless side views *and* an
instance buffer sized from the wrong slot count. On `default` the grasses are
unchanged at 19.1 / 18.3 / 19.2 MB.

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

A carpet entry has its own ring table, because a 0.18 m tile is not a 1.2 m
plant and the same metric radii would put the whole mat in ring 0:

| ring | radius (m) | carpet lattice | triangles |
| --- | --- | --- | --- |
| 0 | < 4 | 9x9, displaced | 128 |
| 1 | < 14 | 3x3, displaced | 8 |
| 2 | rest | 2x2, flat | 2 |

9x9 over 0.18 m is 2.2 cm cells, which is about what the height field carries
after band-limiting; below that the lattice only samples noise. Ring 2 is again
one flat quad — the baseline's geometry — and it still resolves the ground under
each of its four corners, so the mat stays continuous everywhere.

## The carpet (Sphagnum) path

The moss pass, smallest change first. Every one of these is inside the carpet
branch; the upright-plant path is byte-for-byte the same code it was.

1. **Enumerate `carpet_div²` slots per cell, not `SCATTER_MAX_PER_CELL`.** The
   cull dispatch and its loop bound now come from `standEntrySlots(entry)` (484
   for the bog moss) via `ring.z`. Before, it visited 128 of 484 — the first ~6
   of 22 tile rows in every cell — and the bog rendered as bands of moss with
   bare peat between them. This is the single biggest fix and it is four lines.
2. **Capacity sized from expected survivors, not from slots.** The two numbers
   are different: every slot must be *evaluated*, but the buffer only has to
   hold the entry's share of the wetness axis. The nominal share is 1/3, but
   wetness is damped on slopes so it is not uniform — sampling the scatter's own
   field over ±96 m, the middle zone reaches **0.575** of the nodes inside a
   120 m box. Sizing for 1/3 would have clamped ~40% of that zone away. Sized at
   `wetWidth × 2` with 6% slack, and verified by eye at the worst spot.
3. **A mat is a ground-parallel lattice, not a card.** No camera-facing quad, no
   vertical geometry, no azimuth snapping — those are meaningless for something
   0.07 m tall and 0.18 m wide, and vertical cards for a cushion slice through
   the ground and through each other. Width comes from
   `stand_table[i].footprint_m`, never from `height_scale` (which would have
   made it 3.5x too small); yaw and scale come from the scatter and are used as
   given, so the lattice invariant holds by construction. Overscale is 1.0:
   tiles abut exactly, so there is no overlap to z-fight and no dark seam line.
4. **Terrain-fitting rung 3: per-vertex conforming.** Every lattice vertex
   resolves its own `terrain_sample(xz)` — height and (nx, nz) in one bilinear
   fetch. Rungs 1–2 are not merely cheaper here, they are *wrong*: neighbouring
   tiles would fit different planes and crack apart at their shared edge, and at
   0.18 m a tile is far too small to fit a plane to anyway. Per-vertex keeps the
   mat C0 across the whole bog because abutting tiles share their edge vertices
   exactly. The same fetch gives the ground frame the fragment shader lifts the
   baked normal into, so moss on a slope lights as moss on a slope.
5. **A carpet-specific alpha reference (0.06).** The grass 0.4 punches holes in
   a mat: the baked tile square is 92–97% covered, but the mip chain pulls
   distant tiles toward the mean and they fail the test outright. No
   camera-inside fade either — a mat you are standing on must not open a hole.
6. **Crevice shading from the height channel.** The top view's baked sky
   occlusion is 1.0 by construction (nothing stands above a straight-down
   capture), so a carpet has no AO at all — but `surf.b`, the front-surface
   distance, *is* the height of each texel above the peat at 1.3 mm resolution.
   Darkening the low texels is exactly the cavity shading between capitula, and
   it is the single largest "reads as 3-D" contributor at close range. Applied
   mean-preserving (`carpetCrevice`, default 0.55) against the species' own
   coverage-gated mean, so it redistributes light instead of dimming the bog.
7. **The relief itself**, and the one place this needed real thought. The DISP
   atlas the plant path uses is 32 texels per tile, flood-filled and blurred
   twice — over a 0.18 m tile that is 18 texels, and **56–68% of its variance
   survives 4-fold rotation**, i.e. is identical on every tile no matter how the
   stand rotated it. Tiled in lockstep at 0.18 m, that 5 mm ridge occludes ~6 cm
   of the tile behind it at a grazing angle, and the bog reads as **woven
   basketwork** — unmistakable at `cam=grazing`, and the first version did
   exactly that. Two changes fixed it: resample the field from the *full-rate*
   height channel (136 texels across the tile, alpha-gated) into a 16² tile-square
   field, which drops the rotation-invariant share to ~0.25 while carrying more
   relief; then subtract the rotation-invariant component outright, mean
   preserved. Nothing is invented — this removes a signal, and specifically the
   only signal a periodic tile cannot help repeating. What is left is 0.5 cm RMS
   of irregular cushion lump that the 90° rotations decorrelate.
8. **Crop the textures to the tile square** (see below) — which turned out to be
   a correctness fix, not just a density one.

### The distance-band bug this pass nearly shipped

Whole-frame comparison said the mat matched the billboard baseline. Split by
distance band at `cam=grazing` it did not: `debug=albedo` matched 001 band for
band (77 / 72 / 66 / 71 luma), but the lit render was **15% darker in the far
band** and only 3% darker up close. The cause is the mip convention trap in
CLAUDE.md, in a form worth recording: only **26-34%** of the baked top tile is
inside the periodic tile square, and the SURF atlas is box-filtered, so its
height channel falls from 0.78 to **0.34** at deep mips — the crevice term then
darkened the whole far field. The albedo was immune purely because *its* mip
filter is coverage-weighted. Cropping the carpet's textures to the tile square
at load time fixes it at the root: every mip level averages moss only, and the
crevice term converges to exactly 1.0 at distance. After the fix the lit ratio
against 001 tracks the albedo ratio to within 0.5% in every band, i.e. the light
term is distance-flat.

(020 is now slightly *brighter* than 001 in the far band, because 001 samples a
sub-rect of the shared atlas and its deep mips bleed in the neighbouring side-view
tiles. That difference is 001's, not this one's.)

## Status

working — verified by headless screenshot at `grazing`, `topdown`,
`inside-plant`, `far-horizon`, `carpet-close`, on two ridged slopes and at the
worst-case wetness-zone spot, at `debug=normals|lighting|albedo|coverage|depth`,
and on the `default`, `bog`, `dense-mixed` and `scaling-100m` stands. No console
errors and no toasts.

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

## Findings — the moss

**Before / after, from screenshots.** Before: the bog was ~26% covered, in
literal stripes (128 of 484 slots), and what did draw was *upright cards* —
1.5 cm-wide vertical slabs stabbed through a 7 cm cushion, showing dark spiky
edges from above and nearly nothing edge-on. Species VRAM 31 MB, over budget.
After: a closed mat everywhere the stand puts one, at life size, with the three
micro-habitat states interlocking across the wetness field — from `topdown` the
bog reads as real ecology (pale dry crests, dark wet hollows, Calamagrostis
bands in the hollows). At `carpet-close` it is a continuous moss texture with
crevice depth between the capitula. Across the two ridged-slope views the mat
follows the relief with no bare peat, no floating tiles and no buried edges, and
at 1 m straight down over the steepest flank there is no cracking along the
grid. VRAM 13.6 MB per state.

**Against 001-billboard-smoke** (`#/ab/001-billboard-smoke/020-curved-cards?stand=bog`),
honestly: at 5–15 m the two are near-indistinguishable, and both show the same
faint tile grid — that grid is *baked into the source tile* (its periodic seam),
not something either renderer adds. Closer in, 020 has visible cushion relief
and clearly stronger crevice/tip contrast; 001 is flatter and muddier. At
distance 020 is the more honest of the two (see the distance-band section). So:
better, but by a modest margin, and the margin is mostly the crevice shading
rather than the geometry.

**Why the geometry margin is small, which is the honest verdict.** The moss
tile's *surface* is only 0.4–0.7 cm RMS away from flat at scales a lattice can
carry; its 9 cm peak-to-peak is a few deep gaps down to the peat, and its real
intricacy — the individual capitula — lives at sub-centimetre scale. So a
vertex-displaced patch can express the cushion's undulation and nothing more.
This representation renders moss *well* (a closed, terrain-following,
correctly-lit, correctly-sized mat with genuine if slight relief), but it cannot
render the mesh's intricacy as geometry at any sane vertex budget; that detail
has to come from the albedo and the normals, exactly as it does for the
billboard. The 3-D that curved cards buy on a 1.2 m Calamagrostis clump is a
25 cm stand-off; on Sphagnum it is 5 mm.

**Still bad / not done.**

- The tile grid is faintly visible at close range in both renderers. It is in
  the baked top view (a coverage/colour seam where the periodic geometry wraps),
  so fixing it means a bake change — a wrap-aware dilation, or capturing the
  tile with its neighbours' overflow — not a renderer change.
- The moss albedo is dark and low-contrast at every distance. That is the bake
  faithfully reproducing the source mesh's vertex colours; nothing here touches
  it.
- Bog `cull` is now 1.6 ms rather than 0.5 ms, because three carpet entries each
  evaluate 484 slots per cell and each pays for `scatter_wetness` (4 heightmap
  loads) at every one of them. That is 3x redundant across the three entries and
  it is the obvious next target, but it needs either a shared pre-pass or a
  harness-side change, so it is out of scope for this pass. It is also exactly
  what the baseline pays.
- `slope_align` is not consulted: for a carpet `carpet_div > 0` already means
  "full conformance", and per-vertex conforming has nowhere to put a 0..1 blend
  without breaking C0 between tiles. The bog's Calamagrostis (`slopeAlign` 0.3)
  is still drawn bolt upright by the plant path — a real gap, but a grass one.
- No re-bake was done, deliberately (moss bakes are ~2 minutes each). The one
  thing a re-bake would buy is a sharper top tile: the capture is 512 px for the
  whole 0.31 m square, so the 0.18 m tile gets ~271 of them. Keeping the top
  tile at its 1024 px supersampled rate for carpet species would double the
  close-range sharpness for 0.4 MB.

**Params.** `carpetCrevice` (default 0.55) is the only new one; `curvature`
drives the mat's relief amplitude as well as the plants'. Swept both at
`cam=grazing` and `carpet-close`: crevice 0 is flat and muddy, 0.9 over-darkens
the tile seams, 0.55 sits between. Overscale was never introduced — tiles abut
exactly, so there is nothing to hide and nothing to z-fight.

**No regression on `default`.** Same meadow, verified by before/after screenshot
at `cam=grazing`, plus `topdown` and `inside-plant`. Σp50 6.5 → 6.8 ms at
`grazing` with fifteen sibling agents on the GPU (cards 2.96 → 3.02); the plant
path's only changes are two extra vertex outputs and a slightly different
instance-buffer size, and species VRAM moved 19.2/18.4/19.3 → 19.1/18.3/19.2 MB.
`scaling-100m` still renders 134 M plants at Σp50 4.8 ms.
