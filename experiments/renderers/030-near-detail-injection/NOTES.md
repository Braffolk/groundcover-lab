# near detail injection — depth-reprojected cards

## Idea

A billboard is a poster because it only stores *what* is at each texel, never
*where*. So the bake stores where: next to the albedo atlas there is a geometry
atlas whose blue channel is the **signed depth of the first surface along that
capture's axis**. That one channel is enough to turn a card back into geometry at
runtime, analytically, with no marching.

For a capture `k` the three view axes `(u, v, f)` are *linear functionals of world
position* (the bake hands the runtime the axis vectors with `1/extent` and
`1/scale` folded in, so a dot product per axis in the vertex stage is the whole
setup). A straight eye ray therefore stays straight in `(u, v, f)` space, and
finding where it meets the stored height field is one division:

```
dir = cam_axc - axc              // eye ray, in view-axis coords (no normalize needed)
t   = (zb - axc.f) / dir.f       // zb = 2*depth - 1, from ONE texture tap
uv' = uv + 0.5 * t * dir.uv      // read colour / normal / occlusion there
```

One step is the classic parallax offset; a second exact sample gives one Newton
refinement. That is all of it. What it buys, against the brief:

- **parallax inside the silhouette** — near texels (`zb ≈ +1`) shift up to ~20% of
  the card width against far ones (`zb ≈ -1`) as the camera moves. It is the real
  first-order geometry, not a trick: at the capture direction the offset is exactly
  zero and the card degrades to the baked view.
- **a silhouette that changes with view direction** — coverage is read at `uv'`, so
  the outline is *reconstructed*, not rotated. It morphs continuously between the
  15 baked azimuths instead of snapping, and rising off the horizon makes the clump
  lean and foreshorten like real geometry — which is what fixes the baseline's own
  worst artifact, the flat "floating pancake" top card at mid elevations.
- **self-occlusion and ground contact** — the geometry atlas's alpha channel is a
  genuinely volumetric sky-visibility term: the bake voxelizes the source mesh into
  a triangle-area extinction grid (periodic in xz, so a community tile is occluded
  by its own neighbours) and integrates transmittance along 13 cosine-weighted
  directions per voxel. Deep-in-the-clump texels and texels near the soil come out
  dark, canopy tips come out open. The whole meadow gains the interior darkening
  that the baseline fakes with a linear bottom gradient.

**Where the detail is spent.** The cull pass sorts plants into six 3D-distance
buckets of one instance buffer; the bucket picks both the draw order (near to far,
so the biggest occluders write depth first and early-z eats the bands behind them)
and the fragment pipeline:

| tier | 3D distance | taps | work |
|---|---|---|---|
| `fs_near` | < nearDist (10 m) | 3 | 2 relief steps, normal/occlusion re-read at the refined hit, sun transmission |
| `fs_mid` | < midDist (34 m) | 2 | 1 relief step (colour + silhouette parallax); shading normal from the card point |
| `fs_far` | rest of the region | 2 | plain billboard — no dependent fetch, no parallax ALU, and the geometry tap happens *after* the alpha test |

Parallax strength ramps to zero over the last 12 m of the mid band, so the tier
change is not a pop: `fs_far` is exactly what `fs_mid` converges to. Two caps keep
the reconstruction honest, and they are the interesting part of the technique:

1. **A "no surface" sentinel.** Depth `0` means "this capture saw nothing here".
   Past the dilated ring around a silhouette there is genuinely nothing to walk
   onto, so the step is skipped and the gap stays a gap. Without it, empty texels
   act like a surface at mid depth and drag unrelated foliage into every hole —
   the ugliest artifact of the first version.
2. **A screen-space error budget.** A single depth layer cannot resolve what the
   capture never saw, and the residual disocclusion error scales with the offset,
   so the offset is clamped to **14 px on screen** (`uv_limit`, computed per plant
   in the vertex stage). Beyond ~7 m that clamp never binds and the parallax runs
   at full geometric strength; closer in it holds the error to a few percent of the
   plant instead of tearing panicles into confetti.

Sampling is a plain `textureSampleLevel` with a **per-plant mip level** derived
from the card's projected size. Cards are camera-facing, so their screen footprint
is isotropic and one level per plant is right; it also keeps a reprojected uv
(discontinuous wherever the ray crosses a silhouette) away from derivative
hardware, which would pick an absurd mip and sparkle. No dither anywhere: hard
alpha test, depth write, no `frag_depth` (early-z stays alive), and the
camera-inside / region-edge fades erode through the alpha reference.

Plant-count independence is the baseline's: nothing is ever materialized for the
stand, cost is O(region area). 557k (`default`) and 134.2M (`scaling-100m`) plants
render identically.

## VRAM budget math

Per species, HUD-verified (MiB):

| item | MiB |
|---|---|
| albedo atlas 1536² rgba8 + 11 mips | 12.0 |
| geometry atlas 768² rgba8 + 10 mips | 3.0 |
| culled instances, 6 buckets, 16 B each (density 3 → 164 496 slots) | 2.5 |
| entry info uniform (288 B) + indirect args (96 B) | noise |
| **total** | **17.5 / 25** |

Measured: calamagrostis 17.5, elymus 17.1 (density 2.5 → smaller instance buffer),
poa 17.5. Worst defined stand (`dense-mixed`, density 5) ≈ 19.2, still under. The
4× cheaper geometry atlas (192 px tiles vs 384 for albedo) is a deliberate split —
depth/normal/occlusion are low-frequency next to colour, and a `-1` mip bias keeps
the two in lockstep.

Bucket slices are fixed at init and sized for the worst case over the whole
`nearDist`/`midDist` param range, because each bucket is bound as its own storage
window (see Findings); an overfull bucket drops its last plants rather than
scribbling into its neighbour.

The bake needs ~350 MB transient GPU memory (raw vertex/index buffers, two 3072²
targets + depth, two readbacks), all through `ctx.res` tagged `bake-scratch` and
destroyed when the bake ends.

## Bake

`bake.ts` → `mesh/baked/030-near-detail-injection/ndi-v3-<species>.bin`, 11.25 MiB
each (64 B header + 1536² rgba8 albedo + 768² rgba8 geometry). Standard harness
`bakedArtifact`/`commitBake` flow, magic+size validated on load (the dev server
answers missing `/mesh/baked` files with `index.html` at 200, which would otherwise
poison both the OPFS cache and the committed-file path).

Per species, one render pass over the raw GCMESH1 mesh draws **16 orthographic
captures — 15 azimuths at 24° steps + 1 straight-down top view — into a 4×4 grid**,
2× supersampled (3072²), with two colour targets:

- albedo rgba8: authored colour; coverage from the 2×2 coverage-weighted
  downsample.
- geometry rgba8: `rg` = oct-encoded mesh-frame normal (flipped toward the capture
  axis, two-sided foliage), `b` = signed depth along the capture axis, `a` = sky
  visibility sampled from the extinction grid (32×48×32 r8unorm 3D texture, built
  on the CPU from triangle areas and 13-direction transmittance marching, bound to
  the bake shader). 4×4 coverage-weighted downsample, normals averaged as
  *vectors*.

15 azimuths rather than the baseline's 8 is the key bake decision: reprojection
error scales with the angle between the selected capture and the true eye ray, so
halving that angle (22.5° → 12°) halves the disocclusion tearing on close foliage —
worth far more than the 128 texels per tile it costs, at identical VRAM.

Both atlases are dilated into empty space (6 rings albedo, 8 rings geometry); the
geometry dilation is exactly how far a reprojection may legally grow a silhouette,
everything past it is the depth-0 sentinel. Mips are generated on the GPU at load:
coverage-weighted for albedo, and for geometry the normals are decoded, averaged as
vectors and re-encoded (a box filter over oct codes drifts toward the tile mean — a
known wart of the baseline's normal mips).

A cold bake of all three species — including fetching 374 MB of raw meshes and
~20 M CPU ray-march steps for the occlusion grids — takes well under a minute here;
afterwards nothing touches the raw meshes at all.

## Status

**working** — verified by headless screenshots on `default` at all four standard
cams, all five debug views, plus `scaling-100m` and `dense-mixed`, plus the A/B page
against the baseline at grazing / topdown / far-horizon. `npx tsc --noEmit` clean,
zero console errors or warnings.

## Findings

**Perf (contended — 15 sibling agents shared this GPU, so only the SAME-FRAME A/B
ratio is quotable).** `#/ab/001-billboard-smoke/030-near-detail-injection`,
1280×800, `cull + cards` on each side:

| cam | A (billboards) | B (this) | ratio |
|---|---|---|---|
| grazing | 2.75 | 3.63 | **1.32×** |
| topdown | 1.78 | 2.00 | **1.12×** |
| far-horizon | 2.52 | 2.96 | **1.17×** |

(An earlier run of the same three cams gave 1.28× / 1.14× / 1.32× — that spread is
the contention, not the method.)

Solo total GPU Σp50 on the same contended machine: 6.4 ms grazing, 3.6 topdown,
6.2 inside-plant, 5.4 far-horizon, 5.5 on `scaling-100m` (134.2M plants — identical
work and identical VRAM to the 557k default, which is the plant-count-free test) — inside the 9 ms ceiling with room, but these
are shapes, not claims. No `results/` bench JSON is claimed for the same reason;
rerun `#/bench/030-near-detail-injection?stand=default&spline=orbit-low` on an idle
GPU before quoting anything absolute.

Where the 1.2–1.3× goes, measured by forcing every bucket onto one tier:

- all six buckets on `fs_far` → **1.05×** the baseline. So the extra geometry
  channels, the 15-view atlas and the six-bucket draw are free; the 0.05 is vertex
  work (the per-view axis frame costs a few more `rot_y`s).
- the ladder as shipped adds ~0.25×, all of it the dependent tap in the near/mid
  bands. Explicit per-plant mip levels instead of `textureSampleGrad` + 4×
  anisotropy took that from ~0.5× to ~0.25× on its own — anisotropy buys nothing on
  a camera-facing card.
- bucketing by **3D** distance rather than horizontal distance took topdown from
  1.62× to 1.14×: from 40 m up, every plant was landing in the near/mid tiers
  although each covered 20 px.

**Looks — honest verdict: clearly better at every distance the eye spends time on,
with one caveat.** At grazing and far-horizon the wipe is not close. The baseline is
a uniform pink-and-green carpet in which every plant is equally bright; this reads
as a meadow with volume — clumps at visibly different depths, dark canopy interiors,
bright sunlit tips, plants sitting *in* the sward instead of on it. The occlusion
term does most of that and the parallax does the rest: panning sideways slides near
parts of a clump against far parts, and orbiting one plant no longer snaps between
poster views. Topdown is also better than the baseline's known-weakest case,
because the top card now leans and parallaxes instead of reading as a decal.

The caveat is the extreme close-up (camera under ~1.5 m, `inside-plant`): one depth
layer cannot invent what the capture never saw. Before the screen-space offset clamp
went in, panicles at 0.5 m tore into confetti and the baseline was plainly cleaner
there. With the clamp the close-up is coherent and still has more depth than the
baseline, but its silhouettes are a little softer and rattier than the baseline's
crisp cutouts. That is the honest trade; `reliefScale=0` reproduces baseline
behaviour exactly if you want to see the difference in isolation.

Other findings worth keeping:

- **`drawIndirect` silently ignores `firstInstance`** unless the optional
  `indirect-first-instance` feature is enabled — no validation error, the draw just
  behaves as if it were 0. The first version packed all six buckets into one
  instance buffer and addressed them with `firstInstance`; every bucket redrew
  bucket 0, so the meadow simply ended a few metres from the camera and looked
  plausible otherwise. Each bucket now binds its own storage window (offset/size,
  16 instances = 256 B aligned) and indexes from 0. Worth knowing for any
  experiment that wants several indirect draws out of one buffer.
- Sky visibility wants a *lifted* curve, not a linear one:
  `1 - strength·(1 - sqrt(ao))`. Applied linearly at full strength the canopy
  interiors went nearly black and the stand read gloomy and brown; the sqrt keeps
  the contrast that sells the volume while leaving shadowed panicles recognisably
  pink. `aoStrength` 0.55 is the taste setting; 0 disables it. The `albedo` debug
  view is the proof that the browner lit look is occlusion and not a bad bake — the
  raw atlas is pink panicles over bright green blades.
- Empty-space depth needs the sentinel, not a "mid plane" default. That one change
  removed most of the hole-filling mush.
- The top card had to be pushed from ~35° to ~30–46° elevation before it fades in:
  its depth axis is +Y, so at low elevation the eye ray is nearly perpendicular to
  it and the reprojection guard has to clamp, which showed up as little striped
  rectangles floating in the canopy. Discarding fragments whose reprojection leaves
  its tile (instead of letting `tile_uv` clamp and smear the border texel) killed
  the remaining ones.
- Debug views: all five wired through the shared `debug_shade()`, fog only in
  `DEBUG_OFF`. `normals` is genuinely per-fragment and reads pink (+x+z, back at the
  grazing camera) in the near field and green (+y) on top cards; `lighting` spans
  ~0.15 to blown-out and visibly carries the occlusion field; `coverage` is the
  baked alpha the *reprojected* tap resolved to, so it doubles as the alpha-test
  margin; `depth` is card-plane depth, because this technique deliberately does not
  write `frag_depth` — that is what keeps early-z alive, and it is also why
  plant-vs-plant intersections still resolve on the card plane rather than on the
  reconstructed surface. That is the one 3D cue this method does not buy.
- Wind composes with the reprojection for free: the view-axis coordinates are
  computed from the *rest* position and sway is added afterwards, so the
  reconstruction happens in the plant's rest frame and the sway shear carries the
  reconstructed image with it. No inverse-shear maths needed.
- Harness wishlist: a dev-mode warning when a non-zero `firstInstance` reaches
  `drawIndirect` (or the `indirect-first-instance` feature enabled on the device)
  would have saved an hour of bisecting a bug that produced no error at all.
