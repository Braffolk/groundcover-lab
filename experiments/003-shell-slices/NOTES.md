# 003-shell-slices — axis-locked slice-stack shells

## Idea

The canopy as a stack of thin translucent layers — but the stack is **per
plant** and its **axis rotates with the view**. Each species is baked once
into a small 3D voxel volume (premultiplied albedo + coverage, mean octahedral
normal, and a baked top-down sky-visibility channel = the precomputed-raycast
loophole, used as canopy self-occlusion). At runtime every plant of the stand
is drawn as K planar slices through its volume:

- The slice axis is chosen per plant per frame as the plant-local axis most
  aligned with the view direction (Y bias adjustable). Seen from above you get
  classic horizontal shells; at grazing angles the stack flips into vertical
  slice "curtains" — the classic shell-texturing grazing collapse (seeing
  between the layers) is impossible by construction. This is the answer to
  "what happens at grazing angles".
- "How few layers can you get away with": K collapses with distance,
  12 → 4 → 1 over three bands. Slab opacity is renormalized
  (`1 − exp(−σ·cov·slabVox^0.7)`) so one fat slab is as dense as twelve thin
  ones, a 3D mip chain integrates the fatter slab (mip level =
  min(slab level, screen-footprint level)), and a plant-space dithered alpha
  test renders semi-transparent fluff (the calamagrostis heads) as sparse
  pixels instead of solid blobs. Very fat Y-slabs migrate to canopy-top
  height where the visible surface actually lives.
- The logical end state of "fewer layers" is **one** layer: beyond the
  camera-centered plant region (`rRegion`, default 64 m) the entire meadow
  degenerates into a single terrain-draped canopy shell — a polar grid around
  the camera whose fragments tile the baked top-down composites of each
  species (blended by stand density × mottle noise, soil showing through low
  coverage), dissolving in under the stochastically-fading last plant band.

Per-frame cost is bounded by the camera region (~180k plants materialized for
the default stand at rRegion 64, regardless of stand size) and the screen —
never by total plant count. Placement is exactly the stand's: instances come
from `ctx.scene.scatter.region()` over a camera-centered clamped AABB,
rebuilt only when the camera moves > 6 m. No per-frame raymarching: one
explicit-LOD 3D texture sample for color + one for normal/AO per fragment.
Wind: shared `wind_sway` per plant with per-species sway and per-plant phase,
sheared by normalized height (Y-shells shear over each other, curtains bend).
Camera-inside-plant: plants dissolve within 0.5 m (vertex-collapsed at
< 0.22 m).

## VRAM budget math

Per species (measured in HUD, default stand, rRegion 64):

- voxel volume: ~97×149×90 ≈ 1.3M voxels × 4 B × 2 textures × 1.14 (3 mips)
  ≈ 11.9 MB
- top view: 128² × 4 B × 1.33 (6 mips) ≈ 0.09 MB
- region instance buffer: capacity area(128 m)² × density × 32 B ≈ 1.8–5.7 MB
  (density-dependent; counted per species)
- HUD totals: calamagrostis 16.9 / elymus 16.3 / poa 17.0 MB — **under the
  25 MB budget**, including headroom for density-8 stands (~23 MB worst case).

## Bake

`bake.ts` voxelizes the raw GCMESH1 (all three species, including the 6.5M-tri
poa specimen) by barycentric-sampling every triangle (~0.4–1.0 s per species
in-browser, single-threaded JS) into an SSV1 artifact: 64 B header +
premultiplied albedo/coverage rgba8 + oct-normal/sky-visibility rgba8,
committed to `mesh/baked/003-shell-slices/<species>-v3.bin` (~10.4 MB each).
Community tiles are centered on the tile period ("one tile = one plant");
poa is centered on its footprint. 3D mips, 2D top-view composite (with
periodic wrap of overhanging foliage) and its mips are derived at load
(< 1 s). Note: the harness `bakedArtifact` committed-file path was unusable —
the dev server's SPA fallback answers missing `/mesh/baked/...` files with
`index.html` HTTP 200, which gets cached as the artifact. I use a validating
loader (magic + version + size check) over committed file → OPFS → bake
instead. Harness gap worth fixing.

## Status

working — verified by headless screenshots on `default` (grazing, topdown,
inside-plant, far-horizon) and `scaling-100m` (grazing): 134M plants render
with identical VRAM (59 MB total) and no CPU/GPU cost growth.

## Findings

- Axis switching genuinely kills the grazing-angle shell failure; with
  `debugAxis` you can watch stacks flip Y→X/Z as the camera drops. Switches
  pop (stateless per-frame argmax, no hysteresis possible) but at the
  distances where mip blur dominates the pop is hard to see; near plants
  almost never sit exactly on a 45° boundary for long.
- The slab-opacity boost is the central quality dial and it fights itself:
  full linear integration (σ·slabVox) makes low-coverage fluff (calamagrostis
  panicles) solid pink blobs; no integration makes far plants dissolve. The
  compromise — pow 0.7, per-band σ scale (0.5/0.85/1.0), plant-space dithered
  threshold — gives a convincing mid/far field; the near field (< 3 m) still
  reads as chunky voxel "coral" on the fluffy heads. 8 mm voxels cannot do
  better; a near-field-only detail pass would be the next experiment.
- Y-slabs migrating to canopy-top height (deep-slab blend to h=0.7) plus
  toning the far shell by 0.88 removed most of the brightness seam at the
  plant-region → far-shell handover ring visible from topdown; a faint ring
  remains under some sun angles.
- Per-tile random 90° rotation of the far-shell top view breaks tiling but
  introduces grid seams (rotation destroys periodic continuity) — reverted;
  density mottle noise hides repetition well enough.
- Timings in this session were GPU-contended (16 parallel agents on one
  machine; numbers ranged 4.8–33 ms Σp50 for the same scene) — **not
  benchmarked**; run `#/bench/003-shell-slices?stand=default&spline=orbit-low`
  on a quiet machine before quoting anything. Topdown was consistently the
  cheapest view (~5 ms Σ), grazing the dearest (near-band overdraw:
  12 slices × discard-heavy fragments; shrinking slice quads to per-slab
  content bounds would be the first optimization).
- Region rebuild (scatter.region over ±64 m, ~180k plants, 3 entries) costs
  ~10–25 ms CPU but only fires on > 6 m camera moves; a hitch is measurable
  on flythroughs. Incremental per-cell updates would remove it.
- Known artifacts, honestly: voxel-ring moire on stems seen from
  inside-plant; hard rectangular slice borders occasionally visible where a
  curtain cuts the community tile; band K-jumps (12→4) pop slightly on
  approach; dither pattern is world/plant-stable but crawls under wind sway
  (content moves through the volume-space hash).

## Species coverage

All three: calamagrostis-canescens and elymus-repens as periodic community
tiles ("one tile = one plant" at the stand's scatter points — same convention
as the tile mesh's semantics), poa-pratensis as a true single specimen.
