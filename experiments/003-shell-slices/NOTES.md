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
- region instance buffer: capacity area(2·rRegion)² × density × 1.15 × 32 B
  ≈ 1.8 MB at rRegion 64 / density 3 (density- and rRegion-dependent;
  counted per species, grows in place if rRegion is raised)
- HUD totals after the audit: calamagrostis 14.8 / elymus 14.5 / poa 14.8 MB,
  51.7 MB overall — **under the 25 MB budget** with room for density-8
  stands. (Before the audit the instance buffers were sized for the *maximum*
  rRegion 96 regardless of the actual setting: 16.3–17.0 MB, 59 MB overall.)

## Bake

The runtime aux texture is **transcoded from the artifact at load**, not
sampled as stored: the file keeps the compact octahedral pair, but the
texture the shader samples holds `mean_normal × occupancy` offset-encoded in
rgb (128 = exactly zero) plus sky visibility in a — see `buildAuxTexture()`
and the Audit section for why. When the bake version is next bumped this
belongs in the artifact itself.

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
- The normals were poisoned by empty voxels until the audit (see Audit);
  numbers and screenshots taken before it show flatter shading than the
  current build.
- Known artifacts, honestly: voxel-ring moire on stems seen from
  inside-plant; hard rectangular slice borders occasionally visible where a
  curtain cuts the community tile; band K-jumps (12→4) pop slightly on
  approach; dither pattern is world/plant-stable but crawls under wind sway
  (content moves through the volume-space hash).

## Audit (structural waste + debug views)

### Debug views (were entirely missing)

Both shaders now `#include "src/wgsl/debug.wgsl"` and route their final
fragment through `debug_shade(...)`; fog — and the experiment's own
`debugAxis` tint — apply only when `debug_mode() == DEBUG_OFF`. What each
mode is fed:

- albedo / lighting: the *occluded* albedo (`albedo × occ`), i.e. exactly
  what `light_surface()` multiplies, so the lighting view's division back out
  is exact.
- coverage: the slab opacity `a_eff` the fragment resolved to **before** the
  alpha test — how much canopy this pixel really stands for, not a constant 1.
- normals: the per-fragment decoded volume normal, after the two-sided
  camera-hemisphere flip and the 0.35 blend toward up.
- far shell: shell albedo, the terrain-blended canopy normal, and the
  species-blended coverage.

**What the normals view exposed — and the one image-changing fix.** The
volume is 99.0% empty (12 817 of 1 300 770 voxels carry coverage). Empty
voxels were left at oct pair (0,0) — and (0,0) decodes to *straight down*.
Every trilinear tap therefore averaged 7–8 "down" votes against 0–1 real
ones, and every mip level compounded it, so the sampled normal was a readout
of empty space, not of the plant: the normals view was a saturated rainbow
with no plant structure. The stored artifact is unchanged; instead
`buildAuxTexture()` transcodes it at load into `n × occupancy` (offset-encoded
rgb, 128 = exact zero) with sky visibility moved to alpha. Linear filtering
and mips of *that* are by construction the occupancy-weighted mean normal —
empty voxels contribute a zero vector — and the shader decode collapses to a
rescale + renormalize (cheaper than the old oct branch, and more precise:
3×8 bits instead of 2×8). Normals now show plant structure and the expected
up bias; A/B against `000-ground-truth`'s normals view for calibration, its
real geometry is just as high-variance. **This changes the lit image**
(shading is no longer flattened by a downward bias) — it is the deliberate
repair of a broken normal channel, not a structural change.

Lighting view is near-white over most of the frame: `light_surface()` returns
`sun·ndl² + ambient·hemi` which is ≥ 1 for lit surfaces in this scene, so it
clips — the harness terrain reads identically. The scattered black speckles
are the debug helper's `max(albedo, 1e-3)` floor biting on near-black voxels,
not a renderer bug.

### Structural waste found and fixed

- **Instance buffers sized for the wrong region.** Capacity was computed from
  a hardcoded 96 m half-extent (the *max* of the `rRegion` param) instead of
  the actual `rRegion`, so the default setup carried 2.2× the instance VRAM
  it could ever use. Now sized from `ctx.params.rRegion`; the existing
  grow-on-overflow path covers raising the param. 59 MB → 51.7 MB.
- **Uniform buffers rewritten every frame with values that never change.**
  The three band UBOs and the shell UBO depend only on params, fov and
  viewport height — never on time or camera pose — yet all four were
  re-uploaded every frame. Now gated on `onParamsChanged` plus a change in
  `2·tan(fov/2)/height`. `bandK` is updated in the same place the uniforms
  are, so the draw's instance count can never disagree with `band.k`.
- **`pow(slab_vox, 0.7)` evaluated per fragment though it is flat per slice.**
  Slab thickness is constant over a whole slice quad; the vertex stage now
  resolves the power and passes the optical depth in `misc.y`. A transcendental
  moved from every covered fragment to 6 vertices; the fragment maths is
  otherwise unchanged, so the image is identical.
- **Far shell evaluated all three species' mottle noise unconditionally.**
  The per-species blocks computed `wt = weight × noise(...)` and *then* tested
  `wt > 0`, so a stand that does not use a species still paid its 4-hash value
  noise per fragment. Branching on `weights.x > 0` first is exactly
  equivalent (the noise factor is always ≥ 0.3) and drops 2/3 of that work on
  single-species stands.
- **Region rebuild materialized three intermediate point arrays.** The LOD
  bucketing built three `ScatterPoint[]` (~180k pushes at rRegion 64) and then
  walked them again to fill the scratch. Replaced with a counting pass plus a
  direct scatter into the band's slot — same order, same bytes, no
  intermediates. Only fires on >6 m camera moves, but that is exactly the
  hitch flagged below.

### Deliberately left alone

- **Two render passes (`slice-stacks`, `far-shell`) on identical
  attachments.** They fold into one pass trivially and that would save a
  store/reload on tiled GPUs — but it would also collapse the two timing
  labels into one, and the split between slice fill and shell fill is the
  most useful diagnostic this method has. Worth doing if a tiled-GPU target
  ever matters more than the breakdown.
- **Slice quads span the full volume extent.** Shrinking them to per-slab
  content bounds is still the first real fill win (99% of voxels are empty!),
  but it needs per-slab bounds in the artifact and a bake-version bump, and
  it can shift the image. Left as the next experiment, not an audit fix.
- **Far shell rings run to r1 = 900 m while the stand ends at ±128 m**, so
  ~2/3 of the rings are rasterized only to hit the `abs(w.xz) > stand_r`
  discard. Clamping r1 to the stand would change the ring distribution and
  therefore the shell geometry — an image change for a fill saving in the
  cheapest part of the screen. Not worth it blind.
- **Load-time derivation of 3D mips, the top-view composite and its mips**
  (~1 s per page load) belongs in the artifact, but it is init work, not
  per-frame work, and moving it means a bake-version bump. Same for the new
  aux transcode.
- **No frustum culling**: the region is a camera-centered box, so roughly half
  the materialized plants are behind the camera and rejected by the clipper.
  Culling them would require rebuilding on camera *rotation*, which is exactly
  what the move-only rebuild trigger is designed to avoid.

### Verification

Image-neutrality of the structural fixes was measured, not assumed: the
audited build **with the normal repair reverted** was compared against the
pre-audit build at a pinned time and camera
(`#/run/003-shell-slices?cam=grazing&det=1&t=3`, canvas-only crop). Result:
**1 differing pixel out of 91 200, by 1/255 in one channel** — a single
last-bit rounding wobble from `pow()` moving to the vertex stage. Everything
visible in the current build therefore comes from the normal repair alone.

All six debug modes screenshotted headless on `default` at `cam=grazing` and
`cam=topdown`; console clean of WebGPU/shader errors. Frame times were
**not** used to justify anything here — up to four agents shared this GPU
during the audit.

## Species coverage

All three: calamagrostis-canescens and elymus-repens as periodic community
tiles ("one tile = one plant" at the stand's scatter points — same convention
as the tile mesh's semantics), poa-pratensis as a true single specimen.
