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

Bog stand (five entries, HUD, rRegion 64): Sphagnum wet-vigorous 17.8,
late-season 22.6–23.1, sun-exposed 20.5–20.9, calamagrostis 13.9, poa 13.4 MB —
97.6 MB overall, every species inside 25 MB. Where a moss species' budget goes:

- voxel volume 138×61×154 = 1.30M voxels at a **1.52 mm** edge → 5.91 MB per
  texture × 2 (colour+aux) = 11.8 MB including the mip chain
- the two extra carpet mip levels (see below) add 2 477 voxels, i.e. **0.19 %**
  of the volume — 0.02 MB, which is why they are worth having
- top view 0.09 MB
- region instance buffer 5.9–11.2 MB (184k–350k tiles × 32 B). This is the one
  place the split is arguably wrong for a mat: a carpet tile needs x, z and two
  bits of yaw (scale is constant, y is `terrain_height`, phase is 0), so 8–12 B
  would do and the same VRAM would buy a 1.2 mm volume. Not changed here — the
  instance format is shared with the upright path.

1.52 mm voxels are the point of this experiment on this species: 228 ramets per
0.18 m tile means ~12 mm between capitula, so the volume resolves individual
capitulum rosettes with ~8 voxels across each. No other allocation of 25 MB gets
close to that on a 0.18 m tile.

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

Also verified on the `bog` stand (Sphagnum carpet) at grazing, topdown,
inside-plant, carpet-close, far-horizon, a 30° ridge flank, and all four debug
views — see "Moss carpet" below.

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
- Harness gaps hit by the moss pass (all worked around locally, none patched):
  the TS/WGSL scatter twins disagree on carpet scale (see "Moss carpet" change
  1); `carpetScale()` is not exported from `@harness`, so the tiling period for
  the far shell has to be recomputed as `SCATTER_CELL_SIZE / carpetDiv /
  species.tileM`; and there is no way to read the placement seed inside a shader
  without plumbing it through an experiment-owned uniform, which is what blocks
  zoning the far shell by `scatter_wetness()`.
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

All three grasses: calamagrostis-canescens and elymus-repens as periodic
community tiles ("one tile = one plant" at the stand's scatter points — same
convention as the tile mesh's semantics), poa-pratensis as a true single
specimen. Plus the three *Sphagnum palustre* states on the `bog` stand, as
carpets — see below.

## Moss carpet (bog stand)

*Sphagnum palustre* is a 0.18 m periodic community tile, 0.07–0.09 m tall, laid
out by the `bog` stand as a grid-snapped mat: 22×22 tiles per 4 m scatter cell
(484 slots, 0.1818 m step), constant scale, 90°-only yaw, three states zoned
across the wetness field. This experiment already loaded and voxelized all three
(1.52 mm voxels), and the volume was by far the best-suited *storage* for a
cushion of anything in the lab — but almost every decision *around* it had been
tuned for an upright plant, and six of them were actively wrong for a mat.

### What changed, smallest first

1. **Tile scale from `stand_table[i].scale_min`, not from the instance buffer**
   (one `select`). This is a HARNESS TWIN MISMATCH, not a preference:
   `createStandBuffer()` writes the computed `carpetScale()` (4/22/0.18 = 1.0101)
   into the stand table and `scatter.wgsl` reads it, but the TypeScript
   `Scatter.cell()` carpet branch returns `e.scaleMin`, which for a carpet entry
   is the stand's *placeholder* (1.7 / 1.6 / 1.4). Every renderer that
   materializes instances on the CPU — this one — therefore drew moss at 1.4–1.7×
   life size **and at three different sizes for the three states**, so tiles
   buried their neighbours 2×2 deep and the zone boundaries stepped in scale.
   The stand table value is the one that satisfies the documented invariant
   ("scale identical for every tile of that species"), so that is what the vertex
   stage uses when `carpet_div > 0`. Visible effect: the mat stops being a quilt
   of overlapping raised boxes, capitula come out at their real ~12 mm pitch, and
   moss overdraw drops ~2.9× in area.
2. **No region-edge dissolve and no camera-inside fade for a carpet.** Both would
   punch tile-sized holes in a closed surface; the second one opens a hole under
   your feet. (The region fade was inert by accident — carpet points carry
   `phase = 0` — now it is explicit.)
3. **Hard alpha edge for a carpet** (`carpetAlpha`, 0.30) instead of the dithered
   threshold the fluffy grass panicles need, and no value dither on the occlusion
   term. A cushion mat is a closed near-solid surface: stochastic coverage only
   fuzzed its silhouette and screen-doored the mid field, and it stops early-z
   from rejecting the slices underneath — exactly where a mat is deepest. The
   upright path keeps its dither untouched.
4. **Per-vertex terrain conforming** (ladder rung 3, three lines). Every slice
   vertex is displaced vertically by `terrain_sample(world.xz).x - base.y`, scaled
   by `stand_table[i].slope_align`. See the rung discussion below.
5. **The slice axis is locked to Y for a carpet.** The axis flip is this method's
   central trick — a single plant's horizontal shells collapse edge-on, so the
   stack rotates into vertical curtains. For a mat it is backwards: the tiles tile
   the plane, so the union of one horizontal slab per tile is a *continuous*
   canopy sheet that merely foreshortens at grazing, while a vertical curtain cut
   through the middle of a 9 cm cushion paints the cushion's dark bronze interior
   over the whole far field. The flip triggered exactly where it hurt most (a mat
   is most nearly edge-on far away).
6. **Fat Y-slabs migrate to the canopy top much harder for a carpet** (knee 5
   voxels instead of 20, target h 0.9 instead of 0.7). With 9 cm of Y extent even
   the 4-slice band integrates 2.3 cm of cushion, and the middle of a Sphagnum
   cushion is dark pendent foliage, not the olive capitula you actually see.
7. **Two more 3D mip levels for carpet species only** (5 instead of 3, `mip_max`
   is per species so the grasses are untouched). A 0.18 m tile is 2–5 px beyond
   20 m, i.e. ~1/40 of its 138-voxel width; mip 2 was a 40× undersample and the
   distant mat sparkled. Cost: 0.19 % of the volume's bytes.
8. **Two normal bugs that were painting the far field black.** In order of
   severity:
   - The **two-sided hemisphere flip**. `if (dot(n, view) < 0) n = -n` is right for
     a blade (either face may face you) and catastrophic for a mat: its normal is
     up, and `dot(up, view)` changes sign exactly where the ground rises above eye
     level, so everything past ~38 m at grazing was lit by a **downward** normal —
     a hard black ring around the camera, tens of metres wide. A carpet now flips
     into the UPPER hemisphere instead, unconditionally.
   - `buildMips3D` wrote its averages into a `Uint8Array`, which **truncates**. The
     aux texture encodes the mean normal with 128 = *exactly zero*, so any region
     whose leaf normals cancel — i.e. most of a moss cushion at a deep mip — drifted
     to 126–127 in all three channels and renormalized into a confident unit vector
     pointing down-left-back. `decode_normal`'s guard only caught exact zero. Now
     carpet mips round, and a carpet passes a real dead zone (≈6 counts of 8-bit
     precision) below which the direction is quantization noise and the normal
     falls back to canopy-up.
9. **A carpet's baked normals are lifted into the terrain frame**
   (`plant_basis_from_up(up, 0) * n`, with `up` from the same `terrain_sample`
   fetch the conforming displacement already pays for). `slope_align = 1` means the
   mat lies IN the ground plane; without this a cushion on a 30° flank lights
   exactly like a flat one.
10. **The far canopy shell now takes the stand's three heaviest-covering species,
    not the catalog's first three.** That was the same thing while the catalog had
    three species; with the Sphagnum states appended at indices 3–5 the bog's far
    field was built out of calamagrostis + poa (0.5–1.4 emergent plants/m²) and
    draped **0.6 m above the ground** — a floating grass plateau over a moss bog,
    which is what half the topdown frame actually was. Coverage for a carpet entry
    is computed from its grid (`carpet_div²` per 4 m cell × the share of the
    wetness axis it claims), never from `density` (8 for every carpet entry).
    Slots keep catalog order among the chosen three, so a three-species stand gets
    the identical slot assignment — and therefore the identical mottle-noise salts
    — it had before.
11. `carpetOcc` (0.55) — the sky-visibility floor for a carpet, separate from the
    upright 0.42. Honest reason: the visible surface of a mat is quantized to the
    nearest slice, so the sky visibility sampled there belongs a few millimetres
    *inside* the cushion and systematically under-reads. 0.7 matches the
    001-billboard-smoke reference brightness exactly but visibly flattens the
    rosette crevice shading, which is the whole point here; 0.55 keeps the
    structure at ~12 % below the reference.

### Terrain fitting: rung 3 (per-vertex), and why not 1 or 2

Rung 3 is the only correct rung for a *tiled* species, and this representation
gets it nearly free. Each slice vertex takes its own `terrain_sample(world.xz)`
and is displaced vertically by the height difference to the tile's base — a
vertical shear of the volume rather than a rigid tilt. Three consequences:

- **Continuity.** The displacement is a pure function of world xz, so two
  neighbouring tiles agree wherever they meet. Rungs 1–2 fit one plane per tile,
  and two neighbours fit two different planes: the mat cracks along every tile
  edge. Rendered bolt upright (rung 0) a 0.1818 m tile turns the mat into a
  staircase of flat-topped boxes whose step height reaches 0.1 m on the bog's 30°
  flanks — the same order as the 0.09 m cushion itself. That staircase, crossed
  with the tile grid, was the corduroy texture the "before" grazing shot shows.
- **Accuracy for free.** The heightmap texel is 0.5 m, i.e. coarser than a tile,
  so four quad corners capture the local ground almost exactly; the residual is
  the curvature of the finest terrain octave over 0.18 m, ~7 mm, inside a mat
  whose tiles already overlap by 15–30 mm.
- **Zero cost over rung 1.** `terrain_sample()` returns height *and* (nx, nz)
  from one bilinear fetch, so the shading tilt (change 9) is paid for by the same
  four texel loads. A vertical shear also keeps the capitula pointing up, which is
  how moss actually grows on a slope, and it leaves the footprint — hence the
  lattice — untouched.

Overscale was **not** used: the source mesh already overflows its period (0.21–0.24 m
of geometry in a 0.1818 m step at scale 1.01), so tiles overlap by 15–30 mm
without help, and a volume with thickness resolves that overlap by depth rather
than by painting a dark line.

### What improved, measured

Mean scene luma (three fixed boxes, HUD/panel excluded), bog stand, before →
after, with 001-billboard-smoke as the reference point:

| view | before | after | 001 |
|---|---|---|---|
| grazing near | 63.8 | 67.6 | 77.8 |
| grazing mid | 57.9 | 63.4 | 73.1 |
| grazing far | 51.0 | **70.3** | 88.5 |
| topdown | 40.6 | **63.5** | 71.9 |
| 30° slope | 76.2 | 79.2 | 95.8 |
| carpet-close (1 m) | 55.4 | 59.4 | 71.9 |

The numbers understate it; what the screenshots show:

- **carpet-close (1 m straight down)** is where this method wins outright.
  Individual capitulum rosettes are resolved — dark centres, lighter radiating
  leaves, at the correct ~12 mm pitch, in mixed olive/green/bronze. The reference
  at the same camera is a uniform noise field with faint tile-boundary lines and
  no capitulum structure at all. No lattice is visible in 003 at any camera.
- **grazing** went from a dark brown corduroy field with a black ring at ~38 m to
  a continuous olive/khaki carpet with interlocking wetness zones and a smooth
  handover to the far shell.
- **the 30° flank** went from a quilt of raised boxes with dark crevices between
  them to one continuous cushion surface with granular relief, correctly lit for
  the slope. The reference at the same camera is a perfectly flat painted texture.
- **topdown** went from half a mat, dithered against a grass shell floating 0.6 m
  up, to a closed carpet with visible zoning and green Calamagrostis in the
  hollows.
- **far-horizon** sits in a wet hollow and shows the wet-vigorous state as a vivid
  green mat measuring (84,144,33), against the source mesh's own vertex colour
  mean of (70,134,34) — the three states really are distinguishable, and the ones
  near the origin genuinely are ochre/bronze.

### What is still bad

- **Little cushion relief at eye level.** The mat's height field is quantized to
  the slice spacing — 7.75 mm at kNear 12 over a 9.3 cm volume, ~4 usable steps
  across the mesh's 3.3 cm of capitulum relief — so at grazing the surface reads
  as fine granular texture on a smooth sheet rather than as lumpy cushions
  occluding each other. kNear 16 (the max) only gets to 5.8 mm. Resolving this
  properly needs a per-fragment height search, which is the raymarching this
  project's rules exclude.
- **The mat is ~12 % darker than 001 at close range**, for the slice-quantization
  reason under change 11.
- **The far shell is only convincing far away.** It blends the three states with
  mottle noise instead of zoning them by wetness, so it does not match the mat's
  interlocking zones. At the default handover (44–58 m) that reads as a mild
  brightness step; forced inward with `rRegion 32` (tested) the field visibly
  breaks into blotches of shell against mat. The right fix is to zone the shell
  with `scatter_wetness(seed, xz)` from the WGSL twin — it would also replace
  three mottle-noise evaluations and three texture samples with one, so it should
  be *cheaper*. Not done here: it needs the seed plumbed into the shell uniform
  and is a bigger change than this pass allowed.
- **The CPU region rebuild is a 3.7 s hitch on the bog** (measured directly:
  `scatter.region()` over ±64 m for the five bog entries = 532k instances in
  3.7 s, against 148k in 0.33 s for `default`). It only fires on >6 m camera
  moves, but a carpet's 484 slots/cell make it 11× worse than a scattered stand,
  and it is now the single worst thing about this renderer on a mat. The fixes are
  incremental per-cell updates or moving enumeration to the GPU twin; neither is a
  visual change and neither belongs in this pass.
- Zone boundaries are tile-granular, so a single-tile island of another state has
  visibly square edges. That is inherent to a carpet, and the reference shows it
  too.

### Is a slice-stack volume suited to this geometry?

Yes — more so than to the upright grasses it was designed for, once the
mat-specific decisions are made. The reasoning: a *low, wide, periodic* subject
puts the whole 25 MB into a 0.18 m tile, which buys 1.52 mm voxels and therefore
real capitulum structure; and because the tiles tile the plane, the horizontal
shell stack never suffers the edge-on collapse that forces the axis flip on a
single plant. What it cannot do is silhouette: the visible surface is quantized to
the slice spacing, so it renders moss as an intricately textured, correctly lit,
terrain-following surface rather than as a field of distinct cushions. Against
001-billboard-smoke's single flat quad — "very good moss texture down to ~0.4 m,
but no thickness at all" — this has thickness in the shading and in the depth
buffer, and 4-5× the effective texel density at 1 m, but only ~4 quantization
steps of visible relief.

### Verification

`npx tsc --noEmit` clean for this experiment. Headless screenshots (1280×800,
`seed=42&det=1&t=3`), before and after, at: bog grazing / topdown / inside-plant
/ carpet-close / far-horizon / a 30° ridge flank (`cam=21.98,-0.15,-3.49,1.644,-0.35,60`,
found by searching the heightfield for its steepest point near the origin), plus
`debug=albedo|normals|lighting|coverage` on bog grazing, plus default grazing /
topdown / inside-plant. Zero console errors, zero toasts, in every run.

`default` regression check at full resolution, scene pixels only (HUD and panel
masked): **2 969 of 597 030 pixels differ by more than 2/255, mean |Δ| 0.12/255,
and a same-code control run differs in 0 pixels** — i.e. the renderer is
bit-deterministic and the residual is isolated alpha-test edge flips caused by the
two extra vertex interpolants (`carpet`, `tnormal`), the same effect
001-billboard-smoke documented for the same reason. Topdown 46 pixels, inside-plant
1 475. The two normal fixes (change 8) are deliberately **gated to carpet species**
for this reason: applied globally they also improve the grasses' far field, but
they are not visually neutral there — measured on `default` at grazing, the far
band went +24 % and the horizon band −31 %. That is a real bug affecting the
upright path too, and flipping it on is a one-argument change
(`buildMips3D(..., round)` plus the `decode_normal` dead zone); it wants the
owner's decision and a golden refresh, not a moss pass.

**No frame times are claimed.** Up to a dozen agents shared this GPU: across runs
of the identical scene, `default` grazing Σp50 ranged 11.6 → 73.1 ms and the
harness's own base pass (which this experiment does not touch) ranged
0.14 → 3.5 ms. Structurally the moss path got *cheaper* than before this pass —
tiles are 2.9× smaller in area, the dither no longer punches holes in the depth
buffer so early-z works again, and the per-vertex terrain fetch is 4 texel loads
per vertex against a fragment-bound pass — but that needs a quiet machine:
`#/bench/003-shell-slices?stand=bog&spline=orbit-low`.
