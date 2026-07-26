# 004-raycast-lut — ray-answer LUT impostors

## Idea

Push the "precomputed raycast is allowed" loophole to its limit: bake the
ANSWERS of ray queries against each species offline, so that at runtime a
couple of texture fetches answer "what does a ray hitting this plant see".

The 4D ray space (2D direction x 2D offset) is baked as a **24x24 octahedral
grid of full-sphere directions**, each holding a **64x64 slab of ray offsets**
fitted to the plant's baked bounding BOX as seen from that direction (v2; v1
used the bounding sphere and wasted most of a flat carpet tile's slab) —
576 x 4096 = 2.36M precomputed rays per species, stored in two rgba8 atlases
(1536x1536):

- `surf` = premultiplied albedo rgb + coverage
- `geom` = premultiplied depth01 (hit distance along the ray) + octahedral
  normal + height01 (normalized hit height, reused as baked AO)

Baking is plain orthographic rasterization of the raw GCMESH1 mesh (one pass
per direction, 4x4 supersampled, depth test = "first hit of the ray bundle"),
i.e. 2.36M raycasts against 2.2M–6.5M-triangle geometry resolved offline by
the rasterizer.

Runtime (all in one fragment shader on a per-plant camera-facing quad):

1. Build the pixel's true eye ray; transform it into the plant's local frame
   (translate, un-yaw, un-scale) **including the inverse wind shear** — sway
   is modeled as a linear-in-y shear, and lines stay lines under shear, so
   bending the ray instead of the plant is exact, not an approximation.
2. Pick the nearest baked direction cell (1 oct-encode), seed the slab offset
   at the ray's closest approach to the sphere center.
3. Fetch depth there, reconstruct the baked 3D hit, reproject it onto the
   true eye ray, and fetch again — one parallax-correction step that kills
   most of the 9.5-degree direction quantization error (skipped beyond 40m
   where it is subpixel).
4. The final answer gives albedo/coverage/normal/height and a **true world
   hit point** -> `frag_depth`, so plants inter-occlude correctly (no
   billboard cardboard sorting), lighting uses the shared `light_surface`
   with the baked mesh normals (two-sided foliage flip), coverage is
   dithered (hash of pixel coords, deterministic).

Plant-count independence: a GPU cull pass walks only the scatter cells within
`maxDist` of the camera (fixed-size dispatch, `scatter_candidate` WGSL twin,
stand-region clamped exactly like `Scatter.region()`), appends survivors via
atomics and renders with `drawIndirect`. 100 or 1e9 plants cost the same;
`scaling-100m` differs from `default` only in the stand radius uniform.
The same pass rebuilds each survivor's quad exactly as the vertex shader will
(same sphere centre, wind shear, perspective growth, fades) and drops it if
its bounding sphere is outside the side frustum planes or if its fade factor
is zero — both cases would have produced no pixels at all.

Close range: LUT texels are ~2–2.4cm, so <3m plants magnify into mush; a
hash keyed to the (stable, unsheared) local hit cell modulates coverage +
albedo to restore leafy high-frequency structure. Camera-inside-plant
dissolves via the same dither (near fade), per the rules.

## VRAM budget math (per species)

| item | bytes |
|---|---|
| `surf` atlas 1536x1536 rgba8 | 9.44 MB |
| `geom` atlas 1536x1536 rgba8 | 9.44 MB |
| culled instances, scattered entry: `ceil(pi * 96^2 * density * 1.15) * 32B` (96m = maxDist param cap) | 3.20 MB @ density 3 / 2.66 MB @ 2.5 |
| culled instances, CARPET entry: `ceil(pi * 96^2 * (div^2/16) * 0.85) * 8B` | 5.96 MB (bog moss, div 22) |
| uniforms + share of indirect args | < 1 KB |
| **total (default stand)** | **21.0–22.1 MB** ✓ |
| **total (bog moss)** | **23.7 MB** ✓ |

A carpet tile's instance record is **8 bytes**, not 32: it stores the grid node
(cell.xz + slot + the 90-degree yaw index) and the vertex shader rebuilds
position/scale/yaw with the scatter's own arithmetic, since a carpet's scale is
the entry constant and its height is `terrain_sample()` (already needed for the
conforming shear). Life size means ~745k tiles inside the 96m cap for ONE moss
state, which at 32B/tile would have been 23.8 MB of instances alone; at 8B it
is 5.96 MB and the species fits in 23.7 MB. The 0.85 factor is the assumed
worst-case share of the grid nodes one wetness zone can claim inside the disc:
measured over the bog at seed 42 the three states split 16/50/34% globally and
the worst 80m disc anywhere in the stand reaches 63%, so 0.85 covers other
seeds and terrains with the whole disc counted (no reliance on the frustum).

HUD confirms 21.0 / 20.5 / 21.0 MB for the three species on `default`.
Worst case in the repo is `dense-mixed` (density 5): 5.32 MB instances →
24.2 MB, still under 25. Bake-time resources (source mesh upload, up to
~180MB for poa) are transient and destroyed after the bake; they briefly
show in the meter during first load only.

## Bake

- `bake.ts` — in-browser GPU bake via the harness flow (`bakedArtifact` /
  `commitBake`), artifacts committed to `mesh/baked/004-raycast-lut/
  <species>-v2-24x64.bin` (18.9 MB each, header + 2 raw rgba8 atlases).
  All SIX species (3 grasses + 3 Sphagnum states) baked and committed; the v1
  files are deleted. Bake cost: ~5-20s for a grass, ~1-2 min for a 19.8M-tri
  moss tile, then OPFS/committed-file cached.
- **v2 fixed two bake bugs that had been in the artifact from the start** (both
  found while chasing moss artifacts; both affected every species):
  1. *the instance anchor was subtracted twice* — once into `bounds_min/max` on
     the CPU, again in `bake-view.wgsl`. Every species therefore sat off-centre
     in its slab by the anchor, and the ortho frustum clipped whatever fell
     outside. Measured on the committed v1 atlas: the moss tile's nadir view
     held **half** of its diamond (24% of the slab covered instead of 49%), and
     calamagrostis' side views were cut off down one edge. On the mat this read
     as tile-sized holes that looked exactly like a placement bug.
  2. *the slab covered the bounding SPHERE*. Each view now fits the box's
     support along its own slab axes (`rf_extent`), so a 7cm-tall tile seen
     edge-on spends all 64 rows on 9.3cm of cushion (1.5mm/texel) instead of
     30% of them on 33cm of mostly empty air, and the 8-bit depth axis is fitted
     the same way. Header gained the box half-extents at offset 36.
- Gotcha found: `bakedArtifact()` treats the Vite SPA-fallback (200 +
  index.html for a missing committed file) as a valid artifact and caches
  it into OPFS. My loader detects the bad payload (magic/size check),
  rebakes, repairs the OPFS entry (same layout as `src/bake/cache.ts`) and
  commits. **Harness wish**: `src/bake/io.ts` should reject non-binary
  content types or validate against a caller-provided sniffer.

## Debug views

Wired to the global `frame.debug_mode` selector (`view` dropdown / `debug=`):
`albedo` = the un-premultiplied LUT albedo, `normals` = the baked mesh normal
decoded from the geom atlas (yaw-rotated, two-sided flip), `lighting` =
`light_surface` divided back out (includes the height AO term), `coverage` =
the effective coverage the fragment resolved to (`cov_sharp * fade`, i.e. the
value the dither is compared against), `depth` = the reconstructed world hit,
which is the interesting one here: the ramp shows per-plant depth, not quad
depth. Fog is applied only when `debug_mode() == DEBUG_OFF`.

Cross-check against the independent baseline: on `bog` at `cam=grazing`,
`debug=albedo` sampled over the same 500x200 patch of mat gives mean rgb
(0.320, 0.278, 0.066) for this renderer and (0.319, 0.272, 0.067) for
001-billboard-smoke's card bake — two completely different bakes of the same
mesh agreeing to ~1%, which is the strongest evidence available that the LUT's
un-premultiply is right. `debug=lighting` over the same patch means 0.96 with
64% of pixels clipped, against 1.00 / 99% for 001: the sunlit mat exceeds 1.0 in
the light term (sun + ambient) and clips in an sRGB-displayed 8-bit target, but
it is *less* saturated than the baseline, not more.

For a carpet fragment, `coverage` is the raw LUT coverage the hard alpha test
saw (not a dither threshold, which carpets do not use).

Method-specific extra: the `showCells` param flat-tints every fragment by the
baked direction cell it answered from, which makes the 24x24 direction
quantization (and its popping under camera motion) directly visible. It
replaces the old local `debugView` enum, which duplicated the global modes.

## Status

**working** — verified via headless screenshots on `default`
(`grazing`, `topdown`, `inside-plant`, `far-horizon`) and on `bog`
(`grazing`, `topdown`, `inside-plant`, `carpet-close`, a 28-degree ridge flank,
plus `albedo`/`normals`/`lighting`/`coverage`). All stand entries render at the
stand's exact placement (GPU twin of the shared scatter). No console errors, no
validation toasts, typecheck clean.

## Findings

- Baked ray answers + one reprojection step genuinely hold up from all
  angles including top-down and near-grazing; the reconstructed
  `frag_depth` gives correct plant-vs-plant intersection, which reads far
  better than sorted billboards at this density.
- The wind-as-inverse-ray-shear trick is exact for linear shear and costs
  ~6 ALU per fragment; species sway comes from the stand table, phase from
  the scatter hash. (Ground truth bends quadratically with height; linear
  tilt is a documented visual difference.)
- Honest artifacts: (1) <3m plants are soft/blobby — 2.4cm texels; the
  local-hash clutter hides much of it but flower heads can still pancake;
  (2) coverage dithering is white-noise (no blue-noise/TAA in harness), so
  rims and fluffy heads shimmer; (3) no atlas mips (view cells would bleed)
  → some far-field aliasing in motion; (4) beyond `maxDist` (default 80m,
  cap 96m) groundcover fades to bare terrain — a far-field carpet layer is
  the natural next step; (5) 8-bit depth quantizes hits to ~5mm along the
  ray — fine in practice.
- Perf shape: cull pass ~0.15–0.2ms; impostor pass ~4.5ms top-down and
  ~13–20ms at worst-case grazing at 1280x800 **measured while 15 sibling
  agents hammered the same GPU — numbers are contended and not claimable**.
  `frag_depth` disables early-z, so grazing cost is overdraw-bound; the
  obvious lever (not yet built) is a front-to-back pre-pass or splitting
  far plants into a quad-depth pipeline that keeps early-z. No bench run
  for this reason; run `#/bench/004-raycast-lut?stand=default&spline=orbit-low`
  on an idle machine before quoting numbers.
- Species coverage: all six (calamagrostis + elymus community tiles baked as
  clump-impostors anchored at tile centers; poa specimen anchored at its bounds
  center; three Sphagnum states as carpet tiles anchored at the tile center).
- Perf after the moss pass, `default` stand, 1280x800, **still contended by
  sibling agents so not claimable** — the same setup measured Σp50 14.0 / 16.7 /
  17.4 / 29.0 ms across four runs of identical code, which is the size of the
  noise. Before this pass the same camera measured Σp50 31.4 ms. The two changes
  that matter pull in opposite directions: the fixed bake makes plants complete
  instead of ~half-clipped (more shaded fragments), and the box-support quad
  removes ~40% of the empty margin around each grass clump and ~80% around a
  grazing carpet tile. Bog `grazing` lands at Σp50 9.9-22.6 ms across runs with
  the whole 1.1M-tile mat drawn.

## Harness notes (what the interface made easy / hard)

Used: `stand_table[i].carpet_div` (selects the whole carpet path — proxy shape,
fade, alpha test, record layout), `standEntrySlots(entry)` (dispatch sizing and
the capacity formula), `stand_table[i].slope_align` (the shear amount, applied
to the bog's calamagrostis too), `terrain_sample()` (height + gradient in one
bilinear fetch, per vertex for the tile frame and per fragment for the rung-3
displacement), `SCATTER_CELL_SIZE` + `QUARTER_TURN` from the scatter twin (to
rebuild a carpet node from 8 bytes bit-identically).

Gaps found, in rough order of how much time they cost:

1. **`footprint_m` exists, but a carpet renderer needs the tile's real EXTENTS,
   not its period.** Sphagnum's period is 0.18 m while its geometry spans
   0.21 x 0.234 x 0.093 m; a proxy sized from `footprint_m` clips the overflow
   that makes the mat overlap. This experiment gets the true box from its own
   bake, which is fine — but a `MeshInfo` that exposed the source
   `boundsMin/boundsMax` (it already parses the manifest that contains them)
   would let a renderer size proxies correctly *without* a bake round trip.
   Right now `ctx.meshes.info()` gives `tileSize` only, and the alternative is
   fetching a 500 MB binary at runtime.
2. **Nothing in the harness states the GCMESH1 octahedral convention.** The spec
   in `mesh/README.md` says "octahedral normal U/V" and stops there, and the one
   decoder in shared code (`GcMesh.normalAt`, commented "validated visually")
   uses the (x,z)+y form while the data is (x,y)+z — see the chequerboard section
   below for the measurement. Every experiment that decodes normals itself has to
   rediscover this, and the failure mode is subtle: plausible-looking output with
   systematically rotated lighting. A one-line statement of the convention in
   `mesh/README.md` plus a WGSL `gcmesh_normal_decode()` in `src/wgsl/` would
   have saved the whole hunt. (Worth checking `GcMesh.normalAt` and the mesh
   inspector against face normals — I believe they are wrong.)
3. **A `carpet-close`-style bookmark per stand would help.** `carpet-close` is
   at the origin, which on the bog sits in one particular wetness zone; judging
   the three states needs hand-written absolute poses, and those are the exact
   trap the bookmark exists to avoid.
4. `bakedArtifact()` no longer needs the SPA-fallback shim (it content-type
   checks now) — my loader still validates magic/size, which is cheap and worth
   keeping for a format with a version.

## Moss carpet (bog stand)

*Sphagnum palustre* is a 0.18 m periodic tile, 0.07-0.09 m tall, laid out by the
`bog` stand as a grid-snapped mat: 22x22 tiles per 4 m scatter cell (484 slots,
life size, scale 1.01), 90-degree-only yaw, three states partitioning the wetness
axis. What it looked like before this pass: **1.1 m stripes of dark confetti with
3 m of bare peat between them**, no slope conformance, 26.1/25 MB.

What changed, smallest first — all inside this experiment:

1. **Every slot of a cell is now visited.** The cull dispatch was one workgroup
   of 128 threads per cell with `slot = local_invocation_index`, i.e. hardcoded
   to `SCATTER_MAX_PER_CELL`. A carpet has `carpet_div^2` = 484 slots, and
   `slot = i / 22` meant only the first **6 of 22 tile rows** per cell existed —
   the stripes. Now `dispatchWorkgroups(cellsX, cellsZ, ceil(slots/128))` from
   `standEntrySlots(entry)` and `slot = wg.z * 128 + lane`. Capacity is kept as
   a separate number (expected survivors), which is what the "do not conflate
   slots with instances" warning is about.
2. **No near fade for a carpet.** `near_fade` dissolved any plant whose distance
   was under 0.55 of its own bounding radius, which for a mat is the tile you
   are standing on. Forced to 1.0 when `carpet_div > 0`, in both the cull and
   the vertex shader.
3. **Hard alpha test instead of dither for a carpet.** The grass path resolves
   coverage stochastically; a mat has to be a closed depth-writing occluder, so
   a carpet fragment is tested hard against `matCov` (0.3) and the distance fade
   *erodes the reference* instead of dissolving texels. `debug=coverage` on the
   bog is solid white with sparse speckles — no screen door on the ground.
4. **No invented clutter for a carpet.** The grass path multiplies coverage by a
   plant-local hash to fake leaf detail below ~8 m (LUT texels there are ~2.4 cm).
   Carpet texels are ~2-5 mm, so the hash was pure damage: it notched holes in
   a surface that is supposed to be closed. Skipped for carpets.
5. **8-byte carpet instance records** (see VRAM math) — 26.1 -> 23.7 MB.
6. **Terrain conforming as a SHEAR of the ray, ladder rung 3.** See below.
7. **The correct octahedral convention for baked mesh normals.** See below.
8. **Quad sized by the box's support function** instead of the bounding sphere:
   for a tile seen at a grazing angle that is ~5x fewer fragments (a 0.33 m disc
   vs a 0.23x0.09 m rectangle), and ~40% fewer for an upright grass clump. The
   quad *plane* is unchanged, so every pixel still reconstructs exactly the same
   eye ray — the image does not move, only the empty margin goes away. This is
   what paid for the fixed bake covering more real pixels.

### Terrain fitting: rung 3, as a shear of the ray

`slope_align` is 1 for the carpet and 0.3 for the bog's calamagrostis. Both are
applied as a **vertical shear of plant space** — `y += dot(grad, xz)` with
`grad = slope_align * dh/dxz` at the plant's node — never as a rotation:

- a shear leaves the tile's xz footprint *exactly* on the scatter's grid step,
  while a rotation shrinks it by cos(tilt) and opens gaps in a closed mat;
- lines stay lines under a shear, so the existing wind-shear trick applies
  unchanged: the terrain shear is inverted analytically on the eye ray (two
  extra dot products), which is **exact**, not an approximation, for the
  linearized ground. Ray methods get rung 4 almost for free this way.

The hit is then displaced by the **true** terrain height under its own xz
(one `terrain_sample`, inside 30 m), not by the tile's linearized plane — that
is rung 3, per fragment rather than per vertex. Why not rung 1 or 2: measured on
this terrain, a rigid per-tile tangent plane leaves neighbouring tiles up to
**41 mm apart at their shared edge** (mean 1.5 mm), and the cushion is only
70-91 mm tall, so the worst case is a half-cushion step at every ridge crest.
Rung 3 removes it by construction, because the displacement depends only on
world xz and neighbours therefore agree exactly. The residual against the
linearized plane (mean 1.3 mm, max 34 mm) is clamped to 35 mm so a pathological
terrain cannot push the surface out of the quad, and the quad is padded by the
same amount. Beyond 30 m the linearized plane is used instead: the difference
there is sub-pixel.

### The bug that made moss look like a chequerboard

With gaps and slope fixed, the mat came out as a hard **chequerboard of bright
and dark tiles** at every distance. `debug=albedo` was smooth, `debug=normals`
showed whole tiles reading +X or +Z, and forcing a single yaw made the pattern
vanish — so it was the normals, and the 90-degree rotation set was only the
amplifier.

Root cause: **GCMESH1 stores the standard octahedral encoding, `(u,v) = (n.x,
n.y)` with `n.z` derived, and this shader decoded it as `(n.x, n.z)` with `n.y`
derived** — a y<->z swap on every baked mesh normal. Verified two ways: against
face normals computed from raw vertex positions (mean `|cos|` to the geometric
normal 0.75 for the (x,y)+z convention vs 0.57 for (x,z)+y, 120k sphagnum vertex
samples), and by decoding the committed atlas — a straight-down view of the moss
returned a mean normal of **(0, 0.05, 0.97)**, i.e. horizontal, where a cushion
seen from above must be up. The swap is an isometry, so it commutes with the
bake's normal averaging: `rf_mesh_normal()` swaps back at the point of use and
needs no rebake. `debug=normals` at `carpet-close` is now green (up) with
per-capitulum variation, and the chequerboard is gone from every view.
Note this also affected all three grasses, so `default` lighting is fixed too.

The swap was the whole of it, and one intermediate claim here was wrong: a first
pass concluded from winding that the decode also had to be **negated**, and this
file said so. `tools/probe-oct-normal.ts` settled it the other way — the meshes
are open shells and are not even wound alike, so winding proves nothing, while a
4 mm ray fired along ±n from sampled vertices finds material on the −n side on
all three species. So **+n is outward and there is no negation**, and
`rf_mesh_normal()` is a plain y↔z swap with no sign flip. The swap is an
isometry, so it commutes with the bake's normal averaging and needs no rebake —
which is why 004's committed atlas survived the repo-wide wipe.

### Normal-field LOD (the one deliberate cheat)

A carpet texel is 2-5 mm, so past a few metres it is subpixel and the single
normal the LUT returns is an aliased sample of a much richer distribution —
noise in a still, shimmer in motion, and there is no mip chain to fall back on
(view cells would bleed across atlas tiles). For carpets only, the normal is
therefore filtered toward the mat's own surface normal over 6-25 m
(`smoothstep`), which is the correct filtered answer for a sub-texel footprint.
Only the NORMAL converges: albedo, coverage, depth and the height-AO keep their
per-texel detail at all distances. Tuning was done by eye against the
alternatives: no LOD at all is noticeably noisier in the mid field, and an
incidence-corrected metric (distance / cos of incidence, which is the
theoretically right footprint) flattened the whole grazing view and read as
painted ground, so plain distance won.

### What improved / what is still bad

Before/after screenshots read at `cam=grazing`, `topdown`, `inside-plant`,
`carpet-close`, a 28-degree ridge flank at eye level, an eye-level 20-degree
view, and `albedo`/`normals`/`lighting`/`coverage` (bog), plus `grazing`,
`topdown`, `inside-plant` on `default`:

- **Closed mat.** The stripes and the tile-sized holes are gone at every camera;
  `carpet-close` (1 m straight down) shows continuous cushion with fine
  mottled green/ochre capitulum texture and no visible tile boundaries.
- **Real 3D, near.** Inside ~6 m the per-fragment normals give genuine cushion
  relief, and because the LUT reconstructs a true world hit the mat writes real
  depth: overlapping tiles interleave instead of z-fighting, and the grass stems
  emerge *through* the mat correctly.
- **Slope.** The ridge-flank view shows the mat following a 28-degree slope with
  no stair-stepping, no buried or floating edges and no cracks at tile borders.
- **Zones read as ecology** — wet green hollows, ochre flanks, bronze crests.
- **Still bad #1: at a grazing angle the mat is smooth, not intricate.** Past
  ~10 m the normal LOD has converged and the mat reads as textured ground rather
  than cushions. Partly honest (a 7 cm mat at 30 m projects to ~1 mm of
  thickness) and partly the LOD's cost.
- **Still bad #2: nadir texel size.** 4.9 mm at the tile's own scale, so below
  ~0.7 m the cushion starts to smear. The fix is not more slab resolution at
  this budget but a *hemispherical* direction set for carpets (rays never hit a
  mat from below), which would free half the atlas and pay for a 96- or
  128-texel slab at the same 18.9 MB. Deliberately not done here: it needs a
  second layout in the bake and the runtime, and a third rebake of three
  2-minute meshes.
- **Still bad #3: faint tile-scale tonal blocks** survive in the LOD transition
  band (~6-25 m) — the per-tile mean normal still differs slightly between the
  four yaws. Much weaker than the original chequerboard, visible if you look for
  it on a uniform slope.
- **No overscale.** Left at 1.0: the mat is closed without it, and with real
  thickness an overlap would be paid for in overdraw (the pilot's dark-lattice
  problem does not arise here because the depth is real, but there is nothing to
  gain either).

### Is a ray-answer LUT suited to a low, wide, intricate mat?

Yes — better than to the tall grasses it was designed for, with one caveat.
What a mat needs is exactly what this representation gives: a real per-pixel
depth (so the cushion occludes and interleaves properly and the silhouette is
not a card), per-pixel mesh normals, and a ground-conforming query that costs
two dot products because the deformation is a shear. Nothing had to be faked;
the flat-carpet case even *helps* the format, because a box-fitted slab spends
its texels on 9 cm of cushion instead of on a 33 cm sphere. The caveat is
resolution allocation: a 4D field at 25 MB buys ~2.4M rays, and spending half of
them on directions that can only see a mat from underneath is waste that a
carpet-specific layout should reclaim. Under ~0.7 m the mat smears; from 1 m to
the fade band it is a convincing, closed, terrain-following Sphagnum carpet.

## Audit

Structural review pass (code-only reasoning; frame times were contended and
deliberately ignored). Found and fixed:

1. **No frustum culling anywhere.** The cull pass materialized the *whole*
   disc of radius `maxDist` around the camera and drew every plant in it —
   at a ~86-degree horizontal FOV that is roughly 4x more quads than can
   possibly be on screen, all paying vertex shading (cos/sin + `wind_sway`),
   primitive setup and an instance-buffer slot. The cull shader now rebuilds
   each survivor's quad *exactly* as `vs_main` does (same `wc`, same wind
   shear, same `rw`, same `grow`) and rejects it when its bounding sphere
   (`rw * grow * sqrt(2)`, the quad's circumscribed radius) is behind any of
   the four side frustum planes. Planes come from the shared `view_proj` via
   Gribb–Hartmann, normalized on the CPU into the cull UBO that was being
   rewritten every frame anyway.
2. **Plants that render nothing were still drawn.** `near_fade * dist_fade
   == 0` (camera inside the plant, or past the fade band in true 3D
   distance — the old gate was xz-only) makes *every* fragment discard. Those
   are now rejected in the cull pass, which also kills the worst case where a
   camera-inside plant grows its quad up to 5x and covers the screen with
   fragments that all discard.
3. **Per-frame upload of constants.** The 48-byte draw UBO per stand entry
   held only bake constants (sphere centre/radius/topH) plus params, and was
   rewritten for every entry every frame. Now written once and on
   `onParamsChanged` only.
4. **Dispatch sized past the stand.** The cull dispatch grid was
   `camera ± maxDist` in cells, unclamped, while the shader immediately
   returns for cells outside the stand region. The grid is now clamped to the
   stand's cell range on the CPU — no behaviour change, but on
   `close-quality` (±24m) it is a 41x41 → 13x13 workgroup grid.
5. **Dead per-fragment work.** The plant-local "clutter" hash (`hash3` +
   a hit reconstruction) was computed for every fragment and then multiplied
   by `near_w`, which is exactly 0 beyond 8.33m — i.e. for the overwhelming
   majority of fragments in any grazing shot. It is now inside
   `if (near_w > 0.0)`. The same block also reconstructed the baked hit point
   a second time with an expression identical to the one used for
   `frag_depth`; that is now computed once and shared.
6. **Dead file**: `shaders/main.wgsl` was the untouched `_template` billboard
   shader, imported by nothing. Deleted.

Verification: the frustum + fade rejection was temporarily disabled and the
same deterministic frames (`det=1&t=8`) re-shot; masking the HUD, the images
differ in 2–45 scattered pixels out of 1M (max delta on isolated pixels), with
no clustering at the screen edges — that residue is depth-tie ordering, since
culling changes the atomic append order of instances. `topdown` and
`far-horizon` (the angles where a too-tight bound would show as a black band
at the frame edge) are clean.

Deliberately left alone:

- **`frag_depth` kills early-z**, so grazing is overdraw-bound. A depth
  prepass or a far-field pipeline that keeps early-z is the real lever, but
  that is a redesign and needs measurement — out of scope for an audit.
- **Instance-buffer capacity** is still sized for the full `maxDist` disc
  (3.2 MB at density 3) even though frustum culling now leaves ~1/4 of that
  live. Shrinking it would save ~2 MB/species but needs a defensible bound
  over all FOVs and camera heights; not worth the overflow risk here.
- **Per-fragment `cam_d`** (used for the 40m parallax cutoff and the clutter
  weight) is genuinely per-fragment, not uniform across the quad — moving it
  to a flat vertex output would change the image.
- **Cull enumerates the square of cells** around the camera and distance-tests
  per candidate; the ~21% of the square outside the circle is inherent to a
  cell grid and cheap to reject.
- **No atlas mips** — documented above; view cells would bleed.
