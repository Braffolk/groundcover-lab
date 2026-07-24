# 004-raycast-lut — ray-answer LUT impostors

## Idea

Push the "precomputed raycast is allowed" loophole to its limit: bake the
ANSWERS of ray queries against each species offline, so that at runtime a
couple of texture fetches answer "what does a ray hitting this plant see".

The 4D ray space (2D direction x 2D offset) is baked as a **24x24 octahedral
grid of full-sphere directions**, each holding a **64x64 slab of ray offsets**
across the plant's bounding sphere — 576 x 4096 = 2.36M precomputed rays per
species, stored in two rgba8 atlases (1536x1536):

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
| culled instance buffer, `ceil(pi * 96^2 * density * 1.15) * 32B` (96m = maxDist param cap) | 3.20 MB @ density 3 / 2.66 MB @ 2.5 |
| uniforms + share of indirect args | < 1 KB |
| **total (default stand)** | **21.0–22.1 MB** ✓ |

HUD confirms 21.0 / 20.5 / 21.0 MB for the three species on `default`.
Worst case in the repo is `dense-mixed` (density 5): 5.32 MB instances →
24.2 MB, still under 25. Bake-time resources (source mesh upload, up to
~180MB for poa) are transient and destroyed after the bake; they briefly
show in the meter during first load only.

## Bake

- `bake.ts` — in-browser GPU bake via the harness flow (`bakedArtifact` /
  `commitBake`), artifacts committed to `mesh/baked/004-raycast-lut/
  <species>-v1-24x64.bin` (18.9 MB each, header + 2 raw rgba8 atlases).
  All three species baked and committed; first bake ~5–20s per species
  (576 passes over the full source mesh), then OPFS/committed-file cached.
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

Method-specific extra: the `showCells` param flat-tints every fragment by the
baked direction cell it answered from, which makes the 24x24 direction
quantization (and its popping under camera motion) directly visible. It
replaces the old local `debugView` enum, which duplicated the global modes.

## Status

**working** — verified via headless screenshots on `default` stand:
`grazing`, `topdown`, `inside-plant` (near-fade dissolve works),
`far-horizon`. All three species render at the stand's exact placement
(GPU twin of the shared scatter). No console errors, typecheck clean.

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
- Species coverage: all three (calamagrostis + elymus community tiles baked
  as clump-impostors anchored at tile centers; poa specimen anchored at its
  bounds center).

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
