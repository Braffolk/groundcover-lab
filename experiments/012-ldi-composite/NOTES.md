# 012 LDI composite

## Idea

Layered depth images as a groundcover representation. Per species, the source
mesh is baked ONCE into 5 key capture directions (4 side views at azimuth
0/90/180/270 deg, elevation 20 deg, plus one straight-down top view), each
depth-peeled into 4 layers. A layer stores albedo+coverage and oct-normal +
16-bit linear capture depth. At runtime a compute pass evaluates the scatter
WGSL twin over a bounded camera-centred cell window (regionRadius), frustum
culls, and compacts survivors into one instance array consumed by two indirect
draws:

- NEAR (< layerCullDist): 12 cards per plant — the 2 side stacks nearest the
  camera azimuth + the top stack, x4 layers. Cards are FIXED to the capture
  basis at each layer's mean baked depth; fragments write true per-texel baked
  depth via frag_depth, so the union of stacks composites through the plain
  depth test. No sorting, no blending, no cross-fade weights: an off-axis
  stack foreshortens toward invisibility, which IS the view weighting (side
  cards vanish from above exactly as top cards turn face-on). A one-tap
  parallax reprojection (sample own depth once, shift UV, resample — a
  precomputed lookup, not marching) tightens texels within 30 m.
- FAR: 2 cards (front peel layer of nearest side + top stack), rasterizer
  depth only — no frag_depth, so early-z eats the overdraw where the field is
  deepest. Near draws first to prime the depth buffer. The cull pass compacts
  far survivors downward from the top of the shared instance array while near
  fills upward from 0; the descending mapping is applied in `vs_far`, so both
  indirect draws keep `firstInstance = 0` (see Audit).

Per-frame cost is bounded by regionRadius (cells around the camera) and screen
coverage — the stand's total plant count never enters any loop: 557k (default)
and 134.2M (scaling-100m) issue byte-identical work (verified, see Findings).
No source geometry is touched after the bake. Wind = shared `wind_sway` on
card corners, roots pinned, per-species stand sway. Lighting = baked normals
yaw-rotated, two-sided flip, shared `light_surface` + fog. Camera inside a
plant: coverage-threshold fade over 0.4–0.95 of the plant's bounding radius.

### Where LDIs break for thin foliage (and the fixes that landed)

1. **Naive depth peeling wastes layers.** The first 3–4 surfaces under a pixel
   of grass are often the same tuft 2 mm apart; the crown interior never gets
   captured and other views show a hollow shell. Fix: peeling with a MINIMUM
   SEPARATION (~3 cm, `0.04 * boundR`): layer L keeps the nearest surface at
   least that far behind layer L-1, forcing successive layers through the
   crown.
2. **Bilinear taps at silhouettes blend into nothing.** Empty texels hold
   black albedo and FAR depth, so edge samples smear dark fringes and scatter
   "beads" of geometry behind the plant (dramatic close up). Fix: bake-time
   dilation — flood albedo/normal/depth (frontmost neighbour wins) 6 texels
   into empty space, alpha stays 0. This one change took close-range quality
   from broken to usable.
3. **Layer gaps off-axis.** Between two capture azimuths both stacks are ~45
   deg foreshortened and inter-layer space opens up. Fix: draw the stack UNION
   with per-texel depth (they interleave correctly in the depth buffer), plus
   the one-tap parallax reprojection near the camera.

## VRAM budget math

Per species (HUD-confirmed 17.1 MB for density-3 entries, 15.9 MB for 2.5):

- atlas0 (albedo+coverage) 1280x1024 rgba8: 5.24 MB
- atlas1 (oct normal + depth16) 1280x1024 rgba8: 5.24 MB
- instance array, worst-case regionRadius 128: 65^2 cells x 128 slots x
  density/8 x 1.15 margin x 32 B = 6.9 MB at density 3
- 2 indirect + uniform: < 1 KB

Total 17.4 MB at density 3 — under the 25 MB budget (would exceed only at
density 8 + radius 128, which no stand uses). Bake-time transients (mesh
buffers up to 214 MB for poa, strips, readbacks) live outside `ctx.res` and
are destroyed after the bake — they trip the expected "untracked" dev warning
once per fresh bake.

## Bake

`bake.ts`: 4 render passes (one per peel layer), each drawing the full GCMESH1
mesh 5x (one viewport per capture direction) with the previous layer's depth
strip as peel input; min-separation discard in the fragment shader; ortho
framing from bbox projection per direction. Readback, CPU dilation (see
above), per-(dir,layer) mean-depth stats into the header, packed as
`<species>-ldi-v2.bin` (10.5 MB) and committed via `commitBake` — in
`mesh/baked/012-ldi-composite/`. Fresh bake ~2–4 s per species in-browser;
`bakedArtifact()` was bypassed because the dev server 200-serves index.html
for missing files (validated magic instead).

## Status

working — verified by headless screenshots at grazing, topdown, inside-plant,
far-horizon (scaling-100m), plus A/B vs 000-ground-truth (reference patch).
Audited since (see Audit): debug views wired, dead far-draw path repaired,
per-frame uniform traffic and a retained host-memory copy of the atlases
removed. Re-verified by screenshot at grazing/topdown/inside-plant/far-horizon
in every debug mode, console clean.

## Findings

- **All-angle behaviour**: grazing and topdown both read as a dense meadow
  with per-species color character; the same bake serves both because the top
  stack takes over exactly as the side stacks foreshorten out. Inside-plant is
  legible grass (blades, not billboards) with the documented fade.
- **Plant-count independence**: default (557k) vs scaling-100m (134.2M) show
  the same ldi-layers cost envelope — identical dispatch + indirect work by
  construction. Screenshots confirm no per-plant setup exists anywhere.
- **Timings are contended** (15 sibling agents on this GPU while developing)
  — ldi-layers read 30–45 ms p50 with the base pass itself inflated 10x, so I
  do NOT quote absolute numbers; no bench JSON is claimed. Structure-level
  perf work done regardless: near/far split (frag_depth only near, early-z
  far), foreshortening cull (edge-on cards degenerate beyond 12 m), parallax
  gated to 30 m, compute-side compaction so vertex work only runs on real
  plants. Re-bench solo before quoting.
- **Known artifacts, honestly**: close-up (<1.5 m) calamagrostis heads go
  "melted candy" — 6 mm texels + dilation smear; region-edge plants erode
  per-card rather than dissolving smoothly (binary alpha + threshold fade);
  the field ends at regionRadius (96 m default) with fog only partly covering
  the cutoff on long sightlines; frag_depth in the near pass disables early-z
  there (the price of true layer interleaving).
- **Not stochastic**: hard alpha test everywhere, opaque depth writes; fades
  are coverage-threshold based, no dither.
- A/B: `#/ab/000-ground-truth/012-ldi-composite?stand=default&cam=grazing&seed=42`
  (A is a reference patch and ignores the stand — compare plant character at
  the patch, not field density).

## Debug views

Both fragment stages route through `debug_shade()` (`src/wgsl/debug.wgsl`) and
apply fog only when `debug_mode() == DEBUG_OFF`, so `view=` /
`debug=albedo|normals|lighting|coverage|depth` work here:

- **normals** — the baked oct normal decoded from atlas1.rg, yaw-rotated into
  world space and flipped toward the viewer (thin foliage is two-sided). Real
  per-texel foliage normals, not a card normal: grazing reads as dense
  per-blade noise, topdown as mostly +y green with blade speckle, which is
  what a 6 mm-texel capture of a 2M-tri tuft should look like.
- **lighting** — `shaded / albedo`, i.e. the shared `light_surface()` term
  alone. It saturates to white over most of the frame, but so does the
  harness terrain at this sun/ambient (checked side by side against
  `000-ground-truth`'s terrain in the same view), so this is the shared
  model's range, not a skipped or doubled lighting term; the dark speckle is
  foliage whose normal faces away from the sun.
- **coverage** — `baked alpha * fade`, exactly the value the alpha test uses:
  binary white inside foliage with a grey bilinear rim at silhouettes and the
  smooth ramps of the inside-plant / region-edge fades. No dither anywhere.
- **depth** — reconstructed from `world_texel`, i.e. the parallax-corrected
  per-texel position that also drives `frag_depth`, so this view is a direct
  read of what the depth buffer receives.

Method-specific state is a param of THIS experiment (`inspect`), applied only
when the global view is off: `dir` colours by which of the 5 baked capture
directions won the fragment (top capture = purple, essentially absent at
grazing — the foreshortening weighting working), `layer` by peel layer, and
`path` by draw path (orange = near 12-card frag_depth, blue = far 2-card
early-z). `path` is the fastest way to see what `layerCullDist` is buying.

## Audit

Structural/debug pass over the first-generation code. Found:

1. **The FAR draw never executed at all.** `far_args` was seeded with
   `firstInstance = capacity` and the cull pass walked it down with
   `atomicSub`. WebGPU only allows a non-zero `firstInstance` in an indirect
   draw with the optional `indirect-first-instance` feature (the harness does
   not request it), so the driver silently no-ops the whole draw. Everything
   past `layerCullDist` was therefore simply missing: with
   `p.layerCullDist=10` the field ended 10 m from the camera and the rest was
   bare terrain. **Fixed** by keeping the descending compaction but expressing
   it on the shader side (`slot = capacity - 1 - instance_index` in `vs_far`,
   `slot = capacity - 1 - atomicAdd(far_count)` in the cull), leaving
   `firstInstance = 0`. This deletes the `farTemplate` buffer and its
   per-frame 16-byte `copyBufferToBuffer` as a side effect (the far args are
   now init-written once and only the instance count is cleared per encode,
   exactly like the near args). **This one legitimately changes the image** —
   the 70–96 m band of the field now renders instead of being empty.
2. **The whole 400-byte uniform block was rewritten every frame, per entry**,
   including 80 floats of bake-time capture bases and the stand/atlas
   constants, and it rebuilt them through `[...d.right, d.halfW, ...]` array
   spreads (15 throwaway arrays per frame). **Fixed**: the block is now split
   into a static half written once at create time and a contiguous dynamic
   pair (`region`, `tune` — camera cell window + the four params), so the
   per-frame traffic is one 32-byte `writeBuffer` per entry from a preallocated
   scratch and zero allocation. Same values reach the GPU; no image change.
3. **~31 MB of host memory pinned for nothing.** `SpeciesGpu` held the whole
   unpacked `LdiBaked`, whose `atlas0`/`atlas1` are views into the 10.5 MB
   packed artifact, for the life of the session — although the bytes are dead
   the moment `writeTexture` has run. **Fixed**: only the header (`LdiMeta`)
   is retained. Not VRAM, so the HUD never showed it.
4. Debug views were not wired at all (the plants ignored `view=`). **Fixed**,
   see above. The existing shading was otherwise sound — real decoded baked
   normals, one pass through the shared `light_surface()`, no premultiplied
   light in the atlas — so the normal view is unchanged by the debug work.

Deliberately left alone:

- **Cell-level frustum/circle rejection in the cull pass.** The dispatch is a
  square cell window culled to a circle, so ~21% of the cells cannot produce a
  survivor, and off-frustum cells are the majority at grazing. But `ldi-cull`
  is two orders of magnitude cheaper than `ldi-layers` on every camera I
  looked at, so this optimises the wrong pass and would need a conservative
  per-cell height bound to stay exact. Not worth the risk.
- **Instance capacity sized for `regionRadius = 128` regardless of the current
  value** (6.9 MB of the 17.4 MB budget at density 3, ~55% unused at the
  default radius 96). `regionRadius` is a live param the harness mutates in
  place without recreating the experiment, so the worst case has to be
  allocated up front. Correct as it stands.
- **Per-fragment `uni.dirs[dir_i]` loads and the `uni.atlas` divisions in
  `atlas_uv_for`.** Uniform across the primitive and foldable, but that is
  ALU-level, not structural.

Suggestions for a later (measured) pass, not done here:

- With the far path alive again, `layerCullDist = 70` looks far too generous:
  `inspect=path` at grazing shows ~95% of the screen on the 12-card
  frag_depth path, which is the pass that has no early-z. Re-tune it solo.
- `inspect=layer` shows peel layer 0 winning almost every visible pixel in a
  dense stand; layers 1–3 survive only in the gaps. A distance-graded layer
  count (4 layers up close, 2 mid, 1 far) instead of today's binary near/far
  switch is the obvious next structural win, but it changes the image and
  needs a bench + a look in motion.
