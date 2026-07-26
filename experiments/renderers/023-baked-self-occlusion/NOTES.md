# self-occlusion cards

## Idea

**One flat, view-aligned quad per plant. Shading and a single depth tap do all
the work that geometry would otherwise do.**

The seed was "baked self-occlusion and visibility — let shading, not geometry,
make a flat thing read as solid and deep". Two precomputed fields, both indexed
by the same 25 baked view directions, turn a billboard into something that
reads as a solid, deep, self-shadowing clump:

1. **Per-texel surface depth → one-step parallax.** The geometry atlas stores,
   for every texel of every baked view, how far behind that tile's near plane
   the first surface sits (`q`, 8 bits over the tile's own depth range). At
   runtime the fragment intersects **its own eye ray** with that near plane in
   closed form (no marching), takes **one** depth tap, and slides along the ray
   to the depth it reports. Consequences:
   - near parts of the clump shift against far parts as the camera moves —
     real parallax *inside* the silhouette;
   - the alpha test runs at the *warped* coordinate, so the silhouette changes
     shape with view direction instead of rigidly rotating;
   - it also cancels most of the error between the baked view direction and the
     actual one, which is why 8 azimuths do not pop the way a plain impostor
     does (a depth-correct warp of two neighbouring views converges on the same
     true image, so the residual at a tile switch is second order).
   The offset magnitude is smoothly saturated at 0.16 tile widths (~12 cm):
   a one-iteration fixed point is accurate while the shift is small and turns
   into liquid smears once it is large, and this is the one knob that decides
   between "depth" and "melting".

2. **Baked visibility → shading that says "solid".** The same atlas texel
   carries `ao` (the fraction of the upper hemisphere it can actually see) and
   an oct-encoded shading normal leaned 32% toward its mean unoccluded
   direction. `ao` feeds both the ambient term and an ambient-aperture cone
   test for the sun (`cos(aperture) = 1 - ao`, axis = the stored normal), so
   interiors stay dark, tips catch light, and **the self-shadowing moves when
   the sun moves**. Small floors (7% sun, 10% sky) stand in for light that
   bounces around inside a canopy — without them occluded texels go black and
   read as holes rather than depth.

The visibility solve is what makes this work at all: an *isolated* tuft of this
mesh sees ~87% of the sky everywhere (grass is mostly air), which is visually
worthless. So the bake occludes each plant with **its 8 canopy neighbours**,
offset by the mesh's own periodic tile size — exactly how the community tile is
meant to be repeated. That drops openness to 0.35–1.0 with a real vertical
gradient, and a canopy-depth exponent (default 2.6, physically the compounding
of extra canopy layers in a stand denser than the bake's single-species
lattice) expands it into the range the eye needs.

**Runtime shape.** One compute pass evaluates the shared scatter over a
camera-centred cell region (cell-level region + frustum rejects first, then
exact per-plant tests) and compacts survivors into **two** indirect draws:

- **near tier** (< `nearSplit`, default 14 m) — `vs_near`/`fs_near`: eye ray and
  tile UV resolved per fragment, depth warp on. **3 taps** (geometry at `uv0`
  for `q`, albedo at `uv1`, geometry at `uv1`).
- **far tier** — `vs_far`/`fs_far`: the *same* tile UV resolved at the six quad
  vertices instead of per fragment (at that distance the eye ray across a plant
  is near-parallel, so the projective map is affine to well under a texel), no
  warp, implicit-derivative sampling. **2 taps, no ray maths per fragment** —
  the same tap count as the billboard baseline.

The warp fades out over the last 40% of the near band so the two tiers agree at
the switch. Near is drawn first, giving coarse front-to-back for early-z. Hard
alpha test + depth write throughout: no dither anywhere, no blending, no
`frag_depth`.

**Carpets take a third shape.** A species the stand lays out as a mat
(`carpet_div > 0`, the bog Sphagnum) cannot be a camera-facing card: cards would
slice through the ground and through each other, and per-tile view alignment
destroys the lattice that makes a mat a mat. So a carpet entry draws **one
ground-parallel quad exactly one periodic tile across**, per-vertex conformed to
the terrain, textured from the **straight-down view only** — and the same
per-texel depth field becomes a **tangent-plane parallax offset** measured from
the quad's plane, i.e. classic one-step parallax mapping rather than a warp
along the eye ray. That is the one thing this method can give a mat that a flat
card cannot: capitula above the plane and hollows below it slide against each
other as the camera moves. Details in **Findings → Sphagnum carpet**.

**O(1) in plant count.** Nothing is ever materialised for the whole stand.
Verified: `default` (557k plants) and `scaling-100m` (134.2M plants, ±2048 m)
render at the same cost and the same VRAM.

## VRAM budget math

Per species, HUD-verified:

| item | size |
|---|---|
| albedo atlas 1920² rgba8 (5×5 × 384 px tiles) + full mip chain | 18.75 MiB |
| geometry atlas 640² rgba8 (128 px tiles) + mips | 2.08 MiB |
| culled instances, near list (π·44²·density·1.2, 16 B each) | 0.33 MiB (density 3) |
| culled instances, far list (π·112²·density·1.06, 16 B each) | 1.95 MiB (density 3) |
| entry-info uniform (192 B), tile table (1600 B), indirect args | noise |

**Default stand: 23.1 / 25 MB per species** (elymus 22.8). Worst defined stand,
`dense-mixed` at density 5: **24.6 / 25 MB**. No overrun anywhere.

**Carpet species spend the budget somewhere else entirely.** A mat samples
exactly ONE of the 25 baked views, so the other 24 are dead VRAM — 20 MB of an
atlas that a plant with no silhouette can never use. A carpet species therefore
uploads only the straight-down tile (384² albedo + 128² geometry, both mipped),
and the instance lists are sized from `carpetDiv²·wetWidth`, not from `density`:

| item (per Sphagnum species, bog stand) | size |
|---|---|
| straight-down albedo tile 384² rgba8 + mips | 0.79 MiB |
| straight-down geometry tile 128² rgba8 + mips | 0.09 MiB |
| culled instances, near (π·44²·12.6, 16 B) | 1.47 MiB |
| culled instances, far (π·112²·12.6, 16 B) | 8.0 MiB |

**Bog stand: 10.3 / 25 MB per moss species** (HUD-verified), down from 26.9
before — the only thing that was over budget anywhere in this experiment. The
grasses in the same stand are unchanged at 21.9 / 21.3. Cropping to one tile
also gives the mip chain a tile of its own instead of one shared across the 5×5
grid, so distant tiles never blend a neighbouring view's imagery in.

Byte-for-byte the albedo atlas spends what the baseline spends on its
albedo+normal pair, but on 25 views instead of 9, so per-view resolution is
384 px/tile instead of 512 — 513 px/m across the clump (slightly *better* than
the baseline's 502, because each tile uses its own tight silhouette box) and
318 px/m vertically (worse). The geometry atlas at a third of that resolution
is the honest weak point; see Findings.

Bake transients (all tagged `bake-scratch` via `ctx.res`, destroyed at the end):
mesh vertex+index buffers (135 MB + 78 MB for poa), a 3840×768 supersampled MRT
strip (~35 MB), a 2048×1024 r32float visibility atlas + depth (~12 MB), four
storage atlases (~32 MB), readbacks (~18 MB). Peak ~290 MB for poa.

## Bake

`bake.ts` → `mesh/baked/023-baked-self-occlusion/selfocc-v5-<species>.bin`,
15.6 MB each (64 B header + 25×16 f32 tile table + 1920² rgba8 + 640² rgba8),
through the harness `bakedArtifact`/`commitBake` flow with OPFS caching and
auto-commit. Every load is magic-validated because the dev server answers
missing `/mesh/baked` files with `index.html` at HTTP 200, which would poison
both stores; a poisoned OPFS entry is rebaked and repaired in place.

Stages:

1. **Fit.** Plant-local frame (origin = clump centre in XZ, ground at y=0),
   AABB, bounding sphere, and one **tight silhouette box per view** from a
   400k-vertex stride sample, inflated 8% + 8 mm and then clamped to the
   *exact* projection of the mesh AABB so a box can never clip real geometry.
   Tight per-view boxes mean no tile area is wasted on margin and the runtime
   can map a fragment ray straight into tile UV space.
2. **Visibility probes.** 32 directions, uniform in solid angle over the upper
   hemisphere, each rendered as an orthographic depth map into an 8×4 atlas of
   256² tiles (r32float storing `dot(p - centre, d)`, depth-tested so the
   surface closest to the light wins). Drawn with **9 instances** — the plant
   plus its 8 canopy neighbours (community tiles use their own periodic tile
   size; poa, a finite specimen, uses its footprint × 1.6, which lands near the
   0.58 m spacing the default stand's 3 plants/m² actually produces). An
   occluder of P along a probe direction projects to the same (u,v) as P, so
   the neighbours need no larger ortho box. Submitted in chunks of 8 probes so
   no single command buffer runs long.
3. **View capture.** 25 tiles in 5 row passes, 2× supersampled into a
   3840×768 two-target strip: albedo+coverage, and mesh normal (flipped toward
   the bake camera — foliage is two-sided) + normalized view depth `q`.
4. **Resolve** (compute, per row):
   - albedo: coverage-weighted 2× downsample;
   - geometry: 6× downsample where `q` is coverage-weighted **averaged** (the
     warp wants a smooth field) but the normal is **point-picked** — the most
     opaque, most central covered sample in the block. Averaging normals over a
     6×6 block mushes per-blade facets into one near-vertical mean, and because
     the light term is non-linear in the normal that shows up as a flat,
     uniformly bright canopy. Picking one costs zero extra bytes and was the
     single biggest looks fix in this experiment;
   - then the texel's 3D position is reconstructed from the tile frame and `q`,
     the 32 probes are fetched (one texel each), `ao` and the bent normal fall
     out, and the canopy height ramp (`ao^(1.7 - 0.7·h)`) is folded in here
     rather than per fragment.
5. **Dilate** (compute, one pass): empty texels take the nearest covered ring
   (search radius ≤ 3, clamped to the owning tile) so filtering and mips never
   pull background black or a bogus depth across a silhouette. Coverage stays
   zero — the alpha test still cuts the true edge.

Mips are generated on the GPU at load time (coverage-weighted for albedo, box
for geometry). Full first bake of all three species, including fetching 374 MB
of raw meshes, is ~2–3 min on this machine under heavy GPU contention;
subsequent loads read the committed artifacts.

**View layout.** Azimuth resolution follows foreshortening: 8 azimuths at 6°
and 30° elevation, **4** at 52° and 74°, plus one straight-down tile = 25.
Near the horizon the silhouette swings hard with azimuth and needs 8 steps; at
52–74° the plant is compressed into its footprint and azimuth barely matters,
so those rows spend their tiles on *elevation* instead. Worst-case elevation
error is ~11° everywhere. The first version used a uniform 8×3 + top grid
(21° worst case) and the symptom was unmistakable: a hard **density ring**
across the whole top-down view where the tile switched between the 48° row and
the straight-down tile. Re-spending four tiles fixed it completely.

## Status

**working** — verified by headless screenshots at `grazing`, `topdown`,
`inside-plant`, `far-horizon` on the default stand, at all five debug views
(`albedo` / `normals` / `lighting` / `coverage` / `depth`), on
`close-quality`, `dense-mixed` and `scaling-100m`, and on the A/B page against
`001-billboard-smoke`. Zero console errors or warnings in every run.
`npx tsc --noEmit` clean.

The **`bog` stand** (Sphagnum carpet) was added in a later pass and is verified
the same way: `grazing`, `topdown`, `inside-plant`, `carpet-close`, a knee-height
oblique, a 0.55-gradient hillside at 1.1 m and at 4 m, `debug=normals` /
`lighting` / `coverage` / `albedo`, and the A/B page against
`001-billboard-smoke` at three cameras. No toasts, no console errors. The
`default` stand is **pixel-identical** before and after that pass (mean |Δ| 0.00
/255 at `grazing` and `inside-plant` over the whole frame below the HUD).

## Findings

### Sphagnum carpet (`bog` stand)

Where it started: every moss tile was drawn as an **upright, camera-facing card
0.21 m wide**, and only **128 of the 484** slots per cell were evaluated, so the
mat came out as brown ribbons of standing cards covering roughly a quarter of
the ground with bare peat between them. All five failure modes at once.

The fixes, smallest first:

1. **Drive the cull's slot count from `carpet_div²`, not `SCATTER_MAX_PER_CELL`**
   (one uniform field, `info.carpet.x`). 128 → 484 slots. This alone is the
   difference between a quarter of the mat and all of it. Instance capacity is
   sized from a *different* number — `carpetDiv²/16 · min(1, wetWidth·1.25)` per
   m², i.e. the expected survivors of the three-way wetness partition, which is
   ~3× less than sizing for every slot.
2. **A ground-parallel carpet quad** (`vs_carpet` + `fs_carpet_near/far`), sized
   from `stand_table.footprint_m` (never `height_scale` — that would make a
   0.24 m-wide cushion 0.07 m wide), taking the stand's 90°-step yaw and
   constant scale **as given**, with **no camera-inside fade** (a mat you stand
   on must not open a hole) and a carpet-specific alpha reference of **0.06**
   instead of the grass 0.4, so mipped distant tiles cannot dissolve. Overscale
   stays at 1.0: tiles abut exactly, so there is no overlap to z-fight and no
   need for a depth bias.
3. **Per-vertex terrain conforming — rung 3 of the ladder.** `terrain_sample()`
   under each of the four corners (height and ground normal in one bilinear
   fetch), and the baked normal is lifted into that ground basis rather than
   merely yawed. Rungs 1–2 are not merely cheaper here, they are *wrong*: a mat
   is tiled, and neighbouring tiles that each fit their own plane crack apart
   along their shared edge. Only per-vertex keeps the surface C0. Checked on the
   bog's ridged flanks at 1.1 m and 4 m — the mat follows the relief with no
   floating edges, no buried edges and no cracks.
4. **Parallax relief from the same depth field.** Instead of the eye-ray warp
   (which needs a near plane the ray actually crosses — hopeless for a
   ground-parallel plane at grazing), the carpet fragment does textbook
   tangent-space parallax: read `q` at the un-offset UV, convert it to a depth
   below the quad's plane, and slide the sample along the view ray by
   `-(V.xz/V.y)·depth/footprint`. Scale cancels, so it is a pure tile fraction;
   the magnitude is smoothly saturated at half a tile and `V.y` is clamped at
   0.22 so a grazing ray cannot demand an unbounded offset. **The offset wraps
   (`fract`) — the tile is periodic, so it walks into the neighbour's imagery
   instead of clamping at an edge.** Measured effect: parallax 1 vs 0 at the
   same frozen knee-height camera differs by mean **10.95/255** over the mat
   while the mean luma is unchanged (56.0 vs 56.3) — displacement, not a global
   shift. The plane sits at 0.74 of the capture height (the mesh's mean
   capitulum apex), so the offset moves the imagery *both* ways.
5. **Upload only the straight-down tile** for a carpet species — see the VRAM
   section. 26.9 → 10.3 MB per species.
6. **Make the baked tile actually periodic, at load time.** This was the biggest
   remaining artifact and it is worth stating carefully. The bake renders ONE
   mesh instance, and a community tile overflows its own period (0.24 m of moss
   inside a 0.18 m step). So the captured square holds this tile plus *its own*
   overhang, and the overhang of the four neighbours — which in a real mat grows
   back into this square — was never drawn. Show that square as one quad and
   every tile edge is short of coverage: the mat gets a visible lattice of thin
   dark seams, and it did, at every distance. **No rebake was needed, because
   everything missing is already in the image**: the mesh is periodic, so the
   neighbour's overhang at *p* is the mesh's own overhang at *p* ± period. A
   9-tap composite over the lattice (the more-covered sample wins — a top-down
   view of a mat sees the fuller surface) fills the deficit *and* makes the
   result exactly periodic, so the sampling window can then be any window one
   period wide and its two edges match by construction. Mean coverage over the
   sampled window goes **0.951 → 0.984** (wet-vigorous), **0.853 → 0.934**
   (late-season), **0.886 → 0.930** (sun-exposed) — a few points on the mean,
   but all of it concentrated in the edge strips, which is exactly why it read
   as a lattice. Gone in `debug=coverage` and in the render. It also removes a
   second bug for free: the tile square's true phase inside the capture box is
   not derivable from the artifact (see Harness wishlist), and after the
   composite the phase no longer matters.
7. **Smooth the shading normals for a carpet, and only for a carpet.** Sphagnum
   leaflets are far below one geometry texel (1.6 mm), so the bake's
   *point-picked* normal — exactly right for a grass blade, which spans many
   texels, and the single biggest looks fix in the original experiment — is
   per-texel noise here. It turned the sun-visibility cone into salt and pepper:
   `debug=lighting` was a pixel-level black/white check pattern. A 5×5 average
   of the decoded vectors, done once at load, fixes it. This is safe *in this
   one case* despite the standing rule that octahedral normals are not
   mip-averageable: that trap needs opposing front/back faces, and a
   straight-down capture of a mat has every normal in the upper hemisphere.
   Capitula are ~15 texels across, so their shape survives the filter.
8. **Carpet-specific occlusion constants.** The bake occludes each plant with a
   ring of 8 neighbours at the mesh's own periodic spacing; for a scattered
   grass that is an approximation of the stand, but **for a carpet it IS the
   stand**, so the `canopyDepth` compounding (2.6) does not apply — the carpet
   path uses exponent 1.0. The residual floors also had to move: 7 % sun / 10 %
   sky is right for a deep canopy, but a moss cushion is a *surface*, where a
   crevice wall sits millimetres from a lit capitulum of the same bright green
   and short-range interreflection is strong. At the grass floors the mat came
   out 21 % darker than the billboard baseline and read as mud; at 26 %/30 % it
   lands at luma 62 vs the baseline's 71, keeping the crevice contrast that
   makes it read as a cushion. `occlusion=0` (plain `light_surface`) is visibly
   *worse* — a smooth featureless wash — which is the useful evidence that the
   baked visibility is what carries the moss structure here.

**Before / after, from the screenshots.** Before: `carpet-close` (1 m straight
down) showed a grid of dark rectangles — cards seen edge-on — over bare ground;
`grazing` showed brown ribbons of standing cards on green terrain; the slope
views showed the same ribbons draped nowhere. After: `carpet-close` is a
continuous, granular cushion with capitulum-scale clumping and darker crevices,
no grid; `grazing` and the knee-height oblique show a closed mat with the three
wetness zones reading as ecology; `topdown` (42 m) shows smooth zoned fields
with no speckle; the 0.55-gradient hillside at 1.1 m and the 4 m view across it
show the mat following ridges and hollows continuously, and `inside-plant`
keeps solid ground under the camera.

**Against `001-billboard-smoke`, same frame** (`#/ab/001-billboard-smoke/023-baked-self-occlusion?stand=bog&cam=carpet-close`):
the baseline is a smooth, uniform texture with its tile grid still faintly
visible; this side is granular, has visible capitulum clumping and darker
crevices, and has no grid at all. That is the honest win: **structure and
seamlessness**, not resolution — the baseline's albedo is at a slightly higher
texel density than this one's 384 px tile.

**What is still bad.**

- **No thickness at grazing.** A single ground-parallel quad has no silhouette.
  At `grazing` and at the horizon the mat is a painted surface: the parallax
  offset saturates (the clamp at `V.y = 0.22` is doing all the work) and the
  0.09 m of cushion height contributes nothing to the outline. Nothing short of
  standing geometry off the ground fixes this, and that is not what "cards"
  means. Vertical cards are not an option — they slice the ground and each other.
- **Zone boundaries are tile-quantised** into visible stair-steps at mid
  distance. That is the stand's per-node wetness partition, identical in the
  baseline, not a renderer artifact.
- **The geometry atlas is the resolution floor.** 128 px over a 0.21 m box is
  1.6 mm/texel — about 15 texels per capitulum, which is enough for relief and
  AO but not for a crisp capitulum edge. With the carpet now spending 0.9 MB
  where it used to spend 20.8, there is ~14 MB of headroom for a carpet-specific
  bake at 1024² per channel (0.2 mm/texel, 8× the relief detail). That is the
  single biggest remaining lever and it is a bake change, deliberately not taken
  in this pass.
- **`debug=lighting` still saturates to white** on the lit tops. That is the
  shared model (`sun_color` 1.15 + `ambient` 0.25 ≥ 1 at full sun), identical on
  the grass path; the crevice structure is clearly legible underneath it.
- **Three cull dispatches evaluate the same 484 nodes.** The three moss states
  partition one grid, so each entry re-evaluates every node and keeps a third.
  Folding them into one dispatch with three output lists would cut the carpet
  cull cost ~3×. Structural, not done here.

**Is this representation suited to moss?** Partly, and honestly so. The parts of
the method that are *fields* — baked per-texel openness, bent normals, surface
depth — transfer beautifully to a cushion: they are what makes it read as an
intricate 3D surface rather than a printed texture, and they beat the baseline
visibly. The part that is a *card* does not transfer at all: the whole point of
a view-aligned card is to fake a silhouette, and a mat's silhouette is the thing
it does not have. So what survives for moss is "baked self-occlusion", not
"cards" — and the result is a very good ground surface with real relief
response, sitting on a proxy that can never have thickness.

**Perf (same-frame A/B only; sibling agents shared the GPU).** B measures
~1.26–1.35× slower than A for identical work, so raw ratios below are upper
bounds. `bog`: cull+cards B/A **0.75** at `grazing`, **0.78** at knee height,
**0.86** at `carpet-close` — i.e. the carpet path is *cheaper* than the
billboard baseline on this stand even before the correction. `default`: B/A
**1.41**, unchanged from this experiment's documented ~1.4×. Solo, `default`
`grazing` reads Σp50 **8.43 ms** (8.31 before the carpet work) — inside the 9 ms
budget and within run-to-run noise.

### Does it beat the billboard baseline on looks? Yes at every camera except one.

Judged with matched crops at a 12° FOV close-up (camera 2.8 m out, 19° above
horizontal — the scale where a card either reads as a plant or as a poster),
link:
`#/run/023-baked-self-occlusion?stand=default&seed=42&cam=2.00,1.50,2.00,-0.7854,-0.3398,12.0`

- **This method reads as lit, layered vegetation; the baseline reads as a
  texture wall.** Individual blades separate from the interior because the
  interior is genuinely darker (baked openness), and the panicles keep their
  internal structure instead of flattening into pale pink blobs. Mean luminance
  over the same crop: baseline 67.8, this 77.7 — brighter *overall* yet with
  visibly more local contrast, which is the giveaway that the depth is coming
  from occlusion rather than from the baseline's high-frequency normal noise.
- **Parallax is real and measurable.** `p.parallax=0` vs `1` at the same frozen
  camera differs by a mean of 10.1/255 over the crop, and the difference image
  traces *individual stems and panicles* in the near field and fades to exactly
  zero at the far tier — i.e. the displacement is depth-proportional per
  structure, not a global shift. That is internal parallax by construction: the
  offset is `q · D / |w·b|` along the eye ray, so a texel on the front of the
  clump moves not at all and one at the back moves the full depth range.
- **Silhouette genuinely changes with view direction**, on two mechanisms: 25
  captures across 5 elevation bands are different *images*, not one shape
  rotated; and the alpha test samples the warped coordinate, so the outline
  morphs continuously between tile switches rather than snapping.
- **Top-down is now on par with the baseline** — uniform, dense, no ring (see
  the view-layout note above). At 62–82° elevation this method shows the
  canopy from a genuinely different angle where the baseline has only its one
  horizontal top card.
- **`inside-plant` is where the baseline still wins.** At <1 m the geometry
  atlas is 5.8 mm/texel and the shading blurs into thick painted strokes while
  the baseline's full-resolution normal atlas keeps crisp blades. Mitigated (a
  wider camera-inside erosion, which the rules sanction, plus the warp offset
  limit) but not fixed. **The fix is a geometry atlas at albedo resolution, and
  it does not fit in 25 MB alongside 25 views** — that is the real trade this
  experiment is making, and it is the first thing to change if the budget ever
  moves.

### Performance — ratios only, the GPU was contended

15 sibling agents were rendering on this GPU throughout, so **no absolute
millisecond figure here is worth anything and no `results/` bench JSON is
claimed.** What is meaningful is the A/B ratio, because both sides render in
the same frame on the same device:
`#/ab/001-billboard-smoke/023-baked-self-occlusion?stand=default&cam=grazing`.

Total GPU (cull + cards), B/A, median of 4–8 samples:

| cam | ratio |
|---|---|
| grazing | **1.45–1.51** |
| topdown | **1.16** |
| inside-plant | **1.43** |
| far-horizon | **1.43** |

So roughly **1.4× the billboard baseline** — within the ~1.5× ceiling, not at
parity. Decomposition at grazing (`b.nearSplit=0`, i.e. everything on the flat
far path): **1.19×**, so the far tier, which owns most of the screen, is close
to parity with the champion. The remaining cost is entirely the near tier's
per-fragment ray + third tap, and it is directly priced by `nearSplit`:
8 m → 1.40, 12 m → 1.48, 14 m → 1.45–1.51, 20 m → 1.57. 14 m is the chosen
default because a 1.15 m plant at 14 m is still ~57 px tall, so a 12 cm
parallax offset is still ~6 px of visible motion.

Two structural changes took the far tier from 1.57× to 1.19×, both worth
copying: resolve the tile UV at the **vertices** instead of per fragment when
the eye ray is effectively parallel, and give the low-resolution geometry atlas
its own non-anisotropic sampler.

### Plant-count independence

`scaling-100m` (134.2M plants, ±2048 m) renders the same image at the same
VRAM and within noise of the same `cards` time as `default` (557k). Cost is
region area, not plant count — by construction, since the cull dispatch is
sized from the camera-centred cell rect clamped to the stand.

### Debug views

All five route through the shared `debug_shade()`, and fog is applied only when
`debug_mode() == DEBUG_OFF`:

- **normals** — the point-picked mesh normal leaned toward the bent normal,
  decoded from the geometry atlas and yaw-rotated into world space. Reads as
  per-blade facet spread over a green (+Y) bias, which is exactly what a
  visibility-leaned foliage normal should look like; before the point-pick fix
  it was a near-uniform green wash, and the ordinary render looked flat in
  precisely the same way.
- **lighting** — `shaded / albedo`, i.e. the shared model's two terms with the
  baked visibility folded in. It saturates >1 in the lit areas (the shared
  sun+ambient does that, same as the baseline notes) but the **occluded
  structure is clearly legible**: bright tips, dark interiors, dark bases. The
  function is identical to `light_surface()` when both visibilities are 1.
- **coverage** — the baked alpha the fragment resolved to; with a hard alpha
  test that doubles as the alpha-test margin. Binary white inside foliage, grey
  only on the bilinear rim. No dither anywhere.
- **depth** — the card plane's distance, which is exactly what lands in the
  depth buffer (nothing writes `frag_depth`), so the view is honest about what
  depth-tests rather than about the parallax-corrected surface.
- **albedo** — the raw baked capture. Deliberately excludes the visibility
  terms, which belong to the light side.

### Honest artifacts

- **`inside-plant` softness** (above) — the technique's real weakness.
- **Disocclusion.** A view cannot store what it never saw; where the warp
  reveals a region the baked view had hidden, the nearest surface stretches
  into it. Bounded by the 0.16-tile offset limit, and grass tolerates it, but
  at grazing angles in the near field it shows as slight smearing along blades.
- **Azimuth quantisation at high elevation.** Rows 3 and 4 have only 4
  azimuths (45° steps). Between 52° and 82° elevation a slow orbit shows a soft
  swap; it is much less visible than the density ring it bought away.
- **Region edge.** Groundcover ends at `regionRadius` (110 m default, cap 112),
  eroding out over the last 14% with fog covering most of it. Long sightlines
  at `far-horizon` still show the band.
- **Wind is a shear, not a bend.** The card's imagery is anchored to the
  at-rest quad, so sway shears the texture rather than bending each blade.
  Correct to first order and free; visible only if you look for it.
- **No stochastic anything.** Hard alpha test, opaque depth writes, coverage
  fades via the alpha reference. Nothing here relies on dither.

### Harness wishlist

- A dev-server 404 (rather than the SPA `index.html` at 200) for missing
  `/mesh/baked/**` would let every experiment drop the magic-validation shim.
- A scalar/aux debug channel would have saved a round trip: diagnosing "the
  canopy is uniformly bright" needed a Node script to dump the baked atlas,
  because there is no way to visualise a custom per-texel field in-app. The same
  gap cost another round trip on the moss: "why is the mat muddy" needed the
  baked `ao` and `q` channels dumped to PNG outside the app.
- **`MeshInfo` exposes `tileSize` but not the mesh bounds or the tile origin.**
  To locate a periodic tile square inside a baked view you need the capture
  box's offset from the tile — i.e. the mesh AABB centre — and the only way to
  get it is to load the 500 MB binary or to re-fetch `mesh/raw/<id>/manifest.json`
  by hand. `MeshInfo.bounds` (and `tileOrigin`, even if it is `(0,0)` for every
  current mesh) would be three lines and would save every carpet renderer the
  same detour. This one bit hard enough to change the design: rather than
  guessing the phase, the tile is made periodic at load time so the phase stops
  mattering.
- **A periodic top-down capture would be better done once in the harness than
  five times in five experiments.** Every renderer that draws a community tile
  as one quad hits the identical bug — the mesh's own overhang is present, the
  neighbours' is not, so the mat grows a seam lattice — and every one of them
  has to discover it from a `debug=coverage` screenshot. Either bake the top view
  with the 3×3 periodic ring (the visibility bake in this experiment already
  draws 9 instances, so the machinery exists), or document the lattice composite
  as the standard fix next to the `carpet_div` notes.
- **`slope_align` has no natural insertion point for a camera-facing card**, and
  the bog's calamagrostis (0.3) is consequently still drawn bolt upright here.
  Tilting a view-aligned card means transforming the eye ray into a tilted plant
  frame, which is a real change to the impostor mapping rather than a basis
  swap. Not a defect in the harness — just worth knowing that the primitive that
  is easy for geometry (`plant_basis`) is not easy for an impostor.
- `standEntrySlots()` is exported and correct, but the trap it guards is on the
  GPU side, where the equivalent is `stand_table[i].carpet_div²`. A named WGSL
  helper (`stand_entry_slots(i)`) next to `scatter_candidate` would put the fix
  where the bug happens.
