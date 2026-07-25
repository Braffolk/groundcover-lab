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

## Findings

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
  because there is no way to visualise a custom per-texel field in-app.
