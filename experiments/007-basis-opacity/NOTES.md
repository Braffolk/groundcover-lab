# 007 basis opacity — Fourier appearance cards

## Idea

Encode the **view-dependent appearance of a plant in a small set of basis
coefficients evaluated in closed form per pixel** — no view atlas, no
nearest-view blending.

Two shapes, chosen per stand entry: upright plants get a camera-facing card over
a Fourier fit of their *appearance* (below); a **carpet** entry
(`stand_table[i].carpet_div > 0`, the bog's Sphagnum) gets a ground-parallel
tile-sized quad over a periodic top-down capture, with the same Fourier basis
applied to the tile's *geometry* (its horizon) instead. See "Moss carpet".

Each species is baked once into four small `rgba8` array textures (256², one
layer per elevation ring, E=4 rings from 8° to 82°). Every card texel stores,
per ring, a truncated **Fourier series in view azimuth** of the functions an
impostor normally stores per view:

- coverage/opacity `w(θ)` — order 3 (7 coefficients): the silhouette
- luminance modulation `g(θ)/ḡ − 1` — order 1 (2 coefficients): front/back-lit
  color shift
- relative depth `d(θ)` — DC + cos term (baked; not yet consumed at runtime)
- albedo rgb and mesh-frame normal — coverage-weighted DC, normal oct-encoded

At runtime every plant is ONE camera-facing card (shared scatter WGSL twin
over a bounded cell region around the camera → per-frame cost independent of
total plant count; empty slots, out-of-range cells and fully faded plants emit
degenerate quads). The vertex shader computes the plant-local view azimuth
`θ`, runs the double/triple-angle recurrences once per card (`cos2θ = 2c²−1`,
`cos3θ = c·c2 − s·s2`, …) and the elevation-ring coordinate; the fragment
shader samples the coefficient textures at the card uv (two ring layers,
lerped) and evaluates the series **in closed form** from those flat harmonics.
Because the reconstruction is an analytic function of the view angle, view
interpolation is inherently C∞-smooth: there is no 4-view blend, no ghosting
between discrete impostor views, and no popping — the angular parallax lives
inside the coefficients (as controlled blur, see honesty notes).

Elevation is handled by discrete rings + lerp rather than more basis terms;
the view elevation is clamped to [8°, 82°], which also sidesteps the vertical
billboard-basis singularity (from straight above the card tilts at most 8°
from horizontal). Camera below the plant clamps to the lowest ring.

The truncated series is intentionally fuzzy; `alphaGain` sharpens the
reconstructed opacity and a screen-stable hash dither (param `dither`) turns
the soft alpha into stochastic coverage — the same mechanism does the far
fade and the camera-inside-plant fade (`near_fade`/`far_fade` dissolve).

The card quad is cropped to the *provable support* of the baked coverage. The
tile spans ±R (bounding-sphere radius) on both card axes, but since `right` is
always horizontal and `up` tilts by the view elevation φ, the mesh bbox can
only project into

    |x| ≤ ext_xz·R,   |y| ≤ (cos φ · ext_y + sin φ · ext_xz)·R

with `ext_xz` = (bbox horizontal half-diagonal)/R and `ext_y` = (bbox half
height)/R. Outside that rectangle the fitted coverage is exactly zero, so the
quad and its uv window are shrunk to it — ~45% less card area at grazing
elevation, identical image.

Wind: shared `wind_sway`, scaled by normalized world height up the plant
(roots fixed), per-species `sway` from the stand table. Lighting: per-texel
baked mesh-frame normal (oct-decoded, DC over azimuth) rotated by plant yaw
into the shared `light_surface` + `apply_fog`.

Debug: the global `view` selector (`debug=albedo|normals|lighting|coverage|
depth`) is honoured — fog and the ring overlay are applied only in the normal
view. `coverage` is the post-gain, post-fade alpha the fragment resolved to.
Method-specific state has its own param, `ringTint`: hue = elevation ring
index 0..3, green channel = the lerp fraction to the next ring (0 = off).

## VRAM budget math

Per upright species: 4 array textures × 256×256 px × 4 rings × 4 B = **4.19 MB**
(+64 B meta uniform per stand entry). HUD confirms 4.0 MB/25 MB per species.

Per carpet species: 3 textures × 512² × 4 B × 4/3 (full mip chain) = **4.19 MB**
— deliberately the same budget, spent completely differently: no view atlas at
all, everything in one 0.35 mm/texel top-down tile (albedo+coverage, cushion
normal+sky visibility+height, horizon fit). HUD confirms 4.0 MB/25 MB for each of
the three Sphagnum states; bog stand total 29.5 MB including harness scene
resources. Still no per-plant buffers at any plant count.
No per-plant buffers exist at any plant count (fully procedural placement).
Total for the default 3-species stand: ~12.6 MB + harness overhead (HUD:
21.5 MB total including harness scene resources).

Transient bake-only resources (source mesh vertex/index buffers ≤ ~214 MB for
poa, one 6144×256 capture atlas ×2 + depth) are created outside `ctx.res` and
destroyed immediately after the single bake submit — they trip the dev
"untracked" console warning once per session, expected.

## Bake

`bake.ts`, fully on-GPU, **zero readback**, one queue submit per species:

1. For each of E=4 elevation rings: render the GCMESH1 source mesh
   orthographically from V=24 azimuths into a V·256 × 256 ring atlas
   (MRT: albedo+coverage, mesh-frame normal + relative depth; depth-tested).
2. A compute pass (`fit.wgsl`) then projects, per card texel, the V angular
   samples onto the Fourier basis and writes the packed biased coefficients
   straight into ring layer `e` of the four persistent textures.

96 mesh draws + 4 dispatches per species; all three species bake in a couple
of seconds per session. The harness `bakedArtifact()` cache is intentionally
bypassed (dev server answers missing bake files with 200 index.html — known
harness issue, see 005's NOTES; and this bake is fast enough not to care).

A **carpet** species skips all of that and bakes instead (see "Moss carpet"):
one top-down ortho capture of the tile square, drawn with **9 instances** = the
3×3 periodic lattice so the image is exactly seamless, into 512² albedo
(`rgba8`) + height (`rgba16float`, transient); one compute pass fits the horizon
and the cushion normal; then 9 mip levels × 3 textures of coverage-weighted
downsampling. **9 mesh draws instead of 96** for a 19.8 M-triangle mesh, i.e. the
whole bog stand (three moss meshes, ~1.5 GB of source geometry) loads and bakes
in ~20 s headless, and the carpet part of it is ~10× cheaper than the ring bake
it replaced.

Bake↔runtime contracts for the carpet: `CARPET_TEX` in bake.ts vs `TEX/MASK` in
horizon.wgsl, `CARPET_PLANE_FRAC` (0.742 of the tile top height = the mean
capitulum apex; all three states' mesh manifests agree to ±0.008), and the
top-down basis `right=(1,0,0)`, `up=(0,0,−1)`, `fwd=(0,1,0)` which makes
`uv = (x, z) / tileM` in the mesh frame.

Bake↔runtime contracts kept in sync by hand: V/TILE in `fit.wgsl`,
RINGS/PHI_MIN/PHI_MAX in `card.wgsl`, θ_j convention and the card basis
(`right = cross(Y, dir)`) in `bake.ts` vs `card.wgsl` (world→mesh azimuth is
θ+yaw — mind the `rot_y` sign convention). The bake also hands the runtime
`extXZ`/`extY` (mesh bbox half-extents in bounding-radius units) for the card
crop; the *capture* extent is deliberately left at ±R so the coefficient
textures are unchanged and the crop is a pure raster-area saving.

## Status

**working** — verified by headless screenshots at `cam=grazing`, `topdown`,
`inside-plant`, `far-horizon` (det=1, seed 42, default stand): all three
species render, wind sways, fades behave. Typechecks clean.

Carpet path verified on the `bog` stand at `grazing`, `carpet-close`, `topdown`,
`inside-plant`, `far-horizon`, two sloped eye-level views across the ridges
(23° slope, looking down it and up it), and `debug=albedo|normals|lighting|
coverage`, before and after. No console errors, no toasts. `default` is
unchanged (see "Moss carpet" → "Not regressing the grass").

Honesty / known artifacts:

- The core approximation: features that slide across the card with azimuth
  (parallax) are high-frequency in θ and get low-passed by the order-3 fit.
  Plants read as slightly "rotationally blurred" aggregates — good for the
  community-tile species (calamagrostis, elymus are ~0.5 m clumps anyway),
  more visible on poa close up. In exchange there is zero view popping.
- Dither speckle on flower heads close up (angular ghosting passing the
  stochastic alpha test); reduced with alphaGain 2.4 / dither 0.4 defaults.
- Top-down reads plausible but flat/uniform: azimuthal averaging at the top
  ring behaves like a rotational blur of the from-above view.
- No mip chain on coefficient textures → some far shimmer. Fourier
  coefficients are linear, so mip-averaging them is actually valid — easy
  future win (+33% VRAM, still ~5.6 MB/species).
- Baked depth (DC + cos) is not yet consumed at runtime: fragments write
  card-plane depth like 005; overlapping cards can cut each other. The
  channels are already there for a frag-depth or parallax-uv experiment.
- Colors are angular-DC + order-1 luminance only: subtle desaturation vs
  ground truth; spatial detail (256 px) is preserved.
- Beyond `regionRadius` (default 80 m) plants dissolve — bare terrain at the
  horizon, same tradeoff as 005.

## Findings

- Bench (this machine, apple-metal-3, headless Chrome 1600×900, orbit-low,
  600 frames):
  - `results/007-basis-opacity__default__p-bcdd1069__apple-metal-3__2026-07-24T16-07-37-207Z.json`
    — fourier-cards GPU p50 **29.9 ms** (overdraw-dominated; composite 26.5 ms
    is the harness pass). Same ballpark as 005's grazing cost, but with 4.2 MB
    instead of 18.9 MB per species and smooth (pop-free) view response.
  - `results/007-basis-opacity__scaling-100m__p-bcdd1069__apple-metal-3__2026-07-24T16-08-51-626Z.json`
    — the ~134M-plant stand: fourier-cards GPU p50 **23.6 ms**, i.e. LOWER
    than the 557k-plant default stand on the same spline, and identical VRAM.
    Cost is purely region/screen-bounded — plant count is free.
- Solo HUD p50 at 1280×800: grazing ~24 ms, topdown ~18 ms, far-horizon
  ~12 ms, inside-plant ~14 ms.
- **All numbers above predate the audit pass below** (card crop + distance
  culls cut raster area substantially). Nothing was re-benched: the audit ran
  with other agents on the same GPU, so every measurement this session is
  contaminated. Re-bench before quoting.
- Overdraw is the whole cost; the "tighter card bounds" lever is now taken
  (bbox-support crop, see Idea). Remaining levers: a depth prepass, hard alpha
  edges instead of dither, early ring-0-only sampling at distance.
- A/B: `#/ab/007-basis-opacity/000-ground-truth?stand=default&cam=grazing&seed=42`,
  `#/ab/007-basis-opacity/005-octa-impostors?stand=default&cam=grazing&seed=42`.
- Harness wishlist:
  - dev server should 404 missing `/mesh/baked/**` files so `bakedArtifact()` is
    usable.
  - the species catalog / mesh manifest should expose the two numbers a carpet
    renderer actually needs about a mat's vertical structure: the tile's **top
    height** and the **mean capitulum apex**. `footprint_m` and `carpet_div` are
    exactly right and were used as given, but `height_scale` is the *nominal*
    height and the GCMESH1 header carries only bounds, so the apex fraction (0.742,
    where the visible surface of the cushion actually sits) had to be read out of
    `mesh/raw/*/manifest.json` by hand and hardcoded as a constant. `tileM` is
    already promoted from the manifest to `SpeciesDesc`; `tileTopH` and
    `apexMeanH` (or a single `surfaceH`) belong next to it.
  - `slope_align` was read and honoured, but for a tiled species it carries no
    information the renderer can act on beyond what `carpet_div` already implies:
    a mat must conform fully or it cracks at the tile edges, so any value other
    than 1 is unusable. It is the right knob for the bog's calamagrostis (0.3);
    for carpets it is redundant.

## Moss carpet (bog stand)

Sphagnum palustre is a 0.18 m periodic community tile 0.07–0.09 m tall, laid out
by the `bog` stand as a grid-snapped mat (22×22 tiles per 4 m cell = **484 slots**,
constant scale 1.01, 90°-only yaw). What this renderer did with it before was
indefensible on three counts, all visible in the before screenshots:

1. **Only 128 of the 484 slots were enumerated** (`ii % SCATTER_MAX_PER_CELL`),
   so ~5.8 of every 22 grid rows existed: at `cam=grazing` the bog was **striped**
   — bands of moss separated by bands of bare peat.
2. Each surviving tile was a **camera-facing card 0.34 m wide and 0.09 m tall**,
   i.e. a vertical postage stamp standing in the mat. At `carpet-close` it read as
   a field of fuzzy dithered discs — the featureless blobs.
3. And it spent its whole 4.19 MB budget on a **24-azimuth × 4-ring appearance
   fit** of a cushion whose appearance is not remotely smooth in azimuth.

### What changed, smallest first

1. **Enumerate every slot** (`standEntrySlots(entry)`, rounded up to the next
   power of two so the shader shifts instead of dividing; the scatter rejects the
   surplus 28 slots itself). This alone removes the stripes. Nothing is stored per
   instance in this renderer, so there is no capacity number to get wrong —
   `slots` only drives the draw's instance count.
2. **A second card shape for carpets** (`shaders/carpet.wgsl`, its own pipeline
   and bind group, `triangle-strip`): ONE ground-parallel quad per tile, exactly
   `stand_table[i].footprint_m * scale` across — never `height_scale`, which
   would make this 0.24 m-wide moss 3.5× too small. 4 vertices instead of 6, no
   camera-facing geometry at all, no wind (`sway` is 0 for moss anyway).
3. **Hard alpha at a carpet-specific reference (0.06), no dither.** A mat is a
   closed depth-writing surface; the grass path's `dither` 0.4 has no business
   there. Measured coverage (debug=coverage at `carpet-close`) is ~0.95 mean, so
   the mat is effectively an opaque occluder with a few percent of genuine gaps
   down to the peat. **No camera-inside fade** either — only the region-edge
   dissolve, eroded through the alpha reference and evaluated at the tile centre
   so all four corners agree (a per-vertex fade emits stretched slivers).
4. **Terrain fitting: rung 3, per-vertex conforming.** Every quad corner gets its
   own `terrain_sample(xz)` — height *and* (nx, nz) from the same four texel
   loads, so the shading basis is free. Rung 3 and not 1/2 for a specific reason:
   neighbouring tiles SHARE corner positions, so per-vertex is the only rung that
   keeps the mat C0-continuous; a per-tile point normal or `terrain_plane_fit`
   leaves a wedge-shaped crack at every tile edge because two neighbours fit two
   different planes. Verified on the 23° ridge views: the mat follows the slope
   with no buried or floating edges and no cracks.
5. **A carpet-specific bake, and a different budget split** (see Bake): one
   top-down capture of the tile's own square at 512² = **0.35 mm/texel**, over the
   3×3 periodic lattice so it is exactly seamless, with a full mip chain and a
   wrapping anisotropic sampler. Same 4.19 MB, ~16× the on-screen texel density
   of a 256² ring atlas of which a mat only ever sampled one ring.
6. **The Fourier basis moved from the appearance to the geometry.** This is the
   interesting part. A 0.18 m cushion seen near-grazing parallax-shifts by *more
   than a tile period*, so its appearance is high-frequency in view azimuth and an
   order-3 fit low-passes it into rotational mush — that is precisely what made
   the old blobs featureless, and no amount of tuning fixes it. What *is* smooth
   in azimuth is the **horizon**: the elevation angle of the highest obstruction
   seen from a texel looking out along θ. So `shaders/horizon.wgsl` sweeps the
   captured height field (16 azimuths × 20 log-spaced radii up to ~48 mm) and fits
   `β(θ) ≈ dc + c1·cos θ + s1·sin θ` per texel, plus the cosine-weighted sky
   visibility. The fragment shader evaluates that in closed form against the sun
   and the view direction *in the tile's own basis*, which is what gives the mat
   self-shadowing between capitula and a view-dependent bias toward the tops.
7. **The shading normal is the cushion, not the leaves.** The mesh's own normals
   are leaf normals, and at 0.35 mm/texel a top view of Sphagnum resolves
   individual leaf tips pointing every which way. Their local mean is ~zero, so
   *any* filtering (a single bilinear tap, let alone a mip) leaves a tiny residual
   that `normalize()` blows up into an arbitrary near-horizontal direction:
   measured, that lit whole tiles almost black and made the field flicker between
   four colours with the 90° tile yaw (`debug=normals` showed four flat colours,
   one per rotation — see it in the before shots). The normal is now the smoothed
   gradient of the height field at the capitulum scale, stored as the xz of a unit
   vector so a box-filtered mip correctly *flattens* with distance instead of
   drifting (the octahedral trap, avoided by not using octahedral at all).
8. **16-bit heights.** The horizon sweep differentiates the height field over
   single texels; at `rgba8` one step is 0.36 mm over a 0.35 mm texel = a slope of
   1.0, i.e. the sweep was reading pure quantisation noise and saturating β
   everywhere. The transient height target is `rgba16float` and every tap is a 3×3
   box at 0.7 mm, which also averages out the leaf tips that are below any scale
   this shading can resolve.
9. **All three geometric terms are relief CONTRAST, not absolute occlusion.**
   Each is measured against the same fit four mip levels coarser (~5.6 mm, one
   capitulum) so it shades the relief without moving the mat's mean brightness,
   and it vanishes by itself in the far field where no relief is resolvable. Why:
   absolute terms cost the mat ~35% of its brightness against the
   001-billboard-smoke baseline (39 vs 72 mean luma at `carpet-close`) because the
   fitted horizon over-estimates occlusion — Sphagnum's top surface is a fuzz that
   light passes between, not a solid height field — and they darkened the *far*
   field most, which is exactly backwards. Absolute view occlusion is worse still:
   at grazing every texel is behind something, so it collapses into one uniform
   darkening and turned the whole bog into a flat brown plane (screenshotted).
   With the contrast form, mid-field grazing luma is 73 vs the baseline's 76.

10. **One surface height for all three states.** Each state's own mean-apex plane
    (6.8 / 5.1 / 5.5 cm) put neighbouring tiles of different states up to 1.7 cm
    apart, and a ground quad has no side wall — so at grazing every zone boundary
    showed a step you looked straight through onto the dark peat, outlining the
    interlocking boundary tiles in hard dark bands (clearly visible at
    `cam=far-horizon` before the fix; `reliefTint` confirmed the bands were gaps
    between carpet fragments, not shading). All carpet entries now share the mean
    plane, and the mat is one continuous surface. The wet state genuinely is 2 cm
    taller, but a flat quad cannot show thickness anyway, so the step was pure
    artifact with no information in it.

`carpetOverscale` exists for the invariant-preserving uniform overscale, and is
**1.0**. Unlike 001's case it cannot even produce an artifact here (the sampler
wraps a periodic capture, so an overscaled fringe reads the next period, not a
stretched edge) — but for the same reason it cannot produce any *benefit* either:
the image is identical, the tiles just overlap and double-draw. Left at 1.0.

### What improved / what is still bad

From the before/after screenshots (bog, seed 42, det=1, t=3.5 — `grazing`,
`carpet-close`, `topdown`, `inside-plant`, `far-horizon`, two 23°-slope views,
and albedo/normals/lighting/coverage):

- **Improved:** the mat is closed and continuous everywhere (the stripes are
  gone); at `carpet-close` individual capitula read as star-shaped rosettes with
  lit tops and shadowed interstices — real cushion structure, where the before
  shot had dithered discs and 001 has a flat mottle; the three micro-habitat
  states read as interlocking zones from `topdown`; the mat follows the ridges
  exactly; `debug=normals` shows genuine capitulum-scale normals near the camera
  that flatten (not drift) with distance; `debug=lighting` shows the relief as
  bright domes and dark crevices; the camera at 0.55 m stands *on* the mat with no
  hole under it. GPU Σp50 on the bog stand also dropped from **15.6–19.1 ms
  before to 9.8–10.7 ms after** on quiet runs (and 14–29 ms on runs where a
  sibling agent was hammering the GPU — `base`, which I do not touch, swung from
  0.13 to 6.06 ms across those, so treat only the direction as meaningful)
  despite drawing 3.8× as many tiles: a foreshortened depth-writing ground quad
  costs far less than a dithered camera-facing card, and the moss bake got ~10×
  cheaper too. The mat is its own `carpet-tiles` pass in the HUD (~5 ms of ~10).
- **Still bad, inherent: no thickness.** A single ground-parallel plane has no
  silhouette, so at eye level the carpet is a very good moss *texture on the
  ground* rather than 7 cm of springy cushion. The horizon terms fake the
  *shading* consequences of relief, not its geometry: nothing breaks the horizon
  line, nothing occludes anything else, and the mesh's 3.3 cm of capitulum relief
  exists only in the depth channel and the normal. This is the same wall
  001-billboard-smoke hit; a shell/volume method should own it.
- **Still bad: the 90° rotation seam.** Rotating a periodic tile by 90° keeps the
  *grid* seamless but not the *content*: continuity across a shared edge would
  need `h(0,v) = h(v,0)`, which a non-symmetric tile does not satisfy. At
  `carpet-close` there are faint 1-px lines on the 0.18 m grid and at 15–30 m a
  mild patchwork of the four rotations. Not fixable inside the stand contract;
  001 reports the same artifact.
- Still bad, cosmetic: at extreme grazing the near-field mat (1–3 m, footprint
  ratios past 10:1) shows streaky ochre aliasing. `maxAnisotropy` 16 made no
  measurable difference over 4 (compared crops), so it is content frequency, not
  filtering; it stays at 4.
- Still bad, not mine: the wet-vigorous state's baked albedo is a strongly
  saturated flat green while the other two are ochre/bronze, so at
  `cam=far-horizon` its zone reads as poster paint with the drier tiles as brown
  patches in it. `debug=albedo` confirms it is the source mesh's vertex colour,
  not the shading. 001-billboard-smoke reports the same thing independently.
- The far field converges to a plain textured plane by design (all relief terms go
  to zero with the mip chain), and beyond `regionRadius` (80 m) the mat dissolves
  to bare terrain — same tradeoff as the grass path.

### Not regressing the grass

`default` renders through the *unchanged* upright path: `card.wgsl` is untouched,
non-carpet entries still draw 6 vertices × `side²·128` instances, and a stand
with no carpet entry adds no pass at all (the HUD on `default` has no
`carpet-tiles` row). The one edit that touches the shared bake is
`capture.wgsl`'s depth channel, now `dot(rel,fwd)·depth_scale + depth_bias` with
`depth_scale = 0.5·inv_r`, `depth_bias = 0.5` — algebraically *and* bit-wise the
old `clamp(d·0.5 + 0.5)` (scaling by a power of two is exact), and in any case
that channel lands in `coeff_b.w` / `coeff_d.w`, which `card.wgsl` never reads.
The lattice offset is `+0` for a 1-instance draw with `period = 0`.

Verified rather than assumed: `default` at `grazing`, `inside-plant` and
`topdown` render the same dense three-species meadow with pink panicles, VRAM is
identical (21.5 MB total, 4.0/25 MB per species), and `fourier-cards` p50 stayed
in its usual band (13.7 ms at grazing, 1280×800). No bench JSON is claimed:
four other agents were rendering on this GPU throughout the session — `base`
alone swung from 0.13 ms to 6.06 ms between runs — so every number here is
contaminated and only the before/after *shape* is meaningful. Re-bench solo
before quoting.

### Is this representation suited to a moss carpet?

Partly, and the split is exactly where the method's name is. The *basis* idea
transfers well, but only after aiming it at the right function: fitting the
appearance in azimuth is hopeless for a low periodic cushion (the parallax is
bigger than the tile), while fitting the **horizon** in azimuth is cheap, smooth,
order-1-sufficient, and buys the two things a flat plane cannot fake — directional
self-shadowing between capitula and a view-dependent bias toward the tops. That
is a genuinely better answer than a plain textured quad, and it costs one extra
texture and five taps.

The *card* idea does not transfer at all and was deleted for carpets: a 7 cm mat
has no silhouette to billboard, and a camera-facing quad in a mat is pure
artifact. What is left is an honest, fast, closed, terrain-conforming, relief-shaded
ground surface with **no thickness**. If the bar is "reads as intricate moss
ground cover from a standing or crouching camera", this passes and beats the
billboard baseline at a sixth of its VRAM. If the bar is "reads as a cushion you
could press with your hand", a single quad cannot get there no matter what is
baked into it, and the honest fix is geometry (a displaced shell), not a better
fit.

## Audit

Structural + debug-view pass over the first-generation code. Verified by
headless screenshot at `det=1&t=3.5` before/after each change.

**Debug views (were entirely missing).** `card.wgsl` now includes
`src/wgsl/debug.wgsl` and returns
`debug_shade(color, albedo, normal, alpha, world)`; fog is applied only when
`debug_mode() == DEBUG_OFF`. What the modes exposed:

- *normals* — genuinely per-fragment and correct: the baked oct-encoded
  mesh-frame normal was already decoded and yaw-rotated, and the view shows
  real per-texel variation with per-plant yaw clustering. No fix needed.
- *lighting* — already goes through the shared `light_surface()`, applied
  exactly once, and albedo is not premultiplied by light at bake time (the
  capture pass writes raw vertex colour). The view blows out to white, but so
  does the harness terrain in the same view — that is the shared model's
  range, not this renderer, so it was deliberately left alone rather than
  "fixed" locally (that would make it incomparable to every other method).
- *albedo / depth / coverage* — sane. Coverage is nearly saturated because
  `alphaGain` 2.4 pushes surviving fragments to ~1; that is the honest
  resolved coverage of an alpha-tested method.
- Added `ringTint` param for the method-specific state the global modes can't
  show: which elevation ring a fragment sampled and how far it lerped.

**Structural waste found and fixed** (all image-preserving; measured 8×
downscaled before/after diff ≤ 1.6/255 mean at all four standard cams, full-frame
mean RGB identical to 0.1/255; the residual is isolated dither-threshold pixel
flips from 1-ulp uv changes, no structure — see the diff image, pure speckle):

1. **Card was 1.5× wider and up to 1.3× taller than the baked plant can ever
   be.** The quad spanned the full bounding-*sphere* radius on both axes while
   the coverage lives inside the bbox projection; every fragment in that margin
   still did 4 array-texture samples (8 bilinear fetches) + the order-3
   evaluation before discarding. Now cropped to the provable support (see
   Idea) — ~45% of the card area at grazing elevation, for calamagrostis
   `ext_xz` = 0.66, `ext_y` = 0.75. Exact, not heuristic: outside it the fit
   integrand is identically zero.
2. **Cards beyond `regionRadius` were fully rasterized before discarding.**
   The enumerated region is a square but the fade is a disc, so ≥21% of cells
   were pure waste. Added a cell-level reject in the vertex shader *before*
   `scatter_candidate()` (so it also skips the placement hashes and the
   terrain heightmap fetch), keyed on the XZ distance to the nearest point of
   the cell — a lower bound on the 3D distance to any plant in it, so it can
   only cull cards whose alpha was already 0.
3. **Faded cards were rasterized.** Added an explicit `fade <= 0` cull per
   plant, which also catches the camera-inside near-fade. Zero visual change:
   `alpha * 0 < threshold` discards every fragment anyway.
4. **The Fourier recurrences ran per fragment on values uniform over the
   card.** `cos2θ, sin2θ, cos3θ, sin3θ` derive only from the per-plant view
   azimuth, so they moved to the vertex stage as flat varyings. Same for
   `l1 = min(l0+1, RINGS-1)`.
5. **Dead interpolants.** `entry` (u32) and `aux.z` (plant radius) were
   written by the vertex shader and never read by the fragment shader.
   Removed.

**Deliberately left alone:**

- *The 128-slots-per-cell enumeration.* At density 3/8 about 62% of instances
  are non-existent plants that still run a 6-vertex shader each. Compacting
  them with a compute prepass + indirect draw is the obvious fix, but it adds
  a per-region instance buffer and needs benchmarking to justify — out of
  scope for a structure-only audit. Written down as the next perf idea.
- *No depth prepass, and the dither.* `dither` 0.4 punches holes in the depth
  buffer and defeats early-z exactly where overdraw is deepest (the CLAUDE.md
  taste rule). Turning it down or adding a prepass changes the visual
  character / needs measurement, so it stays an owner call. Suggestion: try
  `dither=0` with `alphaGain` raised, plus a depth prepass.
- *Per-frame meta uniform writes.* 3 × 64 B/frame, and `origin_cell` genuinely
  changes as the camera moves. Not worth a dirty-flag.
- *Bake-time loops and per-ring view creation.* One submit per session; the
  24-tap loop in `fit.wgsl` IS the projection integral.
- *The lighting-view blowout* (see above) — shared-model behaviour.
