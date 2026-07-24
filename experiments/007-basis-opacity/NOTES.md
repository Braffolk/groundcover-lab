# 007 basis opacity — Fourier appearance cards

## Idea

Encode the **view-dependent appearance of a plant in a small set of basis
coefficients evaluated in closed form per pixel** — no view atlas, no
nearest-view blending.

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

Per species: 4 array textures × 256×256 px × 4 rings × 4 B = **4.19 MB**
(+64 B meta uniform per stand entry). HUD confirms 4.0 MB/25 MB per species.
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
- Harness wishlist: dev server should 404 missing `/mesh/baked/**` files so
  `bakedArtifact()` is usable.

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
