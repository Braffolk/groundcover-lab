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
total plant count; empty slots emit degenerate quads). The vertex shader
computes the plant-local view azimuth `θ` and the elevation-ring coordinate;
the fragment shader samples the coefficient textures at the card uv (two ring
layers, lerped) and evaluates the series **in closed form** using the
double/triple-angle recurrences (`cos2θ = 2c²−1`, `cos3θ = c·c2 − s·s2`, …).
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

Wind: shared `wind_sway`, scaled by normalized world height up the plant
(roots fixed), per-species `sway` from the stand table. Lighting: DC normal
rotated by plant yaw into `light_surface` + `apply_fog`.

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
θ+yaw — mind the `rot_y` sign convention).

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
- Overdraw is the whole cost; obvious levers: tighter card bounds (crop the
  quad to the baked coverage extents — could be a tiny CPU readback of ring-0
  coefficients, or a per-θ Fourier silhouette radius), a depth prepass, or
  early ring-0-only sampling at distance.
- A/B: `#/ab/007-basis-opacity/000-ground-truth?stand=default&cam=grazing&seed=42`,
  `#/ab/007-basis-opacity/005-octa-impostors?stand=default&cam=grazing&seed=42`.
- Harness wishlist: dev server should 404 missing `/mesh/baked/**` files so
  `bakedArtifact()` is usable.
