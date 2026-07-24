# 008-relief-tap — secant relief cards

## Idea

The relief/parallax-mapping family, pushed to a FIXED tiny tap count: what can
depth-augmented cards buy without any per-frame marching.

Each species is baked once into a small 5x5 hemi-octahedral fan of
**depth-augmented orthographic captures** (25 views, horizon ring to zenith):
albedo+coverage, plus a signed **heightfield** h(u,v) — the distance of the
foliage surface from the plant's mid-plane along the capture axis — plus baked
local-frame normals. At runtime every plant is one camera-facing card whose
fragments intersect the TRUE eye ray with the selected view's heightfield
using exactly three texture taps and one division (secant / false position):

1. tap h0 at the card point; error e(0) = f·o − h0,
2. analytic reprojection: slide along the ray to the plane f·p = h0
   (t1 = (h0 − f·o)/(f·d) — this alone is the classic one-step parallax),
   tap h1 there; e(t1) = f·(o + t1 d) − h1,
3. secant solve t2 = t1 − e1·t1/(e1 − e0) — the root of the linear model
   through two EXACT samples of the ray-vs-heightfield error function — and a
   final surface tap (albedo/normal/height) at t2.

No loop, no march: the tap count is a compile-time constant (param
`reliefMode` exposes 1/2/3-tap variants for A/B). The solved hit is snapped
onto the sampled height sheet and written as real `@builtin(frag_depth)`, so
plants inter-occlude by their reconstructed 3D surfaces (no billboard
cardboard), and the interior of a clump genuinely opens up with parallax as
the camera moves — within one view cell the motion is continuous, not a
crossfade.

Two more tricks:

- **Stochastic view selection.** Instead of blending the 4 nearest view cells
  (4x the taps, ghosting), each pixel picks ONE view with probability equal
  to its bilinear weight, from the shared deterministic PCG hash of its pixel
  coords. Expectation equals the bilinear blend; at a node direction all
  pixels agree; in between, view mismatch appears as foliage-friendly grain
  rather than double edges. `viewSelect=nearest` shows the popping variant.
- **Wind as inverse ray shear** (borrowed from 004, composes perfectly with
  relief): sway is a linear-in-height shear, and lines stay lines under
  shear, so the eye ray is inverse-sheared into the un-swayed baked frame —
  exact for the model, zero cost in the bake.

Plant-count independence: a GPU cull pass walks only the scatter cells within
`maxDist` of the camera (fixed-size dispatch of the shared `scatter_candidate`
WGSL twin, stand-region clamped exactly like `Scatter.region()`, plus
sphere-vs-frustum test), appends survivors via atomics, renders with
`drawIndirect`. 557k (default) and 134.2M (scaling-100m) plants issue
identical work — verified visually, same VRAM, same frame shape. Cards are
sized to the species' PROJECTED tight box, not the bounding-sphere square
(~2x overdraw saving on these tall-narrow species).

Camera-inside-plant: dithered dissolve when the camera enters the bounding
sphere (near fade), same dither for the far fade at `maxDist`.

## VRAM budget math (per species)

| item | bytes |
|---|---|
| albedo atlas 1280x1280 rgba8 | 6.55 MB (6.25 MiB) |
| geom atlas 1280x1280 rgba16float (h, oct normal) | 13.11 MB (12.5 MiB) |
| culled instances `ceil(pi*88^2*density*1.15)*32B` (88 m = maxDist cap) | 2.69 MB @ d3 / 2.24 MB @ d2.5 |
| ubos + indirect args share | < 300 B |
| **total (default stand)** | **21.3 / 20.9 / 21.3 MiB** ✓ (HUD confirms) |

Worst stand is dense-mixed (density 5): 4.47 MB instances → ~23.0 MiB, still
under 25. Bake-time resources (source mesh upload, up to ~214 MB for poa,
render targets, readbacks) are transient, created outside `ctx.res` and
destroyed right after the bake — they trip the dev "untracked" console
warnings once per session, expected.

## Bake

`bake.ts` renders the GCMESH1 source mesh 25 times (one ortho draw per
hemi-oct node, viewport + dynamic-offset uniform) into the two atlases; a
depth32 buffer resolves self-occlusion, normals are flipped toward the
capture axis (two-sided foliage). After readback, covered texels are
**dilated** 4 rings into empty neighbours per tile (raw byte copies, coverage
kept 0) so parallax reprojections that land just past a silhouette read a
plausible height/color instead of background; empty background height is −1
(far side) so rays travel THROUGH gaps to whatever is behind. ~1–3 s for all
three species, in-browser at first load, per session. The harness
`bakedArtifact()` cache is intentionally not used — the dev server's SPA
fallback answers missing `/mesh/baked` files with 200 index.html and poisons
the cache (same harness limitation 004/005 hit; fix belongs in
`src/bake/io.ts`).

## Status

**working** — verified by headless screenshots on `default` at `grazing`,
`topdown`, `inside-plant` (dissolve works), `far-horizon`, plus
`scaling-100m` at grazing (134.2M plants, identical cost/VRAM) and
`close-quality` for the 1/2/3-tap comparison. All three species render at the
stand's exact placement via the scatter twin. Typecheck clean, no console
errors.

## Findings

- The tap ladder is clearly visible up close (`close-quality`, inside-plant
  cam): `flat-1tap` reads crisp but poster-flat (it IS a plain single-view
  impostor); `linear-2tap` recovers most of the interior parallax;
  `secant-3tap` adds depth layering on stems/heads. The cost of the secant on
  a *discontinuous* foliage heightfield is silhouette fray/chatter where the
  linear error model straddles a depth gap — foliage tolerates it, a smooth
  surface would not.
- Stochastic view selection kills the 4-view crossfade for one tap's price;
  the residual inter-view mismatch shows as stable grain, worst midway
  between horizon nodes (22.5° apart on the outer ring).
- frag_depth from the reconstructed hit makes plant-vs-plant intersections
  read correctly (no sorting), but disables early-z: grazing overdraw is the
  perf ceiling, exactly as in 004.
- Honest artifacts: (1) 25 views can't store what a view never saw —
  dis-occluded interior regions stretch (dilated skirts); (2) alpha-test +
  white-noise dither = shimmer in motion, no blue-noise/TAA in the harness;
  (3) no atlas mips (view cells would bleed) → distant sparkle; (4) beyond
  `maxDist` (default 64 m, cap 88) groundcover ends — visible as a meadow
  edge at far-horizon; a far carpet layer is the natural complement;
  (5) calamagrostis/elymus source meshes are ~0.52 m periodic community
  tiles, so their card is a baked clump repeated per scatter point — density
  reads busier than ground truth; poa (true specimen) is the cleanest.
- Perf (HONESTLY CONTENDED — 15 sibling agents hammered the same GPU during
  every measurement, so these are shapes, not claims): relief-cards pass p50
  at 1280x800 ~13–45 ms depending on cam; the tight-box cards + frustum cull
  took topdown from 62 → 13 ms p50 under the same load. `flat-1tap` was
  roughly half the cost of `secant-3tap` in the close-up test. No bench JSON
  recorded for this reason — run
  `#/bench/008-relief-tap?stand=default&spline=orbit-low` on an idle machine
  before quoting numbers.
- A/B vs ground truth:
  `#/ab/008-relief-tap/000-ground-truth?stand=default&cam=grazing&seed=42`.
