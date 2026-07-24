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
(far side) so rays travel THROUGH gaps to whatever is behind.

The result goes through the standard harness bake flow (`bakedArtifact()`:
OPFS cache → committed `mesh/baked/008-relief-tap/views-v1-g5t256-<species>.bin`
→ in-browser bake, then `commitBake()`), so a normal load never fetches a raw
source mesh at all. Artifact = 64 B header (`'R8TP'`, version, atlas px, grid,
centre / radius / halfExt / topH) + albedo rgba8 + geom rgba16float = 18.75 MiB
per species, magic+size validated on load — the SPA-fallback HTML that poisoned
other experiments' caches can only cause a rebake here, never a bad atlas.

## Status

**working** — verified by headless screenshots on `default` at `grazing`,
`topdown`, `inside-plant` (dissolve works), `far-horizon`, plus
`scaling-100m` at grazing (134.2M plants, identical cost/VRAM) and
`close-quality` for the 1/2/3-tap comparison. All three species render at the
stand's exact placement via the scatter twin. Typecheck clean, no console
errors.

## Debug views

The fragment shader routes its final colour through the shared
`debug_shade()` (`src/wgsl/debug.wgsl`), so the global `view` selector
(`debug=albedo|normals|lighting|coverage|depth`) works here exactly as it does
on the terrain; fog is applied only in `DEBUG_OFF`. What each mode reads:

- **albedo** — `surf.rgb` of the solved hit times the depth-in-clump AO ramp
  (the same value handed to `light_surface`, so **lighting** — which divides
  it back out — is exact).
- **normals** — the baked per-texel local normal at the hit, oct-decoded and
  yaw-rotated. It is genuinely per fragment and genuinely high frequency:
  one texel of a 256px view covers several blades of a 2M-tri plant, and
  stochastic view selection means neighbouring pixels can read different
  captures. The lighting term measured over a grazing crop spans 0.13 → 1.0
  (mean 0.76), i.e. real half-Lambert variation, not a constant.
- **coverage** — the bilinear alpha that was alpha-tested, so it is bounded
  below by `covThresh` (0.38) by construction: white = solid interior, grey =
  silhouette texels sitting on the threshold.
- **depth** — from the reconstructed hit written to `frag_depth`, so it shows
  the relief displacement, not the card plane.

Method-specific state lives in the experiment's own `inspect` param (`off` /
`height` = the solved hit's relief height along the capture axis / `view-cell`
= which of the 25 baked views each pixel sampled — the stochastic dither shows
as two-colour grain, and the cell boundaries sweeping across the meadow are
the view quantization made visible).

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

## Audit

Structural pass (no frame times used — the GPU was shared with other agents;
everything below is argued from the code and checked by screenshot).

**Fixed**

1. **The committed baked artifacts were dead weight.** `mesh/baked/008-relief-tap/`
   already held all three view sets, but `bake.ts` never looked at them: every
   page load downloaded the raw GCMESH1 source mesh (poa-pratensis alone is
   229 MB), re-rendered 25 full-mesh ortho captures per species, read two
   1280² atlases back, and ran the 4-ring CPU dilation over 1.6 M texels ×3 —
   all to reproduce bytes that were sitting in the repo. It is now on the
   normal harness flow (OPFS → committed `.bin` → bake → `commitBake`), with
   the mesh load moved *inside* the bake closure so a cache hit fetches no
   triangles at all. Measured effect on load: with the old code a fresh
   `far-horizon` load was still rendering black 8 s in; with the artifact it
   is up in ~2 s. The reason the first version gave for skipping the cache
   (Vite's SPA fallback returning `index.html` at HTTP 200) is now handled in
   `src/bake/io.ts`, and the loader magic+size validates anyway.
2. **Per-fragment recomputation of per-card constants.** The fragment shader
   ran `to_mesh()` twice per pixel — each one a `rot_y` with its own
   `cos`/`sin` — plus `hemioct_encode` + normalizes to pick the view cell,
   for quantities that are identical over the whole card. All of it moved to
   the vertex stage: the eye position in mesh frame, the view-cell coords
   (`g0`, frac), and `cos(yaw)`/`sin(yaw)` are `@interpolate(flat)` varyings
   now. The card point itself is an *affine* function of world position, so
   passing `o_u` as an ordinary perspective-correct varying reproduces it
   exactly instead of re-deriving it — the fragment stage no longer contains
   a single trig call. Also drops `yaw`/`scale`/`fade`/`root`/`sway` from
   perspective interpolation to flat, which is what they always were.
3. **`drawUbo` was re-uploaded every frame for every species** although not
   one of its 24 floats depends on the camera or on time (bake constants +
   params). Now uploaded only when its contents actually change.
4. **Cull dispatch was not clipped to the stand region.** The dispatch is
   sized by the camera's `maxDist` box; cells outside the stand aabb were
   launched only for the shader to reject them with the same `lo/hi` test.
   The CPU now intersects the box with that region (identical predicate, so
   the surviving cell set is unchanged) and skips the dispatch entirely when
   the window is empty.

**Verified unchanged.** Deterministic (`det=1&t=3.5`) crops at `grazing`,
`topdown`, `inside-plant` and `far-horizon`, old code vs new: RMSE 0.5 %–1.2 %,
visually indistinguishable — i.e. the committed artifact is equivalent to a
fresh bake, and the vertex-stage hoisting only moved float rounding around.

**Deliberately left alone**

- `node_basis()` still runs per fragment. It has only 25 possible results and
  could be a small uniform table, but the choice is per-pixel (stochastic view
  selection), the replacement trades ~20 ALU for a dynamically indexed uniform
  read, and which wins is hardware-dependent — that needs a bench, which this
  session cannot honestly run. Same verdict for folding the three per-pixel
  PCG draws into fewer rounds.
- `frag_depth` + `discard` defeats early-z, which is the method's real perf
  ceiling at grazing angles. Fixing that means a depth prepass or giving up
  reconstructed depth — a redesign, not an audit item.
- The dithered near/far dissolve stays: it *is* the fade mechanism (CLAUDE.md's
  dither rule allows exactly that case), and it only runs inside the fade
  bands.
- The instance buffers stay sized for `MAX_DIST_CAP` rather than the current
  `maxDist`, so the `maxDist` slider never reallocates GPU memory mid-session;
  the 25 MB/species budget still holds (21.3 MiB worst case on `default`).

**Open question for a future pass.** The normals debug view shows just how
high-frequency the baked normal field is — one texel spans several blades, and
bilinear filtering across an oct-encoded discontinuity can point anywhere. A
bake-time normal filter (or storing a filtered plus a detail normal) would calm
the lighting noise, but it changes the visual character, so it is a design
change rather than an audit fix.
