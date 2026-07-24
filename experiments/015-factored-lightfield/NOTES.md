# 015 factored quantized light field

## Idea

A per-species light field — "what do you see looking at this plant from
where" — stored whole, but brutally factored and quantized to fit VRAM:

**L(direction, image offset) ≈ Appearance3D( Hit(direction, image offset) )**

- **Geometry factor (all the view-dependence):** a 24x24 hemi-octahedral grid
  of ORTHOGRAPHIC depth maps — 576 views, 128px tiles, 8-bit depth in a single
  rg8 3072x3072 atlas (r = min-depth, 1.0 reserved as miss sentinel; g =
  fractional coverage from a 2x2-supersampled bake). This is a precomputed
  answer table for "which surface does this ray hit" — the sanctioned
  precomputed-raycast loophole; 4.3 deg between neighbouring views.
- **Radiance factor (view-INDEPENDENT):** 3D albedo + normal volumes
  (aspect-fit ≤96 per axis, rgba8), splatted on CPU from the raw GCMESH1
  vertices (2M–6.5M vertices ≫ voxel count), dilated so quantized hit points
  always land on data. albedo.a carries per-voxel luminance sigma, re-injected
  at runtime as deterministic hash speckle (~1.5cm cells, stable under wind
  and camera) so voxel-averaged interiors keep sub-voxel texture.

Runtime: every plant is ONE camera-facing card. Per fragment: un-shear the
wind lean, project into the 4 views nearest the per-fragment eye direction,
1 coverage sample + 1 depth gather each, reconstruct each view's 3D hit,
blend hits by (bilinear x coverage) weight, optionally one eye-ray
reprojection tap round (<45m), then ONE appearance fetch at the blended hit.
Because colour is a function of the 3D hit point only, view interpolation can
never double-image colours — the classic impostor crossfade ghost is
structurally impossible; view blending only softens geometry. The
reconstructed hit writes frag_depth, so plants inter-occlude and intersect
terrain correctly without sorting.

Naive 4D storage at identical angular/spatial resolution (albedo, normal,
depth, coverage ≈ 10B/ray x 576 x 128^2) would be ~94MB/species; the
factorization stores 19–23MB — and buys 4x the angular density of
005-octa-impostors (576 vs 144 views) in comparable memory.

Placement: scatter WGSL twin evaluated in the vertex shader over a bounded
camera-centred cell window (`regionRadius`, default 96m) — 100 plants and the
134M-plant scaling stand issue byte-identical GPU work (verified, see
Findings). No source geometry exists at runtime; no marching anywhere (the
refine tap is a second lookup, not a loop).

Wind: shared `wind_sway` at the plant root, per-species stand sway. The card
leans, and the fragment shader un-shears its sample coordinates by the
interpolated lean — which is what makes the SPRITE lean (sampling at the
displaced position, as some earlier experiments do, cancels the lean except at
the card border). Reconstructed hits are re-sheared before frag_depth/light.
Verified: 66% of field pixels change between t=2 and t=5 (mean |d| 36/255).

Lighting: volume normals (yaw-rotated, two-sided flip) + shared
`light_surface` + fog; height-based AO stand-in (roots 0.6). Camera inside a
plant: coverage fades out below 1.1 x bounding radius (`near_fade`), and at
the region edge (`far_fade`) — honest coverage erosion through the hard alpha
threshold, no dithering anywhere.

## VRAM budget math

Per species (HUD-confirmed):

- depth+coverage light field: 3072 x 3072 rg8unorm = **18.87MB**
- albedo volume rgba8 (aspect-fit): calama 62x96x58 = 1.38MB · elymus
  77x96x79 = 2.34MB · poa 38x96x39 = 0.57MB
- normal volume rgba8: same dims again
- info uniform: 96B

Totals (HUD): calamagrostis **20.6/25MB**, elymus **22.5/25MB**, poa
**19.1/25MB**. No per-plant buffers at any plant count. Transient bake peak
adds ~8MB strip targets + mesh vertex/index buffers (up to ~180MB for poa for
a few hundred ms; destroyed and un-counted after bake).

## Bake

In-browser, per session, ~0.4–1.6s per species (no committed artifact — the
dev server's SPA fallback poisons `/mesh/baked` fetches with index.html, as
005 discovered; the OPFS path would cache poisoned bytes, so like 005 I bake
fresh each load):

1. 576 ortho views rasterized 2x2 supersampled (256px) into a one-row strip
   (6144x256 r8 + d32), chunked ≤160M tris/submit with
   `onSubmittedWorkDone()` awaits (poa = 6.5M tris x 24 draws/row), then a
   downsample pass writes min-depth + fractional coverage into the final
   atlas row. Fractional coverage is what keeps sub-texel seed-head fluff
   wispy instead of snapping to its convex hull.
2. CPU vertex splat + 4 dilation passes + sigma into the volumes (~100-400ms).

## Status

**working** — verified by headless screenshots at grazing / topdown /
inside-plant / far-horizon, on `default` and `scaling-100m` stands, t=2 and
t=5. All three species render (calamagrostis, elymus, poa at their stand
entries).

## Findings

- **Renders correctly from all standard cams.** Topdown reads as a proper
  meadow carpet (the hemisphere atlas covers straight-down); inside-plant
  shows real parallax between stems with the near-fade working; grazing holds
  with no billboard cardboard (frag_depth from reconstructed hits).
- **Plant-count independence:** `default` (557k) vs `scaling-100m` (134.2M)
  screenshots are near-identical in content and cost — the drawn work is the
  same bounded cell window by construction (side^2 x 128 instances x entries).
- **The no-colour-ghosting claim holds:** view transitions show geometric
  softening only; there is no double-image crossfade anywhere because all 4
  views index the same 3D radiance.
- **Known artifact (family trait):** calamagrostis' fluffy panicles
  reconstruct as smooth pale "balloon" lobes close-up — first-hit depth
  cannot represent sub-texel porosity, and the volume averages fluff colour.
  004-raycast-lut shows the identical signature (see its thumbnail). The 2x2
  supersampled fractional coverage + sigma speckle thins and textures them,
  but at <3m they are still lobes, not wisps. Would need either stochastic
  transmittance (dithering — rejected per taste rules) or a porosity channel.
- **8-bit depth (≈6mm over the bounding diameter) is invisible** at normal
  ranges; the speckle hash masks the residual quantization banding.
- **Perf:** timings during development were contaminated (15 parallel agents
  on one GPU — CLAUDE.md forbids quoting them), so no bench JSON is claimed
  yet. Structural note for the audit: frag_depth disables early-z; a near/far
  pipeline split (rasterizer depth beyond ~40m, where the card-vs-hit depth
  error is subpixel) is the obvious next win, as is dropping `refine` (halves
  taps) which is already distance-gated at 45m.
- Directions below the horizon clamp to the hemisphere ring (terrain
  groundcover is effectively never seen from underneath).
