# 005 octa impostors

## Idea

Hemi-octahedral impostors. Each species is pre-rendered, once, into a grid of
orthographic views that tile the whole upper hemisphere of viewing directions
(horizon all the way to straight down). At runtime every plant is a single
camera-facing card. For each fragment we:

1. take the viewing direction in the plant's local (yaw-unrotated) frame,
2. hemi-octahedrally encode it to grid coordinates,
3. pick the 4 nearest baked views and, for each, **project the fragment into
   that view's orthographic basis** to get its atlas UV (this per-view
   reprojection is the parallax that keeps off-axis views correct), and
4. bilinearly blend the 4 sampled albedo/coverage/normal values.

Because the atlas spans the full hemisphere, the plant reads correctly from
grazing angles AND from directly overhead — the case pure camera-facing
billboards cannot handle. Normals are baked in the plant's local frame and
rotated by the per-plant yaw at runtime, so lighting is view-consistent.

Per-frame cost is independent of the stand's total plant count: placement is
evaluated procedurally in the vertex shader (the `scatter.wgsl` twin) over a
**bounded region of cells around the camera** (`regionRadius`), so 100 plants
and 134M plants (scaling-100m stand) issue the identical bounded draw. Nothing
touches source geometry per frame — the 144 captures are baked up front.

Wind: the whole card leans via the shared `wind_sway`, weighted by vertical
card position (roots fixed, top sways), using each species' stand `sway`.

## VRAM budget math

Grid 12x12 = 144 views, 128px tiles -> 1536x1536 atlas.
- albedo atlas: 1536*1536*4 B (rgba8) = 9.44 MB
- normal atlas: 1536*1536*4 B (rgba8) = 9.44 MB
- per-entry meta uniform: 64 B
Total ~18.9 MB / species. HUD confirms 18.0/25MB per species. Under budget.
No per-plant instance buffers exist (fully procedural placement), so cost does
not grow with plant count.

Transient bake-only allocations (source vertex/index buffers, render-target
atlases, readback buffers) are created outside `ctx.res` and destroyed
immediately after the bake, so they never sit in the live budget. They do trip
the dev "untracked createBuffer" console warning — expected, one-off per
session.

## Bake

`bake.ts` renders the GCMESH1 source mesh 144 times with an orthographic
projection built from each hemi-octahedral node direction, one draw per atlas
tile (viewport + dynamic-offset uniform). MRT outputs albedo+coverage and a
local-frame octahedral-decoded normal; a depth buffer resolves self-occlusion.
The two atlases are read back and packed into a single ArrayBuffer.

NOTE: the harness `bakedArtifact()`/`commitBake()` cache is intentionally
bypassed. The dev server answers a missing `/mesh/baked/...bin` request with a
200 `index.html`, which (a) satisfies `committed?.ok` and returns HTML as the
"atlas", and (b) then gets written back to disk by `commitBake`, poisoning both
the OPFS cache and the committed file. Until the harness returns 404 for missing
bake artifacts, we simply bake fresh in-browser each session (~2-3 s for all
three species). This is a harness limitation worth fixing centrally.

## Status

wip — renders correctly and holds from grazing, top-down and (faded) inside.
Verified by headless screenshots at cam=grazing / topdown / inside-plant.

Known issues / honesty:
- calamagrostis and elymus source meshes are periodic *community tiles*, not
  single specimens, so their impostor is really a baked ~0.5 m clump repeated
  per scatter point; density reads busier than ground truth. poa is a true
  single specimen and is the cleanest case.
- Grazing perf is heavy: the octa-impostors pass measured ~34 ms p50 at 1280x800
  in headless Chrome (topdown ~15 ms, less overdraw). Cost is dominated by
  transparent-card overdraw x 8 texture taps/fragment, plus vertex work for the
  bounded candidate set. It is count-independent but not yet cheap; a depth
  prepass / smaller regionRadius / 2-view (barycentric) blend would help.
- Alpha-tested (no sorting), so edges are crisp rather than soft.
- No mip chain on the atlas: distant plants can shimmer; a mip pyramid would
  cost +33% VRAM (still < 25 MB) and is the obvious next step.

## Findings

Bench not yet recorded (perf tuning pending). A/B vs 000-ground-truth:
`#/ab/005-octa-impostors/000-ground-truth?stand=default&cam=grazing&seed=42`.
Top-down is the standout: octahedral views reproduce the from-above appearance
that camera-facing billboards fundamentally cannot.
