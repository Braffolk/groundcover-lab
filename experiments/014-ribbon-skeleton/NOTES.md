# ribbon skeleton

## Idea

A plant is a **nested hierarchy of curved ribbons** whose entire shape lives in
two 12 x 360 textures (67 KB per species) and is unrolled into triangle strips
in the vertex shader. No source geometry, no impostor imagery, no raymarching,
and — deliberately — **nothing camera-facing anywhere**.

**Distillation.** The source mesh is swept once into a cylindrical histogram
around a tuft centre: 32 azimuth wedges x 96 height bins, accumulating
area-weighted position, authored colour, radially-projected area and radial
spread. One ribbon is fitted per wedge:

- centreline = the wedge's area-centroid curve, sampled in 12 steps up to that
  wedge's own top (short blades still get all 12 samples);
- half-width = **coverage-conserving**: the wedge's real silhouette area
  divided by the centreline's arc length, capped by a fraction of the wedge's
  own arc. This one rule is why the representation works at every scale — with
  32 wedges a ribbon is far thinner than its wedge and reads as a blade; with 4
  wedges it saturates its arc and reads as a solid tuft;
- colour = the authored mesh colour (albedo, never premultiplied by light);
- occlusion = `exp(-K · foliage area above this height)` from the real vertical
  area profile;
- fluffiness = surface area per unit wedge volume, which cleanly separates
  blades (sheets crossing a cell) from panicles (hundreds of florets filling
  one). Fluffy wedges get a much tighter width cap (their silhouette estimate
  ignores self-occlusion, so it wildly overstates a structure that is mostly
  air) and a mean-preserving ragged width so a plume is feathery, not a strap.

**LOD by ribbon merging.** Levels are built by merging adjacent wedges
32 -> 16 -> 8 -> 4, so ribbon *i* of level L always has parent *i>>1* at level
L+1. At draw time a ribbon is morphed toward its parent as the plant shrinks on
screen; the morph reaches exactly 1 where the next bucket takes over, at which
point ribbon pairs are **coincident** and drawing half as many is invisible.
Continuous LOD with no popping, and the only fades in the method (region edge,
camera-inside-plant) collapse ribbon *width*, never alpha — no dithering,
no stochastic coverage, hard opaque edges that write depth and occlude.

**Why it survives top-down** (the weak angle of the first attempt): a ribbon's
width axis is the *azimuthal* direction of its own centreline, so ribbons splay
outward like real blades. From the side that is a broad curved blade; from
directly above it is a radial arc, and the 32/16/8/4 wedges tile the full
circle. Nothing rotates with the camera, so nothing spins, shimmers, or goes
edge-on as a group.

**Per frame** (three passes, all bounded by the visible region, never by the
stand's plant count):

1. `cells` — one thread per scatter cell of the camera region rect (clamped to
   the stand's own cell range); region-circle + frustum reject against the
   cell's terrain-derived AABB. Survivors are compacted, and the counter *is*
   the indirect dispatch size of pass 2.
2. `expand` — indirect, one workgroup per visible cell, one thread per scatter
   slot; the WGSL scatter twin regenerates the stand's plants exactly. Each
   plant is classified by **projected screen extent** (height foreshortens as
   the view goes overhead, the XZ footprint does not — a top-down camera lands
   on the cheap buckets automatically) into 5 LOD buckets and compacted with
   workgroup-aggregated atomics into that bucket's instance region. The
   bucket's `instanceCount` is written straight into its draw args.
3. `ribbons` — 5 `drawIndexedIndirect` per stand entry, 32/16/8/4/4 ribbons at
   12/8/6/4/2 samples. Two-sided opaque strips, depth-written, `cullMode:none`,
   normals from the real ribbon frame (tangent x azimuthal width axis) with a
   cross-blade curl so blades shade as channels, not flat cards.

## VRAM budget math

Per species:

| item | size |
|---|---|
| geom atlas `rgba16float` 12 x 360 (xyz + half-width) | 34.6 KB |
| shade atlas `rgba16float` 12 x 360 (rgb + packed ao/fluff) | 34.6 KB |
| **representation total** | **67.5 KB** |
| per-frame instance scratch (see below) | 2.22 MB @ density 3 |
| **charged per species** | **~2.3 / 25 MB** (HUD reads 2.2 for calamagrostis, 1.8 for elymus) |

The instance scratch is 5 LOD regions of 32 B records, sized for the *largest*
region radius the params allow: `pi · 128² · density · 0.45` records, split
3/7/16/24/50 % across the buckets. It scales with visible area and density, not
with the stand's plant count (the 134M-plant `scaling-100m` stand allocates
exactly the same buffers as the 20k-plant `close-quality` one). A packed 16 B
record (camera-relative f16 position + quantised yaw/scale/phase) would halve
it; 32 B was chosen for clarity since the budget is nowhere near tight.

Unattributed shared scratch: cell list 68 KB, strip index buffer 6 KB, uniforms
~4 KB, indirect args 300 B. Whole-experiment total ~7 MB on the default stand.

## Bake

`bake.ts::distill` runs in-browser from the raw GCMESH1 vertex/triangle arrays —
**~210 ms per species** (huge meshes are stride-subsampled with a prime stride;
every histogram cell still receives hundreds of triangles). Output is
`RIB2`-magic-validated before it is trusted, committed to
`mesh/baked/014-ribbon-skeleton/ribbons-v4-s32-k12-va6-<species>.bin`
(138 KB each, f32 on disk, converted to f16 at upload).

6 variants per species. For the periodic community tiles (calamagrostis,
elymus) each variant gathers a full tile's worth of foliage around a different
centre, folded periodically so edge-crossing foliage stays with its tuft — so
the variants are genuinely different pieces of the source mesh. The finite
specimen (poa) uses jittered centres plus a wedge-frame rotation.

## Status

**working** — verified by headless screenshots at grazing / topdown /
inside-plant / far-horizon on the default and close-quality stands, seed 42,
plus all six debug views. Zero console/shader errors. All three species render.

## Findings

- **Debug views drove three real fixes.** (1) `normals` showed correct
  per-fragment frames from the start, but `lighting` showed the canopy sitting
  ~4x darker than the terrain: the AO term was far too strong (`exp(-1.25·x)`,
  bottoming out at 0.29) and opaque blade normals are horizontal, which
  half-lambert punishes. Fixed by softening AO to `exp(-0.7·x)` with a 0.3 floor
  and adding a `canopyLift` bias of the *visible* face toward the canopy normal
  (applied after the two-sided flip, in the fragment shader) — the standard
  stand-in for the multiple scattering one opaque blade cannot show.
  (2) The first bake's fluffiness metric compared radial spread against ribbon
  width and was saturated at maximum everywhere, so every ribbon got the soft
  volumetric normal; replaced by surface-area-per-wedge-volume, which is
  bimodal exactly where it should be (calamagrostis: 0.19 fluff low down, 0.80
  in the panicle). (3) `coverage` correctly showed the region-edge width fade
  as the only place coverage leaves 1.0.
- **Coverage numbers** (side-on silhouette, units of plant height², measured by
  the offline distiller against the mesh's own projected area): calamagrostis
  L0 0.065 / L3 0.130 vs the mesh's two-sided 0.265; elymus 0.152 / 0.185 vs
  0.370; poa 0.080 / 0.145 vs 0.360. Plants therefore get *denser* as they
  recede (coarse levels close up), which is the right direction for hiding LOD
  but means the near field is the airier end. `widthScale` is the knob.
- **Top-down is now a strength, not the failure mode.** No pinwheel, no
  rotating cards, no hard region circle — the far fade is a width collapse into
  grass-coloured terrain and the `close-quality` stand's real 24 m boundary is
  the only visible edge.
- Remaining weakness: at very close range the fluffy panicle wedges are still
  single flat quads, so a few read as pale straps despite the ragged widths.
  Splitting a fluffy wedge into several thin ribbons would need a level with
  more than one ribbon per wedge.
- **No timings quoted.** Up to four agents shared this GPU during the session,
  so HUD and bench numbers were meaningless; no bench was recorded. Structural
  cost is bounded by the region: worst case (far-horizon, default stand) is
  roughly 2M triangles, dominated by the two far buckets — `detail` scales
  every screen-size threshold if that needs to come down.
- A/B against ground truth:
  `#/ab/014-ribbon-skeleton/000-ground-truth?stand=calamagrostis-pure&cam=grazing&seed=42`.
