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
2. `expand` — indirect, one workgroup per visible cell, sweeping this entry's
   scatter slots 128 at a time (`standEntrySlots`: 128 for a scattered entry,
   `carpet_div²` = 484 for the bog moss); the WGSL scatter twin regenerates the
   stand's plants exactly. Every slot is evaluated; capacity only holds the
   expected survivors. Each
   plant is classified by **projected screen extent** (height foreshortens as
   the view goes overhead, the XZ footprint does not — a top-down camera lands
   on the cheap buckets automatically) into 5 LOD buckets and compacted with
   workgroup-aggregated atomics into that bucket's instance region. The
   bucket's `instanceCount` is written straight into its draw args.
3. `ribbons` — 5 `drawIndexedIndirect` per stand entry, 32/16/8/4/4 ribbons at
   12/8/6/4/2 samples. Two-sided opaque strips, depth-written, `cullMode:none`,
   normals from the real ribbon frame (tangent x azimuthal width axis) with a
   cross-blade curl so blades shade as channels, not flat cards.

### Second path: MAT mode (Sphagnum carpets)

A cushion is not a tuft, and feeding one through the tuft distiller produces
something worse than useless (see Findings). A species whose periodic tile is far
wider than the plant is tall (`heightScale < 0.8 · tileM`, true for all three
Sphagnum states and for nothing else in the catalog) is baked instead as a
**radial fan of the tile SQUARE carrying the cushion's surface relief**:

- ribbon `i` of a level runs OUTWARD from the tile centre along the wedge
  mid-angle, with half-width `r·tan(arc/2)`, so the wedges of one tile abut
  exactly along their shared rays;
- its outer radius reaches far enough to cover its whole slice of the tile
  square (`half·cos(arc/2)/cos(|off| + arc/2)`, `off` = angle from the nearest
  side normal), so **the mat is closed by construction** — no lattice of holes,
  and at 32 wedges the overshoot into the neighbour tile is ~2-9 %;
- height along the ribbon is the cushion's own surface, filtered to the fan's
  sample spacing; colour is the authored colour weighted by height² (what you see
  of a cushion is its top layer, and this mesh authors the interior much darker);
- the renderer draws these with a real surface frame (radial slope x azimuthal
  slope from the neighbouring wedges), a per-vertex terrain shear, no width fade
  anywhere, and no blade curl. `carpet_div > 0` in the stand table is the switch.
- LOD keeps 16 wedges out to ~7 m and never drops below 8, spending its budget on
  radial samples instead: a 4-wedge fan cannot tile a square (`tan(45°) = 1`
  puts its corners a full tile width out), so it would either hole or overlap 2x.

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
region radius the params allow: `pi · 128² · perM2 · 0.45` records, split
3/7/16/24/50 % across the buckets. It scales with visible area and density, not
with the stand's plant count (the 134M-plant `scaling-100m` stand allocates
exactly the same buffers as the 20k-plant `close-quality` one). A packed 16 B
record (camera-relative f16 position + quantised yaw/scale/phase) would halve
it; 32 B was chosen for clarity since the budget is nowhere near tight.

**Carpet entries** are sized from the grid, not from `density`: `perM2 =
carpet_div²/16 · wetWidth` = 10.1 tiles/m² for the bog moss (30.25 grid nodes/m²
of which the entry's wetness interval claims a third). That is 233 k records =
**7.9 MB per moss species** (HUD confirms 7.9/25 for each of the three states),
split 1.2/3/6/10/90 % — a life-size 0.18 m tile is in the farthest bucket
essentially always, and the near buckets still get several times their average
share because the wetness field is 12 m noise: a camera standing inside one zone
sees the FULL grid density for the first few metres. Sizing the near buckets from
the zoned average overflowed bucket 2 by ~10 % in a wet hollow and dropped tiles.

Unattributed shared scratch: cell list 68 KB, strip index buffer 8 KB, uniforms
~4 KB, indirect args 300 B. Whole-experiment total ~7 MB on the default stand,
~35 MB on `bog` (three carpet species).

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

`bake.ts::distillMat` is the Sphagnum path — **~200 ms per species** once the
479 MB mesh is in the cache, same artifact format and size, key
`ribbons-mat3-s32-k12-va6-<species>.bin` (`MAT_VERSION` is bumped independently
of `ATLAS_VERSION`, so iterating on the cushions never re-bakes the grasses). It
sweeps the triangles into a periodic 48 x 48 height/colour field over the tile
square (3.75 mm cells) and samples that with a footprint-sized box filter — which
is a proper mip: a coarse level reads a wider window and comes out smoother
rather than aliasing. Three details that mattered:

- **surface height, not mass centroid**: per cell, `mix(mean, mean-of-local-maxima,
  0.72)`, which lands within a millimetre of the mesh manifest's own
  `capitulumApexMeanH`. The mean alone is the mid-height of the moss mass and sits
  20 mm too low.
- **relief gain 1.8** about each tile's mean. The mesh's capitulum apexes span
  33 mm, but leaves fill the gaps between capitula, so a canopy envelope filtered
  to the fan's 9 mm sample spacing only preserves ~9 mm of that. 1.8x puts part of
  it back and is still below the truth; without it the cushions are nearly flat.
- **rim taper 0.5** toward one globally shared boundary height. The height field
  is periodic, so a same-variant/same-yaw neighbour already agrees at a seam — but
  90-degree turns and different variants do not, and a 3 cm step repeating on a
  0.18 m lattice reads as a grid. The taper touches only the outer two samples.
- **per-variant fan twist** of 15 degrees. Yaw is 90-degree-only (lattice rule), so
  without it every tile's wedges point the same four ways and the field reads as a
  lattice of identical flowers.

## Status

**working** — verified by headless screenshots at grazing / topdown /
inside-plant / far-horizon on the default and close-quality stands, seed 42,
plus all six debug views. Zero console/shader errors. All three species render.

**Sphagnum / `bog` stand (2026-07-25)** — verified by before/after headless
screenshots at `cam=grazing`, `topdown`, `inside-plant`, `carpet-close`, a
0.30 m moss-eye pose, `far-horizon`, and two poses across the ridged slopes
(`32,-52` looking down a 23-degree slope and `55.7,-55.7` looking up it), plus
`debug=normals/lighting/coverage/albedo`. `default` re-verified at `grazing` and
`inside-plant`: **pixel-identical** to before (5 of 420 000 viewport pixels
differ at 1 % fuzz, i.e. nothing). No console errors, no validation toasts.

## Findings — Sphagnum carpets (moss round)

What the moss looked like before, in one sentence: **32 near-vertical pickets
arranged in a ring fence per tile, on a quarter of the tile positions, lit from
underneath.** Fixes, smallest first:

1. **Every slot must be evaluated** (`expand.wgsl`). The pass ran one workgroup of
   128 threads per cell with `local_invocation_index` AS the scatter slot, so a
   carpet's 484 slots were cut to 128 — rows 0-5 of each 22 x 22 grid. Three
   quarters of the mat was missing, in 4 m-periodic stripes that looked exactly
   like a placement bug. Now the workgroup loops `ceil(slots/128)` chunks with the
   slot count coming from `standEntrySlots(entry)` on the host. Grass entries are
   one chunk, so nothing changed for them.
2. **No fade for a carpet.** Both fades collapse ribbon *width*, which for a mat
   means punching holes in a closed surface. The camera-inside fade would open a
   hole under your feet; the region-edge fade would dissolve the mat into a
   lattice. Both are skipped when `carpet_div > 0`; the mat simply ends at the
   region radius, which at 96 m against similar-coloured terrain is not visible.
3. **Per-tile colour jitter cut from ±14 % to ±5 %.** At 0.18 m spacing, per-plant
   brightness jitter is not mottling, it is a checkerboard.
4. **Two-sided flip against the view vector, not `@builtin(front_facing)`** — for
   carpets only. The strip winding is fixed in parameter space, and for a fan
   whose centreline runs outward rather than upward the winding-based flip returns
   the normal of the face you cannot see: `debug=lighting` showed the moss at
   ~0.0 against fully lit terrain, and `debug=normals` showed it magenta/blue
   (pointing down). This is the single change that took the mat from black to lit.
   **Note for the owner:** the same flip looks inverted for the tuft path too —
   switching it globally removes most of the near-black blade shards at
   `inside-plant` on `default` and moves 014 visibly closer to 000-ground-truth
   (which shows bright yellow-green blades where 014 shows black ones), and it
   would let `canopyLift` come down from 0.4. That is a change to `default`, so it
   is deliberately NOT in this diff — one line in `fs_main` if you want it.
5. **Per-vertex terrain conforming, rung 3** (see below).
6. **Mat-mode bake** — the radial fan of the tile square described above. This is
   the change that does the real work, and it was not optional: the tuft
   distiller's per-wedge centroid curve is *degenerate for a cushion*. Foliage
   spread evenly over a disc has its area centroid at ~2/3 R at every height, so
   all 12 samples of a wedge sit at r = 0.068 m while y climbs 0 -> 0.081 m. That
   is a vertical picket, not a blade: 32 of them per tile formed a fence around a
   thin annulus, covering maybe a quarter of the ground, with horizontal normals
   (hence black), and the hole in the middle of every tile showed bare peat.

### Terrain fitting: rung 3 (per-vertex), by vertical shear

`slope_align` is 1 for the carpet entries and `carpet_div` already says "mat", so
full conformance. The mechanism is one `terrain_sample(world.xz)` per vertex,
replacing the vertex's height with `ground + (y - base_y)` — a pure vertical
shear, and it is the only rung that is correct here, for two reasons:

- rungs 1-2 (rigid tilt from a point normal or a plane fit) rotate the footprint,
  so on a slope a tile no longer covers its grid step and the lattice opens gaps;
  worse, neighbouring tiles fit different planes and crack apart at their shared
  edge. The shear keeps the footprint exactly on the grid, and two tiles sharing
  an edge evaluate the *same* ground height there, so the surface is continuous by
  construction — no seam handling needed at all.
- the terrain heightmap is 0.5 m per texel, so within a 0.18 m tile the field is
  bilinear-smooth; a shear is therefore not an approximation of the tilt, it *is*
  the tilt, and it costs 4 texel loads.

The same fetch returns `(nx, nz)`, so the shading normal is sheared with the
ground for free: `n' = (n.x + n.y·nx/ny, n.y, n.z + n.y·nz/ny)`, which maps an
up normal exactly onto the terrain normal. It runs for every bucket, including
the far one; gating it by distance saved little and introduced a visible step
where the gate flipped.

### What improved, from the screenshots

- `carpet-close` (1 m straight down): before, a widely-spaced grid of dark
  pinwheel rings with bare terrain between them and half the frame empty; after, a
  closed mottled cushion mat with rosette texture and no bare ground anywhere.
- `grazing`: before, thin moss stripes every 4 m over bare terrain; after, one
  continuous carpet whose wetness zones read as ecology (green hollows, ochre
  flanks, brown crests) with Calamagrostis emerging through it.
- moss-eye (0.30 m): reads as a field of low rounded cushions with real
  silhouette, crevices and cast shading — the thing a single flat quad cannot do.
- the two sloped views: the mat drapes over 23-degree slopes continuously, no
  buried or floating edges, no cracks between tiles, and the texture flows over
  the relief.
- `debug=lighting` went from ~0 (black) to 0.85-1.0, i.e. the same range as the
  terrain; `debug=normals` from saturated magenta/blue to up-ish green with a
  gentle relief wobble; `debug=coverage` is a solid 1.0 across the whole mat, so
  it is a real depth-writing occluder with no dither anywhere.
- VRAM 7.9 MB per moss species against 001-billboard-smoke's 25.8 MB.
- side by side with the reference:
  `#/ab/001-billboard-smoke/014-ribbon-skeleton?stand=bog&cam=carpet-close&seed=42`
  (also worth flicking at `cam=grazing` and the 0.30 m pose
  `cam=0,-7.398,0,0.6,-0.35,60`). 001 is a perfectly flat plane with finer texture;
  014 has thickness, silhouette and terrain-following relief.

### What is still bad

- **The radial signature.** One fan per tile means the facets are organised
  radially around each tile centre, so a tile reads as an 18 cm rosette. Real
  Sphagnum capitula are 1-2 cm. The per-variant twist and the 4 yaws decorrelate
  neighbours, and at 1 m and beyond it passes as mottling, but at 0.3 m the
  rosettes are recognisable and regular. Making the texture finer needs more than
  one fan per tile (e.g. four sub-fans on a 2x2 sub-grid of the tile) — a real
  extension, not a tweak.
- **Relief is ~15 mm, not 33 mm.** Limited by the fan's own sample spacing, see
  the gain note above. Fine detail between capitula is gone; what survives is
  cushion-scale bumpiness.
- **Fine texture is coarser than 001-billboard-smoke's.** Its top-view crop has
  genuine per-texel moss detail; the fan carries one colour per 9 mm facet. 014
  wins on structure and loses on texture — the honest summary is that 001 is a
  photograph of the ground and 014 is a low-poly model of it.
- **No aggregation for the far field.** ~130 k of the ~1.1 M life-size tiles are
  in view at once, and each still costs 32 vertices in the farthest bucket. Bog
  frame time is roughly 3-4x `default`'s. The structural fix (merge 2x2 tiles into
  one instance beyond ~20 m) is exactly what the lattice invariant forbids in its
  simple form, so it was left alone.
- `radiusFrac` for a mat is the tile half-diagonal over the plant height, which is
  1.4-1.9, so the screen-extent metric overestimates a tile's size by ~1.3x and
  spends slightly finer LOD than needed.

### Is this representation suited to a cushion at all?

**Yes, with one qualification.** Nothing about "curved ribbons unrolled from a
texture" requires the ribbons to stand up: reparameterised outward instead of
upward, the same 60 rows per variant become a displaced fan that tiles a square
exactly, conforms per-vertex, writes solid depth and carries real relief — it
expresses the one thing a billboard cannot (thickness), at a third of the VRAM.
The qualification is angular: a fan puts its detail on rays from a centre, so the
*finest* structure it can show is a ~9 mm facet that is long in the azimuthal
direction and organised radially. That is well matched to a rosette-shaped
cushion at arm's length and mismatched to the 1-2 cm capitulum weave you see with
your nose in it.

### Harness interface feedback (from this round)

- **`standEntrySlots` is TS-only.** The WGSL side has `stand_table[i].carpet_div`
  and I derive `carpet_div²` from it, but the "slots per cell" rule now lives in
  two places (`standEntrySlots` in TS, my own `select(128, div*div, ...)` in WGSL).
  A `scatter_slots(entry_index)` helper in `src/wgsl/scatter.wgsl` would remove the
  chance of the two disagreeing — and `scatter_candidate` already knows the rule,
  it just returns `exists = false` instead of exposing it.
- **`footprint_m` was not what I needed, and `heightScale` nearly was.** This
  renderer stores geometry normalised by plant height and multiplies by
  `height_scale · scale`, so what it needs is the tile size *in units of the
  species' nominal height* — `footprint_m / height_scale`. That works out, but only
  because `heightScale` in the catalog happens to match the mesh's own top height
  to within 1 %. If it did not, a carpet tile would silently over- or under-cover
  its grid step by that ratio. A stand-table field for the carpet's world grid step
  (`SCATTER_CELL_SIZE / carpet_div`) would make "exactly fill your step" checkable
  in the shader instead of inferred.
- **`slope_align` says how much but not *about what*.** For a mat the useful
  question is not "how much do I tilt" but "am I a surface that must stay
  continuous with my neighbours" — which is `carpet_div > 0`. Rungs 1-2 of the
  ladder are actively wrong for any tiled species, and that is worth stating in the
  primitive's doc comment next to `terrain_plane_fit`, not only in CLAUDE.md.
- **A `stand=bog` camera bookmark inside a wet-vigorous hollow would help.**
  `carpet-close` and `far-horizon` happen to land in the dry/ochre zones at seed
  42, so judging the green state means hand-writing an absolute pose. Something
  like `carpet-wet` (terrain-relative, 0.3 m up, inside a `wetCenter=5/6` patch)
  would make the three states comparable across renderers.
- **Timings were unusable.** Up to four agents shared this GPU; `p95` swung by 20x
  between identical runs, so no bench was recorded and no frame time is quoted
  anywhere above. `default` was verified for equality by pixel diff instead.

## Findings — grasses (original round)

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
