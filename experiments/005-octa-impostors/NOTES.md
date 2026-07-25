# 005 octahedral view-set impostors

Rebuilt from scratch (the previous attempt under this id — 144-view atlas,
4-view per-fragment blend, plants rendered upside down — was discarded whole).

## Idea

One plant becomes **121 pre-rendered views** covering the entire upper
hemisphere of viewing directions, and every plant on screen is **one quad** that
picks and blends between those views as the camera moves.

**The view set.** Directions are parameterised hemi-octahedrally: a unit
direction `d` (plant → camera, `d.y ≥ 0`) maps to
`uv = (n.x + n.z, n.z - n.x)·0.5 + 0.5` with `n = d / |d|₁`. The square's
boundary is the horizon, its centre is straight down, its corners are the ±X/±Z
horizon views. An 11×11 grid of nodes over that square is the view set: 32
distinct horizon azimuths (11.25° apart) up to an exact top-down view. Each
node is one orthographic capture, stored as **one layer of a texture array**,
not a tile of an atlas — layers cannot bleed into each other under minification,
so the whole thing is mipped with no gutters and no atlas-edge rules.

**Each view has its own extents.** A view's ortho box is the plant's local AABB
*projected onto that view's basis* (`ext = |axis|·half`), so a tall thin side
view and a squat top view each fill their tile completely. Nothing is wasted on
the empty margin a shared bounding sphere would force (≈2.5× on these plants),
and the runtime knows every view's exact basis, which is what makes step 3
possible.

**Selection is per plant, in the vertex shader.** A plant subtends a fraction of
a degree, so its viewing direction is effectively constant across its own
silhouette — computing the octahedral cell, its 3 surrounding nodes and their
barycentric weights once per plant (not once per fragment) costs nothing and
makes the layer index primitive-uniform.

**Blending is a reprojection, not a cross-fade.** The card is the AABB projected
along the *actual* view direction. In the fragment shader the card position is
turned back into a local-frame offset and projected into **each selected view's
own orthographic basis** to get that view's uv. The three images therefore line
up on the card plane instead of sliding against each other, which is what keeps
the blend sharp and is also the parallax that makes an off-axis view usable at
all. Colour is accumulated coverage-weighted; the normal is taken from the
dominant view only (1 tap, not 3 — neighbouring views disagree far less about
normals than about silhouettes). 4 texture taps per fragment, not 8.

`viewBlend=nearest` switches to a single view with a hard switch — the built-in
A/B for what the blend actually buys (ghost vs. pop).

**Per-frame cost is O(visible region).** One compute pass per stand entry
evaluates the shared scatter over the camera-centred cell region (clamped on the
CPU to the stand's cell range), one workgroup per cell so the region and frustum
rejects are workgroup-uniform and a cell that cannot contribute exits before a
single hash round or heightmap fetch. Survivors are compacted into **four
distance buckets** drawn near-to-far: impostors are hard alpha-tested with depth
write, so the near bucket lays down solid depth and hi-z rejects most of the far
buckets' fragments — exactly the overdraw that killed the previous attempt at
grazing angles. No dithering anywhere; the near/region fades erode the alpha
reference instead (CLAUDE.md taste rule).

Wind: the shared `wind_sway` shears the card by height fraction squared,
matching 000-ground-truth's weighting, with each species' stand `sway`.

**Carpet species take a different shape but the same view set** — see "Moss
carpet" below. Short version: for `stand_table[i].carpet_div > 0` the plant is a
periodic mat tile, so it draws ONE ground-parallel quad of exactly one grid step
instead of a camera-facing card, and the view set is used as a *precomputed
raycast* rather than as a billboard.

## VRAM budget math

**Tile size is 256 px (`TILE` in `bake.ts`), chosen by the project owner: 128 px
was visibly too blurry up close.** That deliberately spends well over the 25 MB
soft budget — the README allows exceeding it when the result justifies it, and
sharpness is the whole point of a view-set impostor. The 128 px numbers are
kept below for comparison.

Per species, 121 views × 256 px tiles, 5 mip levels (256 → 16 px):

| item | bytes |
|---|---|
| albedo+coverage rgba8 array, mipped | 121 × 87 296 texels × 4 B = **42.2 MiB** |
| oct normals rg8 array, mipped | 121 × 87 296 × 2 B = **21.1 MiB** |
| per-view basis table (right/up + extents) | 121 × 32 B = 3.8 KiB |
| culled instances (default stand, density 3, 4 buckets, R=128 m) | 185 824 × 16 B = **2.84 MiB** |
| entry uniform + indirect args | 288 B |
| **total** | **≈ 66 MiB** (2.6× the 25 MB soft budget) |

Per **carpet** species (bog stand, HUD-verified **64.1 MiB** each) the same
atlas dominates and the instance buffer is much smaller than a scattered
entry's, because a carpet tile is a 4 B record instead of 16 B:

| item | bytes |
|---|---|
| albedo+coverage + oct normals, as above | **60.4 MiB** |
| culled tiles, 4 B each, 958 560 tiles for R = 128 m clipped to the ±96 m stand | **3.66 MiB** |
| **total per Sphagnum state** | **64.1 MiB** (HUD agrees exactly) |

Where the moss budget actually goes, since the brief asks: **94 % of it is the
view set, and the carpet only ever samples 81 of its 121 layers.** The ground
proxy clamps to the interior of the octahedral square (the horizon ring is
degenerate for a ground plane — see below), so the 40 boundary layers are dead
weight for moss: uploading only the interior 9×9 sub-grid would cut 60.4 → 40.4
MiB per state, 60 MiB across the three. Not done here because it needs a
per-species `GRID_N` and node-offset in the shader and the sampling change is
worth nothing visually. What is NOT starved is tile resolution: the 0.18 m tile
crop is ~175 of 256 texels ≈ 1.0 mm/texel of world, and at the `carpet-close`
bookmark (1 m up) a tile is only ~135 px on screen, so 256 px is already
slightly oversampled at the closest sane viewing distance. 128 px would be the
wrong economy; fewer view directions is the right one.

At the original 128 px the same table totalled **≈ 17.9 MiB / 25 MiB**
(albedo 10.07 MiB + normals 5.04 MiB + instances 2.84 MiB), and the HUD agreed:
17.9 MiB for calamagrostis and poa, 17.5 MiB for elymus (density
2.5 → smaller instance buffer). The dense-mixed stand (density 5) pushes the
instance buffer to ≈4.7 MiB, still ≈20 MiB total. Instance capacity is a real
bound, not a guess: bucket *j* covers an annulus and scatter density is capped,
so `density × area × 1.15 + 2048` cannot overflow in practice, and it is sized
from `REGION_MAX`, never from the stand's plant count.

Transient bake-only allocations (source vertex/index buffers, 768² chunk
targets, readbacks) are tracked with tag `bake-scratch` and destroyed before the
bake returns, so they never sit in the live budget.

## Bake

`bake.ts` renders the raw GCMESH1 mesh once per view node with an orthographic
projection built from that node's basis — the *same* `view_right()` rule the
runtime uses, so reprojection is exact rather than a reconstruction that could
drift. MRT: albedo+coverage and the local-frame octahedral normal, flipped
toward the bake camera so a 2× supersample of a two-sided blade averages
agreeing normals instead of cancelling. Views are rendered 9 at a time into a
768² target (transient VRAM stays ≈7 MB) and resolved on the CPU:
coverage-weighted downsample, 4 dilation passes (colour spreads into empty
texels, alpha stays 0), oct-encode. Mips are then generated on the GPU, one
render pass per (layer, level), coverage-weighted.

Committed artifacts (45.4 MiB each, `mesh/baked/005-octa-impostors/`):
`views-v2-11x256-{calamagrostis-canescens,elymus-repens,poa-pratensis}.bin`
plus `views-v2-11x256-spaghnum-palustre-{wet-vigorous,late-season,sun-exposed}.bin`
for the bog stand. A fresh browser profile loads them straight from disk —
verified (the moss work needed **no re-bake**: the carpet path is a different way
of *using* the existing view set, not a different capture).

The 128 px and 512 px variants exist on disk but are deliberately NOT committed:
the 128 px set is superseded, and each 512 px file is 190 MB, over GitHub's
100 MB per-file limit. Change `TILE` and reload to re-bake any variant.

## Status

working — verified by headless screenshots at `cam=grazing`, `topdown`,
`inside-plant`, `far-horizon`, on the default, close-quality, dense-mixed and
scaling-100m stands, with every debug view. Console clean of WebGPU/shader
errors.

The **bog** stand (moss carpet) is verified separately — see "Moss carpet"
below — at `grazing`, `topdown`, `inside-plant`, `carpet-close`, `far-horizon`, a
30° sloped view, a 0.45 m low eye and a 0.35 m top-down, in `albedo`, `normals`,
`lighting` and `coverage`, with no console errors and **no on-screen toasts**
(one intermediate state did produce a `struct member carpet not found` toast with
a silently missing draw and a plausible-looking image — scrape `.toast`, not just
the console).

`stand=default` is **unchanged by the moss work**: masked before/after pixel
diffs at `cam=grazing` (`det=1&t=3`, HUD/panel/toolbar excluded) differ in
**4 of 179 142 sampled pixels** with a mean delta of 0.000/255, against a
same-code control run that differs in 2. That is expected: the carpet lives in
its own shader modules and pipelines, the upright path is byte-identical, and a
stand with no carpet entry never even creates the carpet pipelines.

Debug views, and what they showed:

- **albedo** — the authored mesh colours exactly as captured (pink heads, green
  blades), unfogged. No light baked into the atlas, so nothing is shaded twice.
- **normals** — real per-fragment normals decoded from the rg8 view atlas, with
  per-blade variation. Directional structure changes with azimuth at grazing,
  and goes dominantly +Y from `topdown` (blades seen from above face up), which
  is the check that the normal atlas is actually decoded and yaw-rotated rather
  than faked.
- **lighting** — bright but *the same brightness as the harness's own terrain*
  in the same mode (sun 1.15 + ambient with half-lambert saturates); the point
  is that it is neither flat nor double-applied. Lighting goes through the
  shared `light_surface()`.
- **coverage** — blended atlas alpha, dark at silhouettes near the cutoff,
  solid inside blades. Meaningful, and it is the same number the alpha test
  uses.
- **depth** — correct near→far ramp; per-plant flat patches are visible because
  an impostor card *is* flat in depth. Honest.
- **viewTint** (own param) — paints each plant by its dominant view id: a wide
  spread of ids at grazing, and a clean radial pattern from `topdown` (view id
  changes with elevation as you move out from the point directly under the
  camera). That is the view set doing its job, visible.

Known limits / honesty:

- 128 px tiles: plants within ~3 m are visibly soft. 121 directions × 128 px is
  the trade this experiment deliberately makes (the thesis is angular coverage);
  the same budget could buy 64 views at 176 px instead.
- calamagrostis and elymus are periodic *community tiles*, so their impostor is
  a baked ~0.5–0.6 m clump repeated per scatter point (the mesh bounds overhang
  the tile, so the card is ~0.75/0.96 m wide). Density therefore reads busier
  than the tiled reference. poa is a true single specimen and is the clean case.
- Views from *below* the horizon are not baked; they clamp to the horizon ring.
  Only the camera-inside case really produces them and that is faded out.
- The 3-view blend ghosts slightly on fast azimuth changes; `viewBlend=nearest`
  trades it for a pop. Judge in motion, not in a still.
- Mip alpha is compensated by relaxing the alpha reference with an estimated
  LOD (one line in the VS), not by a proper coverage-preserving mip rescale.
  Distant coverage is therefore approximate.
- Per-plant maths is recomputed by all 6 vertices of the quad rather than stored
  by the cull pass. That is deliberate: ~80 ALU ops × 6 is cheaper than widening
  the instance record from 16 B to 32 B and re-reading it per vertex.

## Moss carpet (bog stand)

*Sphagnum palustre* in three micro-habitat states is a 0.18 m periodic community
tile, 0.07–0.09 m tall, laid out by the `bog` stand as a grid-snapped mat: 22×22
tiles per 4 m scatter cell (484 slots), constant scale 1.0101 (life size),
90°-only yaw, the three states partitioning the wetness field. ~1.13 M tiles.

Rendered as an upright plant it was indefensible, and the *before* screenshots
say why: a 0.65 m camera-facing card per tile on a 0.18 m grid, of which only
**128 of the 484 slots per cell were ever evaluated** — so about a quarter of the
mat existed, as rows of floating slabs over bare ground, ignoring the slope.

### The idea: the view set is a precomputed raycast, not a billboard

Every baked view is an *orthographic* capture along a known direction `d`, so
`view_k(uv)` is the colour of the first surface hit by a ray in direction `d_k`.
Projecting a point into view `k`'s basis **drops the `d_k` component**, so every
point along one ray maps to the same `uv`. Therefore any proxy surface the ray
crosses gives the same answer: take the tile's own ground plane, project the
fragment's ground position into the view of the *actual* viewing direction, and
the texel you get is exactly what the camera would see along that ray. The
proxy's height is irrelevant to the colour.

So one flat, terrain-conforming quad reproduces the cushion's real relief,
parallax and inter-capitulum occlusion, view-dependently, at 4–5 taps per
fragment. What it cannot reproduce is **depth** (the fragment reports the proxy
plane, off by up to the 3.3 cm of relief along the ray) and the mat's
**silhouette against the sky** at a terrain crest. Nothing else — this is not an
approximation of the relief, it is the relief, sampled on a plane.

Note the corollary, which is why no prism/skirt geometry was added: a *closed*
mat's tiles tile the screen, so any skirt hanging below the plane is occluded by
the plane itself. Thickness is only visible at the mat's own silhouette, i.e. at
crests, at ≤ 7 cm. It is not worth geometry.

### What changed, smallest first

1. **Evaluate every slot.** `cs_cull_carpet` dispatches a third workgroup
   dimension, `ceil(standEntrySlots(entry) / 128)` = 4 for the bog carpet, so all
   484 slots per cell are visited. This alone quadrupled the mat.
2. **4 B per tile instead of 16 B.** A carpet tile is fully described by its grid
   node: `slot 9b | cell x within the region 7b | cell z 7b | quarter turn 2b`.
   Position, scale and yaw are recomputed in the vertex shader from the stand
   table. 958 k tiles cost 3.66 MiB instead of 14.6 MiB.
3. **Capacity from the grid, not from `density`.** `carpetCapacities()` sizes
   each bucket from `carpetDiv²/16` nodes per m² × a measured worst-case zone
   share (1/1/0.8/0.85 by bucket), with the annulus clipped to the stand's own
   square. `density = 8` would have under-sized the far buckets by ~25 % and
   dropped tiles as holes.
4. **One ground-parallel quad of exactly one grid step**, sized from
   `footprint_m × scale_min` (never from `height_scale`, never quantized — the
   packed 12-bit scale would have shrunk the tile 0.1 % and opened a hairline
   crack at every edge). Yaw from the scatter's quarter turn, built as exact
   ±1/0 cosines. No camera-facing card, no near fade (a mat you stand on must not
   open a hole under you), region-edge fade only, measured from the tile centre.
5. **Terrain fitting: ladder rung 3, per-vertex.** Every corner takes its own
   `terrain_sample(xz)` (height and (nx,nz) in one bilinear fetch). Rung 3 and
   not 1/2 for the reason CLAUDE.md gives: neighbouring tiles share corner
   *positions*, so per-vertex is the only rung that keeps the mat C0-continuous;
   a per-tile plane fit is not cheaper here, it is wrong. The tile's shading and
   view **basis** is per-tile (one extra `terrain_sample` at the tile centre),
   because view selection has to be primitive-uniform or the two triangles of a
   quad would disagree; the resulting parallax discontinuity between neighbours
   is ~1 mm. `slope_align` is not read: `carpet_div > 0` already means "full
   conformance", and a mat has no defensible partial-tilt answer.
6. **The view direction is taken into the tile's ground frame** before selecting
   the octahedral node, so the cushion's relief leans with the hillside instead
   of standing bolt upright on it.
7. **The horizon ring of the view set is excluded.** The octahedral square's
   boundary is the horizon, and a horizon view is *degenerate* for a ground
   proxy: `up` is exactly +Y there, so the whole plane projects onto a single
   line of texels. Blending it in at eye level put ~50 % weight on one stretched
   row of pixels. Clamping the octahedral coordinate one node inside bounds the
   sampled elevation at ~19.5°; below that the parallax simply stops deepening,
   which is the standard graceful failure of parallax mapping at grazing.
8. **A carpet-specific alpha reference (0.06 vs the grass 0.38).** A mat is a
   closed surface: the mip chain pulls a tile's coverage toward its own mean, and
   at the grass reference whole distant tiles fail and punch tile-shaped holes.
   `debug=coverage` now shows the mat as solid white — a depth-writing occluder,
   which is what the taste rule wants (more opaque, not dithered).
9. **The normal is used as a zero-mean bump field, not as an orientation.** This
   was the single biggest visual fix and took the longest to diagnose. Two facts:
   the bake flips every stored normal toward *its own* view node, and at
   0.8 mm/texel this moss's normals are pure per-texel leaf noise with no
   structure above ~2 texels (dumped the atlas to check). So the only coherent
   signal in a tile's mean normal is the flip bias, which follows whichever of
   the 121 views that tile picked — and a mat picks ~20 views in one screenful
   (every tile a different quarter turn at a different angle). The field rendered
   as a hard **chequerboard of flat patches**, blatant in `debug=normals`. Fix:
   subtract the atlas normal's own local mean (one extra tap at mip 4 ≈ 13 mm of
   tile), keep the tangential part, and perturb the *ground* normal with it. The
   mat's mean normal is then the ground normal — the honest aggregate for a
   carpet, identical between neighbours — while the roughness that makes the
   surface read as moss instead of paint survives. Subtracting the mean from the
   same view cancels that view's bias exactly, and the residual decays to zero
   under minification, so the far field shades as ground with no shimmer.
   Measured at detail 0 / 0.5 / 0.9: 0.9 (the default) is visibly the roughest
   with no per-tile bias; the pre-fix version at 0.9 was a chequerboard.
10. **The proxy plane height comes from the GRID, not from the species mesh.**
    Each state has its own canopy height (9.1 / 6.9 / 7.2 cm), and using it put
    neighbouring states' planes 1.6 cm apart. At a grazing angle a 1.6 cm step
    exposes ~6 cm of bare peat along the lower tile's far edge — a third of a
    tile — so the mat showed a scatter of convincing tile-shaped "holes" exactly
    where the wetness jitter interlocks two states. (I checked the harness first:
    an f32-exact mirror of the carpet partition over 139 876 nodes reports
    **0 unclaimed and 0 double-claimed**, so the scatter fix is good and the
    holes were mine.) Now `lift = 0.38 × footprint_m` = 6.8 cm for every carpet
    entry sharing a grid, which is also the mean capitulum apex of all three
    meshes.
11. **The sampled crop is inset 10 % inside the tile's period**
    (`carpetCropInset`). The source mesh overhangs its 0.18 m period by
    1.3–2.1 cm, so in a real tiling the outer ~10 % of a tile is covered partly
    by the *neighbour's* overhang, which a single-tile capture cannot contain.
    Sampling to the crop edge therefore renders a coverage-deficient band along
    every boundary: bare peat strips up close and a woven diagonal lattice over
    the whole mid-field at grazing (very visible at `cam=grazing` and on the
    sloped view). Measured 0 / 0.04 / 0.08 / 0.10 — the lattice fades out
    monotonically and is gone at 0.10. The cost is a ~25 % magnification of a
    2–5 mm noise texture, which is invisible; the geometry is untouched (the quad
    is always exactly one grid step), so the lattice invariant holds either way.
    The proper fix is a bake that renders the mesh periodically replicated ±1
    period, i.e. 9× the draws and ~18 min per moss species — and it would change
    the grass captures too, so it is deliberately not done.

Overscale was **not** used: tiles abut exactly at 1.0. With the ray-exact crop
there is no reason to reach for it, and the pilot measured it worse.

`viewBlend=nearest` is worse for a carpet than for a plant and the default
`blend3` stays: a single view is marginally crisper but its coverage has gaps
that the 3-view blend fills, so the 0.45 m low-eye view gains visible dark
strips (compare `/tmp` shots `005final-bog-low` vs `005final-bog-nearest`). For a
ground proxy each of the three views is individually exact, so blending them
costs only a slight softening of the parallax, not a ghost.

### What improved, what is still bad

Before/after headless screenshots at `stand=bog`: `cam=grazing`, `topdown`,
`inside-plant`, `carpet-close`, `far-horizon`, a level view across the steepest
30° slope in the region (`cam=37.65,-0.08,-57.73,-2.5640,0.0299,60`), a 0.45 m
low eye and a 0.35 m top-down, plus `debug=albedo|normals|lighting|coverage`.

- **Coverage**: before, a quarter of the mat as floating slabs over bare ground;
  after, a closed continuous carpet from under the camera to the horizon, on flat
  ground and across the ridged slopes, with the calamagrostis emerging through
  it. `debug=coverage` is solid white over the mat.
- **Terrain**: the 30° slope view shows the mat hugging the hillside with no
  buried or floating edges and no cracks between tiles; before it was sparse
  slabs cutting through the ground.
- **Zones**: `topdown` reads as a proper bog mosaic — green wet hollows, khaki
  late-season flanks, rust sun-exposed crests, interlocking tile by tile.
- **Texture**: at 0.35–1 m the mat is a continuous fine moss speckle at
  ~1.0 mm/texel with real surface roughness in the light term, and no visible
  tile structure at all.
- **Still bad #1, inherent to the subject**: beyond ~2 m the mat is a smooth
  coloured surface. This moss's visual structure is 2–5 mm (capitula are ~5 mm
  apart, leaves ~1 mm), so at 3 m a capitulum is 3 px and at 10 m the whole
  tile is 4 px. No renderer resolves that; the honest ceiling is "correctly
  coloured, correctly shaded, correctly conformed mat". The one thing that
  *could* have carried mesoscale shape into the distance is a capitulum-scale
  normal, and this bake does not contain one (see #9) — it is leaf noise all the
  way down.
- **Still bad #2**: single-tile blockiness at zone boundaries. The wetness jitter
  interlocks the three states node by node, so an area whose wetness sits within
  ±0.03 of a boundary gets a scatter of isolated single tiles of the other state,
  which a discrete tiling can only draw as a hard 0.18 m square. Placement, not
  rendering — but worth knowing it is visible.
- **Still bad #3**: a very faint residual seam at some tile boundaries even at
  inset 0.10, and a 1-texel dark line where two states meet. Same crop-edge
  cause; a periodically-replicated bake is the real fix.
- **Still bad #4**: no distance aggregation. Life size means ~1.13 M tiles on the
  bog stand and the renderer draws every one of them as its own quad out to
  `regionRadius` (128 m by default). It holds up (see below) but a cell-level
  aggregate for the far buckets is the obvious next win.

### Is a view-set impostor suited to a moss carpet?

Better than I expected, and for a reason that generalises. The *impostor* part —
a camera-facing card — is worthless here and had to be deleted, exactly as in
001. But the *view set* part is not a billboard trick: 121 orthographic captures
are a precomputed, O(1) directional raycast of the tile, and a ground-parallel
proxy plane turns that into per-pixel-correct relief with correct occlusion and
parallax, at 4 taps and 6 vertices per tile. That is strictly more than a
top-view card can express, and it costs the same. Where it stops being
distinguishable from a top-view card is not the method's fault: this species'
detail is 2–5 mm, so past ~2 m every method converges on the same coloured mat,
and the 3.3 cm of relief that *would* show at 0.3–1 m produces almost no
lighting signal because the baked normals hold no structure at that scale.

So: suited, yes, with an honest caveat — on this subject the win over
001-billboard-smoke is real but modest (closed mat, no lattice, view-correct
relief, ~1.4× the texel density, per-tile 4 B records), and it is a win in
correctness rather than in something you would point at in a still image. On a
coarser mat — bigger cushions, deeper relief, a species whose features are
centimetres rather than millimetres — the same code would separate from a card
dramatically, because the parallax it computes is exact.

## Findings

**No bench numbers.** Up to four agents shared this GPU while the experiment was
built, so every frame time available to me is contaminated; nothing here is
tuned against measured time and no `results/` JSON is claimed. The structural
claims (region-bounded enumeration, 4 taps/fragment, near-to-far buckets,
count-independence) are from the code, and count-independence is confirmed
visually: the `scaling-100m` stand (134.2 M plants, ±2048 m) renders the same
image with the same VRAM as the default stand (557 k plants).

**Moss timings, same caveat.** On the bog stand (±96 m, ~1.13 M moss tiles, all
drawn individually) the quietest samples were GPU Σp50 1.4 ms at `carpet-close`,
4.1–4.6 ms at `topdown` / `grazing`, 5.2–5.4 ms at the sloped view and 4.8–5.4 ms
at `far-horizon`, of which the carpet cull is 0.4–1.1 ms. Later samples of the
identical frames ran 2–3× slower with sibling agents on the GPU (the harness's own
base pass went from 0.5 ms to 3 ms), so treat all of it as an order of magnitude,
not a number. `default` measured 8.0–9.5 ms Σp50 across five runs in the same
session, i.e. the same band as the 8.8 ms baseline taken before any change — and
the pixel diff above is the real evidence that nothing changed there.

### Harness feedback from the moss round

- `standEntrySlots()` and `stand_table[i].carpet_div / footprint_m / scale_min`
  were exactly what was needed and are what this renderer is driven from. No
  complaints: the three things a tiled mat needs (slots, footprint, one constant
  scale) are all first-class.
- **`slope_align` had no natural insertion point.** For a mat the useful signal
  is already `carpet_div > 0` ⇒ conform fully; `slope_align` would only matter if
  a carpet could be partially upright, which is not a thing. It is read by
  nobody here, deliberately.
- **The base pass draws the terrain coarser than `terrain_sample()` evaluates
  it.** `terrain-draw.wgsl` uses a 256×256 grid over 256 m (1 m triangles) while
  `terrain_sample()` is a bilinear evaluation of a 0.5 m heightmap, and the
  heightfield's finest FBM octave is 2 m / ±0.14 m. The two surfaces therefore
  differ by up to ~10 cm in curved spots, which is more than a 7 cm mat's whole
  height: anything that conforms to `terrain_sample()` can be *buried* by the
  drawn terrain. It cost a wrong hypothesis here (the 25 cm lift probe) and it
  will bite every carpet renderer. Either draw the terrain at heightmap
  resolution, or expose the drawn surface as a primitive
  (`terrain_height_drawn(xz)` doing the 1 m-triangle interpolation) so a mat can
  sit on the ground people actually see.
- **A periodic tile's capture cannot be seamless without replication.** The
  source meshes overhang their period by 1.3–2.1 cm, so every single-tile bake is
  coverage-deficient in a ~10 % band at each edge and every carpet renderer will
  independently discover the resulting lattice (001 reported it too). The mesh
  export knows the tile is periodic (`geometry.periodic`, `tile.sizeX/Z`) — a
  bake helper that renders a periodic mesh ±1 period, or a documented convention
  that periodic geometry is pre-wrapped into `[0, tile]`, would fix it once for
  everybody instead of costing each experiment a magnification hack.
- **The zone jitter interlocks at node granularity**, which for a discrete
  carpet means isolated single 0.18 m squares of a neighbouring state. Not wrong,
  but a renderer cannot soften it; if the intent is a gradual ecotone, the
  jitter wants to be spatially correlated (a few nodes wide) rather than
  per-node.

A/B links:

- moss carpet vs the billboard baseline:
  `#/ab/001-billboard-smoke/005-octa-impostors?stand=bog&cam=grazing&seed=42`
  (and `cam=carpet-close` for tile-level detail)
- vs ground truth: `#/ab/005-octa-impostors/000-ground-truth?stand=default&cam=grazing&seed=42`
- vs the billboard baseline: `#/ab/005-octa-impostors/001-billboard-smoke?stand=default&cam=topdown&seed=42`
  — top-down is where the view set earns its keep: 001 has a single top card,
  this has a whole neighbourhood of directions around straight-down.
- blend vs hard switch: `#/run/005-octa-impostors?cam=grazing&p.viewBlend=nearest`
