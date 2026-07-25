# dissolving canopy

## Idea

One rule drives the whole method: **how continuous a canopy looks depends on the
viewing elevation, not just on distance.** Looking along a meadow you resolve
individual plants a hundred metres out; looking down at the same meadow you see
one closed surface almost immediately. So the renderer keeps three
representations of the *same* stand and dissolves between them with a field
`d = smoothstep(0.5r, r, dist)`, where the radius `r` itself interpolates from
`dissolveGrazing` (106 m) to `dissolveOverhead` (16 m) by the viewing elevation
of the point being drawn:

1. **Near — a plant is a cloud of 4 sub-tuft splats.** The bake partitions the
   mesh's *triangles* into the four xz quadrants of the plant and captures each
   quadrant from 8 azimuths. At draw time each quadrant is a camera-facing quad
   anchored at *its own* centre, so the four sit at four different world
   positions and depths. That buys what one card cannot have:
   - parallax INSIDE one plant (near quadrants slide against far ones),
   - a silhouette that changes shape with view direction instead of rotating
     rigidly, because the quadrants occlude each other differently,
   - real self-occlusion (they depth-test against each other) plus a shading
     term that darkens the quadrant pointing away from the camera, because it
     is deeper inside the volume,
   - reprojection error that scales with a node's radius, so a quarter-plant
     node is ~2x more faithful off its baked azimuth than a whole-plant card,
   - wind evaluated at each quadrant's own position, so a plant twists instead
     of shearing as one flat card.
2. **Mid — one whole-plant splat** (atlas node 0). Past ~12 m the intra-plant
   parallax is a couple of pixels, so it is not worth paying for.
3. **Far — one continuous canopy shell.** A bufferless two-level snapped grid
   (0.75 m out to ±36 m, 3 m out to ±144 m, clipped to the stand) displaced by an
   analytic canopy relief and textured with ONE tileable canopy texture,
   composited at init from the baked per-species top-down captures at the active
   stand's densities. Its coverage threshold starts above 1.0 near the camera —
   nothing can pass, and those quads are collapsed in the vertex stage — and
   falls to `shellFloor` far away, so the same texture reads as "islands of dense
   clumps" through the hand-off band and as a closed canopy beyond it. This is
   the LOD collapse with teeth: thousands of distant plants become a few thousand
   triangles and 2 texture taps, and from a high camera the splat set empties out
   completely (the cull rejects every plant whose fade reached zero), leaving
   only the surface.

A **carpet species** (stand `carpet_div > 0`, i.e. the Sphagnum mat) does not go
through that ladder at all — it has no silhouette to splat and no canopy to
dissolve into islands. It gets a fourth representation, one ground-parallel tile
quad with the cushion's relief expressed per texel; see "Moss carpet" below.

Two physical corrections make the shell agree with the splats instead of looking
like a different biome:

- **Path-length coverage.** The baked coverage is what the canopy hides looking
  straight *down*; a ray at elevation θ travels 1/sin θ as far through the same
  layer, so its coverage is `1-(1-c)^(1/sinθ)`. One `pow()` closes the far field
  at grazing (a meadow is see-through from above and opaque edge-on) and, since
  it can never exceed 1, it never resurrects the near field the dissolve
  threshold killed.
- **View-dependent albedo and normal.** From above, the visible surface *is* the
  canopy top: use the composited top-view colour and the surface normal. At
  grazing you see the sides of blades, which statistically face the viewer, so
  the shading normal rotates toward the camera (the same "flip toward the bake
  camera" convention the splats' baked normals use) and the albedo becomes the
  baked *side-view* mean hue — split into a flower-head band and everything
  below it, picked per texel by the baked canopy height, and darkened by a
  canopy-interior factor because an opaque shell has to stand for the whole
  self-shadowed volume behind the first blade. Without these the far field came
  out as a bright olive-tan carpet against a pink-green near field.

O(1) in plant count: nothing is ever materialized for the whole stand. Per frame
the cost is the region area (dissolve radius) plus the shell's screen area —
identical for 557k plants (default) and 134.2M (scaling-100m), verified below.

## VRAM budget math

Per species (HUD-verified on the default stand: 16.4 MiB):

- coverage atlas r8 1280x4096 + 6 mips — 6.96 MiB. This is the alpha-tested
  silhouette, so it gets full resolution: ~500 texels/m, i.e. billboard parity.
- albedo atlas rgba8 640x2048 (half res) + 5 mips — 6.96 MiB (a = coverage, so
  the mip chain stays coverage-weighted)
- oct normals rg8 320x1024 (quarter res) + 4 mips — 0.87 MiB
- culled instances, sized for the param maxima (36 m tuft radius, 112 m region)
  at the entry's density, 16 B/instance — 2.30 MiB at density 3, 3.5 MiB at 5
- entry info uniform (336 B) + indirect args (32 B) — noise

Shared by all species (tagged `canopy-shell`, deliberately NOT charged to a
species row): the canopy tile, 2 x rgba8 768x768 + 9 mips = **6.29 MiB**, i.e.
2.1 MiB per species if amortized. Splitting the channels by the resolution each
actually needs is what pays for it: a plain rgba8 atlas at the same silhouette
resolution would be 21 MiB per species.

Totals: default stand 16.4 / 16.0 / 16.4 MiB per species + 6.3 shared → ~18.5
MiB effective per species. Worst defined stand (dense-mixed, density 5): 17.8 /
17.1 / 17.1 + 6.3 → ~20 MiB. Inside the 25 MiB budget in every stand.

**Carpet species spend the budget somewhere else entirely.** A species that only
ever appears as a carpet in the active stand never draws a splat, so its node
atlas is not uploaded at all — 14.8 MiB of side views of a 0.09m cushion, which
is the one thing a mat does not show. (The CPU-side node bake is still read,
because the shell's canopy composite uses its straight-down capture.) What it
buys instead:

- carpet tile rgba8 512² + 7 mips, x2 planes (albedo+coverage, oct normal +
  height + occlusion) — 2.8 MiB, i.e. 2844 texels/m across the 0.18m tile
- culled instances, 484 slots/cell over the region at the entry's wetness share
  — 7.1 MiB

HUD-verified on the `bog` stand: **9.8 / 25 MiB per moss species**, 74.5 MiB
total. For comparison, 001-billboard-smoke on the same stand reports 25.8 / 25
per moss species and 124.2 MiB total, because it keeps the full side-view atlas
for a species that only draws a ground quad.

The bake uses ~150 MB of transient GPU memory (raw vertex/index buffers, a
2560x2048 supersampled MRT chunk, readbacks), allocated via `ctx.res` tagged
`bake-scratch` and destroyed at the end.

## Bake

`bake.ts` → `mesh/baked/026-dissolving-canopy/canopy-v2-<species>.bin`,
11.35 MiB each (34 MiB for the three species; no stale variants).

Per species, in one render pass per two azimuth rows (chunked to cap transient
VRAM):

- **Node atlas**, 5 nodes x 8 azimuths, 256x512 px tiles. Node 0 is the whole
  plant; nodes 1..4 are the four xz quadrants of the *triangle* set, partitioned
  by centroid, so the four tiles reassemble the plant exactly. Each node's box is
  its own content bounds (a cylinder radius for the width, so a camera-facing
  quad is valid from every azimuth) with a 3% margin. Rendered at 2x and reduced
  three ways: coverage 2x2 → full res, colour 4x4 coverage-weighted → half res,
  normals 8x8 coverage-weighted → quarter res; colour and normals are then
  dilated a few texels into empty space so filtering never pulls in background
  black. Normals are flipped toward the bake camera (two-sided foliage). All
  nodes share one azimuth grid (see the findings: rotating them per node broke
  blades that cross a quadrant boundary).
- **Top-down capture** of the whole plant, 160 px: albedo+coverage, oct normal,
  and the height fraction of the topmost surface. `canopy.ts` composites these
  into the shell's tileable canopy texture at init — not at bake time, because
  the species mix and densities belong to the stand, not to a species. Stamps use
  the shared PCG hash (deterministic), one of 8 free orientations (no resampling
  blur), torus wrapping, coverage accumulated as independent occluders
  `1-Π(1-c)`, and colour/normal/height composited front-to-back (pass 1 finds the
  canopy top per texel, pass 2 weights every copy by its coverage attenuated by
  how far below that top it sits).

A SECOND artifact, only for periodic carpet species:
`mesh/baked/026-dissolving-canopy/carpettile-v2-<species>.bin`, 2.0 MiB each
(6.0 MiB for the three Sphagnum states; the v1 files are deleted).

- **One top-down capture of exactly the tile square**, 512 px over 0.18 m
  (2844 texels/m — 1.9x the billboard baseline's cropped top view, 6x the same
  tile inside the 160 px whole-mesh capture above).
- It is genuinely SEAMLESS: the mesh is a periodic community tile whose geometry
  overflows its period (0.21 x 0.23 m of bounds inside a 0.18 m tile), so the
  bake draws it **9 times at the 3x3 wrap offsets** and keeps the tile square.
  What leaves one edge arrives at the other. (`bake.wgsl` reads the wrap period
  from a previously-unused uniform lane, so every other view is untouched.)
- Channels: rgb albedo + coverage; oct normal, the height of the topmost
  surface, and a **baked cavity occlusion** — 8 azimuths x 3 radii of horizon
  obscurance over the tile's own height field, clamped at a 45° horizon and
  torus-wrapped. Occlusion is baked rather than derived from the height in the
  shader because a term derived from the height is non-linear in it and does not
  survive mip filtering: as the chain flattens the height toward its mean, a
  derived cavity term collapses to "open" and the mat brightens with distance.

Mip chains are generated on the GPU at load; tile sizes stay even at every
generated level, so a 2x2 box never crosses an atlas tile. The oct-normal variant
decodes, averages and re-encodes rather than box-filtering oct pairs, and the
carpet's aux variant additionally averages the occlusion channel. Every load
is magic+size validated, because the dev server answers missing /mesh/baked files
with index.html at 200.

## Status

working — verified by headless screenshots at grazing, topdown, inside-plant and
far-horizon on the default stand, the same four on scaling-100m (134.2M plants),
dense-mixed, all five debug views, and A/B wipes against 001-billboard-smoke.
Zero console errors or warnings.

The `bog` stand (Sphagnum carpet) is verified separately: grazing, topdown,
inside-plant, `carpet-close`, a mid-height oblique, a low view across the ridged
slopes, and `debug=normals/lighting/coverage/albedo`. The `default` stand is
**pixel-identical** to its pre-carpet render at `cam=grazing` and
`cam=inside-plant` (2-3 differing pixels out of 250k, and those only where AA
lands on an edge) — the carpet path adds no pipeline and no draw when the stand
has no carpet entry.

## Findings

### Performance (A/B, in-frame, CONTENDED — ratios only, never the ms)

`#/ab/001-billboard-smoke/026-dissolving-canopy?cam=grazing&seed=42` at
1280x800, summing `A/cull + A/cards` vs `B/cull + B/canopy`:

| cam | ratio B/A |
|---|---|
| grazing | 1.22x |
| topdown | 0.66x |
| far-horizon | 1.07x |
| inside-plant | 1.08x |

~1.0x on average, worst case 1.22x, and 1.5x *faster* than the baseline from
above.
15 sibling agents were rendering on this GPU throughout, so absolute numbers are
worthless — the `composite` pass (a fullscreen blit) alone reported 2.5-3.5 ms.
No `results/` bench JSON is claimed for the same reason; rerun
`#/bench/026-dissolving-canopy?stand=default` on an idle GPU before quoting ms.

Sweeps at grazing (`b.shell=0`, `b.tuftRadius=6..20`) moved the ratio by less
than the run-to-run noise (1.21-1.24x), i.e. neither the shell nor the near band
is where the time goes — the far band (one splat per plant out to 106 m) is, and
that is the same work the baseline does with its cards.

Plant-count independence: default (557k) vs scaling-100m (134.2M) at the four
standard cams — 6.5/2.8/6.2/6.5 vs 6.3/3.3/8.4/6.2 ms Σp50, i.e. equal within
the contention noise, same VRAM.

### Looks — does it beat the baseline?

Yes at every camera, with one honest caveat.

- **Near field (grazing, close-down, inside-plant): clearly better.** The splat
  cloud gives visible depth layering — panicles occlude other panicles, stems
  read at different depths, and the back-quadrant shading makes plants look like
  volumes. The baseline's worst artifact is simply gone: no pale floating top
  cards (light beige discs all over its close-down view). Both methods soften
  under magnification (<1 m), and there the baseline is the sharper of the two:
  its silhouette carries ~690 texels/m of one unbroken capture against my ~500
  split across four, so from *inside* the canopy its panicles resolve finer
  spikes than mine. Everywhere further out my depth structure wins.
- **Topdown: better, and 1.5x cheaper.** The baseline's top cards read as
  discrete bright discs; the shell is a continuous clumped canopy with
  height-driven occlusion and a one-tap heightfield sun shadow, and the splats
  are gone entirely (the elevation-driven dissolve rejects them in the cull).
- **Far field: better in coverage, weaker in character.** The baseline stops at
  its region radius and shows bare terrain past it; the shell carries the canopy
  to the stand edge. But a horizontal surface cannot show the *vertical*
  structure that gives a distant meadow its texture, so past ~106 m mine reads as
  a well-textured mat rather than as grass. That is the one place the baseline's
  cards look more alive, and it is exactly why `dissolveGrazing` ended up at
  106 m: the splats are better at grazing, so they keep the field, and the shell
  takes the deep distance and the high-elevation views, where it is better than
  they are.

## Moss carpet (bog stand)

Sphagnum palustre is a 0.18 m periodic community tile, 0.07-0.09 m tall with
3.3 cm of capitulum relief, laid out by the `bog` stand as a life-size mat
(22x22 tiles per 4 m cell = 484 slots, constant scale 1.01, 90°-only yaw). Run
through this experiment's own representation it was indefensible: **four
camera-facing quarter-plant cards standing upright out of a cushion**, sized
from the plant's capture radius, ignoring the slope — and only about a quarter
of the mat present, in bands, because the cull enumerated `SCATTER_MAX_PER_CELL`
(128) of the 484 slots. See `before/bog-carpet-close`: a fan of vertical brown
blades radiating out of bare olive peat.

What changed, smallest first, all inside this experiment:

1. **Enumerate `standEntrySlots(entry)`, not the scatter budget.** The cull's
   per-cell slot count now comes from the entry (`info.cfg3.y`); 128 visited
   only tile rows 0-5 of 22 in every cell. Capacity stays a separate number,
   sized for the entry's expected survivors (its wetness share), not for all
   484 — that is 3x the memory for the same picture.
2. **A carpet gets its own primitive: one ground-parallel tile quad.**
   `carpet_div > 0` routes the entry to `shaders/carpet.wgsl` — 6 vertices, no
   camera-facing card, no near/far band split (a tile is one quad at every
   distance), no camera-inside fade (a mat you are standing on must not open a
   hole under you). Width comes from `stand_table.footprint_m * scale`, never
   from the height; the yaw is snapped back to an exact quarter turn from the
   packed angle and the scale is read straight from the stand, so neighbouring
   tiles abut to the float and the lattice is untouched. Overscale is 1.0.
3. **A carpet alpha reference of 0.06** instead of the params' 0.45. A tile's
   coverage is ~95% up close but the mip chain pulls it toward the tile mean,
   and at the grass reference whole distant tiles fail the test and punch holes
   in ground that is fully covered. At 0.06 the mat is a solid depth-writing
   occluder and only the genuine gaps down to the peat open (`debug=coverage`:
   white with dark filaments, no tile-sized holes).
4. **Terrain fitting: ladder rung 3, per-vertex.** Every corner does its own
   `terrain_sample(xz)` — height and the ground normal's (nx, nz) out of the
   same four texel loads, so the shading basis is free. Rung 3 and not 1 or 2
   for a specific reason: neighbouring tiles SHARE corner positions, so a
   per-vertex fit is the only rung that keeps the whole mat C0-continuous. Any
   rung that fits one plane per tile leaves a wedge-shaped crack at every tile
   boundary, because two neighbours pick two different planes. The baked normals
   are then lifted into the local ground frame (mesh +x -> tangent, +y -> ground
   up, +z -> bitangent), not merely yawed, so a mat on a slope lights like a
   slope.
5. **The budget moved to where a cushion actually shows.** A carpet-only species
   no longer uploads its 14.8 MiB node atlas (40 side-view tiles of a 0.09 m
   mat, 512 px of vertical resolution for 9 cm — the budget was almost entirely
   in the dimension a mat has nothing to say about). It gets a 512 px top-down
   capture of the tile square instead: 2844 texels/m, 2.8 MiB, and the species
   row drops from 19.9 to 9.8 of 25 MiB.
6. **Thickness, which is where a card cannot follow.** The tile capture stores
   the height of the topmost surface and a baked cavity occlusion, and three
   things use them: cavity occlusion (crevices between capitula go dark),
   a one-tap self-shadow along the sun azimuth (capitula cast onto their
   neighbours), and a single-step parallax offset of the sample point, so the
   cushion slides against itself as the camera moves. Together they are what
   turns "a photograph of moss lying on the ground" into "a lumpy mass".
7. **The far field agrees with the mat.** `compositeCanopy` stamped carpet
   entries as random plants at their stand `scaleMin..scaleMax` — but those are
   placeholders for a carpet, so the moss went into the shell at ~2x life size
   and put the shell's surface 0.20 m above a 0.07 m bog. Carpet entries now
   stamp on their own grid, at the carpet scale, from the tile capture, with
   quarter-turn orientations, and the three states partition the grid by a
   NODE-only hash (hashing the entry index in is exactly how the harness's own
   zoning once grew holes and double stacks).

### What it looks like (screenshots, bog stand, seed 42)

- **`carpet-close` (1 m straight down):** a continuous cushion with visible
  capitulum-scale texture and dark crevices, no tile seams anywhere. Against
  001-billboard-smoke at the identical camera the difference is stark: 001 shows
  a dark grid line at every 0.18 m tile boundary (its capture is cropped from a
  whole-mesh top view, so tile edges sample the sparse overhang) and a noisier,
  flatter speckle. Mine has no grid at all — the capture is wrapped, so its
  edges match by construction — and more legible mound structure.
- **oblique / mid (0.3-2.8 m up):** a closed mat draping the terrain, mottled
  green-to-rust where the three wetness states interlock, cushions reading as
  3D masses with lit tops and shaded flanks.
- **across the ridged slopes:** the mat follows the relief with no cracking and
  no floating edges. The pale khaki zones in the mid field are the sun-exposed
  state, and 001 renders the same zones the same colour at the same 4 m-stepped
  zone boundaries — that boundary staircase is the harness's wetness field
  (its slope damping clamps at the heightmap texel scale), not a renderer bug.
- **topdown (42 m):** the elevation-driven dissolve has already handed the whole
  frame to the shell at 16 m; it now reads as a fine moss-coloured mat rather
  than a green meadow.
- **`debug=`:** normals show real per-texel variation around ground-up (not a
  flat up-normal); albedo is the baked capture, unlit; coverage is white with
  thin dark filaments; lighting is bright but varies, and its **near-to-far
  drift is +8%** (222 -> 241 mean over four distance bands of one frame, of
  which the occlusion terms are ~1.5% and the rest is the normal flattening),
  with **albedo flat to within 1%** across the same bands (52.1 / 52.9 / 52.6 /
  52.5). That is the per-band check CLAUDE.md asks for, and it says the stored
  colour and the light term are both behaving.

### Performance (bog stand, own passes only, CONTENDED — ratios only)

15 sibling agents were rendering throughout; the fullscreen `composite` blit
alone read 1.0-4.5 ms across runs, so absolute ms are worthless. Same-window
paired samples of `cull + draw`:

| cam | 026 | 001 | ratio |
|---|---|---|---|
| grazing | 4.23 | 3.90 | 1.08x |
| carpet-close | 2.08 | 1.01 | 2.06x |
| topdown | 2.45 | 4.14 | 0.59x |

Comparable at grazing, 1.7x cheaper from above (the dissolve empties the tile
set at 16 m and only the shell remains), and 2x the baseline at 1 m straight
down — where both are ~2 ms and the difference is the shell's grid vertex stage
plus 4 taps against the baseline's 2. VRAM per moss species is 2.6x lower.

### Is this representation suited to a moss carpet?

Partly, and the honest split is: **the splat cloud that is this experiment's
actual idea is useless for a mat, and the parts around it are good at it.**
Four camera-facing quarter-plant billboards express intra-plant parallax and a
view-dependent silhouette; a 7 cm cushion has neither, and standing them upright
is the single worst thing you can do to it. So a carpet bypasses the method's
centrepiece entirely and uses a fourth primitive.

What the method does contribute is real: the elevation-driven dissolve is a
genuinely good fit for a mat (a carpet IS a continuous surface, so collapsing it
into one at 16 m of elevation costs nothing and saves the most), and the
shell is a better far-field for a mat than for grass, because a horizontal
surface is what a mat actually is. And the top-down tile capture with baked
relief beats the flat baseline card at the one thing the baseline admits it
cannot do — thickness.

### Still bad

- **No silhouette.** The quad is flat; the cushion's 3.3 cm of relief is faked
  per texel, so at a very low grazing angle the mat still ends in a straight
  edge against the terrain and the cushions have no profile against the sky.
  Fixing that needs geometry (a displaced micro-grid per tile), and a displaced
  tile cracks against its 90°-rotated neighbour, so it is not a small change.
- **Single-step parallax has a ceiling.** It only stays coherent while the offset
  is smaller than the lateral correlation length of the height driving it. At
  mip 3 with a 20 mm clamp it combed the near field into vertical streaks at a
  low oblique angle; at mip 5 (11 mm texels) with an 8 mm clamp it reads as
  sliding mounds, but a little directional smear survives on the steepest ones.
- **The far field is flatter than the near field.** ~8% brighter and much lower
  contrast, because the normal and the relief both average out. The baked
  occlusion removed the part of that drift which was a shader-side non-linearity;
  the rest is inherent to a mip chain.
- **The shell's canopy tile repeats every 12 m,** and for a mat that repeat is
  more visible from straight above than it is for grass, because a mat has no
  vertical structure to break the pattern up.
- **The dissolve radii are one pair of numbers for the whole stand.** A mat could
  hand over to the shell far earlier than a grass canopy (a 0.18 m tile is ~2 px
  at 70 m and sub-pixel past ~140 m), but the shell's coverage ramp is
  stand-global, so shortening the carpet alone would open a bare band between
  them. The carpet is therefore drawn to the same radius as the splats and the
  deep field is a lot of sub-pixel quads.

### Things that were wrong along the way (all fixed, all instructive)

- **Parallax destroys the mip level.** The offset varies per fragment, so an
  implicit-derivative sample after offsetting reads a scrambled footprint and
  lands one to two mip levels too coarse: switching parallax on made the whole
  mat visibly SOFTER than switching it off, which is the opposite of the point.
  `textureSampleGrad` with the quad's own (unoffset) uv gradient fixes it
  exactly. Worth remembering: any technique that perturbs a uv must carry its
  gradients by hand.
- **Reading the parallax height at full resolution scrambles the detail.** A
  one-step offset warps the texture by whatever the height says; per-texel
  height warps it per texel, which is mush. Read the height from a deliberately
  coarse mip and the mounds slide while the fine capitula ride along undistorted.
- **The pale khaki mid-field on the bog stand is not a bug** — it is the
  sun-exposed state, and 001 renders the identical zones in the identical
  colour with the identical 4 m-stepped boundaries. I chased it through
  "shell off", "capacity x3" and per-band statistics before comparing against
  the baseline; comparing first would have cost one screenshot.

### Things that were wrong along the way in the original build (all fixed)

- **Two render passes cost a full pass worth of contention noise.** The first
  version timed `tufts` and `shell` separately and measured 2.2x; merging them
  into one pass — which is also the right ordering: near band, far band, then the
  surface behind them, so early-z does the culling — brought it to 1.2x. The
  radial band sort is what makes that ordering free: the cull writes each band
  into its own segment of the instance buffer and each band binds its own view of
  it, because WebGPU forbids a non-zero `firstInstance` in indirect draws without
  an optional feature.
- **A top-down capture of grass is only 5-15% covered.** Blades are edge-on from
  above, so the composited canopy tile is ~49% covered and the shell was
  invisible at grazing (its threshold sat above the coverage) — for a while I was
  looking at bare terrain and calling it the shell. The path-length correction is
  the physically correct fix and costs one `pow()`.
- **The mean of that capture's height channel is a third of the plant height**
  (the wide lower leaves dominate the view from above), which sank the shell into
  the ground. It now uses the 85th percentile of the per-texel canopy top for the
  surface height and the 97th for the height normalization.
- **Two hues, not one.** A single mean side colour makes the far field either
  khaki (flower band only) or green (whole plant). It now mixes the two bands per
  texel by the baked canopy height, with the gain chosen so a fully mip-averaged
  sample lands on a flower-dominant mix — because at grazing the heads are the
  outermost layer a horizontal ray hits, which is also why alpha-tested cards
  keep their heads and lose their stems at distance.
- **Rotating each node's azimuth grid was a bad trade.** Offsetting node j's 8
  views by j/5 of a step spreads the view-switch pop beautifully — but it also
  means two adjacent quadrants reproject their *shared* blades from azimuths up
  to 9 deg apart, and at 1 m that displaces the two halves of one panicle by
  ~30 px. Close up, plants looked like disconnected cauliflower lumps. All nodes
  now share one azimuth grid (bake v2): blades stay connected, and the pop is
  coherent, i.e. no worse than the baseline's single-card pop.
- `macro` is a WGSL reserved keyword (worth adding to CLAUDE.md's list).

### Known limits, left as honest behaviour

- Beyond `dissolveGrazing` (106 m) the far field is the shell, which has no
  vertical structure. With fog it reads acceptably; without fog it would not.
- The shell's relief is analytic value noise, not the actual per-plant canopy, so
  its lumps do not line up with the texture's clumps. Invisible past ~45 m, and
  it is what makes the two grid levels agree exactly on the surface (no crack, no
  swimming) — a texture-driven displacement at two different mip levels would
  not.
- The 12 m canopy tile repeats. A slow macro tint variation and the relief noise
  break it up; at grazing it is invisible, from 42 m straight down you can find
  it if you look for it.
- Magnification below ~1 m goes soft, like every impostor method here.
- The dissolve frontier is camera-relative, so walking forward makes distant
  clumps dissolve in and out. It is spread over ~50 m at grazing, so the
  per-frame change is small, but it is there.
- The per-entry info uniform (336 B x 3) is rewritten every frame although only
  the planes, region rect and params change; the node table never does. Not worth
  the field-order churn.
- The shell's fine grid level (±36 m) is entirely skipped at grazing (the
  dissolve kills everything inside 47 m there) but its 9216 instances still run
  the vertex stage. It is needed for high-elevation views, and a CPU-side test
  would have to know the per-cell elevation, so it stays.

### Harness wishlist

Mostly none — `bakedArtifact` + `commitBake` + the shared scatter twin covered
everything, and `standEntrySlots` / `footprint_m` / `carpet_div` were exactly
the right shape for the carpet work. Four things cost time:

1. **`carpetScale()` is not exported from `@harness`.** `stand.species[i]`
   carries the placeholder `scaleMin/scaleMax` while the GPU `stand_table` gets
   the computed carpet scale, so any CPU-side code that needs the real scale has
   to re-derive `SCATTER_CELL_SIZE / carpetDiv / tileM` itself. Two places in
   this experiment do, and `compositeCanopy` silently used the placeholder until
   I noticed the shell sitting 20 cm above a 7 cm bog. Either export
   `carpetScale`, or resolve the entries' scales in `ctx.stand` so the CPU sees
   the same numbers the GPU does.
2. **No way to know an entry's expected survivor fraction.** For a carpet the
   three states partition the wetness axis, so capacity should be sized by each
   entry's share of the nodes; `wetWidth` is a proxy for it, but the wetness
   field is not uniformly distributed, so the proxy can be off by 2x either way.
   `ctx.scene.scatter.wetness()` exists and I could sample it, but a harness-side
   `expectedSlotsPerCell(entry, seed)` would let every renderer size its buffers
   from the truth instead of a guess.
3. **A periodic tile's ORIGIN and overflow extent live only in the mesh header.**
   `SpeciesDesc.tileM` gives the period, which is enough to size a quad, but a
   wrapped tile bake also needs the tile origin and how far the geometry
   overflows it, and those require loading the 479 MB mesh. Mirroring
   `tileOrigin` (and the bounds) onto `SpeciesDesc` would let a renderer decide
   what to bake without paying for the mesh first.
4. **The wetness field's zone boundaries are hard-stepped at the heightmap texel
   scale.** `scatter_wetness` damps by `flat = clamp((THRESH - slope²)/THRESH)`,
   and the clamp turns a bilinear field into a hard contour, so on the `bog`
   stand the three moss states meet along a visible ~4 m staircase rather than
   interlocking. Every renderer shows the same staircase (001 included), and the
   carpet jitter (±0.03) is far too small to break it up. It is the one thing
   that stops the bog reading as ecology at mid distance.
