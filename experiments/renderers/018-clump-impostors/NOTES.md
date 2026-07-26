# clump impostors

## Idea

A billboard is flat because a plant is drawn as **one** quad. So don't draw one.
That is the whole method, and it has two shapes — an upright plant is cut into
sub-clumps that stand at their real offsets, and a **carpet tile is cut into
ground-parallel shells that lie at their real heights**. The carpet half is
written up under "Sphagnum: the carpet path" below; everything until then is the
grass path, unchanged.

## Idea (upright plants)

Cut the plant into K_SUB = 4 **sub-clumps** and give each its own camera-facing
card, standing at that sub-clump's real offset inside the plant.

Four cards ~0.2 m apart in world space buy every depth cue a single card cannot:

- **parallax inside the silhouette** — the cards are at genuinely different
  depths and positions, so moving the camera slides the near sub-clump across
  the far one. It is not an effect, it is real 3D placement.
- **a silhouette that changes with view direction** — the union of four offset
  shapes re-arranges as you orbit (a sub-clump that was hidden behind another
  swings out to the side). A single card can only rotate.
- **self-occlusion and interleaving** — each card writes depth, so sub-clumps
  occlude each other *and* weave into the neighbouring plants' sub-clumps
  instead of stacking like cutouts. This is the cue that makes a meadow read as
  a volume rather than a deck of cards.
- **ground contact** — four bases touching the soil at four places instead of
  one card standing on a line.
- sub-clumps sway slightly out of phase with each other (`swaySpread`), which is
  what sells them as separate masses in motion.
- as a bonus, the 8-azimuth snap 001 pops on is spread over four cards that
  switch view at four different camera angles, so each pop is a quarter the size.

The cut has to be invisible, and that is where the source meshes cooperate: the
generators emit **one blade at a time**, so a whole blade is exactly a maximal
run of consecutive vertices with no positional jump. Segmenting on a 6 cm jump
threshold yields 158 / 851 / 221 strands for calamagrostis / elymus / poa, with
0.05–0.26 % of triangles straddling a run (those follow their first vertex).
Cutting on **strand** boundaries — never on triangles — means no blade is ever
sliced in half between two cards. Strands are then k-means clustered (weighted by
vertex count, deterministic angular seeding, no RNG) into 4 spatial clumps.

Three more things carry their weight:

1. **Per-view tight crops.** After the capture, every tile is scanned for its
   exact alpha bounding box and the runtime quad spans only that. No empty margin
   is ever rasterized — this is what pays for most of the extra cards, and the
   merged card benefits too (001 rasterizes its margin).
2. **Coverage-preserving alpha, on every mip level.** Splitting a clump splits
   each texel's partial coverage 4 ways, and a hard alpha test is non-linear: two
   half-covered sub-clump texels can both vanish where the one merged texel
   survived. So each tile's true geometric coverage is measured from the
   supersampled capture, and each mip level's alpha is scaled by one scalar (a
   histogram quantile) so the fraction of texels passing the alpha reference
   equals that truth. Sub-clump and merged cards then have the same density, the
   LOD switch does not pop, and grass stops thinning out with distance — with no
   dithering anywhere.
3. **Front-to-back rings.** The cull bins every survivor by distance into
   equal-area rings (3 near + 4 far) and each ring is its own indirect draw,
   issued nearest first, so hard alpha test + depth write let the near rings
   occlude and early-z reject the deeper ones. (Honesty note: the 25 % win I
   first measured for this was an artifact of the `firstInstance` bug below. Its
   real benefit is unmeasured — see Findings.)

Per frame: one compute pass evaluates the shared scatter WGSL twin over a
camera-centered cell region (cell rect CPU-clamped to the stand), rejects whole
cells against the region circle and the frustum before touching the scatter,
frustum-tests each plant, and compacts survivors into the ring lists. Then one
render pass, 7 indirect draws per stand entry. Cost is O(visible region), never
O(plants): the `default` stand (557 k plants) and `scaling-100m` (134 M) measure
identically.

**LOD.** `lodDistance` (default 14 m) is the collapse: inside it a plant is 4
sub-clump cards + 4 top cards; outside it a plant is exactly one merged
whole-plant card + one top card, i.e. a plain billboard. Distant plants therefore
cost a quarter of the vertices and a smaller quad, and the far field is *cheaper*
than 001's because of the tight crop. `lodDistance = 0` collapses the method to a
pure billboard renderer, which makes an honest self-A/B control.

Fragment shader: 2 texture taps, hard alpha test, depth write, no `frag_depth`,
no loops, no marching, no dither. Camera-inside erosion is per **card** (a
sub-clump card can stand 0.25 m off the plant axis, so a per-plant test happily
leaves one card 0.1 m from the eye as a screen-filling smear); top cards erode
from twice that distance and use a **signed** elevation test, which removes the
"pale pancake overhead" artifact 001 shows at eye level.

## Sphagnum: the carpet path

A carpet entry (`stand_table[i].carpet_div > 0`) is not a plant. It is a 0.18 m
periodic community tile, 0.07–0.09 m tall, with 3.3 cm of capitulum relief, laid
out 484 tiles per 4 m cell at 90°-only rotations and one constant scale. Four
camera-facing sub-clump cards are the wrong shape for that in every way, so a
carpet species takes a **different bake and a different draw** and never touches
the clump atlas.

**The shape.** Near: `SHELLS = 4` **ground-parallel quads** through the tile, at
four heights taken from the quantiles of the baked cushion height field, drawn
top shell first. A fragment survives on shell *s* only if the cushion under it
reaches `shell_t[s]`, and the bottom shell's threshold is −1, so the stack is
cumulative: the mat is closed from below, the higher shells write depth first
and early-z rejects the copies underneath, and the relief is real geometry in
the depth buffer rather than a texture. Past `carpetShellDist` (8 m — a 0.18 m
tile stops resolving 1 cm of relief at ~7 m) it collapses to ONE quad at
`farH`, which is exactly the shell the mip-averaged near band converges to, so
the collapse moves nothing. `carpetShellDist = 0` renders the mat as plain flat
quads — the honest self-A/B control for what the shells are worth.

**The lattice, used as given.** Yaw is snapped back to an exact quarter turn and
scale is taken straight from `entry.scale_min` rather than from the packed
instance record: the record quantizes yaw to 1/1024 turn and scale to 1/2047 of
4 m, which at tile scale is a 0.4° rotation and a 0.2 % shrink — about a 1 mm
crack at every tile edge, i.e. a chicken-wire lattice over the whole bog. Width
comes from `footprint_m`, never from `height_scale`. No overscale (the pilot
measured ×1.15/×1.35 as clearly worse, and at 1.0 the tiles abut exactly so
there is nothing to z-fight and no depth bias is needed).

**Terrain fitting: rung 3, per-vertex.** Every quad corner is placed with
`terrain_sample(xz)` — height and (nx, nz) from one bilinear fetch — and the
shell height is added vertically. Neighbouring tiles share corner positions, so
the mat is C0 continuous across every tile border on any slope; a per-tile plane
fit (rungs 1–2) is not merely cheaper here but wrong, because neighbouring tiles
fit different planes and crack apart along their shared edge. The same fetch
gives the ground frame the baked normal is lifted into, so a mat on a slope
lights as a slope. This applies to the far LOD as well, which is the honest
place to spend: it is 4 texel fetches per vertex over ~650 k far tiles.

**What actually makes it read as 3D** (in order of how much it buys):

1. **A dedicated capture.** One straight-down orthographic view of exactly the
   tile square at 1024 px — 5.7 k px/m, ~3.5× the linear texel density
   001-billboard-smoke gets for the same tile out of its 3×3 atlas. The mesh is
   drawn **9 times** at ±1 tile step so the periodic overflow wraps: a single
   cropped copy cuts every overhanging capitulum in half at the border and
   leaves the neighbour's overflow missing, which is exactly the faint grid 001
   shows at `carpet-close` and this does not.
2. **Normals from the height field, not from the mesh.** The captured leaf-scale
   normal is genuine but is white noise at 0.35 mm; it mips straight to "up" and
   lights the cushion perfectly flat. The structure a viewer sees is the
   *cushion*, and that lives in the height field, so the stored normal is the
   gradient of the smoothed apex height with 35 % of the leaf noise left on top.
   `debug=lighting` goes from a flat wash to legible domes and gaps.
3. **A cavity term.** Per texel, the height minus its own ~6 mm neighbourhood
   (one extra `textureSampleLevel` at lod 4 of the aux tile), which darkens the
   gaps between capitula and leaves the domes alone, and fades out by itself
   with distance as both taps converge on the same mip.
4. **The shells.** Real stepped relief and correct depth interaction with
   whatever stands in the mat — but honestly the smallest of the four (see
   Findings).

Normals are stored as a plain hemisphere pair (nx, nz; ny reconstructed), never
octahedral: the capture flips every normal into ny ≥ 0, so the reconstruction is
exact and the channels survive filtering and mipping. Coverage-weighted colour
is normalised at every mip level, so the shader must not divide by alpha.

Alpha reference for a carpet is 0.06, not the params' 0.4 — a mat must not
dissolve with distance — and there is no camera-inside fade, because a mat you
are standing on must not open a hole under you.

## VRAM budget math

Per species, atlas 1280×2464 with (K_SUB+1)×9 = 45 tiles: sub-clump side tiles
160×512, merged side tiles 160×256, top tiles 160×160. Every tile origin and size
is a multiple of 32, so a 6-level mip chain never mixes two tiles into one texel —
the atlas is bleed-free by construction.

- albedo rgba8 (rgb + coverage), 6 levels: 4 204 200 texels × 4 B = **16.04 MiB**
- normals oct rg8 at **half** resolution (640×1232), 5 levels: 1 050 280 × 2 B =
  **2.00 MiB** (halving the normals is what buys the sub-tiles their resolution;
  coverage stays full-res, only lighting is filtered)
- far instance rings, 4 × 46 848 × 16 B = **2.86 MiB** (circle bound at
  regionRadius 128 and density 3, +20 % slack, split over equal-area rings —
  tighter than the region's square bound; ring capacities are rounded to 16
  instances so each ring's byte offset is 256-aligned and bindable)
- near instance rings, 3 × 7 552 × 16 B = **0.35 MiB**
- two uniforms per stand entry (192 B dynamic + 1 616 B static atlas layout):
  noise

**Total 21.25 MiB / 25 MiB** per species on the `default` stand (bench-verified:
21.25 / 20.72 / 21.25). Worst defined stand `dense-mixed` (density 5) fits at
**23.3 MiB**. The static uniform (atlas layout + per-unit capture boxes + 45 tile
rects) is written **once**, never per frame.

Texel density vs 001: a sub-clump tile resolves ~0.30 m across 160 px (533 px/m)
and 1.15 m across 512 px (445 px/m); 001's card is 731 × 445 px/m. So per-blade
sharpness at sub-metre range is ~0.73× 001's horizontally — the honest price of 5
units instead of 1 in the same 25 MiB. See Findings.

Bake transient: ~300–500 MB of GPU scratch (raw vertex+index buffers, two
2560×4928 rgba8 targets + depth, one readback at a time), all tagged
`bake-scratch` and destroyed.

### Carpet species (Sphagnum), per species

The whole allocation is different, because a mat shows exactly one view and the
tile count is 4× a grass entry's:

- carpet albedo rgba8 1024², 11 levels: 1 398 100 texels × 4 B = **5.33 MiB**
- carpet aux rgba8 512² (normal xz + height + coverage), 10 levels: 349 525 × 4 B
  = **1.33 MiB**
- far instance rings, 4 × 195 088 × 16 B = **11.91 MiB** — the real cost, and it
  is the tile count, not the imagery: 484 slots per 4 m cell × the entry's
  wetness band ≈ 12.6 tiles/m², over a 128 m circle
- near instance rings, 3 × 19 680 × 16 B = **0.90 MiB** (sized for the FULL grid,
  not the band: the near disc spans only a few 12 m wetness cells, so one state
  can own all of it, and dropping tiles there would open holes under the eye)
- uniforms: noise

**Total 19.48 MiB / 25 MiB** per moss species, and the HUD reads **19.5 MiB** on
the `bog` stand. (The committed bog bench result says 20.1 MiB — it was run
before the near rings were trimmed from a 1.4 clustering slack to 1.05, which is
the only difference.) Note what the split says: the 45-tile clump atlas
would have spent 18 MiB to give this species **one** usable 160×160 tile out of
45 — 1/123 of the imagery — which is why a carpet species skips it entirely and
why the stale `clumps-v2-spaghnum-*.bin` artifacts were deleted. The 3 moss
species between them cost 21 MiB of committed artifacts instead of 55 MiB.

## Bake

`bake.ts` → `mesh/baked/018-clump-impostors/clumps-v2-<species>.bin`, 17.5 MiB
each (1 KB header + the whole albedo mip chain + half-res oct normals). Steps:

1. Segment vertices into strands (6 cm jump in vertex order).
2. Weighted k-means over strand centroids → 4 sub-clumps; build a cluster-major
   index buffer so each unit is one `drawIndexed` range (the merged unit is the
   whole buffer, no extra memory).
3. Per unit: bbox centre in xz, exact max horizontal radius, y range.
4. Render 45 tiles orthographically at 2× (one command submission per unit — 45
   viewport draws over 6.5 M triangles in one buffer is a watchdog risk),
   albedo+coverage and view-flipped mesh normals.
5. Coverage-weighted downsample; normals straight from 2× to half resolution;
   per-tile true coverage; per-tile tight alpha rect; colour dilation clamped to
   each tile.
6. CPU mip chain with the per-level coverage calibration described above (this is
   why the albedo chain ships baked instead of being generated at load — a box
   filter would undo it). Normal mips are still generated on the GPU, but by
   decoding, averaging as vectors and re-encoding, so a distant tile keeps a real
   mean direction instead of an average of oct coordinates.

Artifacts are magic+size validated on every load because the dev server answers
missing `/mesh/baked` files with `index.html` at HTTP 200; a poisoned OPFS entry
is rebaked and repaired in place. First bake of all three species ≈ 60 s.

### Carpet bake

`carpet.ts` + `shaders/carpet_bake.wgsl` →
`mesh/baked/018-clump-impostors/carpet-v2-<species>.bin`, 6.7 MiB each
(256 B header + both mip chains). Steps:

1. One orthographic straight-down capture of exactly `[tileOrigin, +tileM]²` at
   2048² (2× supersampled), depth = 1 − height so the depth test resolves the
   topmost surface and its height comes out in the alpha channel for free.
2. The mesh is drawn **9 times**, offset by (−1,0,+1)² tile steps, one submission
   each (~20 M triangles per submission — nine in one command buffer is a
   watchdog risk). The rasterizer clips the rest. This is what makes the texture
   genuinely periodic.
3. Resolve to 1024² albedo (coverage-weighted) and 512² aux; wrapped dilation of
   both so bilinear taps at a coverage edge never pull the cleared value in.
4. Smooth the height twice (wrapped binomial), take its gradient for the cushion
   normal, blend 35 % of the captured leaf normal on top, store as (nx, nz).
5. Shell heights from the height histogram quantiles (0.08 / 0.35 / 0.68 of the
   covered area above them) plus a skirt at 0.05, and `farH` = the shell whose
   band contains the mean apex.
6. Wrapped box mip chains for both textures on the CPU (a 2×2 block never
   crosses the periodic boundary, so the chain itself needs no wrap).

Measured on the three states: coverage 98.5 / 93.8 / 93.3 %, mean apex
73.1 / 52.2 / 56.2 mm, shells 82.2/76.4/70.5, 60.1/55.4/50.5, 64.4/59.5/54.3 mm.
~2 min per species (dominated by fetching and parsing the 479 MB source mesh).

## Status

**working** — verified by headless screenshots at `grazing`, `topdown`,
`inside-plant`, `far-horizon`, plus `debug=normals`, `debug=lighting`,
`debug=coverage`, plus the A/B page against 001 at four cameras and hand-placed
in-canopy close-ups, on `default`, `scaling-100m` and `dense-mixed`. Zero console
errors AND zero harness toasts (see the bug below — toasts are where WebGPU
validation errors surface, and my verification script now scrapes them).
`npx tsc --noEmit` clean.

The carpet path is verified on `bog` at `grazing`, `topdown`, `inside-plant`,
`carpet-close`, `far-horizon`, two sloped views (a ridge from ~8 m up and a
2 m-high near view down a slope), `debug=normals` / `lighting` / `coverage`, the
A/B page against 001 at `grazing` and `carpet-close`, and a `carpetShellDist`
0-vs-8 self control at 0.35 m eye height. Zero console errors, zero toasts.
`default` re-checked at `grazing` and `inside-plant` before/after: visually
identical, and the before/after pixel difference is *smaller* than the
frame-to-frame difference between two captures of the same build (the wind is
not frozen by `time=`, so that is the noise floor).

## Findings

### The bug that invalidated my first round of numbers

The ring draws pass each ring's slice base as `firstInstance` in the indirect
args. **WebGPU rejects a non-zero `firstInstance` in an indirect draw unless the
optional `indirect-first-instance` feature is enabled**, and the harness does not
request it. So six of my seven draws per stand entry were silently dropped: only
near ring 0 and far ring 0 ever rendered. The renderer looked plausible — a
continuous meadow — because the two surviving rings covered 0–10 m and 14–57 m;
what it actually had was a missing annulus at 10–14 m and nothing beyond 57 m.

It hid well because the validation error goes to `device.onuncapturederror` →
the harness's `onError` → an on-screen **toast**, not the console: my screenshot
script was reporting "0 console errors" the whole time. It also produced a
plausible-looking *speedup*, which is how I nearly shipped "front-to-back rings
cut grazing cost 25 %".

What found it: painting near-LOD cards red and sweeping `lodDistance`. At
`lodDistance = 30` the red band stopped at ~17 m and everything from 17–30 m was
bare terrain — geometry that must be there was simply absent. Fix: bind the
instance buffer at the ring's byte offset in a per-ring bind group (ring
capacities rounded to 16 instances so the offsets are 256-aligned) and keep
`firstInstance = 0`. Everything below is measured *after* that fix.

Two earlier conclusions died with it: the "open band inside lodDistance" I blamed
on split coverage was the missing annulus (so `nearAlphaBias`, added to
compensate, now defaults to 1 = no-op and is just a density knob), and the ring
ordering's real benefit is **unmeasured** — I attempted a 1-ring control build to
quantify it, it failed to come up, and I reverted rather than ship a broken
config. Rings cost nothing measurable to keep, so they stayed.

### Speed — modestly slower than 001, worst case 1.35×

The GPU went quiet late in the session, so these are stable, repeatable numbers
(4 consecutive samples within ±0.02 ms), not contended guesses.

**Bench** (authoritative), `?stand=default&spline=orbit-low`, 1600×900, both run
2 minutes apart on the same quiet GPU:

| | cull | cards | renderer | total Σp50 |
|---|---|---|---|---|
| `results/001-billboard-smoke__default__p-86e4907a__apple-metal-3__2026-07-25T01-50-34-363Z.json` | 0.380 | 2.501 | 2.881 | 5.868 |
| `results/018-clump-impostors__default__p-08c1f77a__apple-metal-3__2026-07-25T01-48-48-766Z.json` | 0.491 | 3.398 | 3.889 | 7.244 |

→ **1.35× on the renderer, 1.23× on the total**, and 7.24 ms is comfortably under
the 9 ms bar. `orbit-low` is a grazing-height orbit, i.e. the expensive case.

**Interleaved solo runs at the standard cameras** (cull+cards p50, medians of 3
alternating samples, 1280×800, shipped defaults):

| cam | 001 | 018 | ratio |
|---|---|---|---|
| grazing | 2.54 | 3.37 | 1.33× |
| in-canopy close-up | 2.50 | 2.60 | 1.04× |
| far-horizon | 2.39 | 2.55 | 1.07× |
| topdown | 1.73 | 1.94 | 1.12× |
| inside-plant | 2.69 | 2.68 | 1.00× |

So: parity everywhere except grazing-type views, where the sub-clump band costs
~1/3 more. `lodDistance` is the dial — 10 m measures 1.25× at grazing, 14 m
(shipped) 1.33×, 18 m 1.39×.

**A/B same-frame** ratios are quoted only as a sanity check, because the B slot
of that page is systematically 26–35 % slower for *identical* work (running 001
against itself gives A/A = 1.26–1.35×). Raw B/A for 018 in the final run was 1.26×
(grazing), 1.05× (far-horizon), 0.98× (inside-plant), 0.97× (topdown) — i.e.
consistent with, or flattering relative to, the solo table.

**Plant-count independence:** `default` (557 k) cull 0.45 / cards 1.8–2.9,
`scaling-100m` (134 M) identical, same VRAM. `dense-mixed` (7.7 M, density 5)
rises with density, not count, and stays inside the VRAM budget.

### Looks — better than 001 in the near and mid field, level at distance

Judged on same-camera, same-frozen-time pairs plus the A/B wipe, after the fix.

- **In-canopy (eye at 0.85 m, fov 30) is the clearest win.** 001 is a flat pink
  wall with a handful of giant elymus spikes pasted in front of it — the
  calamagrostis field has no internal structure at all. 018 shows panicles at
  several depths and scales, dark gaps between clumps, green blades crossing in
  front of pink heads, and the spikes sit *inside* the stand at their real
  positions. It reads as standing in a volume.
- **`inside-plant`:** 001 fades out the plant you are inside and leaves a
  backdrop of near-identical spikes, plus a large pale disc overhead (its
  top card seen from below). 018 keeps sub-clumps at their offsets, so blades
  pass at several depths with sky through the gaps, and the signed elevation test
  removes the overhead-disc artifact entirely.
- **`grazing`** (the judging camera): 001 is a fairly uniform pink haze; 018 has
  green stems and shadowed gaps between the panicles and a visible near→far depth
  gradient — more chromatic variety, more layering.
- **Parallax:** a 0.30 m lateral camera step at 14° fov moves near panicles much
  further across the screen than the material behind them and re-orders what is
  visible inside single plants. This is structural, not a trick: the cards really
  are at different world positions.
- **`far-horizon`:** a wash. 001 is slightly sharper per panicle (bigger tiles),
  018 slightly deeper. Past `lodDistance` both draw one card per plant.
- **`topdown`:** identical by construction (everything is past `lodDistance`), at
  1.12× the cost.
- **Where 001 still wins:** per-blade crispness inside ~1 m. Five units share the
  same 25 MiB, so a sub-clump tile gets ~0.73× 001's horizontal texel density and
  the very-close view is grainier. Fixing it needs a bigger budget or fewer
  sub-clumps, and both trade away the thing that makes the method work — even
  halving the normals again only buys 1.10× linear.

### Moss round — what was broken, what the fixes bought

Before this round the renderer treated Sphagnum as a small upright plant. The
`bog` stand was, in order of severity:

1. **~74 % of the mat missing.** The cull dispatch and its slot index came from
   `SCATTER_MAX_PER_CELL` (128), so of the 484 carpet slots per cell only the
   first 128 — the first ~6 of 22 rows — were ever evaluated. On screen that is
   **4 m-pitch stripes of moss over bare peat** (see `grazing` before/after), and
   from above a banded speckle. Both the dispatch and the buffer capacities now
   come from `standEntrySlots(entry)`, and the two numbers are kept apart: every
   slot is *evaluated*, capacity only holds the expected *survivors*.
2. **Vertical camera-facing cards for a mat**, at K_SUB=4 per tile: they sliced
   through the ground and through each other, showed nothing edge-on, and
   billboarded, which breaks the lattice outright. Replaced by the ground-
   parallel shells above.
3. **The budget was spent on views a mat never shows.** 45 side/top tiles at
   18 MiB, of which a carpet used one 160×160 tile. Replaced by the dedicated
   1024² tile capture; the moss's committed artifacts went 55 MiB → 21 MiB and
   its runtime VRAM 26.5 MiB (over budget) → 20.1 MiB.
4. **Slope.** The old path stood every tile bolt upright at the scatter's point
   height; now every vertex is conformed (rung 3).

What that is worth against 001-billboard-smoke on `bog`, from the A/B page and
same-camera pairs:

- **`carpet-close` (1 m straight down) is the clearest win.** 001 is uniform
  red-green noise with a visible dark tile grid every 127 px; this shows
  individual capitula — star-shaped rosettes, orange sun-exposed ones among the
  green — with no grid at all, because the capture wraps the periodic overflow
  instead of cropping it.
- **`grazing`:** both cover the ground now. 001's mat is a flatter, brighter
  yellow-olive wash (its octahedral normals mip toward straight up, so it lights
  like a plane); this one is darker and has visible cushion texture, zone colour
  variation and bumpiness. Mean luminance is 18 % below 001's at `carpet-close`,
  which is the cavity term plus honestly-tilted normals, not a bug.
- **Slopes:** the mat follows the ridges continuously with no cracks at tile
  borders and no floating tiles, near and at ~40 m.
- **`topdown`:** continuous carpet with interlocking wetness zones; before it was
  banded speckle.
- **The shells are the smallest of the four wins.** At 0.35 m eye height,
  `carpetShellDist` 8 vs 0 is a visible but modest corrugation. That is the
  honest physical answer: the cushion's relief is 11.7 mm of expressed step out
  of 33 mm total, and at any distance where the mat covers real screen area that
  is a few pixels. Their real value is that the relief is in the depth buffer —
  it self-occludes and interleaves correctly with the grass standing in it —
  rather than that it looks dramatic.

**Perf.** `results/018-clump-impostors__bog__p-9cba0282__apple-metal-3__2026-07-25T16-56-44-367Z.json`
(`bog`, `orbit-low`): cull 1.20 / cards 3.73 / Σp50 7.05 ms. `default` is
unchanged:
`results/018-clump-impostors__default__p-9cba0282__apple-metal-3__2026-07-25T16-52-47-490Z.json`
gives cull 0.407 / cards 3.644 / Σp50 7.88 ms against the pre-round 0.491 /
3.398 / 7.24 ms — inside the noise of a GPU shared with other agents. Two runs
of the *pre-round* build measured Σp50 7.24 and 5.18 ms with cards at 3.40 and
2.10, i.e. a 40 % spread, and in my run the two passes I do not own (`base`,
`composite`) are also at their highest, which is what a busy GPU looks like.
Structurally the grass path is untouched: same draws, same vertex counts, same
two texture taps per fragment (the carpet's third tap uses an explicit LOD, so
it lives inside the carpet branch and the grass never executes it). Same-frame A/B against 001 on `bog` at `grazing`: A(001) cull 0.81 /
cards 3.64, B(018) cull 1.12 / cards 3.31 — and the B slot is systematically
1.26–1.35× slower for identical work, so the carpet path is at parity or
slightly cheaper than 001's, despite drawing 4 shells in the near 8 m.

**Known limits and risks of the carpet path**

- Far instance rings are sized for the entry's *expected* wetness band
  (band × 1.25), not for the worst case where one moss state owns a whole
  region. Sizing for the full grid would need 26 MiB of rings per species. I
  probed the two worst cases I could find — a dry slope (where the wetness field
  is damped and the driest entry claims nearly everything) and a wet flat — by
  comparing `regionRadius` 110 against 40 at the same pose: the moss pixels are
  **bit-identical** in both, i.e. nothing is being dropped. If a future stand
  does overflow a ring, the failure mode is scattered missing tiles, and the fix
  is an 8-byte carpet instance record (cell + slot index + quarter turn, with xz
  recomputed in the vertex shader) which would halve the ring cost.
- Four shells over a top-heavy height histogram express the top ~12 mm of the
  33 mm relief; texels in the deep gaps get raised to the base shell. More shells
  would need `shell_y`/`shell_t` to grow past one `vec4f` each.
- The mat's own 0.18 m periodicity is visible as a faint repeat at 1–3 m in
  grazing views. Only the 90° rotations break it up; that is inherent to a
  periodic tile drawn at life size.
- The `far-horizon` camera on `bog` looks over a wet hollow where one state owns
  almost every node, so the mat reads as a vivid green lawn with a few brown
  tiles poking through. That is the stand's zoning, not the renderer: 001 shows
  exactly the same thing at that pose (with a stronger grid).

### Notes for whoever reads this next

- **Scrape the toasts, not just the console.** Every WebGPU validation error in
  this harness lands in a `.toast` element and disappears after a few seconds; a
  screenshot-only check will report a clean run while half your draws are being
  rejected. `page.locator('.toast').allInnerTexts()` after each capture is two
  lines and would have saved me an hour.
- The coverage calibration is tuned for `alphaRef = 0.4` (the default); moving
  that slider trades part of the LOD density match back. The `coverage` debug view
  shows what it costs, and it is worth stealing for any alpha-tested foliage
  method in this repo — it also stops distant grass from thinning out.
- `lodDistance = 0` renders the method as plain billboards over the same atlas,
  which is the cleanest way to isolate what the sub-clumps are actually buying;
  `carpetShellDist = 0` does the same for the moss shells.
- **A source mesh's own normals can be the wrong normals.** Sphagnum's leaf
  facets are real geometry at 0.35 mm and they are useless: filtered they average
  to "up" and light the cushion flat, unfiltered they are speckle. The normal a
  viewer actually needs is the *cushion's*, and the cheapest honest way to get it
  is the gradient of the height field the same capture already produced. Any
  method baking a dense, self-similar mat should check `debug=lighting` for this
  specifically — a flat lighting view over obviously bumpy geometry is the tell.
- **Screenshots of this harness are not frame-deterministic** (`time=` does not
  freeze the wind), so "did my change alter the other stand" cannot be answered
  by hashing a PNG. Capture the same build twice to get the noise floor first;
  for a carpet the moss pixels *are* static (sway 0), which makes the same trick
  a precise test for dropped tiles.
- Harness wishlist: (1) `indirect-first-instance` in `requiredFeatures` (it is
  widely supported and the workaround costs a bind group per bucket); (2) the A/B
  page's B slot is systematically 26–35 % slower for identical work — a built-in
  self-control, or alternating the A/B encode order between frames, would remove
  a real footgun; (3) the three carpet entries of `bog` each re-evaluate the same
  shared wetness field over the same 484 nodes per cell, so ~2/3 of the cull's
  work is provably redundant — a way to ask the scatter "which entry claims this
  node" once would cut the carpet cull by 3×; (4) `standEntrySlots()` exists in
  TS but has no WGSL twin — `stand_table[i].carpet_div²` has to be re-derived by
  hand, which is exactly the arithmetic the "hardcoded 128" trap is about;
  (5) there is no terrain-relative way to write a URL camera pose, so any view
  other than the five bookmarks has to be found by flying the camera and reading
  the pose back out of the URL — a `cam=x,+2,z,...` "relative to ground" form
  would make sloped-terrain checks reproducible.
