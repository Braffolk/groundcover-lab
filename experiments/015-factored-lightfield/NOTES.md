# 015 factored quantized light field

## Idea

A per-species light field — "what do you see looking at this plant from
where" — stored whole, but brutally factored and quantized to fit VRAM:

**L(direction, image offset) ≈ Appearance3D( Hit(direction, image offset) )**

- **Geometry factor (all the view-dependence):** a hemi-octahedral grid of
  ORTHOGRAPHIC depth maps in a single rg8 3072x3072 atlas (r = min-depth, 1.0
  reserved as miss sentinel; g = fractional coverage from a 2x2-supersampled
  bake). This is a precomputed answer table for "which surface does this ray
  hit" — the sanctioned precomputed-raycast loophole. The angular/spatial split
  is a per-species PROFILE, because grid x tile = 3072 either way:
  - upright plants — **24x24 views (576) at 128px**, 4.3 deg apart: thin stems
    seen edge-on change fast with direction, so angular density wins;
  - carpet tiles (`carpetDiv > 0`, the bog's Sphagnum) — **12x12 views (144) at
    256px**, 7.4 deg apart. A cushion is nearly a height field: 3.3cm of relief
    with nothing standing off it, so the parallax between neighbouring views is
    ~2mm, while 256px on a 0.24m tile puts **0.9mm on a texel** and resolves
    single capitula. Same atlas, same 18.9MB, 4x the spatial detail where this
    species is actually looked at.
  Each view is framed on the **box support along its own axes**, not on the
  bounding sphere (`half_u` in the uniform, 1/support per view in the bake).
  For the 0.21 x 0.09 x 0.23m Sphagnum slab the old sphere frame spent ~70% of
  every tile on empty space; the grasses gain too (a top view of calamagrostis
  gains ~4x in texel area).
- **Radiance factor (view-INDEPENDENT):** 3D albedo + normal volumes
  (aspect-fit ≤96 per axis, rgba8), splatted on CPU from the raw GCMESH1
  vertices (2M–6.5M vertices ≫ voxel count), dilated so quantized hit points
  always land on data. albedo.a carries per-voxel luminance sigma, re-injected
  at runtime as deterministic hash speckle (~1.5cm cells, stable under wind
  and camera) so voxel-averaged interiors keep sub-voxel texture.

Runtime: every plant is ONE camera-facing PROXY card — a window, not the
surface: the fragment reconstructs a 3D hit and writes its true depth, so
nothing on screen is a flat quad. Per fragment: take the
rest-pose (wind-un-sheared) sample point straight off the interpolator,
project into the 4 views nearest the per-fragment eye direction,
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

Carpets additionally **collapse with distance** (see "Moss carpet" below): past
a few metres a 0.9mm texel is far below one pixel, so the mat stops being
sampled from the light field and becomes what it honestly is at that range — a
closed slab of cushion tops, ray-intersected analytically, one volume tap, no
gathers. Blended over a distance band; nothing dithered.

Placement: scatter WGSL twin evaluated in the vertex shader over a bounded
camera-centred cell window (`regionRadius`, default 96m) — 100 plants and the
134M-plant scaling stand issue byte-identical GPU work (verified, see
Findings). No source geometry exists at runtime; no marching anywhere (the
refine tap is a second lookup, not a loop).

Wind: shared `wind_sway` at the plant root, per-species stand sway. The card
leans, and the fragment samples in the plant's REST pose — which is what makes
the SPRITE lean (sampling at the displaced position, as some earlier
experiments do, cancels the lean except at the card border). The rest-pose
sample point is affine in the card corner, so it is emitted per vertex and
interpolated rather than un-sheared per fragment. Reconstructed hits are
re-sheared before frag_depth/light. Verified: 66% of field pixels change
between t=2 and t=5 (mean |d| 36/255).

Lighting: volume normals (yaw-rotated, two-sided flip) + shared
`light_surface` + fog; height-based AO stand-in (roots 0.6). Camera inside a
plant: coverage fades out below 1.1 x bounding radius (`near_fade`), and at
the region edge (`far_fade`) — honest coverage erosion through the hard alpha
threshold, no dithering anywhere.

Debug views: the fragment shader routes through `debug_shade()` (albedo =
volume colour incl. sigma speckle, pre-AO; normal = the final per-fragment
normal after relief/collapse blending, frame rotation and the two-sided flip;
coverage = the blended coverage x fade, i.e. the exact quantity the alpha test
compares; world_pos = the reconstructed hit, so `depth` shows the
reconstruction and not the card). Fog is applied only in `debug=off`.
Method-specific state lives in the `showViewGrid` param: it replaces albedo
with a stable per-hemi-oct-node colour blended by the same 4-view weights,
which makes the angular grid and the width of its blend region directly
visible.

## VRAM budget math

Per species (HUD-confirmed):

- depth+coverage light field: 3072 x 3072 rg8unorm = **18.87MB** (every
  profile — the angular/spatial split never changes the size)
- albedo volume rgba8 (aspect-fit): calama 62x96x58 = 1.38MB · elymus
  77x96x79 = 2.34MB · poa 38x96x39 = 0.57MB · Sphagnum 86x38x96 / 90x26x96 /
  96x30x93 = 1.25 / 0.90 / 1.02MB
- normal volume rgba8: same dims again
- info uniform: 256B

Totals (HUD): calamagrostis **20.6/25MB**, elymus **22.5/25MB**, poa
**19.1/25MB**, Sphagnum wet **20.4**, late-season **19.7**, sun-exposed
**20.0/25MB**. No per-plant buffers at any plant count, on any stand —
including the bog's 1.13M life-size moss tiles. Transient bake peak adds ~8MB
strip targets + a second copy of the atlas for readback + mesh vertex/index
buffers (up to ~420MB for a 19.8M-tri Sphagnum tile for a few seconds;
destroyed and un-counted after bake).

Note the allocation question the moss round asked: for a 0.09m-tall cushion the
old split spent 18.9MB on 576 views of a **bounding-sphere** frame, i.e. most
texels on empty space at 2.6mm each. The same 18.9MB now buys 144 views at
0.9mm — the budget moved from angular resolution the mat does not use into the
detail you actually look at, without adding a byte.

## Bake

Through the harness bake flow now (OPFS cache -> committed artifact -> fresh
bake), one gzipped `.bin` per (species, profile):

1. `grid` rows of ortho views, each rasterized 2x2 supersampled into a one-row
   strip (`grid*2*tile` x `2*tile` r8 + d32), chunked ≤60M tris/submit with
   `onSubmittedWorkDone()` awaits (a Sphagnum tile is 19.8M tris x 12 draws per
   row), then a downsample pass writes min-depth + fractional coverage into the
   final atlas row. Fractional coverage is what keeps sub-texel seed-head fluff
   wispy instead of snapping to its convex hull.
2. CPU vertex splat + 4 dilation passes + sigma into the volumes (~0.1-1.5s;
   11.5M vertices for a moss tile).
3. The atlas is read back and packed with the volumes into one artifact
   (96B header + atlas + 2 volumes), gzipped through `CompressionStream`.

GPU bake time is 0.5-2.6s for a grass and 1.5-5s for a 19.8M-tri Sphagnum
tile, but **fetching and parsing 479MB of source mesh dwarfs it**, so the
carpet profile sets `commit: true` and the three moss artifacts are committed
(4.8-6.1MB each gzipped, 16MB total — the raw 21.4MB compresses well because
the miss sentinel is a constant plane over the empty parts of every view).
The grass profiles stay OPFS-only: a 64MB mesh and a ~1s bake are not worth
21MB of repo each. Every load validates a `QLF1` magic, so an SPA-fallback
`index.html` can never be mistaken for an artifact (the trap that made 005 and
the earlier version of this experiment bake per session).

## Status

**working** — verified by headless screenshots at grazing / topdown /
inside-plant / far-horizon, on `default` and `scaling-100m` stands, t=2 and
t=5, and on the `bog` stand at grazing / topdown / inside-plant /
carpet-close / two eye-level sloped views plus normals / lighting / coverage /
albedo. All six species render (three grasses, three Sphagnum states).

## Findings

- **Renders correctly from all standard cams.** Topdown reads as a proper
  meadow carpet (the hemisphere atlas covers straight-down); inside-plant
  shows real parallax between stems with the near-fade working; grazing holds
  with no billboard cardboard (frag_depth from reconstructed hits).
- **Plant-count independence:** `default` (557k) vs `scaling-100m` (134.2M)
  screenshots are near-identical in content and cost — the drawn work is the
  same bounded cell window by construction (side^2 x slots-per-entry
  instances x entries).
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
  yet. Structural note: frag_depth disables early-z; a near/far pipeline split
  (rasterizer depth beyond ~40m, where the card-vs-hit depth error is
  subpixel) is the obvious next win, as is dropping `refine` (halves taps)
  which is already distance-gated at 45m.
- Directions below the horizon clamp to the hemisphere ring (terrain
  groundcover is effectively never seen from underneath).

## Audit (structural + debug-view pass)

**Debug views (were entirely missing).** The fragment shader never included
`src/wgsl/debug.wgsl`, so every `debug=` mode drew the plants exactly as
`off` while the terrain around them switched — the method could not be
inspected at all. Now wired per the template (see Lighting above). What the
views exposed once wired:

- `normals` — real per-fragment normals, as claimed: coherent gradients along
  stems and over the panicle lobes, mostly +Y from `topdown` and mostly
  downward from `inside-plant` (undersides of leaves, which is the correct
  answer given the two-sided flip toward the viewer). No constant/flat
  normals, no undecoded atlas. Nothing to fix.
- `lighting` — goes through the shared `light_surface()`, applied exactly
  once; the shared model peaks at `sun_color + ambient` ≈ 1.36, so sunlit
  fragments clip to white in this view exactly as the harness terrain does.
  Not a double-apply: albedo is passed to `debug_shade` pre-AO, so the view
  is the honest light x AO term.
- `albedo`, `coverage`, `depth` — all meaningful; `depth` in particular
  confirms the reconstructed per-fragment hit depth (plants read as coherent
  silhouettes at their own distance, not as flat cards).

**Structural waste found and fixed.** Per-fragment work that is constant
across a card moved into the vertex stage: three `rot_y()` calls per fragment
(each with its own `sin`/`cos` of the plant yaw) became one `cos`/`sin` pair
per plant passed as a flat varying; the camera position in unit-sphere local
space (`cam_unit`) and the camera distance used to gate the `refine` tap were
being recomputed per fragment from constants and are now computed once per
plant; the rest-pose sample point is affine in the card corner, so the
per-fragment un-shear (`world - sway*t_lean`, rotate, divide) collapses into
one interpolated varying. Six of the seven varyings are now
`@interpolate(flat)` (they were all being perspective-interpolated), and the
unused `entry`/`scale` varyings are gone. Nothing about the technique or the
sampling changed.

**Verified image-equivalent.** Compared against the pre-audit shader at
`?det=1&t=2` (fixed time + paused, so runs are bit-reproducible — two runs of
identical code differ in 0 pixels): grazing/topdown/inside-plant/far-horizon
differ in 0.02-0.14% of plant pixels by more than 8/255. That is the
renderer's own numerical noise floor, not a behaviour change: a control that
merely scales the card by 1.000001 (strictly larger, cannot clip anything)
flips 240-415 pixels the same way. The method amplifies ULP-level vertex
differences because four of its steps are discontinuous — which 2x2
`textureGather` footprint is picked, the 8-bit depth step, the ~1.5cm speckle
hash cell, and the hard alpha test.

**Tried and rejected: a tighter card.** The obvious fill win is that the card
is sized by the bounding-sphere diameter while the plant's silhouette is much
smaller (bbox support along the card axes is ~1.5-2.6x less area). It was
implemented (exact box support function of the yaw-rotated bbox along the card
axes, x the perspective/view-tilt magnification `D/(D-half_depth)`, clamped to
the old radius) and then reverted: the `refine` tap re-queries the light field
at a point slid ALONG THE EYE RAY, up to ~2 unit-sphere radii from the
fragment's own sample point, so with a ~4-8 deg quantized view direction a
fragment can legitimately resolve geometry ~0.3 radii outside its own
projected footprint. Padding for that leaves only ~1.1x for calamagrostis and
elymus (poa keeps ~1.5x) — and measurably clipped real coverage at every pad I
tried (0.3-1% of pixels, 10-20x the noise floor). Worth revisiting only
together with the `refine=false` path or a near/far pipeline split.

**Deliberately left alone.**

- *Per-frame uniform rewrite:* `update()` rewrites all 96 bytes of the info
  UBO per entry per frame though only `origin_cell`/`side` change. Splitting
  it saves 3x84 bytes of queue writes and no draw calls — below the noise.
- *Vertex-stage candidate enumeration:* `side^2 x slots` instances per entry
  run the scatter twin 6x per plant, and most slots are empty or off-screen.
  A cell-level frustum+region reject now kills most of them before the
  scatter's 8 heightmap taps (added in the moss pass), but the vertex launches
  remain: 21M/frame on the bog. The compute prepass was previously rejected on
  VRAM grounds assuming a 16B instance record; a **4-byte** record (cell
  offsets + slot, with `scatter_candidate` re-run in the vertex stage for the
  survivors) is 1.55MB/species at the bog's worst-case survivor count and stays
  inside 25MB. That makes it a free win rather than a trade — see "Moss carpet
  / What is still bad".
- *Per-fragment view-basis reconstruction:* `tap_view` rebuilds each hemi-oct
  node's ortho basis (decode + 2 crosses + 3 normalizes) 4-8x per fragment
  from what is a bake-time constant. A 576-entry basis table would trade ALU
  for a buffer read; that needs measuring, so it stays a suggestion.
- *No early-z:* the shader writes `frag_depth`, which disables early-z for a
  fill-heavy pass. WGSL has no conservative-depth attribute, so the only fix
  is the near/far pipeline split noted above — a behaviour change, out of
  scope for an audit.
- *Bake still runs per session (~3s for three species).* The stated reason
  (dev-server SPA fallback poisoning `/mesh/baked` fetches) is now handled by
  the harness — `bakedArtifact()` detects the HTML fallback. Committing the
  artifact would need a readback of the 18.9MB atlas per species (~63MB in
  the repo) and buys load time, not frame time, so it is left for the owner
  to decide.


## Moss carpet (bog stand)

*Sphagnum palustre* arrives as a 0.18m periodic community tile, 0.07-0.09m tall,
19.8M tris, laid out by the `bog` stand as a life-size mat: 22x22 grid-snapped
tiles per 4m scatter cell (**484 slots**), constant scale 1.01, 90-degree-only
yaws, three states partitioning a wetness field. ~1.13M moss tiles over ±96m.

Before this pass the renderer drew **horizontal stripes of speckle over bare
peat**: it enumerated `SCATTER_MAX_PER_CELL` (128) slots per cell, which for a
22x22 carpet is grid rows 0-5 — a 1.09m band of every 4m cell — and anchored
each tile on the MESH ORIGIN, which for a periodic tile is a corner, so the four
90-degree yaws threw their squares into four different quadrants around the grid
node. What survived was blocky, untilted and gap-ridden.

What changed, smallest first (all of it gated on `stand_table[i].carpet_div > 0`
so the grasses and the `default` stand keep their exact behaviour):

1. **Slots per cell come from the entry, not the global budget.**
   `carpet_div^2` = 484 in the vertex stage, `standEntrySlots(entry)` for the
   instance count. This alone is the difference between a quarter of the mat and
   all of it.
2. **Anchor on the periodic tile centre** (`footprint_m * 0.5`, and the task's
   guarantee that the tile origin is mesh (0,0)) instead of the mesh origin, so
   a quarter turn rotates the tile square about its own centre and every yaw
   still lands on the grid. Also: `footprint_m` is the only correct width source
   here — `height_scale` would have been 3.5x too small.
3. **Carpet alpha reference 0.15** instead of the `coverage` param (0.38). A mat
   is a closed surface; at 0.38 marginal texels punched speckled holes in it.
   Hard-edged, depth-writing, no dither.
4. **No camera-inside fade for a carpet** — a mat you stand on must not open a
   hole under you. Verified at `inside-plant`: continuous under the camera.
5. **Ground-tilted plant frame** (`slope_align`): the whole plant basis becomes
   `plant_basis_from_up(mix(up, terrain_normal, align), yaw)`, from ONE
   `terrain_sample` (height + normal in the same four texel loads). This also
   gives the bog's calamagrostis its intended 0.3 lean; entries with align 0
   keep the exact literal yaw matrix, so `default` is arithmetically unchanged.
6. **Proxy card hugs the slab, not the sphere.** The card is still camera-facing
   — but it is a WINDOW, not the surface: every fragment reconstructs a 3D hit
   and writes true depth, so the lattice invariant is untouched (the sampled
   tile keeps the stand's 90-degree yaw and constant scale; nothing is
   billboarded but the query window). Sized from the exact box support of the
   yaw-rotated bbox along the card axes x 1.25 instead of the bounding sphere:
   for a 0.24 x 0.09 x 0.24m slab in a 0.33m sphere that is ~3x less fill and no
   quad hanging metres below ground.
7. **Per-view box-support framing** (see Idea): 0.9mm texels instead of 2.6mm.
   This is the single biggest reason the mat reads as intricate rather than as a
   blob, and it improves the grasses as well.
8. **Depth-gradient relief normals** (`carpetRelief`, default 0.85). The volume
   normal of a dense cushion is close to noise — 2.4mm voxels average every leaf
   orientation, and `debug=normals` showed literal rainbow salt-and-pepper. The
   2x2 depth gather each view already fetched gives the mesoscale surface
   gradient for free; blended over the 4 views it is the honest normal of the
   geometry the light field describes. Measured: it also comes out *brighter*
   than the volume normals (region luma 50.0 vs 44.6), because the volume's
   random normals point away from the sun as often as toward it.
9. **Distance collapse** (`collapseNear` 14m, `collapseFar` 36m, scaled by up to
   1.8x as the view flattens). Past a few metres a 0.9mm texel is far below one
   pixel and the light field can only alias — one point sample per tile,
   flickering between capitulum and gap. Beyond the band the tile becomes what
   it honestly is at that range: a closed slab (the periodic square, from the
   ground to the mean capitulum height, `0.74 * tile topH`, which the three
   manifests agree on to 1.5%), ray-intersected analytically, shaded from one
   volume tap at the top plane, with the **area-weighted mean of the visible
   slab faces** as its normal. Neighbouring slabs abut exactly, so the far field
   is continuous and quiet, and it costs zero gathers. Nothing is dithered; this
   is coverage that becomes MORE solid with distance, not less.
   Calibration: at eye level the collapsed and light-field renderings of the
   same view agree to 3% in mean luma (49.2 vs 47.8) — the mean-face normal is
   what buys that; a flat up normal was ~20% too bright, which is the classic
   "far field gets brighter" failure.
10. **Cell-level frustum + region reject before the scatter.** 484 slots x 2401
    cells x 3 entries is 3.5M candidates per frame, and each `scatter_candidate`
    on a carpet costs 8 heightmap taps (wetness + height). Six CPU-computed
    inward planes vs a conservative cell box kill most of them for ~40 ALU and
    no memory traffic. `topdown` on bog went from 177ms to 85ms in the same
    contention window.

### Terrain fitting: rung 1, and why that is not a compromise here

Rigid tilt from the point normal at the tile's own node. The ladder's warning
about per-tile plane fits cracking at shared edges is about footprints that span
real relief — this tile is **0.18m across while one heightmap texel is 0.5m**
(`size 256 / resolution 512`), so the ground under a tile is a single bilinear
patch and its tangent plane tracks it to ~3mm (the bilinear cross term over
0.09m). Neighbouring tiles therefore agree to a few mm at their shared edge,
inside geometry that already overlaps by 30% (0.234m of mesh on a 0.1818m step).
Per-vertex conforming would cost 6 terrain fetches per tile x 1.13M tiles to fix
a 3mm error under a 33mm cushion. If the terrain regime ever gets rougher than
its heightmap texel — bumps at 0.2m scale — this choice flips, and the honest
upgrade for this method is rung 4: shear the *lookup* by the local gradient and
keep the exact height per fragment.

### What it looks like now

- `carpet-close` (1m straight down): a continuous, gapless cushion carpet with
  cushion-scale clumping and green/ochre/rust mottling; no tile seams, no grid,
  no holes. Clearly better than 001-billboard-smoke's flat top-view quad at the
  same camera (which is smooth and featureless), though 001 is ~30% brighter.
- `topdown` (42m): a closed mat with the three states forming organic zones —
  bright green wet hollows, khaki crests, olive flanks. Before: stripes.
- `grazing` / eye level on the ridges: the mat covers everything, follows the
  slopes with no terracing and no staircase between tiles, grass emerges through
  it. Near-mid field carries a fine material grain; beyond the collapse band it
  is smooth.
- `debug=normals`: up-dominated with fine variation (was rainbow noise).
  `debug=coverage`: 1.0 across the collapsed field; in the near field a faint
  regular lattice appears where two overlapping tiles both present a marginal
  edge texel — visible in the debug view, not in the shaded image.

### What is still bad

- **The mid-field grain is undersampled, not textured.** The atlas has no mip
  chain, and it cannot easily get one: the reconstruction reads depth with
  `textureGather`, which is mip-0 only in WGSL. So between ~1m and the collapse
  band each pixel point-samples one of many texels it covers. The collapse hides
  the worst of it and the band position is a param, but the real fix is a
  prefiltered lookup (coverage-weighted depth mips + a non-gather path), which
  is a different sampling design, not a tweak.
- **Zone boundaries are tile-quantized in the collapsed band.** A collapsed tile
  is one species' flat colour, so the jittered wetness boundary reads as stepped
  0.18m patches at 15-40m. In the near field the cushion texture hides it.
- **Grazing incidence is ~10% brighter than the light field would be** inside
  the blend band (measured 64.7 vs 78.5 luma fully collapsed at the `grazing`
  cam, so ~10% at the blend weights actually used there). A rough mat seen edge
  on genuinely darkens; the mean-face normal only approximates that.
- **Cost.** Same-frame A/B against 001-billboard-smoke on `bog` at grazing:
  A/cards 4.08ms vs B/qlf-cards 136ms (contended; the B slot also runs
  1.26-1.35x slow). That was before the frustum reject and the collapse, which
  together took bog grazing from 113ms to 85ms and slope from 100ms to 41ms in
  comparable windows — still roughly an order of magnitude off the billboard
  baseline. The structural reason is enumeration, not shading: 3 moss entries x
  2401 cells x 484 slots x 6 vertices = **21M vertex invocations per frame**, of
  which ~300k plants survive. Two identified fixes, neither implemented here:
  (a) a compute cull prepass compacting survivors into a 4-byte-per-instance
  buffer (cell offsets + slot, re-running `scatter_candidate` in the vertex
  stage) + `drawIndirect` — 1.55MB/species at the worst-case bog survivor count,
  keeping the species inside 25MB, and it cuts the scatter evaluations ~6x and
  the vertex launches ~10x; (b) beyond the collapse distance, drawing one slab
  per 2x2 block of tiles (a collapsed slab is rotation-invariant and 0.36m of
  ground is still sub-2px at 24m), which is another 4x on the far field but
  needs a per-block zone assignment and is therefore a judgement call about the
  placement contract.

### Reproducing the screenshots

`/tmp/verify-moss-015-factored-lightfield.ts` (playwright, chrome channel,
`--enable-unsafe-webgpu`) captures bog at carpet-close / grazing / topdown /
inside-plant / two sloped eye-level poses / normals / lighting / coverage /
albedo, then `default` at all four standard cams; `/tmp/zoom-015.ts` takes
3x-scale crops where tile-level artifacts would show. All timings quoted above
were taken while ~13 sibling agents shared this GPU (the harness's own
`composite` pass read 5-240ms across runs, so absolute numbers are worthless);
only same-frame A/B ratios and within-run comparisons are used. No bench JSON is
claimed — rerun `#/bench/015-factored-lightfield?stand=bog` on an idle GPU.

### Verdict: is this representation suited to a moss carpet?

Better than expected — with one caveat about which part of the range.

The method's core assumption is that first-hit depth per view captures the
geometry. That is *exactly* true for a Sphagnum cushion: it is a height field
with 3.3cm of relief and no thin structures standing off it, which is the
opposite of the fluffy calamagrostis panicles where this method's known
"balloon lobe" artifact comes from. Close up (<3m) the mat has real thickness,
real silhouette, correct depth against the terrain and cushion-scale relief
shading — the things a flat billboard quad structurally cannot have. So the
answer to "can it express thickness" is yes, and that is where it beats the
reference.

Where it is a poor fit is the *far* field, and for a mat that is most of the
screen: 1.13M tiles at life size, each a proxy card doing 4-8 texture gathers,
prefiltered by nothing. The technique has no natural way to aggregate — which is
why the honest answer there was to stop using the light field entirely past a
few metres and draw the mat as the slab it is. That is a good result visually,
but it is worth being clear that beyond ~15m this renderer is no longer a light
field at all; it is 001's flat-quad idea with a volume-sampled colour and a
mean-face normal.
