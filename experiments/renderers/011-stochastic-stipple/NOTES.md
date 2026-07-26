# 011 — Stochastic thinning cascade ("opacity as statistics")

## Idea

Distant overlapping plants are structured noise; render the statistics, not
the plants. Three hierarchical stochastic levels, all fed by one baked
"coverage card" per species (9-view atlas: 8 azimuths + top; PREMULTIPLIED
albedo·coverage + plant-local normals, full GPU-generated mip chain — so every
mip texel is exactly the fractional-coverage / mean-radiance statistic of its
footprint):

1. **Plant level — stochastic thinning with conserved ensemble coverage**
   (Cook-style aggregate simplification, made continuous). A plant survives at
   distance d iff its scatter slot rank r = (slot+0.5)/128 < keep(d) =
   min(1, (R0/d)^2); survivors widen by 1/sqrt(keep) (capped), so the expected
   total frontal area of the ensemble is preserved while the drawn count falls
   as 1/d². Because survival is keyed to the slot INDEX, enumeration can be
   hierarchical: concentric cell annuli around the camera where ring k doubles
   the radius (quadrupling area) and quarters the slot cap — constant work per
   ring, O(rings) total. Defaults: 4 rings to 256m, ~163k candidate instances
   per stand entry (~0.98M lightweight VS invocations, most early-out) — fully
   procedural via the shared scatter WGSL twin, zero instance buffers, cost
   INDEPENDENT of stand plant count (134M-plant scaling stand renders the
   same work as 20k).
2. **Pixel level — hashed coverage realization.** The mip-averaged alpha is a
   coverage probability; the fragment shader realizes it with a hash threshold
   anchored to (plant id, card texel). Near the camera the threshold is a hard
   0.5 (crisp silhouettes, solid early-z occluders); it morphs to fully
   stochastic between stochStart(25m)..stochFull(70m). Amplified clump cards
   additionally "densify": a_eff = 1-(1-a)^γ with γ grown by the amplification
   — the coverage a card standing in for 1/keep overlapping plants would have.
3. **Field level — mean-field carpet.** Beyond the radius where width
   amplification saturates (widthCap·R0), a terrain-following canopy sheet at
   mean canopy height dissolves in underneath the cards (1m hashed patches):
   species-mottled (2.5m patches weighted by stand density × baked top-view
   coverage), colored by the baked top-view premultiplied mean (the exact
   color the ensemble converges to from above), lit with terrain normals +
   hash perturbation, with a statistical gust-wave shimmer standing in for
   per-plant sway. One draw, ~12k tris, camera-bounded.

Wind: shared `wind_sway` sheared over card height per plant (species sway from
the stand table, per-plant phase from the scatter hash). Lighting: baked
normals rotated by plant yaw, blended toward the statistical canopy normal at
distance, shared light_surface + fog. Camera-inside-plant: cards collapse
within ~0.75m. All three species rendered, exact stand placement via
`scatter_candidate` (positions/scale/yaw/phase bit-identical to the CPU twin).

**Carpet species (`stand_table[i].carpet_div > 0`, i.e. the bog's Sphagnum) take
a fourth shape: a nested SHELL STACK over a periodic relief tile.** None of the
three levels above applies to a 0.18m cushion tile — see "Moss carpet" at the
bottom for what replaced them and why. Level 3 (the mean field) survives and
becomes the mat's far field; levels 1 and 2 (thinning, hashed coverage) are
switched off for a mat by construction.

**Why stochastic alpha is justified here (taste rule):** it is not a soft-edge
dither — it IS the distance-collapse mechanism (the seed direction of this
experiment). The near field (< 25m), where early-z and silhouette crispness
matter, uses a hard 0.5 alpha test with full depth writes; the hashed test
only takes over where cards are small and the coverage statistics are the
signal. The hash is anchored in card-texel space (stable per plant), not
screen space, so it does not swim under camera translation.

## VRAM budget math

Scattered species: two rgba8 768×768 atlases with 7 mip levels ≈ 768²·4·(4/3) ≈
3.1MB each → **~6.3MB per species** (HUD: 6.0/25MB), plus 320B uniform per
stand entry. No instance buffers at any plant count. Mean-field sheet: 176B
uniform, card/shell quad index buffer: 72B. Well inside the 25MB budget.

Carpet species: two rgba8 512×512 tile textures with 8 mip levels ≈
512²·4·(4/3) = 1.4MB each → **~2.8MB per species** (HUD: 2.7/25MB). That is
less than half the card budget and buys 512 texels across 0.18m
(~2800 px/m) instead of the ~750 px/m a 9-view card atlas leaves on a tile's
own square — the reallocation is the point: eight azimuth views of a 7cm-tall
mat are almost entirely empty sky, and the mat never shows them.

## Bake

`bake.ts` renders the raw GCMESH1 mesh (any complexity; poa's 6.5M tris
included) once per species into the 9-view ortho atlas pair on the GPU
(~1-2s), reads back level 0, and commits `mesh/baked/011-stochastic-stipple/
<species>-v1.bin` (4.7MB each, validated magic GCSTIP1 on load — the dev
server's index.html fallback for missing files is detected and ignored).
Mips are regenerated on the GPU at load. Committed for all three grasses.

Carpet species get a different artifact, `<species>-carpet-v1.bin` (2.0MB,
magic GCMOSS1, committed for all three Sphagnum states): ONE top-down
orthographic capture of exactly the tile's own 0.18m period, rendered at 1024²
and box-filtered to 512², two MRT targets, everything premultiplied by
coverage so the runtime mip chain averages statistics:

- `albedo`: rgb = albedo·cov, a = cov
- `nrmh`:   rgb = (n·0.5+0.5)·cov, a = cushion-top height·cov

The mesh is drawn **3×3 times at ±one period** — the tile's geometry overflows
its own period (0.24m of cushion inside a 0.18m tile) and the mat is periodic,
so the neighbours' overhang is exactly what has to wrap in. Cropping without
the wrap shaves every capitulum that leans over a seam. The height channel is
what turns one quad into a shell stack; its inverse CDF (16 samples) is
computed at load (`heightQuantiles`, a 512² pass, ~5ms) so the artifact format
did not need to carry it. Measured: coverage 93-98% of the tile, cushion-top
height 4.5cm..8.5cm (2nd..98th percentile) of a 9.3cm geometry range — i.e.
the visible surface lives in the TOP HALF of the mesh's height range, which is
why evenly spaced shells are wrong and equal-area ones are right.

## Status

working — verified by headless screenshots: grazing, topdown, inside-plant
(default stand), far-horizon (scaling-100m, 134M plants), debugRings ring
placement, and all six `debug=` views. No console errors; typecheck clean.

Carpet path verified on the `bog` stand: grazing, carpet-close (1m straight
down), a 20cm-high oblique pose, topdown, inside-plant, far-horizon, two poses
across the ridged slopes (downhill at 32° and looking up the same hillside),
albedo/normals/lighting/coverage, plus shell-tint (`debugRings`) and 1-shell
control renders. `default` is **bit-identical** to before the carpet work — 0
differing scene pixels at grazing, topdown and inside-plant (HUD/panel masked,
`?det=1&t=3`).

## Debug views

Both fragment shaders (`cards.wgsl`, `carpet.wgsl`) route their final colour
through the shared `debug_shade()` and apply fog only when
`debug_mode() == DEBUG_OFF`. What each view carries here:

- **albedo** — the un-premultiplied mean radiance of the covered area
  (`alb.rgb / alb.a`) times the vertical card AO ramp; carpet: the mottled
  species mean colour. No lighting anywhere in it.
- **normals** — the real decoded per-fragment card normal: the normal atlas is
  premultiplied and mip-averaged exactly like the albedo, un-premultiplied
  (`nrm.rgb / nrm.a`), rotated by plant yaw into world space, then blended
  toward the statistical canopy normal by `sigma`. It shows the expected
  camera-facing hue gradient across the screen (the bake faceforwards each
  view's normals toward its capture direction) with per-leaf variation inside
  it — the same rainbow speckle 000-ground-truth's real geometry produces.
- **lighting** — the shared `light_surface()` term alone. It reads mostly
  blown-out white; that is the harness model (SUN_COLOR 1.15 + AMBIENT 0.21
  clips >1), and 000-ground-truth's lighting view looks identical, so it is
  not a double-applied or skipped lighting term.
- **coverage** — `a_eff`, the densified fractional-coverage statistic the
  hashed test actually realizes (carpet: its dissolve ramp). Near field ~1,
  falling with distance — the technique's core quantity, visible directly.
- **depth** — standard ramp; shows the stipple holes punched through the
  depth buffer past `stochStart`.

`debugRings` (own manifest param) tints cards by cascade ring; it is applied
only in the normal view so it can't pollute the debug views.

## Findings

- Renders a full closed meadow from grazing to horizon; the thinning cascade
  transition is invisible in stills (debugRings shows the bands land at the
  intended distances with no seams — survival is distance-keyed, rings are
  only an enumeration structure).
- scaling-100m (±2048m, ~134M plants) renders with identical per-frame work
  to the default stand; the carpet carries everything past ~256m under fog.
- Carpet color MUST come from the baked top-view premultiplied mean, not the
  vertex-color mean — the vertex mean is biased by the (pink/straw) flower
  heads and produced orange speckle artifacts during dissolve; fixed.
- Bench NOT run / no numbers claimed: 16 experiment agents were hammering the
  same GPU all session, so any timing would be contaminated (observed ~120fps
  at 1280×800 solo-ish early in the session; stipple pass p50 fluctuated
  6.7→16ms purely with sibling load). TODO bench on a quiet machine:
  `#/bench/011-stochastic-stipple?stand=default&spline=orbit-low`.
- Known limitations / open questions:
  - Close-up (< ~3m) cards read flat, as any single-card impostor does; the
    9-view blend smears slightly at mid elevations (classic impostor smear).
  - Temporal behavior of the hashed test under camera MOTION only partially
    assessed (static deterministic captures + reasoning about the card-texel
    anchoring); plant-scale dissolve pop-in is visible by design near the
    survival boundary — smoothed by the 35%-of-keep fade band, but a video
    pass would be the honest check.
  - VS enumeration wastes ~50-60% of invocations on nonexistent/rank-killed
    slots (early-out after ~2 hashes); a compute-compaction prepass would
    roughly halve vertex work if the pass ever shows up hot on target GPUs.
  - Top-view uv de-rotation sign was validated only visually (yaw-symmetric
    clumps make the error subtle either way).
  - Seen from straight above the card normals collapse toward +Y (the bake
    faceforwards the top view's normals toward its capture direction), so the
    topdown normals view is near-uniform green with only fine speckle. It is
    a property of single-sided card capture, not a missing decode — the
    grazing views carry the full normal variation.

## Audit (structural-waste + debug-view pass)

Debug views did not exist here at all: neither shader included `debug.wgsl`,
so both cards and carpet ignored the global `view` selector and always drew
lit+fogged pixels. Both are now wired (see **Debug views** above). Nothing
was broken behind them — normals were already real, per-fragment and decoded
from the mip-averaged normal atlas, lighting already went through the shared
`light_surface()` exactly once, and the bake stores authored albedo (not
pre-lit), so the debug wiring changed no pixel in the normal view.

Structural waste found and fixed (all four verified image-identical; see the
verification note below):

1. **Card quads drawn non-indexed.** `draw(6, N)` shaded 6 vertices per card
   where the two triangles share two corners. Now a 12-byte index buffer +
   `drawIndexed(6, N)` with a 4-corner array: the post-transform cache shades
   **4 instead of 6** vertices per instance — a third off the heaviest stage
   in this method (~163k instances × 3 entries, each doing hashes, terrain
   fetch, billboard basis and wind), for identical triangles.
2. **Per-fragment work that is constant over the card.** The FS recomputed
   `cos(yaw)`/`sin(yaw)` *twice*, plus `smoothstep(stochStart, stochFull, d)`
   and the densify exponent `gamma`, all from flat varyings. They are now
   computed once in the VS and carried flat (`yaw_gs`, `top_vf`); `amp` and
   `dist` no longer need to reach the FS at all. Same varying slot count.
3. **Normal atlas sampled before the alpha tests.** Both atlases were tapped
   up front, then ~half the fragments were discarded by the coverage/hash
   test. Coverage lives entirely in the albedo alpha, so the 2–3
   `textureSampleGrad`s of the normal atlas now happen *after* both discards —
   discarded quads never touch it. (Safe: gradients are taken in uniform
   control flow and the taps use `textureSampleGrad`.)
4. **Carpet drawn before the cards.** The near card layer is a solid, hard-
   alpha-tested, depth-writing occluder and the carpet is a large 54m→800m
   sheet largely hidden behind it. Order can't affect the image (both are
   depth-tested opaque, no blending), so cards now draw first and early-z
   rejects the covered carpet fragments instead of shading them.
5. **Per-frame CPU/uniform churn that depends on nothing per-frame.**
   `carpetStats()` (a pure function of stand + baked meta) was recomputed and
   re-uploaded every frame — now computed once at create. The ring table was
   rebuilt (allocating row objects) every frame and all three 160-byte card
   uniforms were rewritten every frame, although their only time-varying
   field is the camera's 4m cell. Both are now change-driven: a still camera
   with untouched params uploads zero bytes.

Deliberately left alone:

- **The ~50-60% of vertex invocations that early-out** on nonexistent or
  rank-killed slots. Fixing it means a compute compaction prepass — a
  redesign, and its win depends on measurement this session cannot provide.
- **`scatter_candidate()` recomputing the `hash4` + offset hashes** the VS
  already evaluated for the cheap existence/rank rejection. Inlining them
  would duplicate the shared placement twin inside experiment code and risk
  silent divergence from `src/wgsl/scatter.wgsl`; the redundancy is a handful
  of ALU ops on the surviving path only.
- **The ≤6-iteration ring lookup loop** and the 4-entry species-pick loop in
  the carpet: bounded, tiny, and collapsing them is pure ALU shuffling.
- **The carpet's non-indexed 11.5k-vertex draw** (indexing saves ~4k vertex
  invocations — noise next to the base terrain pass).
- **Mip level 6 is generated but `lodMaxClamp: 5` never samples it** (~576
  bytes and one load-time blit). Left for the owner: dropping `MIP_LEVELS` to
  6 would also change the committed bake's shape for no measurable gain.

Verification: `?det=1&t=3` (paused, deterministic) canvas captures at
grazing/topdown/far-horizon, before vs after, diffed with ImageMagick. Two
runs of the same build are bit-exact (0 differing pixels), and before-vs-after
differs by **11 / 3 / 3 pixels** out of 422k in the HUD-free crops — 1-ULP
differences from evaluating `cos`/`sin`/`smoothstep` in the vertex stage
flipping a few fragments across the alpha threshold. Visually identical.

## Moss carpet (bog stand) — shell stack over a periodic relief tile

Sphagnum palustre arrives as a 0.18m periodic community tile, 0.07-0.09m tall
and ~98% closed seen from above, laid out by the `bog` stand as a grid-snapped
mat (22×22 tiles per 4m cell = **484 slots**, life size, 90°-only yaw). Every
assumption in this renderer was wrong for it. Before: a 0.63m **camera-facing**
square per tile (`r_card` is the mesh's bounding diagonal, and the whole card
is as tall as it is wide), only 128 of the 484 slots enumerated, distance
thinning + width amplification applied to a surface that must stay closed, and
the camera-inside fade collapsing every card within 0.75m — so standing on the
mat opened a metre-wide hole in it. It read as rows of green/brown bricks
floating over bare peat, with a hole under the camera.

What changed, smallest first (all inside this experiment):

1. **Enumerate `carpet_div²` slots, not `SCATTER_MAX_PER_CELL`.** A carpet
   entry's per-cell slot count comes from `standEntrySlots(entry)` (484 here),
   and the cell block is a plain camera-centred square of `ceil(mossRadius/4)`
   cells rather than the ring cascade. One line of arithmetic; without it three
   quarters of the mat simply does not exist.
2. **No thinning, no hashed alpha, no camera-inside fade for a mat.** Survival
   by slot rank, `1/sqrt(keep)` width amplification and the stochastic alpha
   test are all illegal on a carpet: the first two break the lattice (the tile
   scale is the stand's, and 90° yaw steps are the only rotation a periodic
   square tolerates), and the third punches holes in a surface whose whole job
   is to be a closed, depth-writing occluder. The carpet path uses two hard
   tests (coverage ≥ `mossCoverRef`, height ≥ the shell's band) and nothing
   else. This is the same taste rule the stipple obeys near the camera, applied
   at all distances because a mat is never an ensemble of separable plants.
3. **Ground-parallel, grid-snapped, tile-sized quads.** Width comes from
   `footprint_m · scale_min` (= the 0.1818m grid step), never from
   `height_scale`. The quad stays axis-aligned in world space and the scatter's
   quarter-turn yaw is applied to the TEXTURE instead (a quarter turn maps the
   square onto itself, so neighbours keep sharing corner positions exactly —
   that is what keeps the mat crack-free). Overscale is **not** used: tiles abut
   exactly at 1.0 and the periodic wrap in the bake means the overhang is
   already inside the texture, so there is nothing an overscale would fix.
4. **Repurposed the bake: one 512² top-down relief tile instead of nine card
   views** (see Bake). ~3.7× the linear texel density on the tile's own square
   versus cropping a card atlas, at 2.7MB instead of 6.3MB per species.
5. **Thickness: a nested shell stack.** Shell *k* is a ground-parallel quad
   sitting at the median height of the *k*-th **equal-area band** of the tile's
   cushion-top height field, and keeps only the texels whose cushion reaches
   that band. The stack is a quantized height field: real silhouette, real
   parallax, honest depth, no transparency anywhere. Two details matter:
   - the bands must be **equal-area quantiles**, not evenly spaced heights. The
     cushion top occupies 4.5-8.5cm of a 9.3cm range, so evenly spaced shells
     put three coincident full-coverage planes in the empty air below the
     cushions and express the relief with one. `debugRings` tints by shell and
     shows the intended contour map of capitulum tops.
   - shell 0's threshold is 0, so it covers the whole tile. Shells collapse with
     distance (each higher one dies ~40% nearer, metric = 3D distance including
     the camera's height over the ground), and because shell 0 never dies, no
     LOD step can open the mat.
6. **Cushion-dome shading from the height field's gradient** (`mossRelief`).
   This is the single biggest visual win at 0.5-2m. Leaf-scale normals average
   toward straight up as the mip level rises — that is what the average of a
   two-sided leaf canopy IS — so a mat 1m away lit with mip-2 normals is a flat
   sheet with fine noise. The height field mips into the *cushion* shape, and
   its gradient taken one screen pixel apart (so the scale always follows what
   the screen resolves) puts the dome light-and-shade back. 0.6 is the default;
   1.2 is visibly bumpier but darkens the mat and starts to look crunchy.
7. **The mean field became the mat's far field.** Tiles are drawn to
   `mossRadius` (32m default) and the sheet takes over just inside that, below
   the cushion tops, so the tiles occlude it wherever they are drawn. Its
   colour now comes from the **wetness field** rather than a per-patch hash
   whenever the stand has carpet entries: the entries' mean colours blended by
   distance to each one's band centre. On a zoned community that is the correct
   first-order statistic, and it makes the tile→sheet boundary invisible —
   `mossRadius` 20 / 32 / 56 are indistinguishable at grazing. With the old
   hash mottle the far field broke into a 2.5m ochre/green checkerboard, which
   was tolerable when the sheet started at 54m and glaring at 20m.
8. **Lifted the mat out of the drawn terrain in hollows** (`mossLift`). The
   base pass draws the terrain as 1m quads while `terrain_sample()` — the
   surface a per-vertex-conforming plant uses — is the finer bilinear
   heightmap. Measured over a 80×80m patch: the drawn mesh sits **up to 7.2cm
   ABOVE** the true surface (≥1cm over 19.8% of the area, ≥3cm over 1.6%),
   because its chords bridge every concavity. A mat conformed to the true
   surface therefore sinks into the drawn ground in hollows and gets swallowed
   whole. Fix: add the local concavity (`terrain_plane_fit(xz, 0.5).h` minus the
   point height, ×1.5) per CORNER — continuous across tiles, assumes nothing
   about the base pass's tessellation, mean lift 4mm, worst residual 4.2cm
   (below the base shell's 6.5cm, so nothing is swallowed). Costs 4 extra
   terrain fetches per carpet vertex. On *this* terrain it is nearly invisible
   (the ≥6.5cm case is 0.001% of the area); it is in for the regime where it is
   not, and it is a param so it can be switched off.

### Terrain fitting: ladder rung 3 (per-vertex), plus a plane fit for the tilt

Every quad corner gets its own `terrain_sample(xz)` — one bilinear fetch for
height and (nx, nz) together. Rung 3 and not 1 or 2 for a structural reason,
not a cosmetic one: neighbouring tiles **share corner positions**, so a
per-vertex fit is the only rung that keeps the mat C0-continuous. Rungs 1-2 fit
one plane per tile, two neighbours pick two different planes, and the mat cracks
along every shared edge. The shells are offset along the ground normal rather
than straight up, so a cushion on a slope stands out of the slope; where
`mossLift > 0` that normal comes from the plane fit over ±0.5m (rung 2 for the
*tilt*, rung 3 for the *height*), which is smoother than the point normal and
already paid for. `slope_align` (1 for carpets) scales the tilt and is honoured
rather than assumed.

### What is better, what is still bad

Better: the mat is closed everywhere including directly under the camera, at
life size, on the slopes, and at 1m it reads as cushions with light and shade
rather than a texture swatch. Against 001-billboard-smoke on the same stand
(`#/ab/001-billboard-smoke/011-stochastic-stipple?stand=bog&cam=carpet-close`)
011 has visibly more cushion structure and no tile grid, at 2.7MB/species
against 001's 28.5MB (which is over budget). Where 001's flat quad shows
smooth low-contrast moss, the shell stack shows relief, and its gaps fall to
lower shells instead of showing terrain-coloured dots.

Still bad / honest limits:

- **The relief is small because the geometry's relief is small.** Area-weighted,
  the cushion top spans ~1.5cm; the manifest's 3.3cm `reliefH` is the extreme.
  Four shells over that band give ~5mm steps — visible as parallax when moving
  and as a soft fringe at grazing, but nobody will mistake it for the mesh.
- **At grazing from 20cm the mat is noisy mush.** 512 texels over 0.18m
  compress to almost nothing vertically at 9° elevation; 001 is smoother there
  and arguably no worse. This is the one view where the extra texel density
  does not help.
- **Distant zone boundaries read as flat colour blocks.** Beyond ~8m a tile is
  under 20px, so every tile is at its mean colour, and where the late-season
  zone interlocks into the wet-vigorous zone you get hard-edged ochre patches
  in green (they have a dark rim because the late-season cushion is genuinely
  2.3cm shorter). This is the stand's zoning rendered faithfully, not a
  renderer bug: 001 shows the same patches at the same world positions, and the
  CPU scatter twin confirms 0 unclaimed and 0 double-claimed nodes over 34,848
  nodes near that camera (the old carpet-jitter hole bug really is fixed).
- **Tile repetition** is hidden by the three species zones and four rotations,
  but there is exactly one tile per species; a second tile variant per species
  would cost 2.7MB and remove the last of it.
- Beyond `mossRadius` the mat is a smooth mean-colour sheet. It matches in
  colour, but the detail transition is visible from directly above (topdown at
  42m shows a faint disc edge).

### Is this representation suited to moss?

Partly — with an honest asterisk. The stipple's own machinery (stochastic
thinning, hashed coverage realization, width amplification) is *unusable* on a
carpet: all three assume separable plants whose ensemble can be resampled, and
a mat is one surface. Two thirds of the technique is therefore switched off for
this species, and what renders the moss is a different method living in the
same shaders — a shell-stacked relief tile — plus the one level that does
transfer (the mean field, which is genuinely the right far-field answer for a
closed zoned ground cover). As a *renderer of moss* the result is good: closed,
life-size, terrain-conforming, cheap (2.7MB/species, ~1.2ms of the pass at
32m), and it beats the flat-quad baseline on structure. As *this experiment's
idea applied to moss*, it is a graft, and NOTES should say so rather than
pretend the cascade scaled down to a cushion.

### Perf

Timings this session are unusable as absolutes (up to 16 sibling agents on the
same GPU; the harness's own base and composite passes moved by 3-5× between
identical runs). Relative, same-run measurements on the bog at grazing:
stipple p50 ≈ 3.9ms with `mossRadius=8`, 5.1ms at 32 (default), 6.5ms at 56;
shells 1 → 4 costs ~0.3-0.5ms; the 3D-distance shell LOD cut topdown from 6.2
to 3.6ms. On `default` the stipple pass is bit-identical work (proven by 0
differing pixels) and its p50 wandered 3.7-5.1ms across five identical runs
while the untouched composite pass wandered 2.4-3.3ms in lockstep — i.e. all
of that is contention. No bench JSON is claimed; rerun
`#/bench/011-stochastic-stipple?stand=bog&spline=orbit-low` on an idle GPU.

### Harness feedback from this round

- **`ctx.stand.species[i].scaleMin` is NOT the carpet scale.** `carpetScale()`
  is applied only when building the GPU stand table (`createStandBuffer`), so
  the TS stand object still carries the placeholder 1.4-2.5 for carpet entries
  — and `new Scatter(terrain, seed, stand)` reads `e.scaleMin`, which means the
  CPU twin places carpet tiles at 1.7× while the GPU twin uses 1.0101×. Any
  renderer that materializes instances on the CPU gets a mat at the wrong
  scale. Normalizing the stand once (or exposing `standEntryScale(entry)`)
  would fix both the divergence and the guesswork.
- **The base pass's terrain tessellation is coarser than the placement
  surface** (1m quads vs the bilinear heightmap; up to 7.2cm of gap, numbers
  above). Every renderer that follows CLAUDE.md's rung-3 advice inherits a mat
  that sinks into the visible ground in hollows, and each one has to invent the
  same lift. Either tessellate the base pass to the heightmap texel grid, or
  expose the drawn surface (e.g. `terrain_drawn_height(xz)`) so conforming
  geometry can agree with what the eye sees.
- `stand_table` has `footprint_m`, `carpet_div` and `slope_align` and they were
  exactly what was needed; nothing else was missing for the carpet path.

### One diagnosis worth recording (it cost an hour)

The ochre patches in a green hollow looked exactly like missing tiles, and the
first two hypotheses (the fixed carpet-jitter hole bug; the mat sinking into
the coarse drawn terrain) were both wrong. What settled it: the patches survive
`p.carpet=0`, `p.mossRadius=8`, `p.mossLift=0/2` and the shell tint unchanged,
appear at identical world positions in 001-billboard-smoke, and in
`debug=albedo` measure (0.416, 0.416, 0.133) — the late-season tile's own
coverage-weighted mean colour to within 1.5%. They are simply that species'
zone, interlocking into the wet zone the way the boundary jitter intends, seen
past the distance where a 0.18m tile keeps any texture. Sampling a debug view's
pixels against the baked per-species means is a fast way to identify *which*
species a suspicious patch is, and worth doing before changing any geometry.
