# micro splats

## Idea

A plant is not a card and not a blob cloud — it is **a few thousand tiny
oriented surfels welded to the real mesh surface**, drawn as hard-edged
elliptical micro-cards that go through the ordinary depth buffer. Nothing is
camera-facing inside ~25 m, so the parallax, the view-dependent silhouette, the
self-occlusion and the ground contact are not tricks: they are what real 3D
geometry does for free. The whole technique is about making that geometry *tiny
and cheap enough* that a screenful of it costs about what a screenful of
billboards costs.

**Bake (once per species, from the raw GCMESH1).**

1. The triangles go into a **perfectly balanced kd-tree**: 13 median splits along
   the longest axis of the current centroid box. Every leaf is therefore a
   spatially compact, equal-population patch of surface. Because the splits are
   balanced, *every tree level is a complete clustering of the plant* — level 13
   is 8192 patches, level 0 is 1 patch. **That is the LOD ladder, obtained for
   free**, and because area moments are additive a coarse splat is the exact
   union of the fine ones underneath it (no separate coarse bake, no popping
   between unrelated representations).
2. Each node's area-weighted moments give a centroid and a covariance. The two
   largest PCA axes span the splat's plane, the smallest is its normal
   (sign-locked to the area-weighted mesh normal). Half-extents are 1.9σ, so the
   ellipse *hugs* the patch. This is where the previous splat attempt
   (002-splat-reconstruct, owner rating 2/5) went wrong: big camera-facing
   ellipses detached from the surface. Here they are small, oriented by the real
   surface frame, and sitting exactly where the surface sat.
3. `projected area / ellipse area` is the patch's **coverage ratio**. Each splat
   stores it plus a 4-bit index into a 16-tile **stamp atlas cut out of a real
   orthographic render of the plant** — 16 64px windows picked so their measured
   coverages span [0,1], transposed so the source's mostly-vertical blades run
   along the tile's U axis, which is the splat's major PCA axis. A 6px splat
   therefore still shows individual blades and a blade-shaped silhouette instead
   of a flat chip. Mips are built per tile with **Castano-style alpha rescaling**
   (each level's alpha is scaled so the fraction of texels above the alpha
   reference matches mip 0), which is why the hard alpha test survives all the
   way to the horizon instead of dissolving the far field away.
4. A 24³ area-density grid, suffix-summed along +y, gives every *triangle* a
   canopy-occlusion value. Colour is accumulated **pre-multiplied by that
   occlusion and divided by its sum**, so `colour × ao` reproduces the patch's
   true mean radiance at every tree level — a distant plant takes the colour of
   its lit outer shell instead of the average of shell and dark interior. This
   single change is what turned the far field from a dark brown crust into a
   meadow.

**Per frame (two passes, both region-bounded).**

- `cull` (compute): one workgroup per half-cell of a camera-centered cell region
  (already clamped to the stand's cell range on the CPU). Whole-cell
  region/frustum rejects are workgroup-uniform, so most of the window exits
  before a terrain texel is fetched. Survivors run the shared
  `scatter_candidate` WGSL twin (bit-identical placement), get frustum-culled,
  and pick their LOD from **one `log2`** of projected pixel height — the ladder
  is geometric (each level halves the splat count and moves the switch out by
  √2), so there is no threshold table walk and no loop. Per-plant work that the
  draw would otherwise redo in every one of a plant's ~32k vertex invocations —
  yaw sin/cos, the wind displacement (4 transcendentals), the coverage fade — is
  finished here, once, into a 32-byte record appended to that LOD's bucket +
  indirect args. Fully faded plants are dropped outright.
- `splats` (render): 12 LOD × 3 entries `drawIndexedIndirect`, **near LOD first
  across all species**, so early-z has the nearest opaque geometry down before
  the deep field behind it is rasterized. 4 indexed vertices per splat. The
  fragment shader is **one texture tap**, a hard alpha test, an ellipse cut, a
  two-sided normal bent across the minor axis (each micro-card reads as a curved
  blade, not a flat chip), and the shared `light_surface` + fog. No marching, no
  per-pixel loops, no frag_depth, no blending, no dither.

O(1) in plant count: no global plant array exists anywhere; the cull window is
fixed by `regionRadius`, and each plant's cost collapses 2× per √2 of distance.

## VRAM budget math

Per species (HUD-tagged; measured 5.7 / 4.9 / 4.7 MB of 25 MB for
calamagrostis / elymus / poa on the default stand):

- splat hierarchy: 16383 splats (8192+4096+…+1) × 16 B = **262 KB**
- stamp atlas: 256² rgba8 + 3 mips = **340 KB**
- LOD buckets: 12 per-level instance buckets, each sized from the ring the level
  owns at `REGION_MAX = 128 m` widened ±2.6/1.6× so the `lodBias` slider cannot
  starve one, × the entry's density × 0.6 frustum fraction + 1024 slack. For
  density 3 that is ~169k records × 32 B ≈ **5.4 MB**, and it is the only term
  that matters. The far bucket dominates (91k of the 169k), which is correct: it
  is the only level that ever holds tens of thousands of plants.
- per-entry uniforms + indirect args: ~4.5 KB
- shared, not per species: 98 KB static quad index buffer

Splat record, 16 B: pos u16×3 | oct normal u8×2 | tangent angle u8 |
half-extents u8×2 (sqrt-encoded against a per-level max) | stamp 4b + coverage
4b | rgb u8×3 | canopy occlusion u8. The tangent angle is stored in a branch-free
Duff orthonormal basis built from the *dequantized* normal, and `splats.wgsl`
rebuilds that basis with the identical formula, so the PCA major axis
reconstructs exactly.

## Bake

`loadSpeciesSplats()` in `bake.ts`, in-browser from the raw GCMESH1, ~10-20 s per
species (kd-tree quickselect over ≤1.5M sampled triangles dominates), committed
via `commitBake` to:

- `mesh/baked/025-micro-splats/msplat-v2-calamagrostis-canescens.bin`
- `mesh/baked/025-micro-splats/msplat-v2-elymus-repens.bin`
- `mesh/baked/025-micro-splats/msplat-v2-poa-pratensis.bin`

596 KB each ('MSP1' format, layout documented at the top of `bake.ts`); the v1
variant was deleted, nothing else is left lying around. Triangles are sub-sampled
to ≤1.5M with a uniform stride (area-unbiased). Community tiles and the single
poa specimen go through the identical path; the capture box (`rXZ`, `y0`, `y1`) is
measured exactly as 001 measures it, so both put the same plant in the same place
at the same size and the A/B is honest. `bakedArtifact()`'s committed-file fetch
gets the SPA `index.html` at HTTP 200 when a file is missing, so every artifact is
magic-validated and a poisoned OPFS entry is re-baked and repaired.

## Status

**working** — verified by headless screenshots at grazing / topdown /
inside-plant / far-horizon on `default`, plus `close-quality`, `scaling-100m`
(134.2M plants) and all five debug views. Console clean everywhere.

## Findings

### Speed

Bench JSONs (1600×900, `orbit-low`, seed 42 — **the machine was shared with 15
parallel agents, so treat every absolute number here as an upper bound**):

- `results/025-micro-splats__default__p-d3087eb3__apple-metal-3__2026-07-25T03-28-56-857Z.json`
  — base 0.334 / cull 0.214 / splats 3.892 / composite 2.768 → **Σp50 7.21 ms**
- `results/025-micro-splats__scaling-100m__p-d3087eb3__apple-metal-3__2026-07-25T03-29-23-572Z.json`
  — base 0.384 / cull 0.192 / splats 3.759 / composite 2.512 → **Σp50 6.85 ms**

The two differ by less than the noise: **557k plants and 134.2M plants cost the
same**, at identical VRAM. At 1280×800 the solo run reports Σp50 5.0-7.5 ms
depending on how contended the GPU is that minute.

The only numbers worth quoting are the **same-frame A/B ratios** against
001-billboard-smoke (`#/ab/001-billboard-smoke/025-micro-splats?...`, contended
machine, both sides rendering the same stand in the same frame):

| cam | A own passes | B own passes | own-pass ratio | total Σp50 ratio |
|---|---|---|---|---|
| grazing | 2.03 ms | 3.59 ms | 1.77× | **1.34×** |
| topdown | 0.83 ms | 1.33 ms | 1.60× | **1.27×** |
| far-horizon | 1.04 ms | 2.27 ms | 2.18× | **1.53×** |
| inside-plant | 2.18 ms | 3.98 ms | 1.83× | **1.39×** |

So: ~1.6-2.2× the billboard baseline on its own passes, **~1.35× on the total
frame**, at 110 m region radius on both sides (identical to 001's default).
Not parity — honestly stated, this is "modestly slower", not "free".

What the cost actually responds to, measured by holding the A/B frame fixed and
sweeping one param at a time:

- **splat count**: halving it (`lodBias` 0.7) drops the own-pass ratio from
  2.53× to 1.54×; doubling it (1.4) pushes it to 3.63×. Cost ≈ count^0.55.
- **splat area**: shrinking every splat by 22% (`splatScale` 0.78, 39% less
  rasterized area) saved only 11%.

That is a **primitive-bound**, not fill-bound, renderer — the opposite of what I
assumed going in, and the reason the near field can afford 2048-splat plants
while the horizon cannot afford 8-splat ones. It also means the ellipse cut and
the alpha test are close to free, and that a depth prepass would be a waste.

### Looks — does it beat the baseline?

**Yes on the thing that was asked for, with one honest loss.**

- **The decisive image** is a 6 m oblique A/B on `close-quality`: billboards show
  the *same* camera-facing pink lozenge stamped over and over, all parallel, all
  flat — stickers on a green mat. Micro splats show individually oriented
  panicles (some leaning, some seen from above, some edge-on) with blades weaving
  between them and the canopy interior visibly darker behind them. Parallax
  inside the silhouette, a silhouette that changes shape rather than rotating,
  real self-occlusion, real contact with the soil.
- Moving the camera 0.6 m laterally rearranges *which* blades occlude *which*
  panicles — near parts shift against far parts. That is not a shader trick here;
  the splats are 3D geometry in the depth buffer, so it cannot not happen.
- **Top-down** is a clean win: 001 needs a faked horizontal "top card" that has
  to be eroded away at low elevations; this method just has geometry, and the
  top-down view is a rich varied carpet.
- **Where it loses:** beyond ~80 m the billboard's mip-filtered card is *smoother*
  than a 4-splat plant, which is grainier (feathery, not blobby — that took
  work — but grainier). And the whole field reads slightly greener/darker than
  001's, because a per-cluster occlusion-weighted mean is a greener statistic
  than the pink-dominant coverage-weighted imagery on 001's cards. Someone who
  wants a bright pink flowering meadow will prefer 001; someone who wants a
  meadow with depth will prefer this.
- **Worst case is `inside-plant`**, as the rules allow: blade-segment splats
  within half a metre of the eye become long dark straps against the sky. They
  are geometrically correct (that *is* a 27 cm blade segment seen from 30 cm) but
  they read badly, so their projected half-extent is capped at 170 px and the
  plant the camera is standing in is faded out by shrinking its splats to nothing.
  001 is also poor here (giant blurred cards), so it is roughly a wash.

### What mattered, in order

1. **Occlusion-weighted colour** (§4 above). Before it, the 25-100 m band was a
   dark brown crust and the method looked clearly *worse* than billboards. After
   it, the same band is a meadow. Nothing else came close to this in impact.
2. **Keeping the hard alpha cut all the way to the horizon.** My first version
   collapsed far levels to solid ellipses (fearing mip dissolution) and the
   horizon was a rim of chunky dark blobs. With coverage-corrected mips the alpha
   test survives, and the horizon becomes a feathery grass line. Only the last
   used level still blends 50% toward solid, with a 1.7× over-coverage shrink,
   because a canopy at that range genuinely is opaque by layering.
3. **A floor of 4 splats per plant.** 1-2 ellipses read as blobs on the horizon
   and saved almost nothing (those splats are ~1px — they are primitives, not
   fill). The bake still ships all 14 levels; the renderer uses 12.
4. **Hoisting per-plant work into the cull pass.** wind_sway alone is 4
   transcendentals, and it was being recomputed by every one of a plant's ~32k
   vertex invocations.
5. Per-plant jitter (±18%) on the LOD switch distance, so a level change reads as
   scattered plants changing rather than a ring sweeping across the field.

### Taste / rules compliance

No dithering anywhere: coverage falls off by **shrinking whole splats**
(camera-inside, region rim), which keeps every remaining edge hard and
depth-writing, and fully faded plants are dropped in the cull pass rather than
punched full of holes. Hard alpha test + depth write + near-LOD-first draw order
so early-z can reject. No `frag_depth`. One texture tap per fragment. Randomness
only via the shared PCG hash, animation only from `frame.time`.

### Debug views

`debug_shade()` is wired in the one place this method produces colour.
`albedo` = the baked cluster colour × the stamp's colour modulation, before
lighting and fog (a green field with pink panicles — the plant structure is
directly readable). `normals` = the per-fragment normal actually fed to
`light_surface()`: the baked patch normal, mirrored to whichever face is being
looked at (thin foliage) and bent across the minor axis; it is genuinely varied
per blade and shows the expected left/right hemisphere split at a grazing camera.
`lighting` = sun + ambient × the baked canopy occlusion (occlusion is occlusion,
so it lives in the light term, never in the albedo). `coverage` = the stamp alpha
this fragment resolved to, which with a hard alpha test is also the alpha-test
margin — it reads essentially binary, which is the wanted answer: hard edges, not
a screen door. Fog is applied only when the view is `off`.

### Not done

- The far field would be smoother with a temporal or sub-pixel component; every
  option I could think of either shimmers or needs history, so it stayed out.
- One triangle per splat (circumscribing the ellipse) instead of a quad would
  halve the primitive count — the measured bottleneck — at +30% rasterized area,
  which the sweep above suggests is a net win of maybe 15%. Untested; it changes
  the draw setup per level, so it wants its own verification pass.
- Colour statistics still read greener than 001's imagery. A fluffy panicle has
  enormous surface area but small projected footprint, so an area-preserving
  surfel set under-covers it relative to an orthographic render. Weighting the
  bake by projected rather than surface area would close the gap.
