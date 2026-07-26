# part-assembled plants (028-exact-silhouette)

Seed: *"get the silhouette exactly right, fill the interior as cheaply as
possible."*

## Idea

A billboard's silhouette is a flat cutout that rotates. The only way a cutout can
have the *right* outline from every direction is if the thing casting it is
actually three-dimensional — so instead of making one card cleverer, this method
**takes the plant apart and puts the pieces back where they belong in 3-D.**

**Bake (once, per species).** The source mesh is PARTITIONED — every triangle
belongs to exactly one part — into `BANDS=3` equal-height slabs x `SECTORS=4`
azimuth sectors around the clump axis (12 parts). Each part gets its own tight 3-D
box and is captured from `PART_AZ=4` orthographic azimuths **using that box**, so
every texel spends itself on a part instead of on the empty margin a whole-plant
card pays for. Two extra tile sets ride along: one straight-down tile per band (the
horizontal fill for steep views) and a whole-plant far set (8 side azimuths + 1
top) which is exactly a billboard pair.

**Runtime.** A plant inside `partRadius` (default 16 m) draws **15 small cards**:
12 upright part cards, each standing at its part's true centre inside the plant,
plus 3 horizontal band cards. Everything else in the region draws the 2-card
whole-plant billboard pair. Both are hard-alpha-tested with depth write, 2 texture
taps per fragment, no fragment loops, no `frag_depth`, no dither.

Why that produces real 3-D at card cost:

- **Parallax inside the silhouette is free and exact.** Part *positions* are real
  3-D and only their *images* are quantized. Each card is camera-facing, so it
  never degenerates edge-on, but its centre is fixed in the plant — when the camera
  moves, the front sector's card slides across the back sector's card and the upper
  band slides against the lower band by exactly the amount the real geometry would.
  The rasterizer and the depth buffer do the work a fragment loop would otherwise
  do.
- **The silhouette reshapes instead of rotating.** The outline is the union of 12
  cutouts whose relative screen offsets change with view direction, plus proper
  foreshortening: as the camera rises the sector offsets project shorter and the
  plant genuinely narrows.
- **Self-occlusion is real (depth test between parts) plus a modelling term.**
  Cards are emitted **nearest sector first**, so the hidden half of the clump is
  early-z rejected rather than shaded. On top of that each part gets a
  mean-preserving sun-side term (`selfShade`: sunward parts brighten by as much as
  shaded ones darken), which is what makes 12 cutouts read as one solid volume.
- **The azimuth pop is spread out.** Each sector's 4 baked azimuths are
  **staggered 22.5° apart** from its neighbours', so orbiting re-quantizes an
  eighth of the plant at a time. The baseline flips a whole plant at once every
  45°; this never flips a whole plant.
- **LOD is structural, not decorative.** 15 cards inside `0.5 * partRadius`, 12
  from there to `partRadius` (the back sector is dropped — it only shows through
  gaps), 2 cards beyond, and each band/top card is killed in the vertex shader
  when its viewing elevation makes it useless. Cost per plant collapses with
  distance, and the whole enumeration is region-bounded, so plant count is free.

## VRAM budget math

Per species, matching the HUD exactly (18.9 MiB on `default`, budget 25):

| item | size |
|---|---|
| near albedo array, 51 layers x 192² rgba8 + 6 mips | 9.55 MiB |
| near normal array, 51 layers x 192² rg8 (oct) + 6 mips | 4.78 MiB |
| far albedo, 9 x 128² rgba8 + 8 mips | 0.75 MiB |
| far normal, 9 x 128² rg8 + 8 mips | 0.38 MiB |
| near instances (16 B/plant, sized for partRadius max 32 m, density 3) | 0.18 MiB |
| far instances (16 B/plant, sized for regionRadius max 128 m, density 3) | 3.30 MiB |
| 2 x entry-info uniform (672 B each) | ~0 |
| **total** | **18.93 MiB** |

51 near layers = 12 parts x 4 azimuths + 3 band tops. On the worst defined stand
(`dense-mixed`, density 5/4/4) the far instance buffer grows and the HUD reads
21.2 / 20.1 / 20.1 MiB — still inside budget. The bake peaks at roughly
300–400 MB of transient GPU memory (raw vertex+index buffers, a 3072² x 2 atlas
target, readbacks), all allocated through `ctx.res` tagged `bake-scratch` and
destroyed when the bake ends.

**Where the resolution went.** Part boxes are tight and nearly square, so 192² per
part buys ~2.0 x 2.3 mm/texel on calamagrostis — the same texel density as the
baseline's 512² whole-plant card (1.6 x 2.3 mm), at 2/3 of its texture bytes,
while carrying 12 view-dependent pieces instead of 1. The trade is angular: 4
azimuths per part (staggered) instead of 8 per plant. Two earlier configurations
are recorded because both were wrong in instructive ways: 2 bands + 6 azimuths at
192² (too few bands → part boxes 0.4 x 0.98 m squeezed into square tiles →
3.8 mm/texel vertically, visibly soft), and triangle-count *quantile* bands
(calamagrostis carries half its triangles in the top 0.2 m of panicles, so one
band spanned 83% of the height — the same failure). Equal-height bands keep the
boxes square, which is the whole reason the tiles stay sharp.

## Bake

`bake.ts` → `mesh/baked/028-exact-silhouette/parts-v3-<species>.bin`, 11.6 MiB
each (512 B header + 51 x 192² rgba8/rg8 near tiles + 9 x 128² far tiles).

1. CPU: per-triangle centroid → band (equal height) and sector
   (`round(atan2(x,z)/90°)`, the same formula the shader uses), then a counting
   sort of the index buffer so every part is one contiguous `drawIndexed` range;
   per-part vertex AABBs, per-band xz AABB + mean height for the horizontal card.
2. GPU: one render pass per tile set, one viewport per tile in a 3072² atlas at 2x
   supersample, two targets (albedo+coverage, mesh-frame normal flipped toward the
   bake camera so both faces of a blade light alike). Every tile is described by an
   orthonormal basis + box centre + half extents, so side, part and straight-down
   views all go through one shader path.
3. CPU: coverage-weighted 2x downsample per tile, 4 dilation passes into empty
   texels (alpha stays 0, so the silhouette is untouched and filtering never pulls
   in background black), oct-encode normals, pack.
4. Upload as texture **arrays**, not atlases: tiles cannot bleed into each other
   under minification, so no gutters and no atlas-edge rules. Mips are generated at
   load time on the GPU, coverage-weighted for albedo and **decode → average →
   re-encode** for normals (box-filtering oct pairs averages a fold of the
   octahedron, not a direction).

Artifacts are magic+config-validated on load: the dev server answers missing
`/mesh/baked` files with `index.html` at status 200, which would otherwise poison
both the OPFS cache and the committed-file path; a poisoned entry is rebaked and
the cache entry repaired. All three species bake in a couple of minutes on first
run (poa's 6.5 M-triangle partition on the CPU dominates); afterwards the
committed `.bin`s load directly. Stale `parts-v1`/`v2` variants were deleted.

## Status

**working** — verified by headless screenshots at all four standard cams, all five
debug views, three extra portrait/orbit cameras, the `scaling-100m` and
`dense-mixed` stands, and the A/B page against the champion. Zero console errors
or warnings in every run.

## Findings

### Speed — 1.02–1.41x the billboard baseline, same frame, CONTENDED GPU

15 sibling agents shared this GPU, so **no absolute millisecond here is a claim.**
The numbers below are the two techniques' passes rendered in the *same* frame
(`#/ab/001-billboard-smoke/028-exact-silhouette?stand=default&seed=42&cam=…`),
the only ratio that survives contention:

| camera | B/A (parts / cards) |
|---|---|
| `topdown` | **1.02x** |
| `far-horizon` | **1.17x** |
| `grazing` | **1.19–1.20x** |
| `inside-plant` | **1.27x** |
| portrait, 2.2 m up / 24° down / fov 25 | **1.37x** |
| eye-level inside the canopy, fov 22 | **1.41x** |

`topdown` is 1.02x because at 42 m altitude every plant is beyond `partRadius` and
the method *is* the baseline there. The worst cases are the ones where the whole
screen is near-ring plants, and they stay under 1.5x.

Bench, also contended, 1600x900, `spline=orbit-low`, seed 42, run back-to-back:

- `results/028-exact-silhouette__default__p-a1a50d19__apple-metal-3__2026-07-25T05-40-05-788Z.json`
- `results/001-billboard-smoke__default__p-86e4907a__apple-metal-3__2026-07-25T05-40-13-680Z.json`
- `results/028-exact-silhouette__scaling-100m__p-a1a50d19__apple-metal-3__2026-07-25T05-40-22-149Z.json`

Pairing the two `default` runs gives **1.22x on the technique pass and 1.15x on
the total Σp50**. Solo run-page Σp50 at 1280x800 stayed between 3.6 and 5.5 ms at
every standard camera and on both stress stands — inside the 9 ms ceiling even
with the neighbours running.

**Plant count is free.** `scaling-100m` (134.2 M plants, ±2048 m) benches
*slightly faster* than `default` (557 k) at identical VRAM, and `dense-mixed`
(7.7 M plants, density 5/4/4) costs 5.5 ms Σp50 at grazing. Nothing is ever
materialized per plant: the cull evaluates the shared scatter over a bounded
camera-centred cell region with workgroup-uniform cell rejects, and the LOD split
happens inside that same pass.

### Looks — better where the mechanism matters, parity elsewhere, honestly softer in one place

**Clear win: any view from above the canopy (roughly 20–45° down).** This is where
the baseline's weakness is structural — its horizontal top card fades in and reads
as flat radiating star-fans of blades pasted on the ground. Matched crops at 2.2 m
up / 24° down and 3.4 m up / 34° down show those fans plainly; the same crop here
shows panicle clusters at three real heights over a darker lower canopy, with no
fans anywhere. Reproduce with
`#/ab/001-billboard-smoke/028-exact-silhouette?seed=42&det=1&t=3&cam=3.50,-4.20,3.50,-0.7854,-0.6000,30.0`
(wipe, then flicker).

**Clear win: `far-horizon`.** The canopy edge against the sky carries individual
panicles at different heights and the mass behind it has dark pockets; the
baseline's edge is a more even wall of vertical strokes.

**Clear win: `debug=depth`.** The baseline's plants are single-depth vertical bands
(one plane per card, every texel of a plant at one depth). Here each plant shows 12
depths and neighbouring plants interleave per pixel. That is the objective evidence
for the layering claim.

**Parallax and silhouette change verified, not assumed.** Orbiting a fixed point at
3–4 m in ~7° steps (`cam=3.00,-5.40,3.00,-0.7854,-0.4200,25.0` →
`3.35,-5.40,2.61,-0.9100,…` → `3.67,-5.40,2.12,-1.0472,…`): interior features move
by *different* amounts — front sectors sweep across back sectors, the outline
changes shape rather than sliding, and no plant ever flips as a whole (the
staggered azimuths cost an eighth of a plant per switch). Run the same orbit with
`p.partRadius=0`, which is the flat-card behaviour, to see the difference isolated.

**Parity: `grazing`, `topdown`, `inside-plant`.** At grazing the two are hard to
separate in a wipe — mine has a bit more local contrast and depth in the mid field
and no pale flecks (see below), but the baseline is equally convincing. `topdown`
is identical by construction. `inside-plant` is a genuine tie: mine layers better,
the baseline's blades are a touch crisper.

**Honest loss: extreme close range (<1.5 m) on the widest community tile.**
`elymus-repens` spreads its foliage over a 0.96 m span, so its sector boxes are
0.66 m wide and a 192² tile gives 3.4 mm/texel horizontally against the baseline's
2.2 mm. Magnified past that, my parts are softer than the champion's card.
Calamagrostis and poa are at parity. The fix is per-species tile sizing (or 6
sectors for the wide tiles), which did not fit in 25 MB alongside 51 layers — the
first thing to try if the budget ever moves.

**Artifacts found and fixed during the build**, worth knowing for anyone reusing
this geometry:

- *Pale horizontal slivers among near plants at eye level.* The horizontal band
  cards were gated on viewing elevation the way the baseline gates its top card
  (35–37°), but elevation is per plant, and a plant 1 m from a 1.35 m camera is
  already at 35°. The result was exactly the baseline's "floating pancake"
  artifact. Gating at 33–51° removed it while keeping steep views dense.
- *Over-darkening.* Stacking a canopy-depth gradient and a subtractive sun-side
  term made the meadow muddy next to the baseline. The sun-side term is now
  mean-preserving (`1 + selfShade * dot(partDir, sunDir)`), which adds modelling
  without losing brightness. Both terms live in the light term, never in albedo,
  so `debug=albedo` shows the baked tiles exactly as captured and `debug=lighting`
  shows sun+ambient x (canopy depth x sun side).

### Debug views

All five are wired through the shared `debug_shade()`. `normals` shows real
per-fragment normals (oct atlas decoded, rotated into world by the plant yaw);
they are noisier than the baseline's on purpose — each part's normals were flipped
toward *its own* bake azimuth, so a plant carries several facing families instead
of one. `lighting` goes through `light_surface()` exactly once (it looks blown out
only because linear values are shown in sRGB). `coverage` is the baked alpha the
fragment resolved to, which with a hard alpha test is also the alpha-test margin.
`depth` is the strongest single argument for the method.

### Wishlist / notes for the harness

- Nothing blocking. `bakedArtifact` was sufficient once magic-validated; a real 404
  for missing `/mesh/baked` files (instead of the SPA fallback) would let
  experiments drop that shim.
- Load-time mip generation for texture arrays costs 636 tiny render passes per
  species (51 layers x 5 levels x 2 textures + the far set). It is one-time and
  invisible in practice, but a shared array-mipgen helper in the harness would save
  every array-based method from writing it.
