# ND moss — Uncharted 4's fuzzy-moss material on a Sphagnum carpet

## What the paper actually says

Source: Andrew Maximov & co., *"The Technical Art of Uncharted 4"* (SIGGRAPH 2016
Advances in Real-Time Rendering), slides 28–58 and 63–65.

The important thing, and the thing that shaped this experiment: **Uncharted 4's
moss is not a geometry technique at all — it is a SURFACE MATERIAL.** The talk's
moss sits on ordinary game geometry (rocks, walls, roots) which owns the
silhouette, and everything moss-like about it happens in the shader. Its recipe,
built up one slide at a time:

- **Inputs**: Colour, Normal, **Heightmap**, Ambient Occlusion — and the AO map
  is unusually dark ("for moss, the AO goes very dark").
- **The observation the whole thing is built on** (slide 44): moss is
  *micro-fibre, not micro-facet* — porous, hugely self-shadowing, transmitting
  light, lighter at the tips than at the base, and strongly view-dependent:
  *"when viewed straight on, you can see down into the shadows between the
  fibers, and viewed from an angle, only the tips are visible."*
- **Light wrap** (slide 51, code on 57): broadens `NdotL` past the terminator to
  fake SSS. Deliberately **ignores the normal map** so it fills the cracks, and
  the artist supplies a **tint** (green, in their example). Cites Penner's
  pre-integrated SSS and McAuley's energy-conserving wrapped diffuse.
- **Fuzz BRDF** (52): a micro-fibre layer over the micro-facet layer with its own
  colour — explicitly *the subpixel version* of the whole phenomenon, because
  "moss details are large enough to be seen in the texture when the camera is
  close, but become subpixel when more than a few meters away. We need to solve
  both of these cases."
- **Parallax occlusion + parallax shadows** (30, 53) from the heightmap.
- **Sun Micro Shadowing** (36–37), verbatim:
  `aperture = 2*ao*ao; microShadow = saturate(abs(dot(L,N)) + aperture - 1); shadow *= microShadow`
- **AO Fresnel / "AO occlusion"** (38–40), verbatim:
  `aoFadeTerm = saturate(dot(vertexNormalWS, viewWS)); ao = lerp(1, ao, aoFadeTerm)`
  — the cracks baked into the normal map *should* be occluded by geometry at a
  glancing angle, and since that geometry does not exist, occlude them in the
  shader. Uses the **geometry** normal, not the normal-map normal.
- **AO saturation** (55): "add back a bit of colour where the ambient occlusion
  gets dark" — bounce light and SSS, "our code was a simple hack".
- **Unified wetness** (63–65): artist-painted wetness + porosity; lerp the base
  colour toward its **square** (porous surfaces darken more), flatten normals as
  the surface saturates, four puddle regimes.

The talk's *other* half (slides 84–119) is the pivot-based vertex-animation /
foliage-wind system — vertex-colour pivots, hierarchical rotation without
hierarchy, a top-down projected 2D wind-gust texture, pivot billboarding. None of
that is reusable here: this harness owns wind, and the bog's Sphagnum has
`sway = 0` (moss does not sway), so the moss path deliberately has no wind at
all and the grasses use the shared wind include.

## Idea

Take the paper's material, but **measure its four maps off the real 19.8M-triangle
cushion instead of authoring them**, and then supply the one thing the paper does
not have to solve — thickness — with geometry, because a Sphagnum cushion *is*
the geometry rather than a coating on someone else's.

Baked per species, for exactly ONE periodic 0.18 m tile:

| runtime texture | rgb | a |
| --- | --- | --- |
| A | albedo | **height** of the topmost surface |
| B | mesh normal (upper hemisphere) | **cavity AO** |

Height comes free: an orthographic top-down render with a depth test keeps the
topmost surface per texel, so its `y` *is* the heightfield. AO is then a wrapping
horizon sweep over that heightfield.

Runtime, three distance bands per carpet entry, so cost falls with distance:

| band | geometry | fragment |
| --- | --- | --- |
| `< patchDist` (7 m) | tile is a **displaced 8×8 patch** (real relief, real silhouette, real depth) | 2 height taps + normal + 1 shadow tap |
| 7 – 26 m | one flat tile-sized quad | same |
| `> mergeDist` (26 m) | one quad per **2×2 phase-locked block** | 2 taps, no parallax |

All three are ground-parallel and terrain-conformed per vertex (ladder rung 3).
Upright plants (bog Calamagrostis/Poa, all of `default`) use the
001-billboard-smoke card path copied **verbatim**, so an A/B against 001 isolates
the moss and `default` measures like the baseline.

### What I took, adapted, or went beyond

**Taken verbatim**: sun micro shadowing (`2*ao*ao` aperture); AO Fresnel's form;
light wrap's design (geometry normal only, tinted, converging to plain `NdotL` as
the tint goes to 0); parallax + parallax shadows; the fuzz layer as the
*subpixel* answer; wetness as "lerp toward the squared base colour, flatten the
normal as it saturates".

**Adapted, and why:**

1. **The tints are measured, not authored.** The bake reports the mean albedo of
   the top height decile (the capitulum tips) and of the bottom 45% (deep
   cushion). Tip colour drives the fuzz sheen, base colour drives the light wrap
   and the AO-saturation bounce. The paper's artist picks these by hand; here
   they come from the mesh, so each of the three micro-habitat states gets its
   own — and the paper's claim that moss is "lighter at the tips" is *confirmed*
   by the bake rather than assumed (tip 0.316,0.596,0.153 vs base
   0.294,0.573,0.142 for wet-vigorous).
2. **`aoFresnel` is a knob, default 0.25, not the paper's implicit 1.0.** Fading
   AO all the way to white at glancing angles is right for moss on a rock, but a
   *carpet* is seen at a glancing angle almost everywhere. Moss's albedo is
   nearly uniform (tip vs base differ by ~7%), so **all** of its structure is
   shading — and at `aoFresnel = 1` the mat measured as flat olive paint at
   `cam=grazing`. This is the one place I knowingly diverge from the slide.
3. **AO applies to indirect light, micro shadow to the sun.** The shared
   `light_surface()` is `albedo * (sun + ambient*hemi)`; I call it and then split
   the two halves apart by subtracting the recomputed sun term, so AO occludes
   ambient while `microShadow * parallaxShadow` occludes the sun. With
   `ao = micro = shadow = 1` this is *exactly* `light_surface()`, and the ambient
   formula is never duplicated.
4. **AO saturation is additive**, as the slide describes ("add back colour"),
   tinted by the measured base colour normalised to luma 1 and scaled by
   `frame.ambient`. My first attempt made it a hue-only tint, which did nothing
   for the real problem: a *physically measured* cavity AO is much darker than an
   authored one (mean 0.60 across the tile), and without a bounce term the mat
   sits at 53% of an AO-free renderer's light and reads as a hole.
5. **Wetness is driven by the stand, not by a painted mask.** The bog zones the
   three states across the shared wetness field, so each entry's own band centre
   *is* its wetness input (`wet_center * param`), deepened per fragment in the
   crevices where water actually pools (`1 - height`). No specular sharpening —
   the shared lighting model has no specular lobe.
6. **Parallax is one offset-limited step + one optional secant refinement**, not
   a march: the project rules forbid marching, so the paper's POM becomes
   2 dependent height taps with the offset bounded by half the relief.

**Beyond the paper:**

- **Thickness by geometry (bands 0/1/2 above).** The paper never needs this. Near
  tiles are displaced patches; the tall capitula (up to 9.1 cm) exist as real
  vertices, write real depth, and cast a real silhouette.
- **The rotation-symmetrised tile boundary** (see below) — the trick I am
  happiest with, and it came out of the Crimson Desert reading rather than the ND
  talk.
- **The 3×3 periodic bake wrap**, which makes the tile *exactly* periodic and
  lets the runtime sample it with `repeat`. 001 reports "faint 1-texel dark lines
  along a tile boundary" from cropping the capture; this removes the cause.
- **2×2 phase-locked merging** with the four per-sub-tile quarter turns packed
  into the instance bits, so a merged quad is pixel-equivalent to four separate
  tiles rather than an LOD that loses the lattice.

### The lattice problem, and the Crimson Desert connection

Per-vertex displacement of a tiled carpet looks impossible at first: the
heightfield is periodic under *translation*, but the stand rotates each tile by a
random quarter turn, and `h(u,0) != h(0,u)`, so two neighbours disagree along
their shared edge and the surface cracks open. Pinning the boundary ring to the
mean plane fixes the crack but leaves a flat crease along every grid line — a
0.18 m quilt, which is exactly the "square block with a flat cutoff" that
Crimson Desert's POM-silhouette work is about.

Pearl Abyss's answer (Black Space Engine "screen-space displacement mapping";
publicly described as silhouette-clipping POM) is to inverse-rotate and
inverse-scale the displacement offset, test it against the mesh's `[0,1]` UV
bounds, and mask out the pixels where the displacement has **overshot the
boundary** — so tall areas overshoot less, low areas more, and the block's edge
follows the height field instead of being a straight cut. Sources:
[80.lv writeup](https://80.lv/articles/how-to-create-rich-surface-details-like-crimson-desert-using-pom-silhouettes),
[SPOM + horizon-trimming shader](https://godotshaders.com/shader/spom-with-horizon-detection-self-shading-silhouette-clipping-parallax-occlusion-mapping-self-shading-horizon-trimming-erosion/).

That exact trick does **not** transfer to a periodic carpet, for a reason worth
recording: overshooting a *tile* boundary is not an error here. The tile tiles, so
the runtime samples with `repeat` and an overshooting lookup lands on
geometrically correct neighbouring moss — masking it would punch holes in a
closed mat rather than reveal a silhouette. (This is also why this renderer needs
no clamping fixups for POM at tile edges at all.)

What does transfer is the *goal* — make the edge follow the material — and it has
a clean solution in the geometry domain: displace the tile boundary by the
**rotation-symmetrised heightfield**, the mean of the field over all four quarter
turns. That field is invariant under the tile's own rotation, and because the
fetch wraps, two neighbours compute bit-identical values at a shared boundary
point. So the patch edge follows real relief, all four rotations agree, and the
mat is crack-free by construction instead of by pinning. The interior blends to
the tile's true field over ~1.6 patch cells.

The one thing I did **not** take from SPOM is pixel depth offset. WGSL has no
conservative-depth qualifier, so `@builtin(frag_depth)` would disable early-z
across the whole mat — and the near band already gets correct depth and correct
intersections for free, because there it is real geometry.

## VRAM budget math

Per moss species, HUD-measured: **48.9 / 25 MB** — about 2× the soft budget,
deliberately, and **this experiment's own choice, not something the owner asked
for** (an earlier version of this section said otherwise; the only precedent is
that the owner raised 001's atlas tile 128 → 256 because 128 "was too blurry",
which is a different renderer's texture). The override is therefore pending the
owner's verdict — see the 1024² fallback at the end of this section. Breakdown:

- moss texture A (albedo+height) 2048² rgba8 + full mip chain: 22.4 MB
- moss texture B (normal+AO) 2048² rgba8 + mips: 22.4 MB
- instance lists, 16 B each, sized from the carpet's own node density
  (`(22/4)² = 30.25` tiles/m²) and each band's disc area, never from
  `SCATTER_MAX_PER_CELL`:
  - LOD 0, `π·12²·30.25·1.1 + 512` = 15.6k → 0.25 MB
  - LOD 1, `π·32²·30.25·1.1` plus 5% of the whole region for zone-boundary
    blocks that fail to merge = 151k → 2.42 MB
  - LOD 2, `(π·96² − π·8²)·30.25/4·1.1` = 240k → 3.84 MB
- entry info 304 B + indirect args 48 B: noise

Why 2048² is worth 4× the memory: the tile is 0.18 m across, so 2048 texels is
**0.088 mm/texel** — a Sphagnum capitulum is 4–5 mm, i.e. ~50 texels wide, which
is what makes individual rosettes resolve at 1 m instead of averaging into fuzz.
For scale, 001's carpet gets ~188 texels across the same tile (0.96 mm/texel);
this is 11× the linear density. At 1024² (also verified working — the code is
resolution-independent, it is one constant) the whole thing fits in 16.9 MB per
species, i.e. inside budget, and looks clearly worse at `carpet-close` but
identical beyond ~3 m.

Capacity is sized for the param *maxima* (`patchDist ≤ 12`, `mergeDist ≤ 32`),
which is why those two params have bounded ranges: a 128 m `mergeDist` would want
25 MB of instances by itself.

Grass species are unchanged from the 001 baseline: 21.3 / 20.7 / 21.3 MB on
`default`, 18.9 / 18.3 MB on `bog`.

Bake scratch: ~500 MB transient (the 184 MB vertex buffer + 238 MB index buffer
of the raw mesh, 4096² targets, readbacks), all tagged `bake-scratch` and
destroyed at the end of the bake.

## Bake

`bake.ts` → `mesh/baked/039-nd-moss/moss-v4-<species>.bin`, **33.55 MB** each
(128 B header + 2048²·4 albedo+height + 2048²·4 normal+AO). Plus
`cards-v1-<species>.bin` (14.2 MB × 3) for the upright path, which is
001-billboard-smoke's bake copied verbatim.

One orthographic top-down pass over exactly the tile square `[0, 0.18]²` of the
mesh frame at 4096² (2× supersampled), then:

1. **Nine instances**, offset by −tile/0/+tile in x and z. The source geometry
   overflows its own period (bounds run −0.019 → 0.215 inside a 0.18 m tile), so
   without the eight wrapped copies the crop misses every neighbour's overhang
   and the tile is not periodic. This is the direct cause of the seam lines 001
   reports, and with the wrap the measured coverage is 93–98% instead of the
   27–37% 001 sees over its whole capture square.
2. **Coverage-weighted 2× downsample** for albedo and for normals-as-plain-xyz
   (never octahedral: this atlas gets a mip chain and octahedral pairs are not
   mip-averageable — they filter toward straight up). Height is a **plain box
   filter over "premultiplied" height** (an empty subsample counts as 0), so one
   channel is both the heightfield and the coverage signal, and mipping pulls a
   gap toward the surrounding surface height instead of dissolving the mat.
   Because the mip filter normalises colour by that coverage, the stored rgb is
   already coverage-weighted and the shader must NOT divide by it again.
3. **Wrapping dilation** (5 passes) of colour and normal into the gaps.
4. **Cavity AO**: 8 azimuths × 5 radii specified in *millimetres* (1.05 → 16.9 mm,
   i.e. sub-capitulum to a few capitula), occlusion per direction =
   `sin²(max horizon angle)`, the cosine-weighted visible fraction.
5. Scalars into the header: mean plane, top, apex, coverage, tip/base/mean colour.

Two AO iterations were thrown away before this one, and both are instructive:
starting the sweep at 1 texel (0.18 mm) gave mean AO **0.26** — at that scale the
cushion's own sub-texel roughness makes the horizon tangent enormous everywhere,
so the "AO map" was a noise field and even capitulum tops read as holes; and a
clamped *tangent* instead of `sin²` saturates the moment any neighbour is higher,
which on a cushion is nearly everywhere (mean 0.48). The shipped version means
0.55–0.60 with bright tops and black crevices.

Bake time: ~2 min for all three moss species (the AO sweep and the dilation are
the expensive parts, ~170 M taps each per species at 2048²) — well under the
"~2 min per species" the brief warns about, because the meshes come from
localhost. `moss-v1/v2/v3` were deleted.

## Status

**working** — verified by headless screenshots I read individually, on
`?stand=bog` at `cam=grazing`, `carpet-close`, `topdown`, `inside-plant`,
`far-horizon`, and two eye-level sloped views across the ridges; with
`debug=normals`, `lighting`, `albedo`, `coverage`, `depth`; and on
`?stand=default` at all four standard cams. Zero console errors, no toasts.

What the debug views showed:

- **normals**: real per-fragment variation at capitulum scale around +Y, not a
  flat up-normal (the octahedral-mip trap is avoided by storing plain xyz).
- **coverage**: the mat is closed — fine height detail everywhere from the camera
  to the horizon, no tile-shaped holes, no dissolve with distance. This view is
  what proved the mat was *drawn* everywhere while it still *looked* flat, which
  is how I found the AO-Fresnel problem instead of hunting a non-existent culling
  bug.
- **lighting**: goes through `light_surface()` once; mean band values 0.73–0.77
  against 001's 0.84–0.99 at `carpet-close`, i.e. the moss is deliberately ~80%
  as lit as an AO-free renderer, the missing fifth being genuinely dark crevices.
- **albedo**: 0.186–0.204 mean vs 001's 0.184–0.206 — the two bakes are
  calibrated identically, so the visible difference is all shading and detail.

## Findings

### Performance

Same-frame A/B, `#/ab/001-billboard-smoke/039-nd-moss?stand=bog`, summing each
side's own passes (`cull + cards` vs `cull + moss`), 3 reps per camera. Raw
**B/A**, before correcting for the B slot's known 1.26–1.35× handicap:

| camera | B/A (raw) | bias-corrected |
| --- | --- | --- |
| grazing | 0.76 / 0.80 / 0.77 | **≈0.58–0.62×** |
| carpet-close | 1.12 / 1.30 / 1.19 | ≈0.86–1.00× |
| topdown | 1.11 / 0.86 / 1.28 | ≈0.66–0.98× |

So: **at or better than the champion everywhere, and roughly 40% cheaper at
grazing** — the angle that matters most, because that is where a life-size carpet
puts a million tiles on screen. The 2×2 merge is why: beyond 26 m it draws one
quad per four tiles and takes the 2-tap fragment path. `default` Σp50 measured
3.6–3.7 ms across the four standard cams, well inside the 9 ms ceiling (it is the
001 card path unchanged). No `results/` bench JSON is claimed: ~33 sibling agents
were sharing this GPU, so per CLAUDE.md any absolute number would be
contaminated.

### Did it beat the billboard reference on moss?

**Yes at close and mid range, clearly; marginally at grazing.**

- `cam=carpet-close` (1 m straight down) is the decisive one. 001 renders a
  fine-grained red-green *fuzz* with the 0.18 m tile grid faintly visible as
  straight seams, and no structure. Mine resolves **individual capitula** — star
  shaped rosettes, clumping, dark crevices between them — and shows no tile seams
  at all. That is the 11× texel density plus the AO/micro-shadow doing exactly
  what the paper promises.
- **Thickness and relief, specifically**: a real but partial win.
  - Real, in the near band: the cushion is *geometry* there, up to 3.3 cm of
    displaced relief per tile, writing real depth, with hard alpha-tested gaps
    opening onto terrain 7–9 cm below. Silhouette and depth are physically
    correct and the surface visibly undulates instead of being a plane.
  - Partial beyond ~7 m: parallax gives motion parallax and self-shadowing, but
    the surface is a plane again, so the mat's *horizon* silhouette is smooth.
    001 has no thickness at any distance, so this is strictly better — but the
    honest claim is "cushion relief out to 7 m, cushion shading everywhere", not
    "thickness everywhere".
- `cam=grazing` is where I expected the biggest win and got the smallest. Mine
  has visibly finer texture and softer zone boundaries (001's tile-quantised zone
  patches read as blocky steps), but at 20–100 m a 0.18 m tile is 2–8 px and both
  renderers converge on mean colour. The paper predicts exactly this ("details
  become subpixel more than a few meters away") and its answer is the Fuzz BRDF,
  which the far path keeps.
- Bonus, not mine to claim as skill: the three micro-habitat states read as much
  more coherent ecology than 001's, because the wetness function darkens the wet
  zone and leaves the sun-exposed crest pale — 001's NOTES call its wet-vigorous
  zone "poster paint", and that is fixed here.

### Still bad / honest limits

1. **Grazing is only marginally better than a flat card.** Beating a card there
   needs either real geometry much further out (vertex cost) or a
   silhouette-clipping march (banned by the project rules).
2. **VRAM is 2× the soft budget** at the requested 2048². One constant reverts it.
3. **The near→mid LOD seam is a ≤4 mm height step** where a displaced patch meets
   a flat quad (the patch boundary sits at the rotation-symmetrised height, the
   quad at the tile mean). Sub-pixel at 7 m; not visible in any screenshot.
4. **Merged far quads leave T-junctions** at their edge midpoints against
   per-tile neighbours (a merged block has no vertex there). Millimetres at 26 m+.
5. **Zone boundaries are quantised to the tile**, and at grazing a boundary
   foreshortens into a visibly stepped edge. That is the stand's lattice, not a
   bug, but a noise-blended boundary would look better.
6. **No specular/sheen lobe** in the shared lighting model, so the paper's
   wetness can only darken albedo and flatten normals, never sharpen a highlight.
7. The fuzz sheen is the term most likely to be over-tuned; it washes contrast at
   grazing, which is why its default is 0.25 rather than 0.5.

### Harness feedback

- **`carpetScale()` is not exported from `@harness`.** A carpet entry's
  `scaleMin`/`scaleMax` on the CPU-side `StandSpecies` object are *placeholders*
  (1.7 for the bog moss) while the real constant scale (1.0101) is computed inside
  the harness when it builds the stand table. Any renderer that needs the scale on
  the CPU — to size geometry, or to convert baked mesh-frame metres into world
  metres, as this one does — has to re-derive
  `SCATTER_CELL_SIZE / carpetDiv / speciesById(id).tileM`. Exporting `carpetScale`
  (or patching the entries in place) would remove a silent-wrong-answer trap.
- **`ctx.meshes` does not expose the source `manifest.json` metadata.** Those
  manifests carry exactly the numbers a moss renderer wants —
  `canopy.capitulumApexMeanH`, `canopy.reliefH`, `geometry.tile.topH/sizeX` — and
  I measured all of them again in my own bake. Arguably better (measured beats
  declared), but a `MeshInfo.manifest` passthrough would save every carpet
  experiment a bake iteration.
- **The dev server answers a missing `/mesh/baked/...` with `index.html` at 200**,
  so every loader needs the magic-validation shim 001 wrote. A 404 would let it go.
- **A `textureSampleGrad` warning belongs in the silent-failure list**: any
  parallax/POM renderer must pass gradients derived from the *unperturbed* UV, or
  the per-fragment offset makes the UV derivative the heightmap's own gradient and
  the sampler drops to near-top mips at grazing angles. (It turned out not to be
  the cause of my flat-at-grazing symptom, but it is a real bug I fixed on the
  way, and it is invisible except as "the far field looks like paint".)
- Not a complaint, but worth recording for other carpet renderers: the
  `carpet-close` bookmark is the most useful camera in the repo for this species,
  and `debug=coverage` is what distinguishes "not drawn" from "drawn but shaded
  flat". I would have chased a phantom culling bug without it.
