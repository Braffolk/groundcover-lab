# 036-deferred-grass-field — the canopy as a per-pixel ray answer

Seed: *"grow the grass in screen space: one fullscreen pass unprojects the scene,
and each pixel asks a baked table what grass the eye ray meets on its way in."*

## Idea

**Nothing plant-shaped is ever rasterized.** The frame contains exactly ONE
extra draw call: a fullscreen triangle (3 vertices, 1 primitive, no depth
attachment, alpha-blended over the terrain the harness already drew). Every
blade in the image is the answer to a per-pixel query resolved from a baked
table, put back onto the eye ray in closed form.

Per pixel:

1. **Unproject.** `textureLoad` the scene depth and inverse-project → `G`, the
   exact point where THIS eye ray meets the ground (the terrain of the base pass
   is the reference for where the canopy stands, so ground contact is exact by
   construction). Sky pixels fall back to the ground plane under the camera.
2. **Inverse-shear for wind.** Sway is a linear-in-height shear of the canopy;
   a shear maps lines to lines, so instead of bending half a million plants the
   *ray* is inverse-sheared into the baked rest frame. 4 ALU, exact for the
   model (see *Wind* below).
3. **Slide up the ray, in closed form.** The table is indexed by where the ray
   crosses a horizontal plane near the canopy top: `A = G - phi * s * (H-align)`
   with `s = |d.xz|/|d.y| = cot(elevation)`. No iteration, one multiply-add.
4. **Ask the table.** `(u, v)` = `A.xz mod tile`, azimuth and `s` pick the
   direction. It answers: albedo, coverage, surface normal, and the **height h**
   of the first thing the ray meets.
5. **Put the hit back on the ray.** `t = (g + h - eye.y) / d.y` — the article's
   `|OB| = |OA| / cos(alpha)`, one divide. That gives a real per-pixel world
   position, so `debug=depth` shows canopy depth, fog and lighting are correct,
   and the canopy meets the terrain exactly where it should.
6. Burial AO, blade-scale relief, shade, blend.

No march, no loop, no per-plant work, and no dithering anywhere: coverage IS
alpha, so a partially covered pixel composites honestly over soil or sky.

### What is rasterized (the no-plant-geometry rule)

| per frame | count |
|---|---|
| fullscreen triangle | **1 primitive, 1 draw, 3 vertices** |
| plant primitives (cards/quads/ribbons/prisms/shells) | **0** |
| vertex or compute work proportional to plant count | **none** |

`default` (557k plants) and `scaling-100m` (**134.2M plants**) issue identical
GPU work: 1.24 ms vs 1.40 ms for the pass, same VRAM, same table. The stand is
seen only by the bake. Verified by screenshot on both stands.

### Texture fetches to resolve one pixel (constant, no loops)

| taps | what |
|---|---|
| 1 | scene depth (`textureLoad`) |
| 2 | `surf`/`geom` of elevation cell e0 — trilinear over (u, v, **azimuth**) + mips |
| 2 | same for cell e1 (the elevation lerp; `elevLerp=false` drops these) |
| 1 | `geom` of the steepest cell = the canopy-top height map (burial AO + the eye-inside-canopy ceiling) |
| 0–1 | terrain height, sky pixels only (the fallback ground plane) |

**6 fetches typical, 7 worst case, 4 with `elevLerp=false`.** The azimuth axis is
the third axis of a 3D texture, so azimuth interpolation — and, in the mip
chain, direction averaging with distance — is free hardware trilinear filtering.

## Table parameterisation (the actual intellectual work)

A **periodic 6 m canopy patch**, baked flat, populated by the stand's own
scatter. Per elevation cell, a pair of 3D textures:

```
surf[e] : 256 x 256 x 16  rgba8unorm  = (albedo * cov, cov)
geom[e] : 256 x 256 x 16  rgba8snorm  = (n.x * c, n.z * c, h01 * c, c)
          ^u    ^v    ^azimuth (22.5 deg cells, wrap-addressed)
```

* 6 m / 256 = **2.34 cm texels**; 16 azimuths; 6 elevation cells at shear
  `s = 0.12, 0.37, 0.75, 1.43, 2.98, 9.99` (elevation 83°…5.7°), spaced
  uniformly in `q = s/(1+s)` so the cells crowd where the answer changes fastest.
  `q` inverts in closed form, so choosing a cell is arithmetic, not a search.
* Everything is **premultiplied by coverage**, which makes a plain box filter the
  correct coverage-weighted prefilter: mips are an honestly averaged canopy, and
  that is the whole anti-aliasing story (no dither, no stochastic coverage).
* 3D mips halve the azimuth axis too, so distant grass is averaged over
  direction as well as area — deliberate: at those distances one pixel spans far
  more than one azimuth cell's worth of parallax.
* **Height, not distance.** Storing the hit HEIGHT makes the table independent of
  the elevation quantisation: the runtime converts it to a distance along the
  *true* ray, so depth stays right where the direction is quantised, and 8 bits
  buy 1.1 cm of height precision instead of 8 cm of range.
* The normal is **point-sampled from the subsample that won the ray**, not
  averaged: averaging a texel full of blades pointing every which way collapses
  to straight up and the canopy lights up like linoleum (that was v1 — visible
  instantly in `debug=normals`).

### Two indexing decisions that made or broke it

1. **Index by the ray's ENTRY into the canopy, not by its ground crossing.** Same
   content, but indexed by the ground crossing the answer for a 2° ray is nothing
   like the answer for a 9° ray through the same ground point (one grazes the
   tips, the other dives to the soil), so clamping the elevation axis painted a
   dark plateau over the whole mid-field. Indexed by the entry, every answer is
   "what is just under where the ray came in", which *converges* as the ray
   flattens: clamping at `s_max` becomes a sub-metre error along the ray.
2. **Put the lookup point on the expected first-hit plane** (`H - alignDepth`)
   and offset each cell by *its own* shear. Two bracketing cells are otherwise
   laterally misregistered by `(s1-s0) * penetration` — metres at the grazing
   end — and their lerp is mush with concentric rings around the camera. This one
   change is the difference between wet cardboard and a meadow.

## VRAM budget math

| item | bytes |
|---|---|
| `surf` mip 0, per elevation (256²×16×4) | 4.19 MB |
| `geom` mip 0, per elevation | 4.19 MB |
| 3D mip chain (halves all three axes → ×8/7) | ×1.143 |
| **6 elevations × both slabs** | **57.5 MB** |
| params UBO | 96 B |

The table is ONE composite of all three species — a ray does not care what it
hits — so its bytes divide across the stand's species: **19.2 MB per species**,
under the 25 MB budget. The HUD says 21.3 MB/species because `textureBytes()`
halves only width/height per mip level, which over-counts a 3D chain; the real
number is 19.2. Nothing is allocated per frame and **nothing scales with plant
count** — the same 57.5 MB serves 557k plants and 134.2M.

Bake-time transients (source meshes up to ~180 MB for poa, stamp atlases ~90 MB,
1024² composite targets) live in `ctx.res`, are destroyed at the end of the bake,
and only exist on a cold first load.

## Bake

Two GPU stages, both in-browser via the harness flow, ~5–7 min cold, then
cached/committed:
`mesh/baked/036-deferred-grass-field/field-v4-default-s42-t256a16e6.bin`
(50.3 MB = 64 B header + mip 0 of all 12 slabs; the mip chain is generated at
load by a compute pass, which is cheaper than storing it).

* **Stage A — stamp library.** Each species' GCMESH1 mesh is rasterized with a
  *sheared* orthographic projection: `uv = p.xz + phi * s * p.y`, depth
  `= topH - p.y`, so the depth test keeps the highest surface on each ray — the
  rasterizer resolves a whole ray bundle per pass, which is the only reason
  2.2–6.5 M-triangle plants can be "raycast" at all. 3 species × 6 elevations ×
  16 yaws = **288 mesh draws**. At the grazing cell one stamp is a 12 m streak
  (2078 texels), so the atlas column count adapts to `maxTextureDimension2D`.
* **Stage B — field composite.** For each of 6×16 = 96 directions, the stand's
  scatter over one 6 m tile (306 plants on `default`) is stamped by an exact
  affine rule: a plant at `p` with yaw `psi` and scale `o` seen from azimuth `az`
  is `p.xz + o * R(az) * stamp[psi - az]`, so ONE stamp library serves every
  azimuth (yaw quantised to 16 steps — invisible in a field). `frag_depth` = the
  hit height resolves which plant the ray meets first, and copies at integer tile
  offsets (9–49 of them) make the answer **periodic**, which is what lets a ray
  travel 15 m horizontally inside a finite table.
* Reduce: 4×4 coverage-weighted supersample → premultiplied slab slices.

## Wind

The canopy leans as a linear-in-height shear `k(x,z,t)` taken straight from the
shared `wind_sway` model evaluated at the pixel's ground point. Because a shear
maps lines to lines, querying the sheared world with ray `(O,d)` is exactly
querying the rest world with `(S⁻¹O, S⁻¹d)`; `S` fixes `y`, so the ground
crossing is unchanged and only the direction bends: `d' = d - k * d.y`. Four ALU,
and *exact* for the model rather than an approximation of it. Because the shear is
evaluated per pixel at the entry point, the gust term produces a travelling
ripple across the field rather than a uniform lean.

**Honest limitation:** one composite table can only be sheared by one amount, so
per-species sway collapses to the stand's density-weighted mean (0.68 on
`default`). A moss-only stand still gets sway 0 correctly, but a mixed stand
cannot have its species lean by different amounts. The alternative — one table
per species, composited by depth — triples both VRAM and taps.

## Status

**working** — verified by headless screenshots at 1280×800 on `default`
(`grazing`, `topdown`, `inside-plant`, `far-horizon`, `debug=normals|depth|
lighting|albedo|coverage`, `t=8` and `t=11.5` for wind, plus `close-quality`) and
on `scaling-100m` (134.2M plants). Typecheck clean; no console, page or
validation errors. All five global debug views are meaningful: `depth` is the
canopy hit (not a proxy), `normals` is per-pixel canopy normals, `coverage` is
the alpha actually blended, and `lighting` divides out exactly because the AO is
folded into the albedo handed to `light_surface`.

Bench (solo page, but a shared machine — the ratios below are the trustworthy
numbers): `results/036-deferred-grass-field__default__p-1ecb7de3__apple-metal-3__2026-07-25T09-59-38-359Z.json`
— `grass-field` p50 1.98 ms at 1600×900, whole-frame Σp50 ≈ 4.3 ms.

## Findings

### Speed: parity with billboards, and free scaling

Same-frame A/B (`#/ab/001-billboard-smoke/036-deferred-grass-field`), contended
GPU, ratio of B (this) to A (billboard cards: cull + cards):

| cam | ratio B/A |
|---|---|
| grazing | **0.97×** |
| far-horizon | **0.91×** |
| topdown | 1.16× |
| inside-plant | **0.79×** |

Cost parity with the champion, from a pass bounded by screen pixels and fully
independent of plant count (134.2M plants: same 1.4 ms), at 19.2 MB/species
instead of 21.3, with real per-pixel depth and normals.

### Looks: it does NOT beat billboard cards, and here is exactly why

Flicker/wipe at `cam=grazing` is decisive and not in my favour. Billboards show
individual calamagrostis plumes with hard edges standing at eye level; this shows
a granular, tufted meadow *surface*. What the ray answer genuinely buys — and all
of it is visible — is parallax within the canopy as the camera moves (the answer
slides against the ground because the hit is metres nearer than the ground
point), a silhouette that changes with view direction (the streaks at grazing
angles are real occlusion by tall plants), self-occlusion through burial AO,
correct contact with the soil, and honest prefiltering into the distance. What it
does not buy is **blade-scale silhouette**, for one measurable reason: a table
texel is 2.34 cm and a pixel at 1 m is 1 mm, so the near field is a 12×
magnification of the table. That is a resolution wall, not a bug — 1.17 cm texels
would cost 230 MB.

The near field is partly rescued by a **sub-texel relief field** (deterministic
world-space value noise, two octaves, displacing the hit height and tilting the
normal by its analytic gradient, faded out with the pixel footprint). That is
honest detail texturing, not baked data, and I flag it as such: with
`reliefAmp=0` the canopy is visibly smoother and blurrier close up. It is not a
dither — it is stable in world space, it only breaks the *partial*-coverage band
(`4c(1-c)`), so solid canopy stays solid and depth stays a hard surface.

### Where it breaks (all four inherent, none hidden)

1. **Placement is a periodic realisation, not the stand's exact plants.** This is
   the technique's fundamental tension with the stand contract and I will not
   dress it up: the table bakes ONE genuine 6 m × 6 m sample of
   `ctx.scene.scatter` (real positions, scales, yaws, species mix, each entry's
   exact density) and tiles it. Every species the stand asks for is present at
   the right density, size distribution and mix — but plant #12345 is not at its
   exact xz, so an A/B diff against a per-plant renderer shows *a different
   arrangement of the same field*, not a match. No table indexed by anything
   smaller than the whole stand can do better; the alternative is per-pixel
   iteration over plants, i.e. the marching this round exists to avoid. Stand +
   seed still fully determine the table (both are in the bake key), so runs are
   reproducible and comparable within a stand.
2. **Eye inside the canopy.** The table answers for the whole ray *line*, so when
   the eye is under the tallest tips (grazing sits 9 cm below them, inside-plant
   is half way down) part of the answer lies behind the camera and the table
   cannot report the next hit down. Grazing degrades gracefully — the hit falls
   back to the *local* canopy-top height from the steep slab, which keeps the
   real height variation instead of ironing it flat — but inside-plant is the
   documented "camera inside a plant" breakdown: the fringe of tips over the sky
   fades out (inventing blobs there looked far worse) and the near canopy
   flattens into an eye-level carpet.
3. **No grass above the terrain silhouette beyond the near field.** Sky pixels
   have no ground reference, so they use the plane under the camera: that gives a
   correct fringe of near tips against the sky (bounded by a segment-extinction
   term, which is why it thins with elevation) but distant ridge lines are shaved
   — a ~10 px bare crest at 100 m. Fixing it properly needs a second rasterized
   carrier surface, which would roughly double the pass.
4. **Habitat zoning collapses.** A stand whose entries use `wetCenter/wetWidth`
   (e.g. `bog`) is sampled at ONE spot of the wetness field, so the tile freezes
   one zone's mix instead of intergrading. Zoned stands would need a zoning axis
   in the table — another multiplier on table size.

### Smaller things learned

* Feeding an 8-bit height channel into a 2 cm-wavelength noise turns the near
  field into contour marbling — the 1.1 cm staircase *becomes* the pattern.
  Anchor detail noise to a smooth quantity (here: the hit plane).
* The raw gradient of a 1 cm-wavelength relief field saturates the normal into
  rainbow noise; soft-normalizing it (`g/(1+0.6|g|)`) caps the tilt near 60° and
  is the difference between grass and static.
* The stand-region test must be applied to the HIT, not to the ground crossing:
  for a near-horizontal ray the ground crossing can be 150 m behind the camera,
  which silently killed the entire fringe of tips.
* `layout` is a WGSL reserved keyword (as CLAUDE.md warns). `dir`, `atlas`,
  `grid`, `info`, `rect`, `top` are all fine.
* Only the `default` stand's table is committed (50 MB each). Any other stand
  bakes on demand in-browser and lands in the OPFS cache; `close-quality` and
  `scaling-100m` were baked, verified and then deleted to keep the repo lean.

### Verdict

A structurally clean answer to the brief — zero plant geometry, one primitive, a
constant handful of taps, plant count genuinely free, real per-pixel depth and
normals, wind expressed exactly — which lands at cost parity with billboard cards
and loses to them on near-field looks because the ray answer is ~12× too coarse
exactly where the eye is most demanding. Worth keeping as the cheapest and most
scalable renderer in the repo and as a measurement of where the resolution wall
of a screen-space ray table sits; not a replacement for the champion at eye level.
