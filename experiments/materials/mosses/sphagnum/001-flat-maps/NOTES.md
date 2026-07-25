# 001-flat-maps — Sphagnum from the measured maps, and nothing else

## Idea

The first `material` experiment, and deliberately the least clever one: take the
measured Sphagnum map set exactly as it was captured, sample it on the
harness-owned preview geometry, light it with the shared `light_surface()`, and
stop. No parallax, no fuzz BRDF, no silhouette work.

Two jobs:

1. **Be the floor.** Every later Sphagnum material — parallax, SPOM, a
   graph-authored one — has to beat *this*, and "beat" only means something if
   the baseline is the honest one. Anything that looks better than flat maps has
   earned it; anything that does not is spending VRAM and ALU for nothing.
2. **Prove the plumbing.** `MaterialStage`, the preview geometry + its uv /
   tangent contract, `src/gpu/image.ts` loading, `src/gpu/texture.ts` mip
   semantics and generated decoders, and all five debug views — end to end, on
   every preview object.

Source maps: `assets/materials/sphagnum-{wet-vigorous,late-season,sun-exposed}/`.
They were measured off the 19.8M-tri source mesh by `039-nd-moss/bake.ts`; that
mesh has been deleted, so **these PNGs are source data and cannot be
regenerated**.

## What the shading actually does

```
uv (metres of surface)  ->  x tilesPerMetre  ->  albedo / normal / ao
n = normalize(mix(N, T*m.x + N*m.y + B*m.z, normalStrength))
out = light_surface(albedo, n, world) * mix(1, ao, aoStrength)
```

Three things in there are decisions, not defaults:

- **The albedo is LINEAR, loaded as `rgba8unorm`, not `-srgb`.** `measured.json`
  says "rgb8, linear", and it checks out: the byte mean of `albedo.png` is
  (0.3035, 0.5839, 0.1467) against a recorded `meanColor` of
  (0.3045, 0.5855, 0.1469). The bake wrote authored vertex colour to an
  `rgba8unorm` target with no conversion, and this lab writes fragment colour to
  a non-sRGB target with no encode. An sRGB decode here would darken the moss by
  a full gamma for no reason.
- **The normal map is MESH-FRAME, not tangent-space.** rgb = mesh (x, y, z),
  upper hemisphere — verified in the data: the green channel's minimum over
  4.2M texels is exactly 128, i.e. `n.y >= 0` everywhere, and its mean is 193
  (mean `n.y` = +0.515), while red and blue sit at 128. `moss_bake.wgsl` laid the
  tile out with `u = mesh x / tileM`, `v = mesh z / tileM`. So the axis order is
  **(T, N, B)** — the map's *green* channel is the surface normal. Reading it as
  the usual (T, B, N) tilts every fragment ~90° and looks merely "off" rather
  than obviously broken, which is why it is written out in the shader.
- **No fog.** A studio backdrop has no atmosphere, and at ~1 m the shared fog
  term is 1.6e-5 anyway; applying it would only tint the object toward a sky
  colour that is not in this scene.

`height.png` is deliberately **not loaded**. A flat-maps material has no
parallax and would never read it, and loading a map you do not sample burns
5.3 MiB while misrepresenting the method's cost. It is the first thing the
parallax material adds.

## Tiling: one isotropic number, never a per-axis fit

The preview uv arrives in **metres of surface** (see the contract at the top of
`src/scene/preview.ts`), so life size is `1 / tileM` — one scalar, applied to
both axes. The only correction is on an axis that closes on itself: a fractional
number of tiles around the sphere or the cylinder butts two different parts of
the tile together at the seam, so the period is snapped to the nearest exact
divisor and **the snapped value is used on both axes**, so tiles stay square.

| object    | uv period       | tile size at `tileScale` 1 | error |
|-----------|-----------------|----------------------------|-------|
| plane     | open            | 0.1800 m                   | 0%    |
| cube-edge | open (6 charts) | 0.1800 m                   | 0%    |
| sphere    | open (6 charts) | 0.1800 m                   | 0%    |
| cylinder  | u wraps @2.199m | 0.1832 m (12 around)       | 1.8%  |

Two earlier drafts of this were wrong, and both were wrong in the same way —
letting the *geometry's* parametrisation set the material's scale:

- uv normalised to [0,1] per object, with the material un-stretching it by a
  **per-axis** factor. That stretched the cylinder's caps by 2.2×, the ratio of
  circumference to height, and it is the general shape of "material fitted to
  the polygons". Metres removed the entire class of bug.
- a lat/long **UV sphere**, whose u compresses by sin(colatitude). No scale
  factor fixes a singularity: the material collapsed into a pinwheel of slivers
  at the pole, and the default three-quarter camera looks slightly *down*, so
  the pole was right in frame. Replaced with a tan-warped cube-sphere — bounded
  5.7% distortion at eight corners instead of an infinite one at two poles, and
  as a bonus the sphere now also sits at exact life size because none of its six
  charts wraps.

## VRAM budget math

One habitat is resident at a time. Measured by the tracker, not estimated:

| map     | source               | GPU format   | mips | bytes      |
|---------|----------------------|--------------|------|------------|
| albedo  | 2048² RGB8 PNG       | rgba8unorm   | 12   | 22,369,620 |
| normal  | 2048² RGB8 PNG       | rgba8unorm   | 12   | 22,369,620 |
| ao      | 2048² gray8 PNG      | r8unorm      | 12   |  5,592,405 |
| params  | —                    | uniform      | —    |         16 |
| **total** |                    |              |      | **50,331,661 B = 48.0 MiB** |

Mip sum is exact: Σ(2048>>k)² over 12 levels = 5,592,405 texels, ×4 B for the
two RGBA maps and ×1 B for AO.

**This is 1.92× the 25 MB/species bar, and that is deliberate.** Three points:

1. 25 MB was calibrated for *a plant species' baked representation* in a scene
   holding several species at once. A material holds one map set; the units are
   not comparable, and the HUD bar has no material-appropriate meaning yet. This
   is worth the orchestrator's attention — see the harness feedback below.
2. 50 MB is simply what an uncompressed 2048² albedo + normal + AO set costs. A
   shipping pipeline would BC7/BC5/BC4 it to **13.98 MB**, comfortably inside
   the bar. WebGPU core has no BC on this adapter path.
3. Cutting it is easy and dishonest. Halving to 1024² costs 12.6 MB and would
   fit — but this experiment's entire job is to be the reference for the
   measured data, and a reference that quietly resamples its source is not one.
   The `macro` bookmark is exactly where the difference would show.

Switching `habitat` reloads asynchronously behind a generation guard and
destroys the previous set, so only one is ever resident. Preloading all three
would be 144 MiB.

## Bake

**None.** The maps are committed source data under `assets/materials/`, loaded
through `loadImageTexture` (albedo, normal) and `decodePng` +
`createTexture2D`/`generateMips` (AO, so it can be `r8unorm` instead of paying
for three channels of nothing). Nothing is written to `mesh/baked/`.

Mip semantics are declared once and drive **both** the filter and the generated
WGSL decoder (`wgslDecoder`), which is prepended to the shader source at build
time:

- albedo — `color-with-coverage` weight `a`. The PNG has no alpha, so a = 1
  everywhere and the filter degenerates to exactly a box over straight colour
  while the decoder is the identity. Declaring `premultiplied-color` instead
  would make the decoder divide by that alpha, which is the shape of the
  017-cached-clusters bug even though it is harmless at a ≡ 1.
- normal — `unit-vector` with hemisphere +Y. `measured.json` is explicit that
  the map is plain xyz*0.5+0.5 *because* it is mipped and octahedral pairs are
  not mip-averageable; `unit-vector` is the filter that matches.
- ao — `scalar-field` / `empty: 'zero'`, i.e. a plain box.

## Status

**working.** Verified by screenshot on all four preview objects, at all four
camera bookmarks, in all five debug modes, on all three habitats, plus A/B and
bench. No toasts, no console errors (one dev-only mip warning, discussed below).

Bench (default `orbit-object` spline, sphere):
`results/001-flat-maps__sphere__p-c5d5102f__apple-metal-3__2026-07-25T23-06-50-756Z.json`
— measured on a shared machine, so treat the absolute milliseconds as a
placeholder until something is compared against it on an idle GPU.

## Findings

- **Debug views are all truthful.** `normals` shows the sphere's normal gradient
  with the map's micro-perturbation grained on top; `lighting` shows a proper
  terminator with cavity AO and peaks below 1 (not blown out); `albedo` is flat
  measured green; `coverage` is solid 1.0, which is the honest answer for opaque
  flat maps; `depth` ramps correctly.
- **`debug=depth` has almost no dynamic range at material scale.** The shared
  ramp is `1 - exp(-d*0.03)`, tuned for a 100 m+ field; at 2-3 m it returns
  0.06-0.09, so the object is nearly black. Readable, but only just. Not fixed
  here — changing `debug.wgsl` would move every renderer's depth view.
- **The DEV mip convention check fires on the sun-exposed albedo, and it is
  quantisation, not a convention error.** Stored covered luma rises monotonically
  0.2499 → 0.2561 across the 12-level chain (drift 2.5%, tolerance 2%) while
  covered alpha is exactly 1.0000 at every level. Round-to-nearest in the 8-bit
  render target biases a skewed histogram upward by ~0.25% per level, and a full
  2048² chain has eleven of them. The check's fixed 2% tolerance does not scale
  with chain length; the other two habitats stay under it. Left alone rather
  than tuned — silencing a correctness check to quiet one's own warning is how
  the next real one gets missed.
- **A flat map has no thickness, and the stage makes that obvious.** At the
  `grazing` bookmark (~4° off the plane) the material is a sliver with no
  parallax and no self-occlusion beyond baked AO, and at `silhouette` the sphere's
  outline is a perfect analytic circle. That is the gap the next material has to
  close, and it is visible in one click rather than argued about.
- **The preview sphere had to stop being a UV sphere.** Reported by the owner
  from a screenshot, and it is the single most visible artifact the first
  version had — see the tiling section. Worth remembering as a class: a
  *singularity* in a parametrisation is not a distortion you can tune away, and
  every tiling material would have inherited it.
- The measured habitats really are different materials, not tints:
  wet-vigorous is saturated green (mean 0.30, 0.59, 0.15), sun-exposed is olive
  (0.31, 0.26, 0.06), late-season is between (0.41, 0.41, 0.13).
