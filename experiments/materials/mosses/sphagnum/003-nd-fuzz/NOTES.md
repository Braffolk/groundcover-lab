# 003-nd-fuzz — Uncharted 4's fuzzy-moss BRDF, as a graph material

## Idea

Port the shading stack of `039-nd-moss` — Naughty Dog's fuzzy-moss material from
*"The Technical Art of Uncharted 4"* (SIGGRAPH 2016, slides 28–65) — onto
`src/material/`, and find out whether the graph abstraction can host a
genuinely non-standard BRDF without being bent.

002-graph-maps proved a *flat* material ports cleanly, and said so itself: its
BRDF is "the honest floor", and the interesting version — a light wrap keyed off
the geometry normal, a fuzz layer, an AO aperture folded into the sun term only,
and a subtraction that splits `light_surface()` in half — was deliberately left
for this one. So this experiment is really two claims:

1. the paper's stack renders correctly on preview geometry, and
2. `material_shade` is `shade_moss` **with the plumbing deleted** — which is the
   test the whole graph/WGSL seam was designed to pass.

Claim 2 held. The seam did not move. What did not fit is smaller and specific,
and it is written up under [What had to be bent](#what-had-to-be-bent).

## The graph

```
albedo-{wet-vigorous,late-season,sun-exposed}   image ─→ albedo      variantBlend(habitat) ─→ channel albedo
normal-{...}                                    image ─→ normal      variantBlend(habitat) ─→ channel normal
ao-{...}                                        image ─→ ao-map      variantBlend(habitat)
                                                                 └─→ occlusion  procedural: pow(ao, aoStrength) ─→ channel ao
height-{...}                                    image ─→ height      variantBlend(habitat) ─→ channel height
                                                                                          └─→ the view-uv stage
tip-color-{...}      procedural const ─→ tip-color        variantBlend(habitat) ─→ channel tip_color
base-color-{...}     procedural const ─→ base-color       variantBlend(habitat) ─→ channel base_color
plane-norm-{...}     procedural const ─→ plane-norm       variantBlend(habitat) ─→ channel plane_norm

[parallax != off]
relief-measured-{...} procedural const ─→ relief-measured variantBlend(habitat) ─┐
parallax-gain         procedural: u.p.parallax_depth ───────────────────────────┴─→ relief-m  combine multiply
                                                                                   └─→ view-uv scale (metres)
```

35 nodes, 7 channels, `viewUv: parallaxOffset`. Node kinds used: **all five** —
`image`, `variantBlend`, `procedural`, `combine`, and `filter` is the only one
absent (nothing here is a neighbourhood gather; 002 gave that kind its user).

### The measured SCALARS are graph data, not shader constants

The decision I am happiest with. `tipColor`, `baseColor`, `planeH` and the
cushion relief are per-habitat measured numbers that change with **the same
selector** as the maps, so they are constant `procedural` nodes behind the same
discrete `variantBlend`. That costs nothing (every variant is live, so the blend
stays live and no texture is created), keeps `habitat` a pure selector instead
of a structural param, and — the part that actually paid off — makes them
**inspectable**: `#/material/003-nd-fuzz` shows `tip_color` and `base_color` as
flat swatches beside the maps they were measured from, and the paper's claim
that moss is "lighter at the tips than at the base" is visible as the difference
between two squares. The alternatives (nine `p.num()` sliders for numbers nobody
should drag, or literals buried in the WGSL) are both worse, and neither is
visible to the graph.

`combine: multiply` exists because `Scalar` is *a literal OR a param OR a node*
and has no arithmetic, so "measured constant × user gain" needs a node. 002's
NOTES flagged `combine` as the least-tested kind, shipped without a consumer;
this is its first real one, and it worked first try.

### Map conventions, declared on the nodes

Both were measured off the source mesh and both are non-obvious:

- **`normal.png` is MESH-FRAME with +Y up, axis order (T, N, B)** — the map's
  GREEN channel is the surface normal, not the conventional blue. Declared as
  `semantics: { kind: 'unit-vector', hemisphere: [0, 1, 0] }` on the node, and
  consumed in `surface.wgsl` as `tangent*m.x + geo_n*m.y + bitangent*m.z`.
  Reading it as (T, B, N) tilts every fragment ~90° and looks merely "off".
- **The albedo is LINEAR** (`srgb: false`), not sRGB: the PNG's byte mean equals
  `measured.json`'s `meanColor` to three decimals.

Neither is a shader comment; the declaration drives the mip filter and the
generated decoder together, so they cannot disagree.

## The port: paper → 039 → here

### Taken verbatim from the paper (via 039)

| slide | term | where it is |
| --- | --- | --- |
| 37 | sun micro shadowing, `aperture = 2*ao*ao; micro = saturate(abs(dot(L,N)) + aperture - 1)` | `shade.wgsl`, character for character |
| 38–40 | AO Fresnel, keyed off the **geometry** normal | `shade.wgsl` (strength adapted, below) |
| 51 | light wrap: broaden `NdotL` past the terminator using the **geometry** normal, tinted, converging to plain `NdotL` as the tint → 0 | `shade.wgsl` |
| 52 | Fuzz BRDF — the explicit **subpixel** answer, tinted by the tip colour | `shade.wgsl` |
| 53 | parallax + parallax shadows | view-uv stage (shadows: `pom` only — see below) |
| 55 | AO saturation as an **additive** bounce | `shade.wgsl` |
| 63–65 | unified wetness: lerp base colour toward its **square**, flatten normals as it saturates | `surface.wgsl` |

The `light_surface()` split is 039's and is preserved exactly: call the shared
model once, recompute the sun half, subtract it to get the ambient half, then
occlude each half with its own occluder — cavity AO on the indirect term, micro
shadow × parallax shadow on the sun. With `ao = micro = shadow = 1` the result
is the shared model and the ambient formula is never duplicated. **Verified
numerically** (see Status).

### Adapted, and why

1. **`aoFresnel` defaults to 0.25, not the paper's implicit 1.0.** The paper's
   moss coats geometry that owns the silhouette, so fading the map's cracks out
   at a glancing angle costs nothing — the rock's own shape carries the read. A
   mat has no such host, and a *sphere* is at a glancing angle over most of its
   projected area, so the term erases the only cue that makes the cushion read
   as 3D (moss albedo is nearly uniform: tip vs base differ by ~7%, so all of
   its structure is shading). Measured on this geometry, at `three-quarter`:
   `aoFresnel = 1` flattens the object's luma spread from **σ 0.141 to 0.129**
   and pushes 19.5% of the lighting view past 1.0, against 14.5% at the shipped
   default. 039 reached the same conclusion by eye on a carpet; here it is a
   number. It stays a param, and 1.0 is the paper exactly.
2. **The AO curve reaches exactly 1.0.** 039 wrote `pow(ao, max(k, 0.01))`,
   which can never quite be an identity, so its `light_surface()` identity was
   arguable rather than checkable. The `occlusion` node returns literally `1.0`
   at `aoStrength = 0`, which is what makes the identity test below falsifiable.
3. **The wetness crevice pivot is measured, not a midpoint.** 039:
   `wet = wetness * (0.45 + 0.55*(1 - h))`. Here the `1 - h` is replaced by
   `saturate((plane_norm - h) / plane_norm)`, where `plane_norm` is the measured
   mean cushion plane in height-map units (0.806 for wet-vigorous — the cushion
   fills only the top fifth of the capture range). Water pools *below the mean
   plane*; above it, wetness sits at its dry floor instead of ramping the moment
   a fragment is off the very top. Same shape, same endpoints, measured pivot.
4. **Wetness is a knob, not a habitat zone.** In 039, `wetness` was multiplied
   by the stand entry's own wetness-band centre, so the effective value varied
   across the bog. A material has no stand, so the param IS the wetness and the
   default (0.7) therefore paints a wetter surface than 039's field average.
5. **Wetness and the normal build moved from the BRDF to `material_surface`.**
   Both change what the surface *is* (albedo, normal) rather than how light
   leaves it, and both are view-independent. 039 has no surface/shade seam so it
   did them inside `shade_moss`; this is the only reordering the port makes, and
   it is the seam doing its job rather than the material fighting it.
6. **`parallaxOffset`'s `limit` carries 039's travel bound** — see the next
   section; this is the one place where a *shared* stage was not expressive
   enough and I had to encode the missing behaviour in an existing knob.

### What could not be reproduced on preview geometry

- **Thickness.** 039's headline claim beyond the paper was that within 7 m a
  Sphagnum tile is a *displaced patch*: real vertices up to 3.3 cm of relief,
  real depth, a real silhouette, hard-edged gaps opening onto peat 7–9 cm below.
  A material draws `ctx.preview.mesh` and nothing else, by rule, so all of that
  is gone. What is left is exactly the paper's own answer — parallax for motion
  relief inside the surface, the fuzz layer for the subpixel regime — which is
  the honest scope of a *material*, and is why the paper's moss lives on rocks.
- **`apexH` is measured but unused at runtime.** It is the mean capitulum apex
  (0.0845 m, 11 mm above the mean plane); 039 used it to place displaced patch
  vertices. There is nothing in a material for it to displace, so it is read,
  reported here, and deliberately not wired — an unreferenced node would be a
  validator warning, and inventing a use would be worse.
- **`cover` (0.984 / 0.938 / 0.933) justifies a decision rather than feeding
  one.** 039 alpha-tested texels below `gapRef` away as real gaps down to peat.
  On a preview object there is no substrate behind the surface, so a discard
  would punch holes in the *object*; `s.opacity = 1.0` and there is no gap test.
- **No specular/sheen lobe in the shared lighting model**, so the paper's
  wetness can only darken albedo and flatten normals, never sharpen a highlight.
  (039 recorded the same limitation.)

## What had to be bent

Four things, in descending order of how much they matter. None of them needed a
new node kind, a change to `src/`, or a special case in the generator — but the
first one is a genuine hole in the ABI.

### 1. `Surface` has ONE tint slot and this BRDF needs two

`material_shade(g, s, vu)` receives `Surface`, and `Surface` is the only channel
from stage 4 to stage 5. The stack needs two measured colours there:

- the **capitulum-tip** colour, which tints the fuzz layer → `s.sheen`. This one
  is a perfect fit: `sheen` is exactly "the microfibre layer's own colour".
- the **deep-cushion** colour, which tints the light wrap and the AO-saturation
  bounce → **`s.emissive`, which is a lie**. It is not emission; it is a
  carrier, flagged with a comment at the assignment and at the use.

The right fix is one more slot on `Surface` — `subsurface: vec3f` (or a generic
`tint`) — beside the `thickness: f32` that is already there. The ABI header says
`geo_n` is kept beside `normal` *because* ND moss's wrap and AO-Fresnel are
defined by that choice; the wrap's **tint** is defined by the paper just as
explicitly, and there is no slot for it. Everything else about the ABI fit this
material better than I expected, which is what makes this gap worth naming.

Two escape hatches were available and both rejected: the generated
`n_<id>_eval` functions are in scope in the shade stage, so it *could* call
`n_base_color_eval(mat_eval_ctx(g))` directly — but that hardcodes a node id
into the BRDF, and the graph is a function of the params, so the node may not
exist. And a `var<private>` written by stage 4 and read by stage 5 works, but is
an invisible side channel. An honest lie in a visible slot beats both.

### 2. `parallaxOffset` never fills `ViewUv.shadow`, so the paper's parallax shadows are only available from `pom`

`ViewUv` carries a `shadow` field and the ABI documents it as "self-shadow along
the light"; `pom` fills it with a second march (`shadowSteps`), and
`parallaxOffset` sets it to 1.0 unconditionally. 039's parallax self-shadow is
*one tap* — step toward the sun by a fixed fraction of the relief and compare —
so it is neither a march nor expensive, but there is nowhere to put it:

- the view-uv stage is generated, not authored;
- a `procedural` node cannot do it either, because **`EvalCtx` has no tangent
  frame**. It carries `uv`, `ddx`, `ddy`, `world`, `geo_n`, `view_ws` — enough
  to project the *view* direction, but a sun-direction tap needs `tangent` and
  `bitangent`, which only `Geo` has. So "sample the height field one step toward
  the sun" is not expressible as a node.

Shipped compromise: `parallax: 'pom'` turns on `shadowSteps` (default 8) and
gives the paper's parallax shadows via a march; the default `offset` has none,
matching 039's stack minus its one shadow tap. Measured, the term is worth
2.6% of mean luma at `three-quarter` (0.3787 → 0.3689). A one-tap
`shadowOffset?: number` on `parallaxOffset` would close this cleanly.

### 3. `parallaxOffset` has no travel bound, and on a periodic tile that is not cosmetic

This is the only thing that rendered visibly *wrong* before it was dealt with,
and it cost the most time, so it is written out in full.

039 bounds the parallax travel at **half the relief in metres** and says why: on
a periodic tile a large offset lands in the *next copy of the same moss*, which
reads as rings of phase-aligned repeats rather than as parallax. The generated
stage has only `limit`, the classic offset-limiting clamp on the denominator:

```
denom = max(|vt.z|, limit);  step = -vt.xy / denom * scale;  uv += step * (1 - h)
```

`vt.xy` is the view direction's tangential part already scaled to **tiles per
metre**, so the worst-case offset is `tilesPerMetre * scale / limit` tiles.
Equating that to 039's bound (`0.5 * relief * tilesPerMetre` tiles) gives

> **limit = 2 × parallaxDepth**, and `tilesPerMetre` cancels

— so a fixed `limit: 2.0` reproduces 039's bound on every preview object and at
every `tileScale`, and the bound automatically tracks the depth gain. Note the
consequence: at `limit ≥ 1` the `max()` can never select `|vt.z|` (it is ≤ 1),
so the stage degenerates to `step = -vt.xy * scale/2`, which is Welsh's
offset-limited parallax (the variant that drops the divide by z entirely) at
half strength — and *the half is the travel bound*.

This matters more here than for a typical POM material because the relief is
**half the tile period** (0.093 m of height in a 0.18 m tile). Measured at
`macro`, where the object overfills the frame:

| `limit` | worst-case offset | what it looks like |
| --- | --- | --- |
| 0.45 (039's `PARALLAX_LIMIT`, no travel bound) | 1.15 tiles | a radial smear ~90 px long; no rosette survives |
| 1.0 (canonical Welsh) | 0.52 tiles | still visibly streaked toward the limb |
| **2.0 (shipped)** | **0.26 tiles** | individual capitula resolve; residual softening only |
| parallax off | 0 | sharpest — the ceiling this is measured against |

A `maxOffset` (in tiles, or in metres) on `parallaxOffset` would say this
directly instead of encoding it in a clamp that no longer clamps anything.

### 4. A node with two consumers is evaluated twice — and this is the first material where that costs something

002's NOTES predicted it: `n_<id>_eval(c)` calls its dependencies *inside*
itself, so a node feeding two consumers is evaluated once per consumer. Here
`height` feeds both the view-uv stage (2 taps: the offset step plus its secant
refinement) **and** the `height` channel, because the paper's wetness pools
water by fragment height. So the default configuration is **6 texture samples
per fragment** — albedo, normal, ao, and height three times — where 002 with the
same stage does 5 and 002 at its default does 3. One of the six is pure
duplication, and no amount of declaring can remove it: the value is needed after
the uv has moved, so it cannot be hoisted, only reused.

Also worth recording, though it is a data-layout fact rather than an abstraction
one: 039 packed albedo+height into one RGBA and normal+AO into another, taking
4–5 taps for the same work. One `image` node is one PNG is one texture, so the
graph cannot express that packing while the assets are four separate files.

## VRAM budget math

Measured by the tracker (`__materialReports['003-nd-fuzz'].textureBytes`), not
estimated. One habitat is resident at a time — that is what `select: 'discrete'`
buys, and it is why three habitats cost one.

| texture | format | level 0 | with mips |
| --- | --- | --- | --- |
| albedo (2048²) | rgba8unorm | 16.00 MiB | 21.33 MiB |
| normal (2048²) | rgba8unorm | 16.00 MiB | 21.33 MiB |
| ao (2048²) | r8unorm | 4.00 MiB | 5.33 MiB |
| height (2048²) | r8unorm | 4.00 MiB | 5.33 MiB |
| **total** | | | **55,924,050 B = 53.33 MiB** |

Plus a 96-byte params uniform (14 params → 16 padded floats + the 8-float
harness header). Nothing is materialized in any configuration, so there is no
compute-pass output and no PNG in the bake cache.

That is **byte-for-byte 002-graph-maps' `parallax != off` configuration** — the
same four maps, so the same argument applies unchanged: the 25 MB bar was
calibrated for a plant species' baked representation and has no
material-appropriate meaning yet, and BC7/BC5/BC4 would bring this set to
~14 MB. The HUD shows `53.3/25MB`; read it as "this material's maps", not as a
verdict. The three measured-scalar channels (`tip_color`, `base_color`,
`plane_norm`) and their nine variant nodes cost **zero bytes** — every one is
live.

## Bake

**None.** The maps under `assets/materials/sphagnum-*` are committed source
data, measured off the 19.8M-triangle Sphagnum cushion by 039's `bake.ts` (NDM1
v4); the mesh has since been deleted and they cannot be regenerated. No node in
this graph is `filter`-kind or `materialize: 'always'`, and the validator report
confirms it: **0 materialized**. The measured scalars come from the same
`measured.json` the maps ship with, read at `create()` and compiled into
constant node bodies.

## Status

**working.** Verified headless (Chrome, WebGPU) at 1600×950, HUD hidden, reading
back the canvas; every number below is over **object pixels only**, masked by
the `coverage` debug view so the identical studio backdrop cannot dilute it.
**Zero console errors and zero `.toast` elements in every run.** `#/`,
`#/run/002-graph-maps` and `#/run/001-billboard-smoke` all still load clean.

### The `light_surface()` identity, measured

With `aoStrength=0` (→ ao exactly 1.0), `microShadow=0`, `parallax=off`
(→ shadow 1.0) and `lightWrap = fuzz = aoSaturation = wetness = 0`, the BRDF
must collapse to the plain shared lighting model — which is exactly what 002 is
when its own occlusion is 1 (`p.aoStrength=0`). Both build the same albedo and
the same normal from the same maps, so this is a clean test of the split:

| camera | object px | differing | differ % | RMSE | max Δ |
| --- | --- | --- | --- | --- | --- |
| three-quarter | 206,315 | 2 | 0.0010% | 0.002 | **1** |
| macro | 1,520,000 | 11 | 0.0007% | 0.002 | **1** |
| grazing | 439,642 | 6 | 0.0014% | 0.002 | **1** |
| silhouette | 643,764 | 2 | 0.0003% | 0.001 | **1** |

The maximum disagreement anywhere is **one 8-bit LSB**, on 2–11 pixels out of up
to 1.5 million. A same-code control (003 captured twice at `macro`) differs on
**0** pixels, so the scene is deterministic and those LSBs are real: they are
float re-association — `sun_part + max(lit - sun_part, 0)` is not bit-identical
to `lit`, and the surface stage renormalises an already-normalised vector once
more. The identity holds to the last bit that can be observed.

### Debug views (all five, plus off)

| view | mean | p01 | p99 | σ | verdict |
| --- | --- | --- | --- | --- | --- |
| off | 0.375 | 0.139 | 0.613 | 0.141 | **0.000% clipped** — nothing is blown out |
| albedo | 0.453 | 0.390 | 0.495 | 0.020 | flat, as a measured moss albedo should be (tip vs base differ ~7%) |
| normals | 0.679 | 0.335 | 0.872 | 0.141 | real per-fragment variation at capitulum scale, over the sphere's own gradient |
| lighting | 0.758 | 0.303 | 1.000 | 0.243 | **not 1.0** — shaded/albedo genuinely varies (the 017 trap is "mean ≈ albedo mean") |
| coverage | 1.000 | — | — | 0.000 | opaque everywhere, by design (no gap discard — see `cover`) |
| depth | 0.141 | 0.075 | 0.192 | 0.030 | smooth ramp over the sphere |

(at `three-quarter`; `macro` agrees — lighting mean 0.677, σ 0.282.)

The lighting view saturates on 14.5% of object pixels, all of them on the
sun-facing cap: the shared model's own `sun + ambient*hemi` already exceeds 1
there (002 clips 1.5% of the same view), and the wrap/fuzz/bounce push more of
the cap past it. The **rendered** image clips on 0.000% of pixels at every
camera, so this is the debug view's [0,1] display of an unbounded ratio, not a
blown-out material. Removing the micro shadow *raises* lighting clipping to
20.4%, which is the term doing its job.

### Against 002-graph-maps (flat map sampling), same maps, same geometry

| camera | differ % | RMSE | mean luma 003 / 002 | σ 003 / 002 |
| --- | --- | --- | --- | --- |
| three-quarter | 100% | 29.5 | 0.375 / 0.271 | **0.141 / 0.101** |
| macro | 100% | 48.4 | 0.341 / 0.253 | **0.172 / 0.154** |
| grazing (sphere) | 100% | 29.4 | 0.302 / 0.221 | **0.148 / 0.111** |
| silhouette | 100% | 39.0 | 0.286 / 0.209 | **0.147 / 0.127** |
| grazing (**plane**) | 100% | 45.3 | 0.557 / 0.335 | 0.043 / **0.055** |

σ is the luma spread over the object — the crude proxy for "how much shading
structure is there". The ND stack is **14–40% higher** on every curved view,
which is the visible claim: more directional shading, more cushion depth. The
plane at `grazing` is the one place it is *lower*, and that is the paper's own
prediction rather than a defect — at 4° off the surface you see only the tips,
the fuzz layer takes over, and texture contrast necessarily falls. 039 recorded
the same and it is why `fuzz` defaults to 0.25 rather than 0.5.

### Term budget (three-quarter, object pixels)

| configuration | final mean | final σ | lighting mean | lighting ≥ 1 |
| --- | --- | --- | --- | --- |
| **full stack (defaults)** | 0.3746 | 0.1410 | 0.758 | 14.5% |
| no AO saturation | 0.3295 | 0.1467 | 0.691 | 10.7% |
| no light wrap | 0.3620 | 0.1483 | 0.736 | 14.3% |
| no fuzz | 0.3716 | 0.1417 | 0.754 | 14.4% |
| `aoFresnel = 0` | 0.3623 | 0.1424 | 0.736 | 12.8% |
| `aoFresnel = 1` (paper verbatim) | 0.4036 | **0.1294** | 0.814 | 19.5% |
| no micro shadow | 0.4480 | **0.1062** | 0.890 | 20.4% |
| no wetness | 0.4080 | 0.1536 | 0.757 | 14.5% |
| no parallax | 0.3709 | 0.1414 | 0.753 | 14.2% |
| `pom` + parallax shadows | 0.3689 | 0.1411 | 0.748 | 13.7% |
| `pom`, no shadow march | 0.3787 | 0.1402 | 0.765 | 15.1% |

The **micro shadow is the single most load-bearing term**: removing it lifts
mean luma 20% and collapses the luma spread by 25% (σ 0.141 → 0.106). That is
one line of the paper, taken verbatim, doing more for the read than everything
else combined. The AO Fresnel result quantifies the divergence at (1).

### Three habitats, one selector

| habitat | mean luma | σ | textureBytes |
| --- | --- | --- | --- |
| wet-vigorous | 0.3746 | 0.1410 | 55,924,050 |
| late-season | 0.2605 | 0.1138 | 55,924,050 |
| sun-exposed | 0.1625 | 0.0731 | 55,924,050 |

Constant VRAM across all three — the discrete blend rebinds one slot. Switching
also moves the tints and the relief, because they ride the same selector.

### Mip-convention verdicts (from the inspector, per level)

| node | semantics | worst drift | reading |
| --- | --- | --- | --- |
| albedo | color-with-coverage | **0.4%** | flat: the filter already normalised by coverage and the decoder does not divide again |
| height | scalar-field | **0.1%** | flat; mean 0.773, which matches `planeH` sitting at 0.806 of the capture range |
| occlusion (live, evaluated on demand) | scalar-field | **0.1%** | flat |
| normal | unit-vector | 25.4% | **expected**: averaging unit vectors and renormalising pulls them toward the +Y hemisphere, which raises green-weighted luma. This is the mip-safe filter, not the octahedral trap (which would collapse to *exactly* straight up) |

### Other objects and modes exercised

Sphere, plane, cube-edge and cylinder; all three habitats; all four cameras; all
six debug modes; `parallax` at `off` / `offset` / `pom`; `pom` with and without
the shadow march. No toasts, no console errors, validator **0 errors,
0 warnings, 0 materialized** in every configuration.

No `results/` bench JSON is claimed: this machine is shared and the brief says
not to quote absolute frame times. The structural cost is stated instead — 6
texture samples per fragment against 002's 3 at its default and 5 with the same
parallax stage — and one of the six is the duplicate `height` read described
above.

## Findings

- **The honest test passed.** `material_shade` is `shade_moss` with the plumbing
  deleted, and the deletions are all plumbing: the tile's quarter-turn rotation
  of a mesh-frame normal into world (a carpet-lattice concern that died with
  carpets), reconstructing `up`/`v` from a hand-rolled vertex output (`Geo`
  carries both), and the wetness function (a surface property, now in stage 4).
  Every BRDF term — the geometry-normal wrap, the fuzz layer, the `2*ao*ao`
  aperture, the AO Fresnel, the additive bounce, and the `light_surface()`
  subtraction — is line-for-line the same code reading `s.` and `u.p.` instead
  of `info.` and a hand-packed varying. **No new node kind, no generator special
  case, no change under `src/`.**
- **The graph absorbed more than the maps.** The thing I expected to fight —
  nine measured per-habitat numbers — turned out to be the cleanest part:
  `procedural` constants behind the existing `variantBlend` cost nothing, keep
  the selector non-structural, and become inspectable for free. If the graph had
  only been able to hold *textures*, this material would have needed a second,
  private mechanism for habitat-varying data, and that is exactly the kind of
  parallel channel that rots.
- **The one real gap is a tint slot on `Surface`** (see [What had to be
  bent](#what-had-to-be-bent) §1). The ABI already keeps `geo_n` beside `normal`
  *because of this material*; it should keep a `subsurface`/`tint` vec3f beside
  `sheen` for the same reason. Until then the wrap tint rides in `emissive`,
  which is a documented lie.
- **The two view-uv gaps are both small and both concrete**: `parallaxOffset`
  cannot report a self-shadow (§2) and cannot bound its travel (§3). The second
  one is the only thing in this port that produced a visibly broken image, and
  the fix — `limit = 2 × parallaxDepth`, derived above — works but encodes a
  travel bound inside a denominator clamp, where nobody will find it. A
  `maxOffset` field would be one line in `codegen.ts` and would have saved this
  experiment an hour.
- **`EvalCtx` deliberately lacks a tangent frame, and that is load-bearing both
  ways.** It is why a node cannot be view-dependent in a way that would break
  materialization — the validator's best rule — and it is also why the one-tap
  parallax shadow cannot be a node. Worth stating in the ABI rather than
  discovering.
- **`combine` finally has a consumer** (`relief-m = relief-measured ×
  parallax-gain`) and needed no fixing. `filter` is the only kind this material
  does not use.
- **A measured pivot beat a hand-tuned constant, cheaply.** Replacing 039's
  `0.45 + 0.55*(1-h)` with the same curve pivoted on the measured mean cushion
  plane took one extra constant node and made the wetness stop ramping across
  the capitula, where water does not sit. Small, but it is the pattern the whole
  "measured, not authored" idea is supposed to produce.
