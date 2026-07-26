# 002-graph-maps — 001-flat-maps, rebuilt as a material graph

## Idea

Port `001-flat-maps` to the new material graph system (`src/material/`) and
prove the port changed **nothing**. 001 stays untouched as the reference; this
one has to be indistinguishable from it and yet contain no pipeline, no bind
group, no mip plumbing and no `debug_shade` call of its own.

The split the system is built on:

- **Channel DATA is a declarative typed graph.** Three habitats × three maps as
  `image` nodes, one discrete `variantBlend` per channel, an optional `filter`
  on the cavity AO, and a `procedural` node for the AO curve.
- **BEHAVIOUR is authored WGSL.** `shaders/surface.wgsl` (28 lines) and
  `shaders/shade.wgsl` (24 lines) are the entire hand-written shader. The BRDF
  is deliberately not nodes — 039-nd-moss' wrap/fuzz/AO-Fresnel/micro-shadow/
  bounce stack is what this seam exists to keep expressible.

`main.ts` is a declaration and one call to `createMaterialExperiment`.

## The graph

```
albedo-{wet-vigorous,late-season,sun-exposed}  image   ─┐
                                                        ├─ albedo   variantBlend(habitat) ─→ channel albedo
normal-{...}                                   image   ─┤
                                                        ├─ normal   variantBlend(habitat) ─→ channel normal
ao-{...}                                       image   ─┤
                                                        └─ ao-map   variantBlend(habitat)
                                                              │
                                          [aoSoftenPx > 0] ao-soft   filter box-blur  (materialized → PNG)
                                                              │
                                                          occlusion  procedural: mix(1, ao, aoStrength)
                                                              └────────────────────────→ channel ao

[parallax != off]  height-{...} image → height variantBlend(habitat) → the view-uv stage (NOT a channel)
```

Node kinds used: `image`, `variantBlend`, `procedural`, `filter`. `combine` is
in the frozen set but this material has no honest use for it — see Findings.

### Three decisions in there that are not defaults

- **The albedo is LINEAR** (`srgb: false`), the normal map is **MESH-FRAME with
  axis order (T, N, B)** — the green channel IS the surface normal — and the
  normal's mip filter is `unit-vector`, not octahedral. All three are carried in
  the `ChannelSpec` declarations in `main.ts` rather than in a shader comment,
  which is the point: the same declaration drives `generateMips` and the
  generated decoder, so the two cannot disagree.
- **The `variantBlend` is `discrete`.** A blending variantBlend would need all
  three habitats resident: 144 MiB for a param nobody moves per frame. Discrete
  binds ONE slot, loads only the active variant, and the generated WGSL is
  selection-independent — so switching habitat rebinds instead of recompiling.
  That reproduces 001's hand-written generation-guarded swap exactly.
- **`height` is not a channel.** The view-uv stage reads that node directly, so
  at `parallax=off` the height map is not reachable from any root and is never
  loaded. 001 made that decision by hand ("a flat-maps material would never read
  it"); here reachability enforces it.

## VRAM budget math

Measured by the tracker, not estimated. One habitat resident at a time.

| configuration | textures | uniform | total |
|---|---|---|---|
| **default** (albedo + normal + ao) | 50,331,645 B | 80 B | **50,331,725 B = 48.0 MiB** |
| `parallax != off` (+ height, r8unorm) | 55,924,050 B | 80 B | 53.3 MiB |
| `aoSoftenPx > 0` (+ materialized blur) | 72,701,265 B | 80 B | 69.3 MiB |
| `aoSoftenPx > 0` + `aoNode=materialized` | 95,070,885 B | 80 B | 90.7 MiB |

001-flat-maps measured 50,331,661 B in the same configuration; the 64-byte
difference is this material's larger params uniform (10 params instead of 4).
**The default is byte-for-byte 001's map set** — the same argument for 1.92× the
25 MB bar applies unchanged (see 001's NOTES; BC7/BC5/BC4 would be ~14 MB, and
the bar was calibrated for a plant species' baked representation, not a material).

Two costs are worth stating rather than discovering:

- **A materialized node costs 4× a scalar's worth of texture.** WebGPU core has
  exactly one 8-bit storage-texture format, `rgba8unorm`; `r8unorm` is not
  storage-capable. So the blurred AO — one channel — occupies 22.4 MB rather
  than 5.6 MB. That is why `aoSoftenPx` defaults to 0 and why the node is
  removed from the graph entirely at radius 0 rather than run as an identity.
- Everything above the default row is **off by default**, so the shipped
  configuration costs exactly what 001 costs.

## Bake

The measured maps under `assets/materials/sphagnum-*` are committed source data
and are loaded, not baked. What IS baked is every **materialized node**, through
the ordinary bake flow with `ext: 'png'` (OPFS → `mesh/baked/<exp>/<key>.png` →
in-browser compute). Nothing is materialized in the default configuration, so
the default has no bake at all.

**The cache key hashes the node's transitive subtree, not the graph.** Measured,
in one browser session, on `ao-soft` at radius 4 (2048², ~0.9 s to bake):

| step | change | key | result |
|---|---|---|---|
| 1 | first load | `ao-soft-4234f3a0` | **baked** (935 ms) |
| 2 | identical reload | `ao-soft-4234f3a0` | cache (131 ms) |
| 3 | `normalStrength=0.5` (unrelated param) | `ao-soft-4234f3a0` | **cache** |
| 4 | `tileScale=2` (unrelated param) | `ao-soft-4234f3a0` | **cache** |
| 5 | `parallax=pom` — adds 4 unrelated NODES | `ao-soft-4234f3a0` | **cache** |
| 6 | `habitat=late-season` (the subtree's selector) | `ao-soft-12f4dd53` | baked |
| 7 | `aoSoftenPx=5` (the node's own declaration) | `ao-soft-77357c74` | baked |
| 8 | back to `aoSoftenPx=4` | `ao-soft-4234f3a0` | cache |

Steps 3–5 are the property that matters: a whole-graph key would have
invalidated the bake in all three, and nobody iterates after that.

One trap the key had to be taught about: a discrete `variantBlend` reads its
selector through the **bind group**, not through `u.p.<name>`, so the textual
`u.p.` scan cannot see it. Without the extra rule in `bakeKeyFor` the key would
have been identical for all three habitats and step 6 would have served
wet-vigorous' blur for late-season.

## Status

**working.** Verified headless on all four preview objects, at all four camera
bookmarks, in all five debug modes, on all three habitats, plus A/B against 001
and a bench. Zero console errors, zero `.toast` elements in every run.

Bench (default `orbit-object` spline, sphere):
`results/002-graph-maps__sphere__p-07fd3751__apple-metal-3__2026-07-26T00-03-33-277Z.json`,
against 001's
`results/001-flat-maps__sphere__p-c5d5102f__apple-metal-3__2026-07-26T00-04-03-406Z.json`
run minutes apart on the same machine. Material pass **p50 0.203 vs 0.210 ms
(0.97×)** — parity, which is the expected answer since the generated fragment
does the same three texture reads and the same arithmetic. p95 differed by 3×
in 002's favourless direction on a shared machine whose *base* pass p95 also
swung 3×; treat the absolute milliseconds as contended and unquotable.

## Findings

- **The port is byte-identical to 001.** Full-resolution HUD-free goldens
  (1600×950, 1,460,800 px) at all four material cameras:

  | camera | 002 vs 001 differing px | RMSE | max channel Δ |
  |---|---|---|---|
  | three-quarter | 0 (0.0000%) | 0.000 | 0 |
  | macro | 0 | 0.000 | 0 |
  | grazing | 0 | 0.000 | 0 |
  | silhouette | 0 | 0.000 | 0 |

  A same-code control (001 captured twice) is also 0 everywhere, so this scene
  really is deterministic and the zeros mean what they look like. The A/B diff
  view agrees independently: `0.00/255 mean · 0.00% px differ` at all four cams.

- **The live-vs-materialized duality holds, and measuring it found a real bug.**
  `aoNode` flips the `occlusion` node between live evaluation and a materialized
  texture; both call the *same generated* `n_occlusion_eval`.

  | configuration | differing px (of 1,460,800) | RMSE | max Δ |
  |---|---|---|---|
  | `aoStrength=1`, macro | 0 | 0.000 | 0 |
  | `aoStrength=1`, grazing | 0 | 0.000 | 0 |
  | `aoStrength=0.5`, macro | 235,013 (16.09%) | 0.243 | **1** |
  | `aoStrength=0.5`, grazing | 59,211 (4.05%) | 0.121 | **1** |

  At strength 1 `mix(1, ao, 1)` is exactly `ao` and the 8-bit round trip is
  exact, so the two paths agree bit for bit. At 0.5 the materialized
  intermediate really is quantised, and the maximum disagreement anywhere is
  **exactly one 8-bit LSB** — which is the entire predicted error, because
  `mix` is affine and a box mip filter is affine, so materializing commutes with
  mip generation. Nothing else differs.

  The bug: the first version measured **99.93% of pixels differing, RMSE 40.9**.
  A materialize pass is encoded during `create()`, *before* the harness has ever
  called `update()`, so it read a zero-filled uniform and baked `ao_strength = 0`
  — no occlusion at all — and then cached that PNG under a key asserting it was
  right. Fixed in `src/material/runtime.ts` (`pushUniforms()` runs before any
  materialize pass); the reasoning is written into the comment there, because
  this class of bug produces a plausible texture rather than a crash.

- **The validator fails loudly at create(), with the fix in the message.**
  Deliberately broken graphs were run to check it, not assumed:
  a channel naming a missing node, a two-node cycle, and a stage `.wgsl` that
  `#include`s `src/wgsl/lighting.wgsl` (the realistic mistake — the generator
  already concatenates it, so it is a duplicate-symbol error). All three produce
  the runner's "Failed to start" panel with the issue code and an explanation.
  The report is also data on `globalThis.__materialReports` for headless probing.

- **`combine` shipped without a consumer here, and that is worth saying.** The
  frozen node set is image / procedural / filter / combine / variantBlend, and
  this material honestly needs four of them. `combine`'s ops (`lerp`,
  `multiply`, `heightBlend`) are implemented and typechecked but nothing in the
  default or the optional paths exercises them, so they are the least-tested
  code in the system. The first material that layers two surfaces should be
  treated as their first real test.

- **Four params exist purely so a stage could be verified at all.** `aoSoftenPx`
  gives the `filter` node kind a real user; `parallax` / `pomSteps` /
  `silhouette` / `parallaxDepth` give the view-uv stage one. Shipping
  `parallaxOffset` and `pom` codegen with no consumer would have meant shipping
  untested code, which is worse than a slightly wider param list. All four are
  off/neutral by default, so the default render is still exactly 001.

  Both view-uv stages work: `pom` gives the cushion real relief at grazing, and
  `silhouette` on the `plane` object eats the near edge into a ragged
  height-field outline instead of a straight triangle edge — a hard discard, no
  dithering. `pdo` (pixel depth offset) is implemented but stays off: WGSL has
  no conservative-depth qualifier, so merely *declaring* `@builtin(frag_depth)`
  costs early-z across the whole surface, and the generator emits the
  declaration only when it is asked for.

- **The DEV mip-convention check fires on the sun-exposed albedo, exactly as it
  does in 001**, and for the same reason: round-to-nearest in the 8-bit render
  target biases a skewed histogram upward by ~0.25% per level over an 11-level
  chain, against a fixed 2% tolerance. Quantisation, not a convention error.
  Left alone — silencing a correctness check to quiet one's own warning is how
  the next real one gets missed.

- **A shared node is sampled once per consumer.** `n_<id>_eval(c)` calls its
  dependencies inside itself (the signature the design fixes), so a node feeding
  two channels is evaluated twice. Nothing in this graph is shared, so it costs
  nothing here — but a material whose height feeds both the view-uv stage and a
  channel will pay for it. Passing inputs as arguments would fix it at the cost
  of the uniform one-argument signature; worth revisiting when something
  actually shares.
