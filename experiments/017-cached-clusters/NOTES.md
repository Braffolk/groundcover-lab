# 017 — Amortized cluster cache (temporal amortization)

## Idea

Cache expensive reconstructions of plant CLUSTERS and reuse them across
frames; refresh only when the view has changed enough. Concretely:

- **Bake (once per species):** two orthographic captures of the source mesh
  (side + top), 256x256 each, albedo+coverage and a plant-local unit normal
  (`rgb = n * 0.5 + 0.5`, flipped into the hemisphere around that view's
  capture axis — see "Lighting bug" below), CPU-dilated and mipped. This is the
  only imagery derived from the source mesh; the mesh is never touched again.
- **Cluster quadtree (CPU, per frame, bounded window):** world-anchored
  clusters sized 8m near the camera doubling to 128m far away (split when
  nearest-distance < 4 x size, coverage to 1024m), so every cached cluster
  subtends ~1/4 rad. Between ~40 and ~200 leaves are visible.
- **Reconstruction (the amortized cost):** a refreshed cluster owns one 176x176
  LAYER of a shared cache array pair (lit color, packed 16-bit depth) and is
  rendered straight into it. A compute pass re-derives exactly the stand's
  plants for that cluster from the scatter WGSL twin (bounded by cluster area,
  never total plant count) and a render pass draws them as crossed cards (side
  texture x2 + view-angle-faded crown card with the top texture; levels >= 3
  collapse to one camera-facing card with a minimum-footprint clamp) through an
  off-axis frustum fitted from the current camera — canonical pose, no wind,
  lit, unfogged.
- **Composite (per frame, cheap):** each visible slot is ONE world quad. The
  fragment shader re-projects every cached texel: unpack slot depth →
  unproject through the slot's stored inverse view-proj → true world position
  → frag_depth from the live camera + fog. Cached clusters therefore
  depth-compose correctly against terrain, the near ring and each other from
  any angle without re-rendering their content. Wind is applied as an
  analytic per-cluster shear (pinned at the card base, phase from the cluster
  hash) so cached imagery never freezes.
- **Invalidation (the core of the method):** a slot is stale when
  |camera_now − camera_at_refresh| / cluster_distance > refreshTau (parallax
  error), or its quadtree level changed, or it was evicted (LRU on
  last-visible frame, pool of 224 slots). Stale slots keep drawing while a
  fixed per-frame refresh budget (default 3, adaptively raised to ~10 during
  warmup/fast motion, hard cap 12 + a cell-work cap) rebuilds the worst
  offenders, prioritized by parallax error x solid angle. Missing children
  fall back to their cached parent while waiting.
- **Near ring:** 8m clusters within `directRadius` (34m) render directly
  every frame — same crossed cards, live per-plant wind (species sway x
  height fraction x phase from scatter), camera-inside-plant alpha fade.
  Handoff is exact: a cluster is either in the direct bitmask or cached,
  decided per cluster, sticky until its slot is ready.

Per-frame cost = direct ring (bounded by directRadius) + one quad per visible
slot (≤ 224) + refresh budget. Nothing scales with stand plant count:
scaling-100m (134M plants) renders at the same GPU cost as default (557k).

## VRAM budget math

Shared across all species (the cache holds mixed-species clusters, so it
cannot be attributed per species):

- cache color array 176x176 x 224 layers rgba8 = 27.8MB
- cache depth array 176x176 x 224 layers rg8 = 13.9MB
- refresh scratch instances 245760 x 32B = 7.9MB
- near instances 65536 x 32B = 2.1MB
- one 176x176 scratch depth + rings + tables ≈ 0.3MB

Per species: card atlases 512x256 rgba8 x2 x1.33 (mips) ≈ 1.4MB.

Total observed in HUD: **62.8MB** for the 3-species default stand ≈
**21MB/species** — inside the 25MB/species budget. Caveat: the cache is sized
by view coverage, not species count, so a single-species stand
(calamagrostis-pure) still uses ~50MB of shared cache and formally blows the
per-species line. Halving SLOT to 128 would put even that in budget at some
sharpness cost; not done.

## Bake

`bake.ts` renders the GCMESH1 mesh on the GPU into the two capture tiles
(one render pass, two viewports), reads back, dilates covered texels 8 steps
(kills black bilinear fringes), packs a 1MB artifact
(`<species>-cards-v4.bin`, committed for all three species). Mips are built
CPU-side at load (alpha-weighted box). Bake runs in ~1s per species when the
artifact is missing.

Artifact v3 -> v4 changed only the normal target: raw octahedral pair in `rg`
became a capture-axis-flipped unit normal in `rgb`. Same 1MB layout, same
byte count. The v3 files are deleted; nothing else reads them.

## Status

working — verified by screenshot at grazing / topdown / inside-plant /
far-horizon (scaling-100m), plus a scripted fly-through (6s W + 2s D) to
exercise invalidation and refresh churn. All three species covered (the two
community tiles + poa specimen; a "plant" is one baked community clump, same
convention as the other impostor experiments). Re-verified after the audit
pass below, including every debug view and a frozen-time before/after diff.

## Findings

- GPU Σp50 ≈ 4.2-4.6ms at 1280x800 on this machine with ~15 other agents'
  dev pages contending the GPU — do NOT quote these as bench numbers; no
  bench JSON recorded for the same reason. Split: composite ≈ 1.9ms,
  fill ≈ 0.08ms, refresh ≈ 0.1-0.6ms (budget-bounded spikes).
- Plant-count independence confirmed: default (557k) vs scaling-100m (134M)
  have identical pass costs.
- Amortization works as intended: with a static camera, refresh cost decays
  to zero (only invalidated slots rebuild); flying at ~15 m/s keeps the field
  coherent — staleness shows as softened/stretched mid-distance cards rather
  than holes, and refreshTau trades that softness against refresh cost.
- The far field is deliberately undersampled (~2-3x vs screen at band-near
  edges): slot res x cluster angular size is the fundamental knob; it reads
  as painterly softness, fog helps.
- Crown (top-view) cards read terribly below ~25 deg elevation (dark spidery
  smears at eye level) — now faded out by view elevation at both direct
  render and refresh time; topdown coverage comes almost entirely from them.
- Known artifacts, honestly: (1) brief parent-fallback double-draw when a
  level splits (both parent card and fresh children for a few frames);
  (2) refresh pops when a stale slot rebuilds after fast motion (the pop IS
  the amortization made visible; budget/tau tune it); (3) snap 90-degree
  turns can show unfilled far patches for ~0.3s (adaptive budget refills);
  (4) beyond 1024m nothing is drawn — at fog f^2 ≈ 0.99 this is a slightly
  barer horizon band on the scaling stand; (5) cached wind is
  cluster-coherent, so the near-ring boundary shows per-plant vs coherent
  sway mismatch in motion.
- A/B: `#/ab/000-ground-truth/017-cached-clusters?cam=grazing` — GT is the
  3-tile reference clump; colors/silhouettes match plausibly (pink-headed
  calamagrostis dominant).

## Audit (structural + debug views)

### Debug views

Both fragment shaders now include `src/wgsl/debug.wgsl` and route their final
colour through `debug_shade(shaded, albedo, normal_ws, coverage, world_pos)`;
fog is applied only when `debug_mode() == DEBUG_OFF`.

The wrinkle unique to this method: the cache stores **already shaded** imagery,
so a debug view cannot be reconstructed at composite time (there is no albedo
and no normal in the slot, only lit RGB + coverage + 16-bit depth). Rather than
spend 13.9MB on a normal channel that only a debug view would read, the debug
view is **baked into the slot at refresh time** (`fs_refresh` runs
`debug_shade`), and `main.ts` bumps a shading generation counter (`shadeEpoch`,
which also covers `baseShade`) when the selector changes, which marks every
cached slot stale. Slots then re-reconstruct under
the normal refresh budget while their stale image keeps drawing — the whole
field converges in ~1s (verified: switching the toolbar picker live at
`cam=grazing` shows a fully converged normal view 1.2s later). The composite
answers COVERAGE and DEPTH live instead (it has the cache alpha and the
re-projected world position), so those two views are exact for the current
frame rather than frozen at refresh time. The selector is read off the URL
(`debug=`) because `FrameInfo` does not expose it to TS; a harness path that
changed `debugMode` without touching the URL would leave cached slots showing
the previous view until something else invalidates them.

What the views exposed:

- normals: **this reading was wrong, and the error cost a release.** The audit saw
  "the field reads mostly +Y green at distance", explained it as an honest
  mip-averaged aggregate normal, and moved on. It was the bug. See
  "Lighting bug" below.
- lighting: saturates to white over most of the field. Partly the shared model
  (`sun_color (1.15) * ndl^2 + ambient * hemi > 1` for up-facing normals, into
  an 8-bit target — the terrain does the same), but the audit used that as a
  blanket excuse. It was mostly the +Y normals feeding the model its maximum
  argument everywhere. Corrected below.
- albedo: unlit, unfogged, clearly darker/more saturated than the shaded view —
  confirms lighting is applied exactly once and that the bake does not
  pre-multiply light into the card albedo.
- depth: the most useful one — the re-projected cached depth is continuous
  across the direct-ring/cache handoff and across cluster boundaries, which is
  the load-bearing claim of the method.
- New param `clusterTint` (`off` / `level` / `slot`): tints cached clusters by
  quadtree level or by cache slot. The direct ring stays untinted, so the
  handoff radius, the LOD rings and slot churn are all visible at once.

### Structural fixes

1. **Cache atlas -> cache array; refresh renders directly into its slot.** The
   cache was one 2816x2464 atlas, and every refresh rendered into a 176x176
   scratch target and then `copyTextureToTexture`'d into the atlas rect — 2
   copies per refresh, up to 24 per frame, plus a scratch round trip. A
   176x176x224 array texture has exactly the same byte count, and an
   array-layer view is a legal 176x176 render attachment, so each refresh now
   renders straight into its own layer. Copies and both scratch colour targets
   are gone; composite indexes the layer instead of doing atlas-rect maths
   (with the same half-texel inset, so sampling is unchanged).
2. **Far refresh jobs draw 6 vertices per plant instead of 18.** Levels >= 3
   collapse to a single camera-facing card, but the indirect args always said
   18 vertices and the vertex shader clipped away the two unused quads. Those
   are the *heaviest* jobs (a level-4 cluster can be ~174k instances), so this
   removes ~2/3 of their vertex work. Zero image change: the extra quads were
   emitted at z>w and clipped.
3. **Near fill dispatch is sized to the direct clusters, not the bitmask
   window.** It dispatched 32x32 workgroups (the full 128m mask window) every
   frame regardless of `directRadius` (34m by default); most workgroups existed
   only to read a bit and return. It now dispatches the bounding box of the
   clusters actually in the direct set (the mask origin stays camera-anchored,
   so bit indexing is untouched), and skips the dispatch entirely when the set
   is empty.
4. **Refresh depth is `storeOp: 'discard'`.** It is scratch and never read
   after the pass.
5. **Quadtree node identity is a number, not a template string.** The
   enumeration walk built ~1-2k strings per frame for map keys (leaves, parent
   lookups, terrain y-range memo). Same equivalence classes, no per-frame
   garbage.

Verified image-identical: with time frozen (`det=1&t=3`), before/after canvas
screenshots at `grazing`, `topdown` and `far-horizon` on `scaling-100m` differ
only in the HUD digits and the new params row — the rendered field is identical
(0.86% of pixels >8, all of it HUD/UI; the run-to-run noise floor for identical
code is 0.09%).

### Deliberately not done

- **Batching the up-to-12 per-refresh render passes into one.** Attachments in
  a pass must share dimensions, so this needs a `12*176 x 176` strip target
  (+3.4MB) plus per-job viewports, and it trades a bigger clear for fewer pass
  setups — a measurement call, not an obvious win. Worth revisiting for one
  concrete reason: at the adaptive budget (10-12) this experiment can encode 14
  timed passes per view, and `PassTimer` only has 32 per frame, so an A/B
  against another heavy experiment can silently drop timings.
- **The composite writes `frag_depth`, which disables early-z for the pass.**
  That is inherent to re-projecting cached depth (WGSL has no conservative
  depth declaration), and the pass is the method's main cost. Not fixable
  without changing the technique.
- ~~**Far-field normal flattening** (mip-averaged octahedral normals) — a real
  quality observation, but fixing it means changing what is baked, which is a
  redesign rather than an audit.~~ **Wrong call.** It was not a quality nicety,
  it was the lighting bug, and the fix was a two-line bake change plus a version
  bump — not a redesign. Done; see "Lighting bug".
- **`levelCap` sizing.** A level-4 job reserves ~174k instance slots of the
  245k scratch, which caps the frame at one big job. It is a correct bound, but
  a tighter per-cluster estimate (from the actual stand density inside the
  cluster's stand-clipped area) would let two big jobs share a frame.

## Lighting bug — TWO independent mip-driven causes

Reported symptom: the whole field uniformly bright and washed out, no
directional light gradient, pink panicles blowing toward white with distance,
oversaturated greens. "It doesn't seem to be using normals properly."

There turned out to be **two** separate bugs, both keyed to mip level (therefore
both monotonic in distance), in two different terms:

1. **The normal atlas** stored raw octahedral codes whose mip average collapses
   to `(0,1,0)`, feeding the *light term* its maximum argument. Fixed in bake
   v3 -> v4, below.
2. **The albedo tap divided by coverage a second time**, inflating the *stored
   colour* by up to 2.42x at the deepest mip. Fixed below in "Second cause".

Fixing (1) alone removed the flatness and most of the clipping but left the
field still too light and *still brightening with distance* — the owner caught
that on review, and the distance-band measurement (below) localised the residual
to the albedo rather than the light term. Lesson worth keeping: "too bright" had
two multiplicative contributors, and the debug views separate them cleanly only
if you measure `albedo` and `lighting` **independently and per distance band**.
The whole-field mean hid it, because 017's light term happens to fall with
distance and partly cancelled the rising albedo.

## First cause: normals (bake v3 -> v4)

### Root cause

**The normal atlas stored raw octahedral codes, and its mip chain averaged
them.** A blade's front and back faces carry opposite mesh normals, which encode
to roughly opposite octahedral pairs, so an alpha-weighted box average of the two
lands near `(0,0)` — and `oct_decode(0,0)` is *exactly* `(0,1,0)`. Every cached
cluster samples this atlas at mip 1.5-3.5 (that is the entire field beyond
`directRadius` = 34m), so almost every cached fragment lit with a straight-up
normal. `light_surface()` then received its maximum argument everywhere at once:
`ndl^2 = 0.86` **and** `hemi = 1.0`, i.e. a multiplier of ~`(1.20, 1.12, 1.02)`
uniformly. That is simultaneously the flat look (no variation to make a
gradient), the blowout (>1 clips in the rgba8unorm cache), and the
worse-with-distance behaviour (higher mip = flatter).

Measured on the committed v3 artifacts (side tile, covered texels, green channel
of the light multiplier):

| mip | mean `n.y` | within 5° of +Y | light: min / mean / max |
| --- | ---------- | --------------- | ----------------------- |
| 0   | 0.28       | 1.8%            | 0.15 / 0.68 / 1.26      |
| 2   | 0.41       | 4.4%            | 0.15 / 0.71 / 1.26      |
| 4   | 0.70       | 16.9%           | 0.17 / 0.85 / 1.25      |

(`elymus-repens` is worse: mean `n.y` 0.59 -> 0.96, mean light 0.84 -> **1.10**,
min light 0.16 -> 0.55 — the spread collapses *and* the mean crosses 1.)

Three smaller bugs stacked on top, all in the same direction:

2. **No two-sided flip.** `000-ground-truth` explicitly flips the normal toward
   the camera ("thin foliage is lit from both sides"); 017 did not. The ~35% of
   fragments whose stored normal faced away dropped to the floor term (~0.15),
   which is why `debug=lighting` was bimodal white-with-black-speckle rather
   than a gradient.
3. **The crossed card's second quad and the far single card used the wrong
   normal frame.** `o.yaw = i0.w` for every quad, but quad 1 is rotated 90°
   about Y and mode-2's card faces the refresh camera at an arbitrary azimuth.
   So both cards of every cross shaded *identically*, and the far card's normals
   were unrelated to the geometry it stands in for — exactly the "plants don't
   shade differently depending on which way they face" complaint.
4. **No self-occlusion term at all.** The shared light model has no shadowing,
   so the root of every card was lit exactly like its tip. Nothing read as being
   in shadow.

Explicitly **not** the cause, checked first and ruled out by reading the code:
there is no double lighting (`light_surface()` is called once, in
`cards.wgsl`; `composite.wgsl` only re-projects and fogs), the cached colour /
alpha convention is self-consistent (refresh writes `a = 1` against a
zero-cleared target, so composite's `rgb / a` is exact), and there is no colour
space mismatch (`getPreferredCanvasFormat()` is `bgra8unorm`, non-sRGB, matching
the cache's `rgba8unorm`).

### The fix

- **`shaders/bake.wgsl`** — flip each mesh normal into the hemisphere around the
  view's capture axis (`view_u.fwd`), then store it as a plain unit vector
  (`rgb = n * 0.5 + 0.5`) instead of an octahedral pair. Once every normal in a
  tile lies in one hemisphere, the box average is a genuine aggregate normal and
  cannot cancel; plain vectors also average linearly, which octahedral codes do
  not do across the fold. Bake version 3 -> 4, same byte count.
- **`shaders/cards.wgsl`** — `decode_normal()` (plain vector, normalised;
  short averages are kept, since a short aggregate is information, and the only
  fallback is the capture axis, never +Y). `card_vertex()` now emits
  `card_yaw` — instance yaw for quad 0, `+90°` for quad 1, the crown card's
  instance yaw, and `atan2(to_cam.x, to_cam.z)` for the far camera-facing card —
  and `shading_normal()` rotates by that and flips toward the viewer.
- **Grounding term** `shade` = `mix(1 - baseShade, 1, height_fraction)` for side
  cards, `1 - baseShade * 0.25` for the crown card, multiplied into the **light**
  term (`light_surface(albedo * shade, ...)`) so `debug=albedo` stays the baked
  atlas colour and `debug=lighting` shows sun x grounding. New param `baseShade`
  (default 0.4).
- Because the cache holds shaded imagery, `baseShade` has to invalidate slots
  the same way the debug selector does — the old `debugEpoch` is now
  `shadeEpoch` and bumps on either.
- Drive-by: the committed-artifact fetch now goes through `assetUrl()`.

## Second cause: the albedo was divided by coverage twice

This is the one that made the field brighten with distance, and it survived the
v4 normal fix untouched (`debug=albedo` was byte-identical before and after,
which should have been the tell).

`buildMips()` in `bake.ts` is an **alpha-weighted** box filter:

```
rgb = sum(rgb_i * a_i) / sum(a_i)        a = sum(a_i) / 4
```

so the stored `rgb` is the coverage-weighted mean colour of the *covered* texels
— already un-premultiplied — at every mip level, and `a` is mean coverage. The
fragment shaders then did:

```wgsl
let albedo = tex.rgb / max(tex.a, 1e-3);   // WRONG: divides by coverage AGAIN
```

which multiplies the albedo by `1/coverage`. Coverage falls monotonically as the
mip chain deepens, and mip level rises monotonically with distance, so the
albedo inflated monotonically with distance. Measured on the v4 atlas
(`calamagrostis-canescens`, side tile, texels that pass that lod's alpha test):

| lod | alpha thr | mean coverage | mean luma `rgb` | mean luma `rgb/a` | inflation | clipped |
| --- | --------- | ------------- | --------------- | ----------------- | --------- | ------- |
| 0   | 0.50      | 1.000         | 0.399           | 0.399             | 1.00x     | 0.0%    |
| 1   | 0.44      | 0.682         | 0.399           | 0.621             | 1.56x     | 19.8%   |
| 2   | 0.39      | 0.674         | 0.404           | 0.626             | 1.55x     | 18.1%   |
| 3   | 0.33      | 0.578         | 0.399           | 0.708             | 1.78x     | 39.5%   |
| 4   | 0.28      | 0.439         | 0.395           | 0.827             | 2.09x     | 63.4%   |
| 5   | 0.22      | 0.323         | 0.377           | 0.911             | **2.42x** | **86.7%** |

The `mean luma rgb` column is flat by construction (0.399 -> 0.377) — that is the
proof that no division belongs there. The fix is one expression, now
`card_albedo_of()` in `cards.wgsl`, with the derivation written next to it:

```wgsl
fn card_albedo_of(tap: vec4f) -> vec3f { return tap.rgb; }
```

The cache texture in `composite.wgsl` keeps *its* divide, and that is not an
inconsistency: `fs_refresh` writes `a = 1` into a zero-cleared target, so a
bilinear tap at a slot silhouette genuinely is premultiplied against black.
Two textures, two different conventions — that is what made the bug easy to
write and easy to miss.

### Distance-band measurement (the method that found it)

Whole-field means hid this bug. What exposes it is mean luma of plant pixels per
**distance band**, in `debug=albedo` and `debug=lighting` separately. At
`cam=grazing` the camera is fixed (eye 1.35m, pitch -0.06 rad, 60° vertical fov),
so screen row maps monotonically to ground distance
(`d = 1.35 / tan(-(pitch + atan(ndc_y * tan(30°))))`); the sample window is
x 360-960, y 340-720 to miss every HUD panel, and sky is excluded by
`b - r > 0.06`. The row->distance map is a lower bound where a ray hits plant
tops rather than ground, but it is identical for every renderer, so band-to-band
comparison is exact.

```
debug=albedo               0-5m    5-20m   20-60m    60m+   far/near
017 v3 (original)         0.554    0.679    0.760   0.795   1.44x
017 v4 (normals fixed)    0.554    0.679    0.760   0.795   1.44x   <- untouched
017 v5 (albedo fixed)     0.416    0.414    0.418   0.437   1.05x
001-billboard-smoke       0.411    0.405    0.397   0.418   1.02x
000-ground-truth          0.196    0.201    0.198   0.198   1.01x

debug=lighting            0-5m    5-20m   20-60m    60m+   far/near
017 v3 (original)         0.899    0.962    0.969   0.970   1.08x
017 v4 / v5               0.703    0.632    0.628   0.609   0.87x
001-billboard-smoke       0.742    0.856    0.902   0.913   1.23x
000-ground-truth          0.992    0.985    1.000   1.000   1.01x

debug=off                 0-5m    5-20m   20-60m    60m+   far/near
017 v3 (original)         0.532    0.708    0.786   0.819   1.54x
017 v4 (normals fixed)    0.399    0.438    0.496   0.516   1.29x
017 v5 (albedo fixed)     0.302    0.269    0.271   0.281   0.93x
001-billboard-smoke       0.315    0.363    0.381   0.412   1.31x
000-ground-truth          0.214    0.238    0.246   0.257   1.20x
```

Reading it: **both reference renderers have a flat albedo (1.02x, 1.01x) and 017
had 1.44x** — that single row is the bug, and it says the drift is in the stored
colour, not the light term. After the fix 017 is 1.05x and its absolute level
(0.416) lands on 001's (0.411). The light-term row separately confirms the v4
normal fix did its job on a *different* axis: 1.08x rising became 0.87x falling.

Note `000-ground-truth` is stand-independent (a 3-tile clump), so its 20-60m and
60m+ bands are mostly bare terrain rather than plants — treat it as a reference
for the near bands and for flatness, not for absolute far-field values.

### Debug views, before -> after

Field region of `#/run/017-cached-clusters?cam=grazing` (luma histogram over
x 400-1270, y 420-700; buckets are 0.0-0.1 ... 0.9-1.0 in percent of pixels):

```
debug=lighting   mean   >0.97   histogram
017 v3           0.918  72.3%    0  1  1  2  2  2  3  4  6 79
017 v4 / v5      0.659  21.6%    0  5  9  9  9  9 10 10 10 29
001 (reference)  0.774  24.6%    0  1  2  3  5  8 12 16 15 37

debug=off        mean   >0.97   histogram
017 v3           0.613   4.6%    1  3  8  8 13 15 16 14 11 12
017 v4           0.412   1.0%    4 15 18 16 16 12  8  6  3  3
017 v5           0.287   0.0%   10 21 25 21 15  4  2  1  0  0
001 (reference)  0.333   0.0%    3 11 30 25 19 10  1  0  0  0

debug=albedo     mean   >0.97   histogram
017 v3           0.613   3.2%    0  2  2  8 17 21 19 14 10  8
017 v4           0.613   3.2%    0  2  2  8 17 21 19 14 10  8
017 v5           0.422   0.0%    0  2  9 36 29 16  4  2  0  0
001 (reference)  0.415   0.0%    0  3  8 33 35 20  1  0  0  0
```

Read the first block as: the light term used to be a **spike at the ceiling**
(79% of the field in the top bucket, 72% clipped) and is now **broad and nearly
flat across the range** — a gradient, and slightly less clipped than 001's. It is
identical in v4 and v5 because `debug=lighting` is `shaded / albedo`, which is
invariant to an albedo scale — a useful internal check that the second fix was
purely an albedo scale and touched nothing else.

The albedo block is the second fix landing: 017's distribution went from a long
bright tail with 3.2% clipping (`... 19 14 10 8`) to `0 2 9 36 29 16 4 2 0 0`,
which is nearly 001's shape (`0 3 8 33 35 20 1 0 0 0`) at nearly the same mean
(0.422 vs 0.415). **Clipping in the final image is now 0.0%**, down from 4.6% in
v3 and 1.0% in v4. Overall 017 now sits slightly *darker* than 001 (0.287 vs
0.333) and above ground truth — the ordering GT < 017 < 001 is where it belongs.

One more note on v3: the shaded field's mean (0.613) equalled its albedo mean
(0.613) exactly — the light term averaged 1.0, i.e. the lighting was doing
nothing but adding noise. That coincidence is what the first fix addressed.

Qualitatively, same-mode side by side
(`#/ab/001-billboard-smoke/017-cached-clusters?cam=grazing&debug=normals`):
017's normals view went from a **single green hue** (i.e. `(0,1,0)` everywhere)
to a full hemisphere spread of pinks/blues/cyans/greens, matching 001's
character. `cam=inside-plant` in `debug=off` now shows a strong root-to-tip
gradient and individual blades shading differently; before it was uniform acid
green top to bottom. `topdown` lost the white/yellow washed-out crowns.

The two contributions isolate cleanly. At **`p.baseShade=0`** (grounding term
fully off) `cam=inside-plant` still shows strong per-blade directional shading —
sunlit blades against shadowed ones, colour in range — so the normals are doing
the directional work on their own; `baseShade` only adds the vertical root-to-tip
depth on top. That is the check worth repeating if this ever regresses: if
`baseShade=0` looks flat again, the normal path is broken, not the shading knob.

### Candidates checked and ruled out

Recorded because each of these was a plausible cause of a distance-dependent
brightening, and knowing they are *not* it saves the next person the work.

- **Does the mip-averaged normal drift in a direction that raises the light
  term, and should it be renormalised after filtering?** It drifts (mean `n.z`
  0.31 -> 0.61 from mip 0 to mip 4, i.e. toward the capture axis) but the drift
  *lowers* the light term, and the measurement proves it: `debug=lighting` per
  band is 0.87x far/near, falling. Renormalising is correct — an un-normalised
  short average would shrink `dot(n, sun_dir)` toward 0 and push half-lambert
  toward its 0.5 floor and `hemi` toward 0.8, which darkens further rather than
  brightening. So this axis is benign and physically reasonable: at a grazing
  view the far field really does show more blade *sides* and fewer tops.
- **Grounding / AO term stronger near than far.** `shade` is
  `mix(1 - baseShade, 1, height_fraction)` — purely geometric, evaluated per
  vertex from the card's own height, with no mip or distance input. Averaging a
  linear gradient over a shrinking card preserves its mean, so there is no bias.
  Confirmed by `p.baseShade=0`, where the distance profile is unchanged.
- **Inter-plant shadowing that exists near and averages away far.** There is no
  shadowing term in this renderer at all (no shadow map, and the shared light
  model has none), so there was nothing to average away.
- **Applied per cluster before the merge.** The composite does not light; it
  re-projects and fogs. `light_surface()` is called exactly once, in
  `cards.wgsl`, for both the near ring and the refresh path.

### The alpha-test threshold ramp is load-bearing — do not flatten it

The other half of the coverage candidate: at high mip a texel is part blade, part
gap, and the hard alpha test writes it fully **opaque**, so gaps get painted as
grass. The threshold ramp `thr = mix(0.5, 0.22, lod/5)` is what decides how much.
Measuring drawn area against true coverage (`texels passing / sum of coverage`,
1.00 = unbiased) on the `calamagrostis-canescens` side tile:

| lod | ramp thr | drawn/true (ramp) | drawn/true (constant 0.5) |
| --- | -------- | ----------------- | ------------------------- |
| 0   | 0.50     | 1.00              | 1.00                      |
| 1   | 0.44     | 1.29              | 1.29                      |
| 2   | 0.39     | 0.89              | 0.75                      |
| 3   | 0.33     | 0.96              | 0.57                      |
| 4   | 0.28     | 1.27              | 0.37                      |
| 5   | 0.22     | 1.86              | **0.12**                  |

The ramp keeps coverage roughly honest (0.89-1.29) through lod 3 and over-covers
at lod 4-5. A constant 0.5 threshold — the "unbiased" choice, and my first
instinct — would lose **88% of far-field coverage** and dissolve the horizon into
holes. So the ramp is doing necessary work against mip coverage dilution and is
staying. Its residual over-coverage at lod 4-5 is a *density* bias, not a
brightness one: with the albedo inflation gone, per-band luma is flat (0.93x)
while the ramp was never touched. Left as-is rather than traded for holes.

### Performance

No structural change across either fix: one extra varying (`shade`), one
repurposed flat varying (`yaw` -> `card_yaw`), one dot+negate per fragment, an
`atan2` on far-card vertices only, and — from the second fix — one **fewer**
divide per fragment. Nothing was added to any pass, dispatch, or attachment, and
the albedo fix is strictly less ALU than what it replaced.

Steady-state HUD readings, default stand, 1280x800, three samples 4s apart once
the cache had converged. Two runs, because contention changed materially between
them — read `base` (the harness's own terrain+sky pass, which is not mine and
should be ~0.4ms) as the contention gauge:

| cam          | Σp50, quiet GPU    | Σp50, busy GPU     | base (quiet -> busy) |
| ------------ | ------------------ | ------------------ | -------------------- |
| grazing      | 4.00 / 4.67 / 3.82 | 7.10 / 7.03 / 6.58 | 0.40 -> 1.0-1.3      |
| topdown      | 1.55 / 4.83 / 3.52 | 4.26 / 3.94 / 3.98 | 0.4-1.6 -> 0.7-0.9   |
| inside-plant | contaminated       | 6.29 / 6.71 / 6.57 | 1.3-2.9 -> 0.8-0.9   |
| far-horizon  | 5.87 / 6.35 / 5.23 | 5.97 / 6.75 / 6.53 | 0.38 -> 0.9-1.0      |

`cam=grazing` at **3.8-4.7ms on a quiet GPU** is inside the band this experiment
recorded before any of the lighting work (4.2-4.6ms in Findings above), and the
6.6-7.1ms readings come with `base` tripled, i.e. they measure the machine, not
the change. Everything stays under the 9ms ceiling even under load. Still no
bench JSON, for exactly this reason — these are HUD readings, not quotable bench
numbers, and ~15 other agents share this GPU.

### Still not right

- **`debug=lighting` still clips over ~22% of the field.** That is the shared
  model itself (`sun_color` 1.15 x `ndl^2` + `ambient` x `hemi` exceeds 1 for
  up-facing normals) written to an 8-bit target, and the terrain and 001 (24.6%)
  do the same. Not fixed here on purpose: the brief was to apply the shared
  model exactly once, not to replace it.
- **Final-image clipping is now 0.0%** of field pixels (was 4.6% in v3, 1.0%
  after the normal fix alone). Nothing left to chase here; `baseShade` and the
  crown card's `1 - baseShade * 0.25` coefficient remain the knobs if the owner
  wants the overall level moved.
- **017's light term falls with distance (0.87x) where ground truth's is flat
  (1.01x) and 001's rises (1.23x).** After the albedo fix this makes the final
  image 0.93x far/near — flat, but *not* showing the mild fog-driven brightening
  toward the horizon that GT does (1.20x). The cause is understood and physical
  (far-field aggregate normals turn viewer-facing, so `ndl` drops for this
  camera/sun pair) rather than a defect, and it is the one place where a
  distance-dependent term still differs from the references. Deliberately NOT
  compensated with a distance knob — that would mask it. If it ever needs
  addressing, the honest lever is what the far card's aggregate normal should be,
  not a brightness ramp.
- **017 now reads slightly darker than 001** (field mean 0.287 vs 0.333) and
  brighter than ground truth (0.214-0.257). That ordering is defensible, but it
  is a *level* judgement rather than a measurement, so it is the owner's call.
- **The two-sided flip is frozen at refresh time** for cached clusters (it uses
  the slot's camera, not the live one). Parallax invalidation bounds the error —
  a slot refreshes long before the viewer crosses a card's plane — but a slow
  orbit at a fixed distance can in principle hold a stale flip. Not observed.
- **Far-field normals converge toward the capture axis** rather than +Y (mean
  `n.z` 0.31 -> 0.61 from mip 0 to mip 4). That is the right aggregate for a card
  impostor and it keeps the light term flat at ~0.48 across the whole mip chain,
  but it does mean the deepest far field shades as a viewer-facing sheet;
  per-plant orientation variety lives in levels 0-2 (out to ~200m), where the
  crossed cards keep their yaw / yaw+90° facings.
- **The far-field alpha test over-covers at lod 4-5** (drawn/true 1.27x, 1.86x).
  A density bias, not a brightness one — see the threshold-ramp section for why
  flattening it would be much worse.
- The 001 comparison numbers here were captured at one point in time; another
  agent was editing `001-billboard-smoke/shaders/cards.wgsl` during this session,
  so re-measure 001 rather than trusting these figures if it has moved since.
