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

## Lighting bug (bake v3 -> v4)

Reported symptom: the whole field uniformly bright and washed out, no
directional light gradient, pink panicles blowing toward white with distance,
oversaturated greens. "It doesn't seem to be using normals properly."

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

### Debug views, before -> after

Field region of `#/run/017-cached-clusters?cam=grazing` (luma histogram over
x 400-1270, y 420-700; buckets are 0.0-0.1 ... 0.9-1.0 in percent of pixels):

```
debug=lighting   mean   >0.97   histogram
017 before       0.918  72.3%    0  1  1  2  2  2  3  4  6 79
017 after        0.659  21.6%    0  5  9  9  9  9 10 10 10 29
001 (reference)  0.774  24.6%    0  1  2  3  5  8 12 16 15 37

debug=off        mean   >0.97
017 before       0.613   4.6%
017 after        0.412   1.0%
001 (reference)  0.333   0.0%

debug=albedo     mean   >0.97
017 before       0.613   3.2%    <- identical before and after, bit for bit
017 after        0.613   3.2%
```

Read the first block as: the light term used to be a **spike at the ceiling**
(79% of the field in the top bucket, 72% clipped) and is now **broad and nearly
flat across the range** — a gradient, and slightly less clipped than 001's.

The second block is the sharpest single piece of evidence: **before, the shaded
field's mean (0.613) equalled its albedo mean (0.613) exactly** — the light term
averaged 1.0, i.e. the lighting was doing nothing but adding noise. After, the
shaded mean is 0.412 against the same 0.613 albedo, a mean light term of ~0.67.
Blowout in the final image fell from 4.6% to 1.0% of field pixels.

The albedo view is byte-identical before and after, which proves the change is
confined to the light term and the bake did not disturb the colour atlas.

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

### Performance

No structural change: one extra varying (`shade`), one repurposed flat varying
(`yaw` -> `card_yaw`), one dot+negate per fragment, and an `atan2` on far-card
vertices only. Nothing was added to any pass, dispatch, or attachment.

Steady-state HUD readings after the fix, default stand, 1280x800, three samples
4s apart once the cache had converged:

| cam          | Σp50 (3 samples) | base | fill | refresh | cc-main | composite |
| ------------ | ---------------- | ---- | ---- | ------- | ------- | --------- |
| grazing      | 4.00 / 4.67 / 3.82 | 0.40 | 0.05 | 0.19 | 1.7-2.0 | 1.5-2.0 |
| topdown      | 1.55 / 4.83 / 3.52 | 0.4-1.6 | 0.03 | 0.07 | 0.6-1.6 | 0.5-1.6 |
| far-horizon  | 5.87 / 6.35 / 5.23 | 0.38 | 0.05 | 2.01 | 1.4-1.9 | 1.4-1.9 |

`cam=grazing` at **~3.8-4.7ms** is inside the band this experiment recorded
before the lighting work (4.2-4.6ms in Findings above) and well under the 9ms
ceiling. The spread within a single camera is contention, not signal: `base` is
the harness's own terrain+sky pass and it moves 0.4 -> 1.6ms across samples of
the *same* view, which is other agents' dev pages hitting the GPU. `inside-plant`
sampled 7.0 / 9.0 / 12.3 in the same run with `base` climbing 1.34 -> 2.87, so
those three are contaminated and not usable. Still no bench JSON, for exactly
this reason — these are HUD readings, not quotable bench numbers.

### Still not right

- **`debug=lighting` still clips over ~22% of the field.** That is the shared
  model itself (`sun_color` 1.15 x `ndl^2` + `ambient` x `hemi` exceeds 1 for
  up-facing normals) written to an 8-bit target, and the terrain and 001 (24.6%)
  do the same. Not fixed here on purpose: the brief was to apply the shared
  model exactly once, not to replace it.
- **~1% of field pixels still clip in the final image**, essentially all of it
  crown cards. The top capture's normals are flipped toward +Y by construction,
  so a horizontal crown card genuinely faces the sun; `topdown` is the brightest
  view for that reason. Physically defensible, but if the owner wants it pulled
  down, `baseShade` is the knob and the crown card's `1 - baseShade * 0.25`
  coefficient is the place.
- **The two-sided flip is frozen at refresh time** for cached clusters (it uses
  the slot's camera, not the live one). Parallax invalidation bounds the error —
  a slot refreshes long before the viewer crosses a card's plane — but a slow
  orbit at a fixed distance can in principle hold a stale flip. Not observed.
- **Far-field normals now converge toward the capture axis** rather than +Y
  (mean `n.z` 0.31 -> 0.61 from mip 0 to mip 4). That is the right aggregate for
  a card impostor and it keeps the light term flat at ~0.48 across the whole mip
  chain, but it does mean the deepest far field shades as a viewer-facing sheet;
  per-plant orientation variety lives in levels 0-2 (out to ~200m), where the
  crossed cards keep their yaw / yaw+90° facings.
