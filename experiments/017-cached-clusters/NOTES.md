# 017 — Amortized cluster cache (temporal amortization)

## Idea

Cache expensive reconstructions of plant CLUSTERS and reuse them across
frames; refresh only when the view has changed enough. Concretely:

- **Bake (once per species):** two orthographic captures of the source mesh
  (side + top), 256x256 each, albedo+coverage and plant-local octahedral
  normal, CPU-dilated and mipped. This is the only imagery derived from the
  source mesh; the mesh is never touched again.
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
(`<species>-cards-v3.bin`, committed for all three species). Mips are built
CPU-side at load (alpha-weighted box). Bake runs in ~1s per species when the
artifact is missing.

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
`debug_shade`), and `main.ts` bumps a debug generation counter when the selector
changes, which marks every cached slot stale. Slots then re-reconstruct under
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

- normals: real per-fragment normals everywhere (baked octahedral normal atlas
  decoded and yaw-rotated) — up close (`cam=inside-plant`) individual blades
  have distinct normals. At distance the field reads mostly +Y green: that is
  the *mip chain* of the normal atlas averaging opposing blade normals toward
  the octahedral origin, i.e. an aggregate foliage normal. Honest, not a
  constant-normal bug — but it means far-field shading is dominated by the
  hemisphere term.
- lighting: saturates to white over most of the field. The shared model gives
  `sun_color (1.15) * ndl^2 + ambient * hemi > 1` for up-facing normals, and
  the debug view is written to an 8-bit target. The terrain does the same
  thing; not specific to this experiment, but it makes the lighting view a
  coarse instrument here.
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
- **Far-field normal flattening** (mip-averaged octahedral normals) — a real
  quality observation, but fixing it means changing what is baked, which is a
  redesign rather than an audit.
- **`levelCap` sizing.** A level-4 job reserves ~174k instance slots of the
  245k scratch, which caps the frame at one big job. It is a correct bound, but
  a tighter per-cluster estimate (from the actual stand density inside the
  cluster's stand-clipped area) would let two big jobs share a frame.
