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
- **Reconstruction (the amortized cost):** a refreshed cluster gets a 176x176
  slot in a shared atlas pair (lit color, packed 16-bit depth). A compute
  pass re-derives exactly the stand's plants for that cluster from the
  scatter WGSL twin (bounded by cluster area, never total plant count) and a
  render pass draws them as crossed cards (side texture x2 + view-angle-faded
  crown card with the top texture; levels >= 3 collapse to one camera-facing
  card with a minimum-footprint clamp) through an off-axis frustum fitted
  from the current camera — canonical pose, no wind, lit, unfogged.
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

- cache color atlas 2816x2464 rgba8 = 27.8MB
- cache depth atlas 2816x2464 rg8 = 13.9MB
- refresh scratch instances 245760 x 32B = 7.9MB
- near instances 65536 x 32B = 2.1MB
- tmp slot targets + rings + tables ≈ 0.5MB

Per species: card atlases 512x256 rgba8 x2 x1.33 (mips) ≈ 1.4MB.

Total observed in HUD: **63.0MB** for the 3-species default stand ≈
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
convention as the other impostor experiments).

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
