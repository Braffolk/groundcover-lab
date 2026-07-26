# Orchestration handbook

Working notes for whoever is driving this repo (including a future me after a
context compaction). Facts and hard-won gotchas only — the rules that bind
*experiment* agents live in `CLAUDE.md`, and the current moss brief lives in
`docs/moss-round-prompt.md`.

## Environment

- **Dev server: `http://localhost:5175`.** Ports 5173/5174 belong to the owner's
  other Vite app. It is started with `npm run dev` in the background and is
  expected to stay up; agents are told never to restart it. Editing
  `vite.config.ts` auto-restarts it, which briefly breaks every running agent's
  screenshots — only do that when nothing is running.
- Owner runs Opus 5 at xhigh and wants agents on the same. Omit `model` in
  `agent()` (inherits the session model) and pass `effort: 'xhigh'`.
- Headless screenshots (the only accepted proof anything works):
  ```
  chromium.launch({ headless: true, channel: 'chrome',
    args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU'] })
  ```
  Scripts must live OUTSIDE the repo (`/tmp/...`) and run with
  `NODE_PATH=<repo>/node_modules npx tsx /tmp/x.ts`. If ESM resolution fails,
  run the script from inside the repo instead (e.g. `.claude/tmp-x.ts`, which is
  gitignored) — `NODE_PATH` does not always apply to ESM.

## Git policy (learned the hard way)

- **Never `git add -A`.** Agents constantly drop large regenerable artifacts and
  the owner's `rating.json` / bench results into the tree. Always stage explicit
  paths. A careless `git add -A` re-added 3×181 MB blobs minutes after I removed
  them and the push was rejected.
- **GitHub hard-rejects any file >100 MB.** Check before pushing:
  `git rev-list origin/main..main --objects | git cat-file --batch-check='%(objectsize) %(rest)' | awk '$1>100000000'`
- Push with `git push --no-thin origin main` (large packs otherwise fail with a
  remote 500), and `git config http.postBuffer 524288000` is already set.
- Blobs only in the tip commit can be removed with `git rm --cached` +
  `git commit --amend` — no history rewrite needed.
- `rating.json` is the OWNER'S verdict. Never create, edit or revert it, and
  never assume a modification to it is an agent's.
- **Baked artifacts are gitignored (settled 2026-07-26).** `mesh/baked/**/*.bin`
  and `*.png` are ignored, matching the raw meshes. 419 MB across 33 files had
  been committed by accident and was untracked; it remains in history, so `.git`
  is ~360 MB until someone decides a rewrite is worth it. Note for anyone
  quoting sizes: `du` on `mesh/baked` measures DISK, which includes ~1 GB of
  untracked artifacts — do not report it as "committed" (I did, and was wrong).
- `goldens/` is gitignored: a full capture is ~400 MB and regenerable with
  `npm run capture:goldens`. Keep a before/after pair OUTSIDE the repo when
  refactoring shared code.

## Project shape

- `experiments/` is a tree, one branch per kind:
  `experiments/renderers/<nnn>-<slug>/` and
  `experiments/materials/<class>/<subject>/<nnn>-<slug>/`. Auto-discovered by
  `import.meta.glob('/experiments/**/manifest.ts')`; an id is the LAST path
  segment, and ratings / bench results / goldens are all keyed by that id, so
  moving a branch never orphans them. Scaffold with `npm run new -- <slug>`
  (`--kind material --group <class>/<subject>` for a material); the number comes
  from a scan of the whole tree. **Pre-scaffold directories yourself before
  launching a fleet** — parallel agents racing `npm run new` collide on
  numbering.
- `000-ground-truth` is the brute-force reference and is stand-independent, so
  it is excluded from every round and is not a usable A/B partner.
- **`001-billboard-smoke` is the champion**: best looking AND fastest (~6 ms
  Σp50 on `default`). It is the baseline every round is measured against, and it
  has already had a moss pass, so it doubles as the worked example for carpets.
  Any change that regresses the `default` stand is unacceptable.
- Stands (harness-owned placement setups, `src/scene/stands.ts`): `default`
  (~557k, the standard), `calamagrostis-pure`, `close-quality`, `dense-mixed`,
  `scaling-100m` (~134 M, the plant-count-independence test). `bog` was deleted
  with the moss meshes at the pivot. Renderers are pure RENDERERS of a stand;
  materials have no stand at all.
- Species: `calamagrostis-canescens` and `elymus-repens` are periodic community
  TILES (0.52 / 0.62 m) and `poa-pratensis` is a single specimen. The three
  Sphagnum states were removed at the pivot and their indices (3-5) are retired,
  never to be reused — the index goes into the GPU stand table.
- Raw meshes are gitignored, so a fresh clone has no `.bin` files at all.

## Harness facts worth not rediscovering

- `StandEntry` (GPU, `src/wgsl/frame.wgsl`) carries `density, scale_min,
  scale_max, sway, height_scale, species_index, wet_center, wet_width,
  carpet_div, footprint_m, slope_align` + 1 pad, 12 floats. Two pads were spent
  on `footprint_m` and `slope_align`; adding more fields grows the stride (safe,
  but recompiles every shader).
- **Always size enumeration from `standEntrySlots(entry)`, never a hardcoded
  `SCATTER_MAX_PER_CELL`.** This is the single most-repeated bug in the repo's
  history: hardcoding 128 silently vanished ~74% of a carpet in nearly every
  renderer. Carpets are deprecated, but the habit is what survives.
- **`SCATTER_MAX_PER_CELL = 128` is load-bearing for density calibration**
  (`128 / (8 × 16 m²) = 1`). Changing it moves every existing stand's placement.
- The wetness field and the whole scatter have **bit-identical TS and WGSL
  twins**; every float step in `scatter.ts` is `fround`ed to match. Any change
  must be mirrored in both, and prefer arithmetic that avoids `sqrt`.
- `carpet-close` is a camera bookmark 1 m straight down. **URL `cam=x,y,z,...`
  poses are ABSOLUTE and terrain at the origin is ≈ −7.7 m**, so a hand-written
  "y=2" is ~10 m up. Bookmarks are terrain-relative; always prefer them.
- Global debug views (`debug_shade()`, URL `debug=albedo|normals|lighting|
  coverage|depth`) are mandatory for every renderer.
- Static deploy: `npx vite build` under base `/groundcover-lab/`, now **~1.5 GB**
  because `mesh/baked/` grew to ~1.4 GB — see the open decisions. WebGPU needs
  HTTPS, so an S3 *website* endpoint will not work — REST endpoint over TLS or
  CloudFront. The old `installAssetBaseShim()` fetch monkeypatch is GONE; every
  asset URL now goes through `assetUrl()`.

## Harness bugs the moss fleet found

All three were reported independently by 3-5 agents with consistent
measurements, and all three are verified in shared code.

**Bugs 1 and 2 are CLOSED as won't-fix (2026-07-25).** Both exist only in the
carpet path, and carpets were deprecated when the lab split materials out from
plant renderers — a tiled ground mat is a material, not a species of plant. No
stand sets `carpetDiv` any more. Do not repair a deprecated path; the notes are
kept because the *shape* of the bug (a TS/WGSL twin silently disagreeing) is the
recurring hazard, not the carpet specifics.

1. ~~**The scatter twins disagree for carpet entries.**~~ `Scatter.cell()`
   (`src/scene/scatter.ts`) returns `scale: e.scaleMin` — the stand's
   PLACEHOLDER (1.7 / 1.6 / 1.4 for the three Sphagnum states) — while
   `createStandBuffer()` writes `carpetScale()` (1.0101) into
   `stand_table.scale_min`, which `scatter.wgsl` reads. So every renderer that
   materialized instances in TS drew the bog moss 1.4-1.7x life size, and the
   "bit-identical twins" invariant was simply false for carpets. Now recorded as
   a deliberate non-fix in a `@deprecated` comment at both sites.
2. ~~**`carpetScale()` is not exported from `@harness`.**~~ Same reason.
4. **FIXED (2026-07-26).** The GCMESH1 octahedral source normal is **(x, y)
   stored, z reconstructed, NOT negated**. `src/mesh/gcmesh.ts`
   (`decodeGcMeshNormal`, which `normalAt` now calls) and its WGSL twin
   `src/wgsl/gcmesh.wgsl` (`gcmesh_normal_decode` / `_u16`) are the only two
   places it lives; **36 experiment ids** now read a source normal through one of
   them (38 shader `#include`s + 4 TS call sites) and **40 hand-rolled y-derived
   copies are gone** — 36 WGSL `oct_decode*` in experiment bakes, 4 in TS bakes,
   and one in the harness's own mesh inspector, which is where "validated
   visually" had been claimed. Re-measure with `tools/probe-oct-normal.ts
   [species]`; all figures below reproduce exactly as of 2026-07-26.

   AXIS — decode the stored pair three ways and compare to the face normal from
   raw dequantized POSITIONS, which needs no convention at all, over 217,113
   triangles strided across the whole mesh (a prefix is one corner of one plant):

   | derived axis | mean \|cos\| | mean cos | |
   |---|---|---|---|
   | x | 0.4225 | 0.0049 | |
   | **y** | 0.5354 | −0.2683 | what `GcMesh.normalAt()` used to do |
   | **z** | **0.7871** | −0.7309 | correct |

   Z wins decisively, independently reproducing 004-raycast-lut's 0.75-vs-0.57,
   and wins by more on the other two species (0.85 elymus, 0.94 poa). A vertex
   normal is a smoothed average, so |cos| below 1 is expected even when right.

   SIGN — **do not take it from that mean signed cos, and this is the trap.** It
   only means something if the CCW cross product points out of the surface, and:
   the same figure is −0.73 on calamagrostis but **+0.71 on elymus and +0.89 on
   poa**, so the three meshes are not wound alike; the signed-volume test that
   would settle winding is invalid because all three are OPEN shells (translate
   the mesh and the "volume" moves, by 330x on calamagrostis); and the
   topmost-vertex-per-column test that appears to confirm it is near-random on
   calamagrostis (mean |n.y| 0.08) and says the OPPOSITE on elymus, where it is
   unambiguous (mean |n.y| 0.72, 93.5% facing up un-negated). A first pass
   negated on that evidence and inverted every source normal in the repo.

   What settles it is occlusion, which assumes nothing about winding,
   closedness or up: a surface has material on ONE side, so fire a 4 mm ray from
   a vertex along +n and −n, skipping its own triangles. +n hits geometry 65% /
   60% / 58% of the time versus −n's 77% / 93% / 69% (calamagrostis / elymus /
   poa) and has the greater free space on all three. +n is outward. Visually the
   same thing shows in the mesh inspector's single-sided `lit` mode: the tile's
   own one-sided ground slab, which must face up, goes black when negated.

   Consequences that were part of the fix: `mesh/baked/**` was wiped except 004,
   014, 025 and 034 (provably convention-independent), and `src/bake/io.ts` now
   carries a `SHARED_BAKE_REV` in the OPFS cache key so a future shared-decoder
   change invalidates every browser's cache instead of silently serving stale
   bytes. An experiment's own `bakeVersion` cannot express a harness change.

   VERIFIED VISUALLY, and this is the part worth keeping. The decisive picture is
   `000-ground-truth`, which draws the raw mesh with real per-vertex normals and
   no bake in between. Rendering it twice — once normally, once with the old
   y-derived decode patched back in at `createShaderModule` so no file had to move
   — and measuring `debug=normals` on the 112k pixels that differ: **mean |n.y|
   0.60 -> 0.26**, and the share of plant pixels whose normal is near-horizontal
   **46% -> 93%** (`cam=inside-plant`; from `carpet-close`, 52% -> 91%). A
   near-vertical grass blade's face normal *must* be near-horizontal, and the old
   decode had a quarter to a half of the canopy claiming it faced straight up or
   straight down. In `debug=off` the old decode's blown-out yellow-green blades
   and pitch-black neighbours become a coherent mid-green.
   Fleet-wide, `debug=lighting` at `cam=grazing`: the share of plant pixels at a
   light term of ~1.0 (blown out) fell in **36 of 40** renderers, often hugely —
   037 98%->42%, 038 78%->30%, 006 33%->7%, 005 38%->11%, 017 15%->2%,
   001 20%->5%. It ROSE in two, and that is a genuine finding rather than a
   measurement artefact: **007-basis-opacity 49%->59%** and
   **010-chord-frustum 52%->54%** (band means +14 and +13). Both AGGREGATE many
   source normals into one vector, and correct near-horizontal normals from
   two-sided foliage cancel far more completely than the old scrambled ones did —
   so 010 hits its `length(n) < 1e-3 -> vec3f(0,1,0)` fallback much more often
   (a degenerate normal answered with "fully lit"), and 007 never flips normals
   toward the capture axis before its basis fit, unlike every other bake in the
   repo. Written up in both NOTES; neither is a decode problem and neither was
   fixed here, because both are method changes needing a rebake and their own
   visual pass.
   Bake staleness was closed on both sides and checked, not assumed. Disk: every
   artifact was re-baked, and for one TS-decode bake (002) and one GPU-shader bake
   (011) a forced re-bake with `mesh/baked` 404ed and OPFS empty reproduced the
   committed files **byte-identically**. OPFS: seeding a cache with `r1__…` and
   un-prefixed legacy entries and then loading an experiment evicts all of them
   and leaves only `r2__…`. Note the gap this exposed — **`017-cached-clusters`
   fetches `mesh/baked/` itself instead of going through `bakedArtifact()`**, so
   it is outside `SHARED_BAKE_REV` entirely; it happens to be safe because it also
   never populates OPFS, but a future shared-decoder bump will not reach anything
   that hand-rolls that fetch.

3. **The drawn ground is not the sampled ground.** STILL LIVE and still worth
   fixing — it has nothing to do with carpets. `basePass.ts` draws
   `TERRAIN_QUADS = 256` over a 256 m terrain (1 m triangles, linear across
   each), while `terrain_sample()` is bilinear over a 512² / 0.5 m heightmap.
   Three agents measured the same thing: the drawn surface sits ABOVE the
   sampled one by >1 cm over ~21% of the bog and up to ~6.5-7.2 cm — more than
   an entire 7 cm Sphagnum tile, so anything conforming to the sampled surface
   gets buried by the terrain the harness itself draws. Preferred fix: raise the
   drawn grid to match the heightmap (512) so there is ONE ground surface; it
   costs 4x terrain verts in the base pass and shifts the base-pass row of every
   historical bench, which is worth saying out loud but does not touch
   experiments' own pass timings.
   There is a second, visual symptom worth knowing: at grazing the 1 m triangles
   read as ~1 m blocky facets through any ground-conforming surface. Seen
   identically in 001 and 039, i.e. it is the harness's ground, not a renderer
   bug — which makes this a look fix as well as a geometry fix.

## Determinism: the lab is NOT bit-deterministic, and now we know by how much

Measured 2026-07-26 during the Stage refactor, by capturing the full golden set
three times — once before, once after, and once more on IDENTICAL code as a
control:

- refactor vs baseline: **77 of 160 goldens differ**, median 18 px of 1.46M
  (0.001%), median RMSE 0.076/255.
- identical code vs identical code: **79 of 160 differ**, median 15 px, and the
  per-file numbers are indistinguishable (93,113 vs 95,078 px on the worst file).

So roughly half the goldens move between two runs of the same code. Causes:
GPU atomic ordering in cull passes changes instance order and flips
exactly-depth-tied fragments; and `det=1&t=2` freezes *time* but not *frame
history*, so any accumulating/temporal method drifts with settle timing —
`016-screen-stamp` moves **6% of its pixels** at grazing this way.

Consequences, both of which have bitten already:
1. **Strict byte-equality goldens are not a usable gate.** Always run a
   same-code control alongside any before/after comparison, and compare the two
   distributions rather than asking "did anything change".
2. A single screenshot pair proves much less than it appears to. When judging a
   visual change, prefer a measurement over many pixels (differing-pixel count
   and RMSE) to an eyeball verdict on one frame.

## Measuring performance under contention

Absolute milliseconds are meaningless while agents share the GPU. Use the
same-frame A/B page `#/ab/001-billboard-smoke/<id>` and quote the ratio.
**Known bias: the B slot measures ~1.26–1.35× slower than A for identical
work** (measured by running 001 against itself), so a reported 1.26× is roughly
parity. A proper comparison needs both orderings or solo benches on an idle GPU
— worth doing now that rounds are finishing.

## Fleet-running lessons

- Workflows: `parallel()` in waves of 4–6. `resumeFromRunId` replays completed
  agents from cache and re-runs only failures — but **changing any prompt text
  invalidates the cache for every agent**, so to fix one agent's brief without
  re-running all, add a per-spec suffix rather than editing the shared template.
- A spend-limit hit mid-run marks agents errored; resuming after the limit
  clears is cheap and safe.
- Generate a fleet script from the reviewed doc so the dispatched prompt cannot
  drift from what the owner approved (see `.claude/moss-round.js`, generated
  from `docs/moss-round-prompt.md` by escaping backticks into a template
  literal).
- Agents reliably report `renderVerified: true` off screenshots they did look
  at, but the *interface feedback* field has been the highest-value output —
  keep demanding it.
- Before believing a speedup, demand a control. One agent nearly reported a fake
  25 % win that was six silently dropped draws.

## The pivot (2026-07-25) — read this before anything below it

The lab was built on one idea: an experiment is a *renderer of a stand*. The moss
round ended that. Forcing a tiled ground carpet and upright plants through one
renderer generalised badly — nearly every renderer independently mis-sized its
enumeration loop and drew a quarter of the mat — and the best moss result
(`039-nd-moss`) turned out not to be a geometry technique at all: Uncharted 4's
moss is a **surface material** on geometry that owns the silhouette.

So experiments now get **kinds**, and the kind being built is **materials**:
graph-authored, channel-inspectable, previewed on standard geometry,
A/B-comparable, with node outputs bakeable to PNG. The full design is in the
approved plan (`~/.claude/plans/read-the-readme-plan-cozy-shannon.md`).

What changed on disk:
- The three Sphagnum meshes (1.5 GB) and the `bog` stand are **deleted**. Their
  measured maps survive as PNGs in `assets/materials/sphagnum-*` — albedo,
  height, normal, cavity AO, plus `measured.json` with the tip/base colours and
  plane/apex heights the shading needs. Those cannot be regenerated; treat them
  as source data now.
- 1.1 GB of baked moss artifacts deleted with them.
- Carpets are `@deprecated` but intact. `tileM`/`footprint_m` are NOT deprecated —
  they are load-bearing for the grasses.
- Species indices 3-5 are retired and must never be reused: the index goes into
  the GPU stand table and the catalog is append-only.

## The materials system as built (2026-07-26)

All phases of the approved plan except the deferred ones are done. What exists:

- **Kinds + a tree.** `experiments/renderers/<nnn>-<slug>/` and
  `experiments/materials/<class>/<subject>/<nnn>-<slug>/`; id = last path
  segment; numbering restarts per branch; the browser renders the tree from
  `entry.taxonomy` with collapse persisted in the URL.
- **A pluggable `Stage`.** `StandStage` is the old behaviour byte for byte;
  `MaterialStage` gives a studio backdrop, an orbit camera and harness-owned
  preview geometry (sphere / plane / cube-edge / cylinder, uv in METRES of
  surface). The `@group(0)` LAYOUT is frozen forever — only its contents vary —
  which is why `light_surface()` and `debug_shade()` work verbatim in a material.
- **A material = a channel graph + two authored WGSL stages.** Five frozen node
  kinds; `meshCapture` deliberately absent. One declaration produces both
  execution modes, so live-vs-materialized is a caching decision. The generator
  emits the `debug_shade` call, which makes the debug views unskippable.
- **`#/material/<id>`** — node graph, every intermediate texture with channel
  isolation / mip slider / texel probe / per-level luma, generated WGSL,
  validator report. Renders itself from the `MaterialDef`, so new materials are
  inspectable for free.
- **Portable export** — a ZIP that renders **bit-identically outside the lab**
  (0 of 684,000 px differ), verified from a page importing nothing from here.
- **Three materials**: `001-flat-maps` (hand-written, the pre-graph reference),
  `002-graph-maps` (the same thing as a graph — pixel-identical), `003-nd-fuzz`
  (the Uncharted 4 BRDF; the port needed no new node kind and no `src/` change).

Known internal gaps, all recorded in the materials' NOTES rather than hidden:
`parallaxOffset` cannot express a one-tap parallax shadow (`ViewUv.shadow` is
never filled, `EvalCtx` has no tangent frame) and has no travel bound, so an
author must set `limit` by hand; a node with two consumers is evaluated twice;
and codegen binds a materialized node's inputs even though only the bake reads
them, which is a dead binding plus wasted VRAM (the exporter strips them, the
lab does not).

## Deferred to the end of the materials plan

- **Silhouette POM is unresolved, and my analysis of it is contested.** After
  seeing 002's `silhouette` option carve the cube-sphere's chart seams (an
  interior "tennis ball" pattern) while leaving the outer edge smooth, I claimed
  uv-bounds SPOM structurally cannot alter a closed mesh's outer silhouette. The
  owner disagrees and reports having seen a UE5 SPOM variant that cuts spheres
  correctly — at the geometrically expected places, not at uv-island borders. So
  treat my claim as WRONG until re-derived: there is an existence proof for the
  behaviour we want, and the current implementation
  (`r.clip = !hit || mat_uv_outside(...)` in `src/material/codegen.ts`) is the
  thing to replace, not the thing to justify. Revisit with the owner's reference
  in hand at the end of the plan; it is explicitly not a blocker before then.

## Owner decisions not yet executed

- ~~**The hierarchy must end up uniform.**~~ DONE (2026-07-26). All 40 renderers
  (000-ground-truth and `_template` included) moved to
  `experiments/renderers/<nnn>-<slug>/` with `git mv`; materials stay at
  `experiments/materials/<class>/<subject>/<nnn>-<slug>/`. Nothing had to be
  edited inside any experiment — ids come from the manifest, `RegistryEntry.dir`
  comes from the glob path, and ratings/bench results/goldens are keyed by id.
  The browser is now a recursive `<details>` tree driven by `entry.taxonomy` /
  `entry.kind`, with per-grid sorting and collapse state in `?closed=`.
- **The 25MB VRAM bar has no material-appropriate meaning.** It was calibrated
  for a plant species' baked representation; the first material measures 48MiB
  for one uncompressed 2048² albedo+normal+AO set (BC7/BC5/BC4 would be ~14MB).
  Needs either a per-kind budget or block compression — owner's call.

## In flight / queued (as of this writing)

- The moss round (`w1hf8i2cc` = run `wf_a5a3432c-55e`) was **stopped by the owner
  at 25 of 33** on weekly-limit grounds, with all completed work kept and
  committed (002–026). 027–034 and 035–038 never got a moss pass and now never
  will — the pivot obsoleted the task. Useful mechanic learned: read a live
  workflow's progress from
  `~/.claude/projects/<proj>/<session>/subagents/workflows/<runId>/journal.jsonl`
  (`started`/`result` records); the task `.output` file stays EMPTY until the
  whole workflow finishes, so do not read emptiness as death.
- `039-nd-moss` landed: Uncharted 4's material measured off the source mesh.
  It is the port target for the first material experiment.
- Still open from before the pivot: delete the `installAssetBaseShim()` global
  fetch monkeypatch by converting ~7 experiments to `assetUrl()`.

## Round history and verdicts

1. **16 methods** (impostors, volumes, splats, screen-space, temporal). Verdict:
   the simplest — billboard cards — won on BOTH looks and speed. Most "clever"
   methods were slower and uglier, and over half violated plant-count
   independence via expensive per-fragment work and overdraw.
2. **Audit + debug views** over 15 experiments: 133 structural problems found,
   64 fixed. Worst were a missing frustum test (drew a full disc), rasterising
   the world out to 2.1 km, 710k scatter evaluations/frame, and two silent
   bugs — an experiment never loading its own baked artifacts, and a
   `firstInstance` indirect draw that WebGPU discarded outright.
3. **16 "beat the billboard"** methods: 16/16 working, 14 claiming better looks,
   ratios clustered 1.18–1.4× (i.e. near parity once the B-slot bias is
   removed). Two admitted honestly they did not win.
4. **4 O(1) precomputed-raycast** methods with zero plant geometry: all four
   work, and all four independently found the same wind solution — sway is
   linear in height, hence a shear, and shears map lines to lines, so you
   inverse-shear the QUERY RAY instead of moving geometry. **None beats
   billboards on looks**: a table at ~25 MB/species affords 1–3 cm texels, and at
   grazing incidence the entry point sweeps along the ray far faster than across
   it, so the honestly prefiltered answer is soft exactly where cards stay
   crisp. They also bend the stand contract — a single-lookup ray answer cannot
   resolve arbitrary per-plant positions, so each tiles one baked window
   (correct densities/scales/yaws, wrong individual positions).
5. **Moss**: 001 pilot succeeded (ground-parallel tile-sized quad, per-vertex
   terrain conforming, tile-cropped imagery, carpet alpha reference) and proved
   a card cannot express cushion thickness. Round for the other 33 is running.
