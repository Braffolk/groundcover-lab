# Groundcover Lab → a multi-kind experiment lab, with a MATERIALS system

> **This is the plan as APPROVED on 2026-07-25, preserved verbatim below the
> status block so it can be held against what was actually built.** It was
> written and agreed in plan mode; the copy that drove the work lived outside the
> repo, which made it invisible to anyone reading the project, so it is checked
> in here.
>
> An earlier plan — the original harness/tooling design from the first session —
> was overwritten in place when this one was written. Its content is not lost in
> any practical sense: it *is* `src/`, and `README.md` records the requirements it
> was built from. It is simply not recoverable as a document.

## Execution status (2026-07-26)

| phase | state |
| --- | --- |
| **A** — rescue the measured maps, delete the meshes and the `bog` stand, deprecate carpets | **done** |
| **B** — experiment kinds + nested discovery, ids decoupled from paths | **done** |
| **C** — material core: graph, codegen, validator, bake-to-PNG, portable export | **done** |
| **D** — material inspector page, browser tree, uniform hierarchy | **done** |
| **E** — two exemplars | ND moss **done**; the silhouette-POM one **in flight** |
| **F** — scene bundles | **not built, by design** — the plan only required that the kind system not preclude it |

Landed beyond the plan, because measurement demanded it: a shared texture/mip
helper that makes the two documented filtering traps unrepresentable, the PNG
codec the repo never had, a corrected GCMESH1 normal decoder (the old one was
measurably wrong), and the deletion of the global `fetch` monkeypatch.

### Where the plan was WRONG, and was corrected during execution

Recording these because a plan that only shows its successes teaches nothing:

1. **`@group(0)` per stage.** The plan gave `MaterialStage` its own bind-group
   layout. That was the dangerous third of the design: the layout is a WGSL
   contract `frame.wgsl` declares and three shared includes depend on, through an
   include mechanism with no conditionals, so a second layout forks those files
   and they drift. The layout is now frozen forever and only its CONTENTS vary —
   `MaterialStage` supplies a 4×4 dummy heightmap and a one-row stand buffer,
   under 1KB, and `light_surface()`/`debug_shade()` work verbatim in materials.
2. **Silhouette POM.** The plan described Crimson-Desert-style uv-bounds clipping
   as the technique. It is not: clipping a marched uv against a uv rectangle
   carves the *chart seams* of a closed mesh — a tennis-ball artifact — and can
   never alter an outline. The accurate form needs a shell/prism geometry stage,
   which the plan listed as stage 1 and never built. That is what is in flight.
3. **"25MB VRAM bar for materials."** Settled by the owner: materials show usage
   with no budget, because the 25MB figure was calibrated for one plant species'
   baked representation.
4. **Baked-artifact storage.** I reported `mesh/baked` as "1.4GB committed"; that
   was `du` measuring disk, including ~1GB of untracked files. The real figure was
   419MB across 33 files, now gitignored alongside the raw meshes.

### Deferred deliberately

`meshCapture` as a node kind (the meshes it would capture are deleted); a shared
BRDF helper library (each material authors its own, which is the point of
graph-for-data/WGSL-for-behaviour, at the cost of copy-paste between materials);
and the scene kind.

---


## Context

The lab was built around one idea: an experiment is a *renderer of a stand* — plants
placed by the harness, drawn by the experiment. That shape has now been pushed to
its limit. The moss round showed that forcing "a tiled ground carpet" and "upright
plants" through one renderer generalises badly: 23 of 25 renderers independently
hit the same carpet bugs, and the best moss result in the repo (`039-nd-moss`)
turned out not to be a geometry technique at all — Uncharted 4's moss is a
**surface material** that sits on whatever geometry owns the silhouette.

So the lab splits along its real seam. Experiments get **kinds**. The kind we build
now is **materials**: complex, possibly very non-standard surface materials (ND
moss's wrap/fuzz/AO-Fresnel stack; Crimson Desert's silhouette-clipping POM),
authored as an inspectable graph so a human or an LLM can open one, see every
intermediate texture, and fix it. Materials are hierarchical (mosses → sphagnum →
N competing attempts), previewed on standard geometry, A/B-comparable, and their
shader-generated textures compile down to PNGs for fast cached use.

Later — sketched, not built — a **scene** kind bundles terrain + its materials +
a stand and plant renderer, to judge a whole regime at once.

## Decisions taken with the owner

1. **The three Sphagnum meshes and the `bog` stand are deleted outright.** Carpet
   code stays, marked deprecated everywhere.
2. **Authoring: typed TS graph + WGSL leaf modules.** Channel *data* is a
   declarative graph the UI renders and diffs; *behaviour* (parallax, BRDF) is
   small WGSL modules with declared params. A material may supply its own
   `shade()` — non-standard is a first-class case, not an escape hatch.
3. **Nested directories, id decoupled from path.** `experiments/materials/mosses/
   sphagnum/003-nd-fuzz/`; the manifest declares the id, so ratings, bench results
   and goldens keyed by id survive a reshuffle.
4. **First delivery = system + two exemplars**: ND moss ported in, plus one
   Crimson-Desert-style silhouette-POM material — two materials that stress the
   abstraction from opposite directions, before N agents build on it.

## The core: what a material IS

Both exemplars were studied to fix this shape, because each breaks a different
assumption of ordinary PBR:

- **ND moss** needs a *custom BRDF*: light wrap that deliberately ignores the
  normal map, a fuzz/sheen layer as the explicit subpixel answer, AO driving
  indirect while a micro-shadow term drives the sun, AO-Fresnel keyed off the
  **geometry** normal, additive AO-saturation bounce, and a wetness function that
  lerps albedo toward its own square. None of that fits a fixed PBR slot list.
- **Crimson Desert SPOM** needs a *view/UV stage with an escape*: ray-march the
  height field, then inverse-rotate and inverse-scale the offset, test it against
  the mesh's [0,1] UV bounds, and **clip** where it overshot — so the silhouette
  follows the height field. It needs UV-transform awareness, an opacity/clip
  output, an optional depth offset, and a second march along the light for
  self-shadowing (light rotated into tangent space *after* the transform).

So a material is a five-stage program, all stages optional except Channels and
Surface:

| stage | when | what it may do |
| --- | --- | --- |
| **0 Inputs** | — | named fields: constant, UI param, texture, or a host field (uv, world pos, curvature, slope, and custom fields like *life stage*) |
| **1 Geometry** | per-vertex | displacement / shell / patch subdivision; declares the preview geometry it needs |
| **2 View-UV** | per-fragment | parallax, POM, **SPOM silhouette clip**, parallax self-shadow, optional depth offset |
| **3 Channels** | per-fragment | the node graph, sampled at the final UV |
| **4 Surface** | per-fragment | assemble the typed `Surface` struct |
| **5 Shade** | per-fragment | standard lighting, a chain of feature modules, or the material's own `shade()` |

**Channels** are open-ended, not a fixed PBR list: albedo, normal, height,
roughness, metallic, AO, cavity, opacity, emissive, translucency/thickness, sheen
colour, anisotropy — plus material-defined ones. A material declares which it
produces; the UI shows exactly those.

**Node kinds** in the graph: `image` (PNG/asset), `procedural` (WGSL snippet),
`meshCapture` (ortho render of a source mesh → albedo/height/normal, as 039 does),
`filter` (dilate, mip, blur, normalize, curvature-from-height, horizon-AO),
`combine` (heightBlend, lerp, mask, triplanar), and `variantBlend` (N variants
mixed by an Input). Every node is *either* evaluated live in-shader *or*
materialized to a texture — same graph, two execution modes. That duality is what
makes "shader output compiles to PNG" a property of the system rather than a
per-material chore.

**Spatially-varying inputs** (the Sphagnum life-stage question): this is standard
practice, not a smell — it is exactly landscape layer blending / UE material
layers / Substance material blending. Two guard rails go in the system: blend
*baked channels*, never N full shading models (cost), and default to
**height-aware blending** rather than a linear lerp, because linear-lerping two
materials reads as mush while height blending keeps each one's structure.

The shading contract in sketch — the part that decides whether both exemplars fit
without special-casing:

```ts
// What stage 2 may return: this is what SPOM needs and ordinary PBR ignores.
interface ViewUvResult { uv: vec2; clip: bool; shadow: f32; depthOffset?: f32 }

// What stage 4 assembles. Open set: a material declares the slots it fills.
interface Surface {
  albedo; normal; geomNormal;      // geomNormal kept separate — ND's wrap and
  height; ao; roughness; metallic; // AO-Fresnel deliberately use it, not normal
  opacity; emissive; sheen; thickness; ...
}

// Stage 5: pick ONE.
shade: stage.standard()                       // harness lighting
     | stage.features([wrapDiffuse, fuzzSheen, microShadow, aoFresnel, wetness])
     | stage.wgsl('shade.wgsl')               // full control, material-owned file
```

Keeping `geomNormal` beside `normal` in the surface struct is not a detail: two of
ND moss's five tricks are defined by using the geometry normal where a normal map
would be the obvious choice, and a system that only carries the shading normal
silently makes that material impossible to express.

**Portability** is a first-class output: `export` emits PNGs + one generated
`material.wgsl` + `material.json` + a usage snippet, depending only on a small
documented ABI (a host-inputs struct and a texture/sampler bind group). Dropping
a material into another WebGPU codebase is a copy, not a port.

## The central architectural decision

Today `LabApp` (`src/harness/loop.ts`) *requires* a stand, always builds terrain +
scatter + stand table, and its shared `@group(0)` hardwires binding 1 = heightmap,
2 = sampler, 3 = stand table — which every experiment pipeline inherits. The base
pass draws terrain and sky unconditionally into view 0 and byte-copies colour and
depth to view 1; that copy is the A/B fairness invariant.

A material preview wants none of that, but wants nearly everything else: the pass
timer, VRAM tracking, the same-frame A/B compositor with one shared camera, the
HUD, params/URL state, capture, and the debug-view contract.

So: **`LabApp` gains a pluggable Stage** that owns scene services and the base
pass.

- `StandStage` — today's behaviour byte for byte, so the 40 existing renderers see
  no change and their bench history stays comparable.
- `MaterialStage` — preview geometry (sphere, plane, and an edge-on object,
  because silhouette POM is only judgeable at an edge), studio lighting, neutral
  backdrop, orbit camera.

The alternative — a standalone material page copying `src/harness/views/mesh.ts` —
is cheaper and touches nothing, but throws away same-frame A/B, which is precisely
the instrument for choosing between eight competing Sphagnums. `mesh.ts` is a
viewer, not a lab: no timer, no VRAM, no params, no URL state, no debug modes.

**Corrections from the design review** (it overturned part of the above, so they
are recorded rather than quietly folded in):

1. **The `@group(0)` LAYOUT is frozen; only its CONTENTS vary per stage.** My
   original idea — give `MaterialStage` a layout without the heightmap and stand
   table — was wrong, and it was the dangerous third of the proposal. That layout
   is a WGSL contract: `frame.wgsl` declares bindings 0-3 by name and
   `lighting.wgsl`, `debug.wgsl` and `terrain.wgsl` all include it, with an
   include mechanism that has no conditionals. A second layout therefore forks
   those three files, and forks drift. The price of keeping it is a 4×4 dummy
   heightmap and a one-row stand buffer — **under 1 KB** — and the return is that
   `light_surface()` and `debug_shade()` work verbatim in material shaders. That
   matters concretely: ND moss's trick of splitting the shared lighting model by
   subtraction only exists *because* the model is shared. One exported
   `FRAME_LAYOUT_ENTRIES` const, one construction site, and a comment saying why.
2. **The Stage does NOT own the base-pass copy to B.** That copy is the A/B
   fairness invariant; if a stage could override it, a stage could silently break
   fairness. Stages supply only `encodeBase(...)`; the copy stays in `LabApp`.
3. **Do not bump `HARNESS_API`.** It is checked for exact equality, so a bump
   greys out all 40 manifests at once. Refusing to bump forces additive-only
   changes — a useful constraint, and the design fits inside it.
4. **Capture goldens for all 40 experiments before touching `loop.ts`.** Only 2
   have goldens today. `npm run capture:goldens` already exists; run it before and
   after and diff to zero. That is the whole regression net for this refactor.

## Gaps that must be built, not reused

Exploration turned up four things the material system needs that simply do not
exist yet — worth naming because they are real work, not glue:

1. **No image loading anywhere.** Zero `createImageBitmap`, zero
   `copyExternalImageToTexture`. PNG is only ever *encoded*, for thumbnails. The
   entire "materials use PNGs" story starts from nothing.
2. **`.bin` is hardcoded** in `bakedArtifact()` and the dev sink, and bake keys may
   not contain `/`. Baked PNG channels need that generalised (extension +
   subpaths) without breaking the 33 experiments with existing `.bin` artifacts.
3. **No shared mip/texture helper.** Ten-plus experiments hand-rolled mipgen, and
   that produced the two filtering bugs CLAUDE.md documents (coverage-weighted vs
   premultiplied; octahedral normals not being mip-averageable). The material
   system is where that helper finally belongs — with both traps designed out, so
   a material declares a channel's *filtering semantics* and cannot pick wrong.
4. **Bake has no progress surface.** `BakeContext.onProgress` is declared, passed
   by two experiments, and never called by anything; a long bake just looks like a
   hung page. Materials bake more, so this gets wired up.

Also to generalise, minimally: `BenchResult.experiment.stand` is required and the
dev sink rejects a result without a valid stand segment — materials need a
`context` alongside it. The VRAM axis needs nothing: a material passes its own id
as the `species` tag and the existing budget bar works unchanged; renaming it can
wait until it grates.

Two traps the review found that would each have cost a debugging session:

- **`ShaderRegistry.module()` serves the FIRST code ever registered for an id.**
  A generated module reusing a stable id would serve stale code for the rest of
  the session — symptom: "my graph edit did nothing", the worst possible
  debugging experience. Fix before writing any codegen: content-hash the
  generated id, and add a `latestCode(src)` accessor so codegen concatenates the
  hot-updated source rather than the stale imported object. Five lines.
- **`createImageBitmap` silently truncates 16-bit PNG to 8 bits**, and the
  browser cannot *encode* 16-bit PNG via canvas at all. So the height/normal
  precision story needs a small hand-written codec — `CompressionStream('deflate')`
  emits exactly the zlib-wrapped stream an IDAT wants, so it is ~200 lines, and
  samples are big-endian (get that wrong and a heightmap looks like noise with a
  plausible histogram). The image loader must *throw* on a 16-bit PNG rather than
  quietly halve its precision. Rejected shortcut: packing 16 bits across two 8-bit
  channels — it works, but the file stops being viewable as an image, which
  deletes the entire reason to prefer PNG over the `.bin` we already have.

## Execution model

Orchestrated, one subagent at a time, reviewed between phases — with two
exceptions I keep for myself:

- **Phase A I do personally.** It contains the only irreversible step in this plan
  (deleting 1.5 GB of source meshes whose derived maps exist on this machine only)
  and a same-commit constraint that turns a mistake into a fatal page. That is not
  work to hand off.
- **Phase B I do personally** — it is mechanical, spans shared code and tooling
  that agents are forbidden to touch, and every later agent depends on it.

Phases C, D and E go to single subagents. E is the only one that can run two in
parallel, since the two exemplars share no files.

## Delivery phases

### Phase A — rescue, then remove
The moss meshes are 1.5 GB and untracked; the 039 maps baked off them are 96 MB
and **also untracked, existing on this machine only**. Once the meshes are gone
those maps can never be regenerated — and any bump of the bake `VERSION` makes the
existing `.bin` unloadable. So, in order:
1. Convert 039's three baked artifacts into committed **PNGs** under the new
   material's directory. Verified feasible with zero dependencies: each file is
   `NDM1` v4 — a 128-byte header then two RGBA8 2048² planes (albedo+height,
   normal+AO), 33,554,560 bytes exactly — so a plain Node script can slice the
   planes and encode PNG via `node:zlib`. **The header's floats must be extracted
   to JSON in the same pass**: measured tip/base/mean colour, mean plane, apex
   height and coverage. Those scalars are what drive the fuzz sheen, the light
   wrap and the AO-saturation bounce — dropping them would leave the maps intact
   and the material unreproducible.
   This keeps the *measured* maps, which is what made 039 good, while the meshes
   still go. Losing them is the one irreversible step in this plan.
2. Delete `mesh/raw/spaghnum-*`, the three catalog rows in `src/scene/species.ts`,
   and the `bog` stand — **in a single commit**, because `createStandBuffer()`
   calls `speciesById()` and a surviving `bog` with removed species is a hard
   fatal page, not a degraded render.
3. Mark carpets deprecated at the 13 shared-code sites (`carpetDiv`,
   `carpetScale`, `standEntrySlots`, `CARPET_JITTER`, both scatter branches,
   `carpet_div`/`slope_align` docs, the `carpet-close` bookmark). **Leave `tileM`
   and `footprint_m` alone** — they are load-bearing for the grasses; only reword
   the `tileM` doc that currently frames it as a carpet feature.
4. Close out the two carpet-only harness bugs in `docs/ORCHESTRATION.md` as
   won't-fix-deprecated, so nobody repairs a deprecated path.

Stale moss ids in 24 manifests are inert (`species` is never validated — its only
consumer is an A/B coverage warning) and can be swept lazily; `?stand=bog` URLs
already fall back to `default` with a console warning. Two consequences worth
stating rather than discovering: `013` and `034` size a texture array from
`SPECIES.length`, so 6 → 3 shrinks them and shifts their recorded VRAM numbers;
and the 1.08 GB of baked moss artifacts across 33 experiments becomes dead weight
that can be deleted with the meshes, since it is all untracked anyway.

### Phase B — experiment kinds + nested discovery
`kind: 'renderer' | 'material' | 'reference'` on the manifest (extensible; `scene`
later). Registry globs `experiments/**/manifest.ts`, carries `dir` alongside `id`,
and derives `taxonomy` from the path (manifest may override).

The one invariant being broken is **id === directory name**, which is load-bearing
in more places than it looks. All of these switch to the entry's `dir`:
`registry.ts`'s `dirOf()` (path segment `[2]`) and its `thumbnailUrl` convention,
`ratings.ts`'s `/experiments/<id>/rating.json` fetch, the dev-sink `segment()`
validator (which rejects `/` outright, so `__thumb`/`__rating`/`__bake` all need
path-aware validation against the known experiment dirs),
`tools/vite-plugin-static-deploy.ts`'s single-level `readdir`, `tools/capture.ts`,
and `tools/new-experiment.ts`'s `^(\d{3})-` numbering scan (which also gains a
`--kind`). Existing renderers move under `experiments/renderers/` keeping their
ids, so ratings, bench results and goldens stay valid — the ids are what those are
keyed by, not the paths.

The browser gets a recursive group emitter replacing the hard-coded
renderers/references/meshes sections. The only real coupling there is `applySort()`,
which reorders via `grid.appendChild()` against a single container and must become
per-grid; the comparator, `sortValue()` and the `?sort=` writeback are unchanged.

### Revised landing order (from the design review)
The riskless, repo-wide-useful pieces land first, so that a slip in the Stage work
still leaves something valuable shipped:

1. `src/gpu/texture.ts` — the shared mip/texture helper. Pure addition, zero risk,
   pays back across the 20 experiments that hand-rolled mipgen.
2. `src/gpu/image.ts` + the PNG codec + `BakeContext.ext` — additive, testable
   standalone.
3. The `ShaderRegistry` stale-id fix — 5 lines, unblocks all codegen.
4. Goldens for all 40 experiments (the regression net).
5. Stage extraction with the layout frozen — verified by re-capturing goldens and
   diffing to zero.
6. `MaterialStage` + `defineMaterial` + preview geometry + studio backdrop.
7. Graph codegen, driven by porting 039's moss as the first material and SPOM as
   the second.

**The mip helper is where the two documented filtering traps get designed out**,
rather than documented around: one `TexelSemantics` declaration drives *both* the
mip filter and the generated WGSL decoder, so "two textures with two conventions
and you get one wrong" stops being expressible. Octahedral encoding with
`mipLevelCount > 1` throws at texture creation with the 017 story in the message —
a loud throw at init beats a paragraph nobody rereads at 2am. And the generated
reader takes `(uv, ddx, ddy)` with the *unperturbed* gradients, which makes the
parallax-gradient trap unrepresentable instead of merely listed.

### Phase C — material core
Types and graph runtime; WGSL codegen composing the five stages; the node library;
feature modules (wrap diffuse, fuzz sheen, micro shadow, AO Fresnel, AO
saturation, wetness, POM, SPOM silhouette clip); bake-to-PNG with a
graph+params hash cache key; the portable export; and a validator (`no cycles`,
`declared channels exist`, `WGSL compiles`, `debug views route correctly`) that
the material page surfaces as a panel.

### Phase D — material page + A/B
`#/material/<id>` with four panes: **graph** (click a node → its output, channel
isolation R/G/B/A, mip slider, value probe, histogram), **channels** (flat
thumbnail grid, format/size/VRAM), **generated WGSL** (read-only, copyable), and
**preview** — sphere, plane, and an edge-on object so silhouette POM can be judged
at an edge, with sun/env controls and macro/grazing/silhouette cameras. A/B reuses
`CompareCompositor` (wipe/flicker/diff) and the existing `debug_shade` contract.

Reusable pieces already in the repo, so this is less new code than it sounds:
`captureViewPng()` is the GPU-texture→PNG path (currently hard-wired to
`app.views[i].color`; taking a `GPUTexture` instead is a ~3-line change and
immediately yields channel thumbnails), `createFullscreenPipeline()` + `blit.wgsl`
give the channel-isolation preview as one small WGSL variant, and `mesh.ts` is the
working template for a page with its own camera and view modes. The design tokens
are fixed and non-negotiable: five greys plus black and white, no rounded corners
anywhere, emphasis only as inverted fill or a stronger border, `<details>/<summary>`
with a text `▸/▾` for any collapsing.

### Phase E — the two exemplars, in parallel
ND moss rebuilt on the system from the rescued PNGs, and a Crimson-Desert-style
silhouette-POM material (rock or bark). Success = the abstraction needed no
special-casing for either.

### The agent contract must be rewritten too
`CLAUDE.md` is the document every experiment agent is bound by, and it is written
end to end in stand/plant/carpet vocabulary ("Experiments are RENDERERS of a
stand, never placers"). It gets restructured into a **shared core** (TypeScript
only, allowed write paths, determinism, debug views, measurement honesty, the
silent-failure traps — most of which are kind-agnostic and hard-won) plus a
**per-kind section**. The materials section states the graph contract, the channel
filtering semantics, the five stages, what may and may not be hand-written in
WGSL, and how to prove a material works. Without this, the first eight material
agents will each invent their own conventions — which is exactly what the carpet
round cost us.

### Phase F — scene bundles (sketch only, not built now)
A `scene` kind referencing choices by id: terrain regime + material assignments by
slope/altitude/wetness + stand + plant renderer. Designed for now only insofar as
Phase B's kind system must not preclude it.

## Verification

- `npm run typecheck` clean; zero `.js` anywhere; dev server on **5175**.
- After Phase A: `#/` lists no moss meshes, no `bog` in the stand picker; a
  renderer that used to list moss still loads on `default` (stale ids are inert);
  `?stand=bog` warns and renders `default`.
- After Phase B: every existing experiment keeps its id, thumbnail, rating and
  bench history; the browser shows a tree; `#/run/<id>` unchanged.
- After Phase C/D: the material validator passes on both exemplars; every channel
  is inspectable; the generated WGSL compiles standalone; an exported material
  folder renders in a scratch WebGPU page outside the lab.
- After Phase E: ND moss A/B against its own 039 screenshots at macro and grazing;
  the SPOM material judged at a silhouette edge, which is the only place its
  technique is visible.
- Headless verification per the existing pattern (Playwright, `channel: 'chrome'`,
  `--enable-unsafe-webgpu`), scripts outside the repo, and screenshots read — not
  "zero console errors", which CLAUDE.md already records as insufficient proof.
