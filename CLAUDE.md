# Conventions — read before touching anything

TypeScript only. No `.js`/`.mjs` files anywhere, including tools. All tooling is browser-first.

## Experiment KINDS

Experiments come in kinds, and the rules below split into a shared core (everything
outside the per-kind sections — it binds you whatever you are building) and the
rules for your kind.

- **renderer** — draws the plants of a stand. The original kind; everything in
  "Renderer rules" applies.
- **material** — a surface material: a declarative channel graph plus two WGSL
  stages, previewed on harness-owned geometry. See "Material rules".
- **reference** — a stand-independent visual baseline (000-ground-truth).

**Carpets are DEPRECATED.** A tiled ground mat is a *material*, not a species of
plant, and forcing both through one renderer generalised badly: nearly every
renderer in the repo independently mis-sized its enumeration loop and drew a
quarter of the mat. The machinery still exists and is marked `@deprecated`; no
stand uses it. Do not build anything new on it, and do not "fix" it.

## Where experiments live

`experiments/` is a TREE, one branch per kind, and the shape is uniform:

```
experiments/renderers/<nnn>-<slug>/                        renderers (and 000-ground-truth, _template)
experiments/materials/<class>/<subject>/<nnn>-<slug>/       materials
```

**Your ID is the LAST path segment, never the path.** Discovery globs
`experiments/**/manifest.ts` and validates `manifest.id` against the directory
name; ratings, bench results and goldens are keyed by that id, so moving a
branch of the tree never orphans them. Ids must be unique across the WHOLE tree
(a collision is surfaced as a broken card, not silently merged), which is why
`npm run new` allocates its number from a scan of the whole tree.

## Adding an experiment (the only sanctioned workflow)

1. `npm run new -- your-slug` → creates `experiments/renderers/<nnn>-your-slug/` from `_template`.
   A material: `npm run new -- your-slug --kind material --group mosses/sphagnum`
   → `experiments/materials/mosses/sphagnum/<nnn>-your-slug/`.
2. Edit ONLY files inside your experiment directory. Never renumber; your ID is your directory's name.
3. Run it: `npm run dev` → `http://localhost:5175/#/run/<your-id>`.

Allowed write paths: your experiment dir, `mesh/baked/<your-id>/`, `results/` (new files only), `goldens/<your-id>/`. Exception inside your own dir: `rating.json` is the OWNER'S visual verdict — never create, edit, or delete it.
Forbidden: `src/**`, other experiments, `package.json`, any shared file. If the harness is missing something you need, write the need into your `NOTES.md` instead of patching shared code.

## Renderer rules

- Import only from `@harness` (plus your own files). Set `harnessApi` in your manifest.
- The harness owns device/canvas/camera/terrain/scatter/wind and draws the terrain+sky base pass. You append compute/render passes via `ctx.timing.renderPass/computePass` (auto-timed) with `loadOp: 'load'` against the provided color/depth targets.
- Allocate GPU memory only via `ctx.res` (the VRAM tracker) — pass `{ species }` so the 25MB/species budget bar in the HUD is meaningful. The budget is a strong default, not an absolute: stay within it in ~95% of cases; a genuinely novel method may exceed it when the results justify it — document the why and the actual numbers in `NOTES.md`.
- Determinism: randomness only via the shared PCG hash (`@harness`), animation only from `frame.time`. Never `Math.random`/`Date.now` in render paths. The A/B diff view will expose you.
- **Never fetch root-absolute URLs.** Production builds are served under a base path (`/groundcover-lab/`), so `fetch('/mesh/baked/...')` breaks there. Load your baked artifacts through `bakedArtifact()`, and resolve any other asset URL with `assetUrl()` from `@harness`.
- **Debug views are mandatory, not optional.** There is ONE global selector (`frame.debug_mode`, the `view` dropdown in the runner/AB toolbar, URL `debug=`) with modes `off · albedo · normals · lighting · coverage · depth`. Every renderer must `#include "src/wgsl/debug.wgsl"` and route its final fragment through `debug_shade(shaded, albedo, normal_ws, coverage, world_pos)`, skipping fog unless `debug_mode() == DEBUG_OFF` (see `experiments/renderers/_template/shaders/main.wgsl`). A method whose normals or lighting cannot be inspected is a method nobody can trust — and if `normals` shows garbage or `lighting` is blown out, that is a bug in your renderer, not in the debug view. Real per-fragment normals and the shared `light_surface()` are required; unlit or uniformly bright output is a defect.
- **Experiments are RENDERERS of a stand, never placers.** The active stand (`ctx.stand`, URL `stand=<id>`, default `default`) is the harness-owned placement setup: which species, densities, scale ranges, sway, region. Stand + seed fully determines every plant instance via `ctx.scene.scatter` (TS buffers) or the bit-identical WGSL twin (`src/wgsl/scatter.wgsl` + `stand_table`). Render exactly the stand's plants — all its species entries, at their exact positions/scales. Your params may only affect HOW it is drawn, never what/where grows. Stands live in `src/scene/stands.ts` (shared code — don't touch); A/B and bench results are only comparable within one stand + seed.
- The only sanctioned exception is `status: 'reference'` (e.g. 000-ground-truth): stand-independent visual baselines, shown in a separate browser row and clearly labeled. Don't add more without a reason as good as "the raw mesh is a community tile and physically cannot follow per-plant placement".
- You MAY read the raw source mesh (`ctx` mesh catalog, GCMESH1) and invent any novel baked representation. Put baked artifacts in `mesh/baked/<your-id>/` via the bake flow; format is entirely yours.

## Material rules

Everything in the shared core above binds you too — `@harness`-only imports,
`ctx.res` for every allocation, determinism, `assetUrl()`, and **debug views are
just as mandatory**. What differs:

### How a material is authored

- **A material is a GRAPH plus two WGSL stages.** Declare a `MaterialDef` and
  return `createMaterialExperiment(ctx, def)`. You never write a pipeline, a
  bind group, a mip plan or a `debug_shade` call — the generator owns all of
  them, which is how debug views became *unskippable* rather than merely
  mandatory.
- **Graph for data, WGSL for behaviour.** Channel data is nodes; the view-uv
  stage and the BRDF are authored WGSL with fixed signatures
  (`material_surface`, `material_shade`). The BRDF is never nodes — read
  `experiments/renderers/039-nd-moss/shaders/moss.wgsl` for why: it interleaves a
  geometry-normal light wrap, a fuzz layer, AO folded into the sun term only,
  and a subtraction that splits the shared lighting model in half. That is code,
  not a diagram.
- **Node kinds are frozen at five**: `image`, `procedural`, `filter`,
  `combine`, `variantBlend`. `meshCapture` is deliberately absent (the meshes it
  would capture were deleted). Do not add a kind — ask.
- **Declare `TexelSemantics` once, on the node.** It drives the mip filter *and*
  the generated decoder, which is what makes the coverage/premultiply trap
  unrepresentable. Never hand-write a `rgb / a`.
- **Your stage `.wgsl` must not `#include` anything.** The generator wraps
  `material.wgsl` (frame + lighting + debug) around it; an include is a
  duplicate-symbol error, and the validator says so by name.
- **`materialize` is a caching decision, not a different result.** The same
  generated `n_<id>_eval` runs live or baked. If the two disagree that is a bug
  — measure it rather than picking the one that looks better. (One such bug is
  already on record: a materialize pass encoded during `create()` baked a
  zero-filled uniform and cached the wrong PNG under a key asserting it was
  correct.)
- **The bake cache key is the node's transitive subtree**, not the whole graph —
  otherwise touching any node invalidates every expensive bake and nobody
  iterates. Anything a node depends on that is NOT visible as `u.p.<name>` in
  the generated text must be added to the key by hand; a `variantBlend`
  selector is exactly this case.
- Validation errors throw at `create()` and surface as "Failed to start" — but
  the report is still published, so a material that fails validation explains
  itself in the inspector rather than only dying. It is also data on
  `globalThis.__materialReports`.
- **Inspect it at `#/material/<id>`** — the node graph with real dependency
  edges, every intermediate texture with channel isolation, a mip slider, a texel
  probe and per-level stored luma, the generated WGSL, and the validator report.
  This is the debugging surface: "which node produced this" and "is the stored
  luma flat across levels" are answered by looking, not by adding print
  statements. It renders itself from your `MaterialDef`, so you get it for free
  and cannot fail to provide it.

### What the stage gives you

- Declare it with `defineMaterial({...})` (sets `kind: 'material'` for you) and
  type your context `ExperimentContext<typeof PARAMS, MaterialContextExt>`. You
  get `ctx.preview` instead of `ctx.scene`/`ctx.stand` — there is no stand.
- **The graph knows what the maps ARE, so declare it there, not in a comment.**
  The measured Sphagnum maps are mesh-frame with +Y up — axis order (T,N,B), so
  the map's GREEN is the surface normal — and the albedo is LINEAR, not sRGB.
  Both were measured, not assumed. A map read with the conventional
  tangent-space assumption tilts every fragment ~90 degrees and looks merely
  "a bit off" rather than broken.
- Live under `experiments/materials/<class>/<subject>/<nnn>-<slug>/`. Your id is
  the LAST path segment and must be unique across the whole tree.
- **Materials are SHADERS of the preview geometry, never authors of it.** Draw
  `ctx.preview.mesh` with `PREVIEW_VERTEX_LAYOUT` and nothing else — two
  materials on two slightly different spheres turn every A/B into a comparison
  of geometry. The object shown is a runtime choice (`obj=`, the toolbar picker,
  `previewObject` in the manifest) and both A/B sides always see the same one.
- **The preview uv is in METRES OF SURFACE, not [0,1].** A material with a
  real-world period divides by that period ONCE, isotropically, and is at life
  size on every object. A per-axis scale factor is the "material stretched to
  fit the polygons" bug; it has already been made twice. `PreviewMesh.uvPeriod`
  tells you which axis closes on itself (snap to a whole number of tiles there,
  and apply the snapped value to BOTH axes so tiles stay square); `uvBounds` is
  the metre rectangle a displacing method must test its offset uv against. Full
  contract at the top of `src/scene/preview.ts`.
- The preview sphere is a **cube-sphere**, on purpose. A lat/long sphere's u
  compresses by sin(colatitude) and every tiling material inherits a pinwheel
  singularity at the poles; a singularity is not a distortion you can tune away.
- A material has no species, and nothing is renamed for it: list your own
  experiment id in `manifest.species` and pass the same string as `{ species }`
  to `ctx.res`, so the HUD budget bar has one row meaning "this material's
  maps". The 25MB figure was calibrated for a plant species' baked
  representation and has no material-appropriate meaning yet — state your actual
  bytes in NOTES.md rather than reading the bar as a verdict.
- Cameras: `three-quarter` (default), `macro`, `grazing`, `silhouette`. Bench
  splines: `orbit-object` (the default) and `macro-pass`. Bench results record
  the preview object as `experiment.context` where a renderer records `stand`;
  they are only comparable within one object.
- The harness base pass clears the studio backdrop and depth and **does not draw
  the object** — you do, so a silhouette-POM material stays free to discard
  fragments and write its own depth.

## Silent-failure traps (each of these has already cost an experiment a wrong result)

- **`firstInstance` in indirect draws.** A non-zero `firstInstance` in `draw*Indirect` is only legal with the `indirect-first-instance` feature. The harness now requests it when the adapter offers it (`ctx` → `gpu.hasIndirectFirstInstance`), but where it is unavailable such a draw is **silently discarded** — the scene still looks plausible with whole batches missing, and it can even look faster. Prefer binding a per-batch slice of the instance buffer (256-byte aligned) over relying on `firstInstance`.
- **WebGPU validation errors do not throw.** They arrive via `device.onuncapturederror` and surface as an on-screen toast. The harness also mirrors every error toast to `console.error` before de-duplicating it, so a headless script that listens to `console` will see them — but if you scrape the page, read `.toast` elements too. "0 console errors" is not proof of correctness; look at the image.
- **A uniform buffer indexed by a runtime index can be catastrophically slow.** 035-raycast-canopy-volume measured a 1168-byte uniform struct read with a non-constant index at **242 ms/frame — a 62x slowdown** — until the struct was shrunk to 4 vec4 and the bulky part moved to its own storage binding. If a shader is inexplicably slow, look for dynamic indexing into a large uniform before anything else.
- **Verify with a control.** If a change makes something faster, confirm it still draws everything (tint batches, sweep the LOD distance, count instances) before believing the number.
- **Know whether your mip filter already un-premultiplied.** A coverage-weighted mip filter (`rgb = Σ(rgb·a)/Σa`) outputs colour that is ALREADY normalised by coverage at every level; dividing it by alpha again in the shader inflates it by 1/coverage. Coverage falls as the chain deepens and mip level rises with distance, so the error looks like "the field gets brighter the further away it is" — 017-cached-clusters was inflated 2.42x at its deepest mip this way. A plain box filter over premultiplied colour is the opposite convention and *does* need the divide. If a renderer has two textures with two conventions, expect to get one wrong. The test: sample the stored `rgb` luma per mip level — if it is flat, no divide belongs there.
- **Measure per distance band, not per frame.** Whole-field averages hide distance-dependent bugs, and can hide two of them at once: in 017 a rising albedo and a falling light term partially cancelled, so the frame mean looked plausible while both were wrong. Split the frame into distance bands (e.g. 0-5m / 5-20m / 20-60m / 60m+) and compare each band against 001-billboard-smoke and 000-ground-truth, in `debug=albedo` and `debug=lighting` separately — that localises the drift to the stored colour or the light term.
- **Octahedral normals are NOT mip-averageable.** Octahedral encoding is non-linear, and a blade's front/back faces encode to roughly opposite pairs — box-filtering them lands near (0,0), which decodes to *exactly* straight up. 017-cached-clusters lit its whole far field with an up-normal this way: flat, blown out, and worse with distance, while `debug=albedo` looked perfect. If you mip normals, store plain unit vectors flipped into one hemisphere (e.g. around the capture axis) and encode after filtering, or filter in a linear space. Check it: in `debug=lighting`, if the shaded mean equals the albedo mean your light term is averaging 1.0 and doing nothing.
- **Plants must sit on the slope, not ignore it.** The scatter gives you a position and a yaw only, so a naive renderer stands every plant bolt upright. That is roughly right for tall grass but badly wrong for a mat, which buries one edge and floats the other on any slope. `stand_table[i].slope_align` tells you HOW MUCH a species should conform (a botanical property — moss follows the ground, grass grows upright); **how you achieve it, and at what fidelity, is your decision**, and different representations legitimately want different answers. Available primitives in `src/wgsl/terrain.wgsl` (mirrored on `ctx.scene.terrain` in TS): `terrain_height`, `terrain_normal`, `terrain_plane_fit(xz, radius)` for a least-squares plane over your footprint, and `plant_basis_from_up(up, yaw)` / `plant_basis(xz, yaw, align)` to build a basis. A rough ladder, all valid:
  1. rigid tilt from the point normal — cheapest, fine for small footprints;
  2. rigid tilt from a plane fit over the footprint — better once the footprint spans real relief, and the fit height avoids chasing one sample;
  3. per-vertex conforming — evaluate the height under each vertex and displace, so a patch actually follows a bump instead of hovering over it;
  4. warping the query itself — for ray/volume methods, bend the ray or the lookup coordinate by the local ground shape.
  Do not assume level 1 is enough because it happens to suffice on the current terrain: that terrain is one arbitrary sample, and these methods are meant to generalise across regimes. If you deliberately choose a cheap level, say why in NOTES.md. Test across the terrain's ridged slopes, and use the `carpet-close` camera bookmark (1m straight down) to judge ground-level detail — URL `cam=x,y,z,...` poses are ABSOLUTE and the terrain sits several metres below y=0, so a hand-written "y=2" is not 2m above ground.
  For per-vertex conforming, `terrain_sample(xz)` is usually the primitive you want: it returns height **and** (nx, nz) from a single bilinear fetch, whereas calling `terrain_height` and `terrain_normal` (or `plant_basis`, which re-fetches internally) pays for the same taps twice. Note also that a per-tile plane fit is not merely cheaper than per-vertex — for a tiled species it is *wrong*, because neighbouring tiles fit different planes and crack apart at their shared edge; only per-vertex keeps the surface continuous.
- **DEPRECATED (carpets).** Two traps lived here — that a carpet had more slots per cell than `SCATTER_MAX_PER_CELL`, and that a tiled mat must keep one grid spacing, 90°-only rotations and one constant scale. No stand uses carpets now. What still transfers, and is the reason these are kept at all: **drive every enumeration loop and buffer capacity from `standEntrySlots(entry)`, never from a hardcoded `SCATTER_MAX_PER_CELL`** — hardcoding the constant is what made ~74% of the mat silently vanish in nearly every renderer, and a renderer that asks the entry stays correct if the budget ever changes.
- **Never derive a plant's width from `height_scale`.** It happens to look fine for tall grasses, but anything short and wide comes out badly undersized — a 0.07m-tall, 0.24m-wide ground cushion came out ~3.5x too small, leaving gaps in what should have been closed cover. Use `stand_table[i].footprint_m` (the species' horizontal footprint at scale 1) or extents you measured in your own bake.

## WGSL gotchas

- WGSL reserves many innocent-looking identifiers as future keywords — `meta`, `ref`, `common`, `filter`, `standard`, `premerge`, `auto` among them. If `createShaderModule` fails with "'X' is a reserved keyword", just rename (e.g. `meta` → `atlas_info`). Prefer descriptive names over generic ones from the start.

## Taste rules (from the project owner)

- **Dithering/stochastic alpha is a last resort, not a default.** In ~90% of cases it adds nothing but fuzziness — soft silhouettes, screen-door texture, temporal shimmer that a static screenshot hides. It also HURTS performance in real scenes: dithered coverage punches holes in the depth buffer, so early-z/hi-z stops rejecting the layers behind it and overdraw explodes exactly where groundcover is deepest (grazing angles) — a hard-edged surface writes solid depth and becomes an occluder instead. Prefer hard alpha-test edges, correct depth-tested opaque geometry, or honest coverage falloff. If your technique genuinely needs stochastic coverage (e.g. it IS the distance-collapse mechanism), justify it in NOTES.md and check it in motion, not just at rest.

## Comparing & claiming results

- Compare: `#/ab/<idA>/<idB>?stand=default&cam=grazing&seed=42` (wipe/flicker/diff) — both sides render the SAME stand by construction. A/B timings are contended — never quote them.
- Bench before claiming numbers: `#/bench/<id>?stand=default&spline=orbit-low` at a standard cam/spline. Results record the stand and are only comparable within one; they auto-save to `results/` — link the JSON filenames in your `NOTES.md`. Scale tests use the `scaling-100m` stand, not custom placement.
- `NOTES.md` required sections: Idea / VRAM budget math / Bake / Status / Findings.
- Standard cameras (keys 1–4): `grazing` (impostor killer), `topdown`, `inside-plant` (fade check), `far-horizon` (scaling check).

## Standing practice for the orchestrator

After every large batch of experiment work (a wave of new experiments, or a major rework), run a **structural waste review**: one agent per experiment, auditing pipeline and shader STRUCTURE for wasteful decisions and fixing them. In scope: loops in shaders that shouldn't exist or shouldn't be loops, work recomputed per frame that belongs at bake/init/param-change time, redundant passes or copies, oversized or per-plant-scaling dispatches. Out of scope: ALU-level micro-optimization — structure only. Auditors must NOT trust frame-time numbers while other agents run in parallel (contaminated GPU); they reason from the code and verify correctness by screenshot. First-generation agents reliably leave a few structurally insane things behind; assume they exist and go find them.

## Repo map

- `src/harness/index.ts` — the public API you import; everything else in `src/` is internal.
- `src/wgsl/*.wgsl` — shared WGSL includes (`#include "src/wgsl/frame.wgsl"` etc.).
- `mesh/README.md` — GCMESH1 binary format spec (the canonical source-mesh format).
- `experiments/renderers/000-ground-truth/` — brute-force reference render; A/B against it for visual fidelity (it is exempt from perf/VRAM rules; you are not).
