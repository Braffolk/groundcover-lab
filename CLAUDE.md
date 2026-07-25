# Conventions — read before touching anything

TypeScript only. No `.js`/`.mjs` files anywhere, including tools. All tooling is browser-first.

## Adding an experiment (the only sanctioned workflow)

1. `npm run new -- your-slug` → creates `experiments/<nnn>-your-slug/` from `_template`.
2. Edit ONLY files inside your experiment directory. Never renumber; your ID is the full directory name.
3. Run it: `npm run dev` → `http://localhost:5173/#/run/<your-id>`.

Allowed write paths: your experiment dir, `mesh/baked/<your-id>/`, `results/` (new files only), `goldens/<your-id>/`. Exception inside your own dir: `rating.json` is the OWNER'S visual verdict — never create, edit, or delete it.
Forbidden: `src/**`, other experiments, `package.json`, any shared file. If the harness is missing something you need, write the need into your `NOTES.md` instead of patching shared code.

## Experiment rules

- Import only from `@harness` (plus your own files). Set `harnessApi` in your manifest.
- The harness owns device/canvas/camera/terrain/scatter/wind and draws the terrain+sky base pass. You append compute/render passes via `ctx.timing.renderPass/computePass` (auto-timed) with `loadOp: 'load'` against the provided color/depth targets.
- Allocate GPU memory only via `ctx.res` (the VRAM tracker) — pass `{ species }` so the 25MB/species budget bar in the HUD is meaningful. The budget is a strong default, not an absolute: stay within it in ~95% of cases; a genuinely novel method may exceed it when the results justify it — document the why and the actual numbers in `NOTES.md`.
- Determinism: randomness only via the shared PCG hash (`@harness`), animation only from `frame.time`. Never `Math.random`/`Date.now` in render paths. The A/B diff view will expose you.
- **Never fetch root-absolute URLs.** Production builds are served under a base path (`/groundcover-lab/`), so `fetch('/mesh/baked/...')` breaks there. Load your baked artifacts through `bakedArtifact()`, and resolve any other asset URL with `assetUrl()` from `@harness`.
- **Debug views are mandatory, not optional.** There is ONE global selector (`frame.debug_mode`, the `view` dropdown in the runner/AB toolbar, URL `debug=`) with modes `off · albedo · normals · lighting · coverage · depth`. Every renderer must `#include "src/wgsl/debug.wgsl"` and route its final fragment through `debug_shade(shaded, albedo, normal_ws, coverage, world_pos)`, skipping fog unless `debug_mode() == DEBUG_OFF` (see `experiments/_template/shaders/main.wgsl`). A method whose normals or lighting cannot be inspected is a method nobody can trust — and if `normals` shows garbage or `lighting` is blown out, that is a bug in your renderer, not in the debug view. Real per-fragment normals and the shared `light_surface()` are required; unlit or uniformly bright output is a defect.
- **Experiments are RENDERERS of a stand, never placers.** The active stand (`ctx.stand`, URL `stand=<id>`, default `default`) is the harness-owned placement setup: which species, densities, scale ranges, sway, region. Stand + seed fully determines every plant instance via `ctx.scene.scatter` (TS buffers) or the bit-identical WGSL twin (`src/wgsl/scatter.wgsl` + `stand_table`). Render exactly the stand's plants — all its species entries, at their exact positions/scales. Your params may only affect HOW it is drawn, never what/where grows. Stands live in `src/scene/stands.ts` (shared code — don't touch); A/B and bench results are only comparable within one stand + seed.
- The only sanctioned exception is `status: 'reference'` (e.g. 000-ground-truth): stand-independent visual baselines, shown in a separate browser row and clearly labeled. Don't add more without a reason as good as "the raw mesh is a community tile and physically cannot follow per-plant placement".
- You MAY read the raw source mesh (`ctx` mesh catalog, GCMESH1) and invent any novel baked representation. Put baked artifacts in `mesh/baked/<your-id>/` via the bake flow; format is entirely yours.

## Silent-failure traps (each of these has already cost an experiment a wrong result)

- **`firstInstance` in indirect draws.** A non-zero `firstInstance` in `draw*Indirect` is only legal with the `indirect-first-instance` feature. The harness now requests it when the adapter offers it (`ctx` → `gpu.hasIndirectFirstInstance`), but where it is unavailable such a draw is **silently discarded** — the scene still looks plausible with whole batches missing, and it can even look faster. Prefer binding a per-batch slice of the instance buffer (256-byte aligned) over relying on `firstInstance`.
- **WebGPU validation errors do not throw.** They arrive via `device.onuncapturederror` and surface as an on-screen toast. The harness also mirrors every error toast to `console.error` before de-duplicating it, so a headless script that listens to `console` will see them — but if you scrape the page, read `.toast` elements too. "0 console errors" is not proof of correctness; look at the image.
- **Verify with a control.** If a change makes something faster, confirm it still draws everything (tint batches, sweep the LOD distance, count instances) before believing the number.
- **Octahedral normals are NOT mip-averageable.** Octahedral encoding is non-linear, and a blade's front/back faces encode to roughly opposite pairs — box-filtering them lands near (0,0), which decodes to *exactly* straight up. 017-cached-clusters lit its whole far field with an up-normal this way: flat, blown out, and worse with distance, while `debug=albedo` looked perfect. If you mip normals, store plain unit vectors flipped into one hemisphere (e.g. around the capture axis) and encode after filtering, or filter in a linear space. Check it: in `debug=lighting`, if the shaded mean equals the albedo mean your light term is averaging 1.0 and doing nothing.
- **Never derive a plant's width from `height_scale`.** It happens to look fine for tall grasses, but a ground carpet (Sphagnum: 0.07m tall, 0.24m wide) comes out ~3.5x too small and a mat that should be closed shows gaps. Use `stand_table[i].footprint_m` (the species' horizontal footprint at scale 1) or extents you measured in your own bake. Test against the `bog` stand, which is mostly carpet.

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
- `experiments/000-ground-truth/` — brute-force reference render; A/B against it for visual fidelity (it is exempt from perf/VRAM rules; you are not).
