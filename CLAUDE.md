# Conventions — read before touching anything

TypeScript only. No `.js`/`.mjs` files anywhere, including tools. All tooling is browser-first.

## Adding an experiment (the only sanctioned workflow)

1. `npm run new -- your-slug` → creates `experiments/<nnn>-your-slug/` from `_template`.
2. Edit ONLY files inside your experiment directory. Never renumber; your ID is the full directory name.
3. Run it: `npm run dev` → `http://localhost:5173/#/run/<your-id>`.

Allowed write paths: your experiment dir, `mesh/baked/<your-id>/`, `results/` (new files only), `goldens/<your-id>/`.
Forbidden: `src/**`, other experiments, `package.json`, any shared file. If the harness is missing something you need, write the need into your `NOTES.md` instead of patching shared code.

## Experiment rules

- Import only from `@harness` (plus your own files). Set `harnessApi` in your manifest.
- The harness owns device/canvas/camera/terrain/scatter/wind and draws the terrain+sky base pass. You append compute/render passes via `ctx.timing.renderPass/computePass` (auto-timed) with `loadOp: 'load'` against the provided color/depth targets.
- Allocate GPU memory only via `ctx.res` (the VRAM tracker) — pass `{ species }` so the 25MB/species budget bar in the HUD is meaningful. The budget is a strong default, not an absolute: stay within it in ~95% of cases; a genuinely novel method may exceed it when the results justify it — document the why and the actual numbers in `NOTES.md`.
- Determinism: randomness only via the shared PCG hash (`@harness`), animation only from `frame.time`. Never `Math.random`/`Date.now` in render paths. The A/B diff view will expose you.
- **Experiments are RENDERERS of a stand, never placers.** The active stand (`ctx.stand`, URL `stand=<id>`, default `default`) is the harness-owned placement setup: which species, densities, scale ranges, sway, region. Stand + seed fully determines every plant instance via `ctx.scene.scatter` (TS buffers) or the bit-identical WGSL twin (`src/wgsl/scatter.wgsl` + `stand_table`). Render exactly the stand's plants — all its species entries, at their exact positions/scales. Your params may only affect HOW it is drawn, never what/where grows. Stands live in `src/scene/stands.ts` (shared code — don't touch); A/B and bench results are only comparable within one stand + seed.
- The only sanctioned exception is `status: 'reference'` (e.g. 000-ground-truth): stand-independent visual baselines, shown in a separate browser row and clearly labeled. Don't add more without a reason as good as "the raw mesh is a community tile and physically cannot follow per-plant placement".
- You MAY read the raw source mesh (`ctx` mesh catalog, GCMESH1) and invent any novel baked representation. Put baked artifacts in `mesh/baked/<your-id>/` via the bake flow; format is entirely yours.

## Comparing & claiming results

- Compare: `#/ab/<idA>/<idB>?stand=default&cam=grazing&seed=42` (wipe/flicker/diff) — both sides render the SAME stand by construction. A/B timings are contended — never quote them.
- Bench before claiming numbers: `#/bench/<id>?stand=default&spline=orbit-low` at a standard cam/spline. Results record the stand and are only comparable within one; they auto-save to `results/` — link the JSON filenames in your `NOTES.md`. Scale tests use the `scaling-100m` stand, not custom placement.
- `NOTES.md` required sections: Idea / VRAM budget math / Bake / Status / Findings.
- Standard cameras (keys 1–4): `grazing` (impostor killer), `topdown`, `inside-plant` (fade check), `far-horizon` (scaling check).

## Repo map

- `src/harness/index.ts` — the public API you import; everything else in `src/` is internal.
- `src/wgsl/*.wgsl` — shared WGSL includes (`#include "src/wgsl/frame.wgsl"` etc.).
- `mesh/README.md` — GCMESH1 binary format spec (the canonical source-mesh format).
- `experiments/000-ground-truth/` — brute-force reference render; A/B against it for visual fidelity (it is exempt from perf/VRAM rules; you are not).
