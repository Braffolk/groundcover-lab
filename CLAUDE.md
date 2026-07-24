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
- Allocate GPU memory only via `ctx.res` (the VRAM tracker) — pass `{ species }` so the 25MB/species budget bar in the HUD is meaningful. The budget is a hard rule from the README.
- Determinism: randomness only via the shared PCG hash (`@harness`), animation only from `frame.time`. Never `Math.random`/`Date.now` in render paths. The A/B diff view will expose you.
- Plant placement must come from the shared scatter service (TS buffers or the bit-identical WGSL twin `src/wgsl/scatter.wgsl`) so all experiments place every plant identically.
- You MAY read the raw source mesh (`ctx` mesh catalog, GCMESH1) and invent any novel baked representation. Put baked artifacts in `mesh/baked/<your-id>/` via the bake flow; format is entirely yours.

## Comparing & claiming results

- Compare: `#/ab/<idA>/<idB>?cam=grazing&seed=42` (wipe/side-by-side/flicker/diff). A/B timings are contended — never quote them.
- Bench before claiming numbers: `#/bench/<id>?spline=orbit-low` at a standard cam/spline. Results auto-save to `results/`; link the JSON filenames in your `NOTES.md`.
- `NOTES.md` required sections: Idea / VRAM budget math / Bake / Status / Findings.
- Standard cameras (keys 1–4): `grazing` (impostor killer), `topdown`, `inside-plant` (fade check), `far-horizon` (scaling check).

## Repo map

- `src/harness/index.ts` — the public API you import; everything else in `src/` is internal.
- `src/wgsl/*.wgsl` — shared WGSL includes (`#include "src/wgsl/frame.wgsl"` etc.).
- `mesh/README.md` — GCMESH1 binary format spec (the canonical source-mesh format).
- `experiments/000-ground-truth/` — brute-force reference render; A/B against it for visual fidelity (it is exempt from perf/VRAM rules; you are not).
