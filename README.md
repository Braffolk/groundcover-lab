A fast method for rendering overlapping multi-species varying height with wind etc effects of groundcover using fake geometry (3D obviously). 

Rules:
- No raycasting/raymarching - precomputed raycast is allowed - O(1)
- Targeted for low-mid end devices
- No actual geometry is handled per frame (processed or rendered)
- Max VRAM extra usage per species of groundcover is 25MB (may increase on ultra detail setup). Treat as a strong default, not an absolute: stay within it in ~95% of cases, but a novel method may exceed it if it delivers great results — document why in the experiment's NOTES.md.
- Wind swaying (varying by species, e.g none for moss)
- Proper normals, lighting, etc effects applied on fake geometry
- You may borrow ideas from things like impostors etc
- Scaling number of plants should cost basically nothing (or grass blades for that effect).
- Must render properly from all angles
- You may limit things so that the effect breaks down when camera is inside of a plant (fade it out in that case)
- WebGPU.
- Must support arbitrary complexity input meshes (transformed into whatever will actually be used during rendering). An example mesh of 2m tris has been provided: `mesh/raw/calamagrostis-canescens/` (binary format documented in `mesh/README.md`). It is a complex Calamagrostis canescens plant with fluffy heads and varied colors — deliberately error-revealing, use it as the primary test mesh.
- The rendering must cost nothing for the amount of plants being rendered. 100 and 1 billion should both be doable on a random mid machine (hence the cost is never per plant).

- TypeScript everywhere — no JS anywhere. All tooling, scripts, and configs are TypeScript (run via tsx); no `.js`/`.mjs` files in the repo.
- All tooling runs and is explorable in the browser: experiment browser, single-experiment runner, A/B comparison view with an exactly-shared camera, benchmark runner and results viewer. Every view (experiment, camera pose, params, seed, time) is encoded in the URL, so any state is a shareable, reproducible link.
- Experiments MAY consume the raw mesh binary directly and invent their own completely unique/novel ways of storing and representing single plants. Designing smart formats is not the project goal, but for some experiments it is legitimately part of the technique.
- Placement is standardized: harness-owned "stands" (placement setups — species mix, densities, scales, region) fully determine every plant instance; experiments are pure renderers of the selected stand and never define placement. A/B and bench numbers are only comparable within one stand + seed.

This task requires out of the box brilliant graphical engineering thinking - proxies, approximations, barely known trigonometry hacks etc.

Build different experiments in different subdirectories. Testable in browser. Work must be duable for parallel subagents, each working on a different experiment. Needs exact frame times to be able to compare results (not just the quality of effect).

See `CLAUDE.md` for the exact recipe for adding an experiment without touching shared code.

## Static deploy

The lab builds to a plain static site (S3 + CloudFront, GitHub Pages, any file host).
It is hash-routed (`#/run/<id>`, `#/ab/...`, `#/bench/...`), so **no server rewrite rules are needed** —
one `index.html` is enough.

```bash
npx vite build                      # → dist/, ~1.5GB if mesh/baked is populated locally
GC_DEPLOY_MESHES=1 npx vite build   # → dist/, larger still (adds the raw GCMESH1 source meshes)
npx vite preview --port 5199        # serve dist/ locally at /groundcover-lab/
```

> **Baked artifacts are NOT in git** (`mesh/baked/**/*.bin|png` is ignored, as are
> the raw meshes). They are regenerated per experiment on first run, in-browser
> and OPFS-cached; `commitBake` only writes them to disk so a local session can
> skip the wait. The static-deploy plugin copies whatever is on disk, so a build
> is ~1.5GB on a machine that has baked everything and small on a fresh clone —
> a deploy that wants the artifacts has to bake them first.


- **Base path.** Production builds are served under `/groundcover-lab/` (`base` in `vite.config.ts`);
  the dev server stays at `/`. Runtime asset URLs go through `assetUrl()` (`src/util/paths.ts`), which
  resolves against `import.meta.env.BASE_URL` — use it (exported from `@harness`) for any hand-written
  fetch of `/mesh/...`, `/experiments/...` or `/results/...`. To deploy at a different path, change
  `base`; to deploy at the domain root, set it to `/`.
- **What gets uploaded** (`tools/vite-plugin-static-deploy.ts` copies these into `dist/`):
  the JS/CSS bundle (~1MB), `mesh/baked/**` (~255MB of committed baked artifacts — without them a
  visitor would have to re-bake from raw meshes in-browser), `experiments/**/thumbnail.png` +
  `rating.json` (~9MB, found by walking the experiment tree at any depth),
  `results/*.json` plus a generated `results/index.json`, and
  `mesh/raw/*/manifest.json`. `goldens/` is never uploaded (nothing fetches it at runtime).
- **Raw source meshes are OFF by default.** They are ~357MB (poa-pratensis alone is 229MB) and only
  the `#/mesh/<id>` inspector and reference renders need them. Build with `GC_DEPLOY_MESHES=1` to
  include them; without them the mesh cards say "source .bin not deployed" and the inspector shows a
  plain "source mesh not included in this deployment" message instead of failing.
- **WebGPU requires a secure context**, i.e. HTTPS (or localhost). An S3 *website* endpoint is
  HTTP-only and will therefore **not** work — use the S3 REST endpoint over TLS
  (`https://<bucket>.s3.<region>.amazonaws.com/groundcover-lab/index.html`) or, better, put CloudFront
  in front of the bucket. Serve `.bin` as `application/octet-stream` and enable compression for
  JS/CSS/JSON only (the `.bin` artifacts are already dense).
- **The deployed site is read-only.** The `/__bench`, `/__thumb`, `/__bake`, `/__rating` endpoints are
  dev-server middleware and do not exist in a build, so: rating pips display the owner's verdict (and
  still drive the visual/balance sorts) but cannot be clicked, the runner has no thumbnail/golden
  capture buttons, a bench run downloads its JSON instead of saving it (drop it onto `#/results` to
  compare), and bakes stay in the browser's OPFS cache instead of being committed. Everything else —
  browsing, running experiments, A/B, benching, the results table — works exactly as it does locally.
