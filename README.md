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

This task requires out of the box brilliant graphical engineering thinking - proxies, approximations, barely known trigonometry hacks etc.

Build different experiments in different subdirectories. Testable in browser. Work must be duable for parallel subagents, each working on a different experiment. Needs exact frame times to be able to compare results (not just the quality of effect).

See `CLAUDE.md` for the exact recipe for adding an experiment without touching shared code.
