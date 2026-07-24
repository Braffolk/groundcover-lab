# billboard cards

## Idea

The classic billboard baseline, but over REAL baked imagery instead of tinted
quads. Precomputed per species (in-browser, once): a 3x3 atlas of 512px
orthographic captures of the raw GCMESH1 mesh — 8 side views at 45deg azimuth
steps plus 1 straight-down top view — as albedo+coverage (rgba8) and
oct-encoded mesh-frame normals (rg8), supersampled 2x, coverage-weighted
downsampled, color-dilated 6 texels so filtering never bleeds background.

Per frame: a compute pass evaluates the shared scatter WGSL twin over a
camera-centered cell region (clamped to the stand's exact cell range),
frustum-culls each plant's bounding sphere, and compacts survivors into a
16B/instance buffer + indirect draw args. Each surviving plant then draws 12
vertices: a cylindrical camera-facing side card that samples the baked view
nearest the current azimuth in the plant's yawed frame, and a horizontal top
card with the top-view bake that fades in only for steep viewing elevations
(it is what keeps the top-down cam dense; at eye level it would read as a
floating cutout, so it erodes away below ~35deg). Hard alpha test with depth
write — no dither, no blending; camera-inside and region-edge fades erode
coverage through the alpha reference (edges dissolve first). Wind = shared
wind include, tips displaced by height fraction. Lighting = baked normal
rotated by yaw into world, shared light_surface + fog, plus a small
bottom-shade gradient for grounding.

O(1) in plant count: nothing is ever materialized for the whole stand — cost
is region area (regionRadius param, default 110m, cap 128m), identical for
557k (default) and 134.2M (scaling-100m) plants. Verified: both stands render
at the same ~5.8ms GPU p50 on this machine, same VRAM.

## VRAM budget math

Per species (HUD-verified, MiB):
- albedo atlas 1536x1536 rgba8 + full mip chain: 12.0 MiB
- normal atlas 1536x1536 rg8 + mips: 6.0 MiB
- culled instance buffer, sized for regionRadius max 128 at the entry's
  density (16 B/instance): density 3 -> 3.3 MiB, 2.5 -> 2.75 MiB
- entry info uniform (176 B) + indirect args (16 B): noise

Totals on the default stand: calamagrostis 21.3/25, elymus 20.7/25,
poa 21.3/25 MiB. Worst defined stand (dense-mixed, density 5) fits at
~23.5 MiB. Bake uses ~200-400MB transient GPU memory (vertex+index buffers of
the raw mesh, 3072^2 targets, readbacks) — allocated via ctx.res tagged
`bake-scratch`, destroyed at the end of the bake.

## Bake

`bake.ts` -> `mesh/baked/001-billboard-smoke/cards-v1-<species>.bin`,
13.5 MiB each (64B header + 1536^2 rgba8 albedo + 1536^2 rg8 oct normals).
Flow is the harness bakedArtifact/commitBake infra (OPFS cache, committed
file, auto-commit after a fresh bake). Every load is magic-validated because
the dev server answers missing /mesh/baked files with index.html at 200,
which would otherwise poison both the OPFS cache and the committed-file path;
a poisoned entry is rebaked and the OPFS entry repaired in place. Full
first-run bake of all three species (incl. fetching 374MB of raw meshes) took
~10s on this machine; subsequent loads read the committed artifacts.

Normals are flipped toward the bake camera per view (two-sided foliage), so
cards light consistently from every azimuth. Mips are generated at load time
on the GPU (coverage-weighted for albedo).

## Status

working — verified by headless screenshots at grazing, topdown, inside-plant,
far-horizon (default stand) and grazing on scaling-100m, zero console errors,
plus a t=2 vs t=5 diff confirming deterministic wind sway.

## Findings

- All three species read distinctly from the baked imagery: calamagrostis =
  pale pink fluffy panicles over green stems, elymus = darker upright spikes,
  poa = short fresh-green tufts. Ground truth (000) at grazing shows the same
  green-stems + pink-heads character for the calamagrostis patch.
- The top card is essential for the topdown cam (side cards vanish edge-on)
  but is the technique's weakest artifact: at mid elevations it pops in as
  pale discs (its imagery is brighter than the side views because straight
  down you mostly see fluff). Restricting it to elevation > ~35deg (vs the
  card's own height) removed the "floating pancake" artifacts at inside-plant
  and the pale flecks at far-horizon while keeping topdown dense.
- Camera-inside fade via alpha-reference erosion works: the inside-plant cam
  shows the surrounding stand, not a texture wall.
- Known billboard limits, left as honest baseline behavior: 8-view azimuth
  snap pops when orbiting a single plant; close-up cards go flat and soft
  (512px tiles magnified); plants beyond regionRadius (cap 128m) erode out in
  a fog-assisted band, so far-horizon coverage ends at ~110m by default.
- GPU p50 on this machine, default stand, 1280x800: cull 0.4-0.5ms + cards
  1.3ms (topdown) to 3.1ms (inside-plant); scaling-100m identical to default.
  No `results/` bench JSONs are claimed: other agents were rendering in
  parallel during this session, so bench numbers would be contaminated
  (CLAUDE.md rule). Rerun `#/bench/001-billboard-smoke?stand=default` on an
  idle GPU before quoting numbers.
- Harness wishlist: none — bakedArtifact was sufficient once magic-validated;
  a dev-server 404 (instead of SPA fallback) for missing /mesh/baked files
  would let experiments drop the validation shim.
