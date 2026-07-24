# billboard cards

## Idea

The classic billboard baseline, but over REAL baked imagery instead of tinted
quads. Precomputed per species (in-browser, once): a 3x3 atlas of 512px
orthographic captures of the raw GCMESH1 mesh — 8 side views at 45deg azimuth
steps plus 1 straight-down top view — as albedo+coverage (rgba8) and
oct-encoded mesh-frame normals (rg8), supersampled 2x, coverage-weighted
downsampled, color-dilated 6 texels so filtering never bleeds background.

Per frame: a compute pass evaluates the shared scatter WGSL twin over a
camera-centered cell region (clamped to the stand's exact cell range on the
CPU, so cells outside the stand are never dispatched), rejects whole cells
against the region circle and the frustum before touching the scatter, then
frustum-culls each surviving plant's bounding sphere and compacts survivors
into a 16B/instance buffer + indirect draw args. Each surviving plant draws 12
vertices: a cylindrical camera-facing side card that samples the baked view
nearest the current azimuth in the plant's yawed frame, and a horizontal top
card with the top-view bake that fades in only for steep viewing elevations
(it is what keeps the top-down cam dense; at eye level it would read as a
floating cutout, so it erodes away below ~35deg). Hard alpha test with depth
write — no dither, no blending; camera-inside and region-edge fades erode
coverage through the alpha reference (edges dissolve first). Wind = shared
wind include, tips displaced by height fraction. Lighting = baked normal
rotated by yaw into world, shared light_surface + fog, plus a small
bottom-shade gradient for grounding. All five debug views (albedo / normals /
lighting / coverage / depth) are wired through the shared `debug_shade()`.

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

## Audit (structural waste + debug views)

Debug views — the renderer honoured none of them (it always returned lit+fogged
colour). Now `cards.wgsl` includes `src/wgsl/debug.wgsl`, skips fog unless
`debug_mode() == DEBUG_OFF`, and returns
`debug_shade(color, alb.rgb, n, alb.a, in.world)`. What the modes exposed:

- normals: already real per-fragment normals (oct atlas decoded, rotated into
  world by the plant yaw) — verified rather than assumed. At `grazing` (camera
  at +x+z looking at the origin) near-field normals read pink = +x+z = pointing
  back at the camera, which is what the bake's flip-toward-the-view-camera
  convention should produce; `topdown` reads green = +y from the top card.
  Both are the expected answers, so the yaw rotation sign is right.
- lighting: goes through the shared `light_surface()` exactly once. Sampled
  linear values run ~0.14 (away from the sun) to 1.0+ (toward it) — it *looks*
  blown out only because the view is displayed in sRGB; nothing is unlit,
  double-lit, or premultiplied (the albedo view shows the raw baked atlas).
  `in.shade` (the grounding gradient) is deliberately counted as part of the
  light term, not albedo: it is fake occlusion, so `albedo` stays exactly the
  baked capture and `lighting` shows sun+ambient x grounding.
- coverage: the baked alpha the fragment resolved to. With the hard alpha test
  everything below `erode` is already gone, so the view doubles as the
  alpha-test margin — thin stems show up as the near-threshold greys.
- depth/albedo: correct, region edge clearly visible in depth at ~110m.

Structural fixes (all three verified image-identical: frozen-time
`det=1&t=3` screenshots at all four standard cams, old vs new, differ by the
same handful of speckles that two *identical* runs differ by — the instance
compaction order is `atomicAdd`-nondeterministic, so cards at tied depth win
the depth test arbitrarily):

1. Fully-eroded cards were rasterized anyway. `erode = alpha_ref / fade`, so
   any card with `fade < alpha_ref` has `erode > 1` and *every* one of its
   fragments discards — yet the quad was still rasterized and every fragment
   still did two texture samples before dying. That set is exactly the
   screen-filling one: all top cards at low elevations (the `smoothstep(0.35,
   0.6, elev)` erosion, i.e. every eye-level view), plus cards the camera is
   standing inside (which cover half the screen at `inside-plant`), plus the
   region rim. The vertex shader now emits those behind the near plane.
2. The cull dispatch was sized by the full square region regardless of the
   stand. The stand cell clamp lived in the shader, so on `close-quality`
   (±24m) ~96% of the 400k threads/species existed only to return immediately,
   and on `default` the same happens whenever the camera nears the stand edge.
   The rect is now clamped to the stand's cell range on the CPU (`side_x` /
   `side_z` replace the single `side`), and a zero-area region skips the
   dispatch entirely.
3. The cull evaluated the shared scatter — 4 heightmap texel loads + 7 hash
   rounds — for every candidate in the full 360° region, and only *then*
   frustum-tested. A workgroup is 64 threads and a cell holds 128 slots, so
   every workgroup covers exactly one cell: two workgroup-uniform cell-level
   rejects now run first (cell rect vs the region circle, and a conservative
   cell box vs the 6 frustum planes), and whole workgroups exit before a single
   texel is fetched. The box is exactly conservative: cell footprint in xz,
   `1.05 * frame.terrain_height_scale` in y (the shared FBM's octave amplitudes
   sum to 1.015) plus the tallest card at `scale_max` and the same +0.35 wind
   margin the per-plant test uses. The per-plant tests are unchanged and still
   exact, so the visible set is identical.
4. Minor: `frustumPlanes()` allocated a fresh `Float32Array(24)` every frame;
   it now fills a preallocated one.

Deliberately left alone:

- The `info` uniform (176 B x 3 entries) is rewritten every frame although only
  the planes, region rect and two params change. Splitting it into a constant
  and a dynamic buffer saves three tiny `writeBuffer`s per frame — not worth
  the field-order churn.
- `uploadAtlas()` builds its own mipgen BGL + 2 pipelines per species (3x
  identical). Init-time only, ~6 pipeline compiles at startup.
- No depth prepass and no front-to-back ordering for the cards. Sorting or a
  prepass might pay off at grazing angles where overdraw is deepest, but the
  win is not obvious from the code — needs a bench on an idle GPU.
- The instance buffer is recompacted every frame even when neither camera nor
  params moved. A dirty flag would skip it, but wind does not move plants, so
  correctness would hinge on catching every input — measure first.
- Normal mips are a plain box filter over *oct-encoded* values, so at extreme
  distance a species' normals drift toward one per-tile mean instead of the
  true average direction (visible in the normals view as the far field losing
  its point-at-the-camera character). Fixing that means decoding, averaging and
  re-encoding in `mipgen.wgsl` — a bake/quality change, out of an audit's
  scope, but it is the first thing to try if distant lighting looks flat.
