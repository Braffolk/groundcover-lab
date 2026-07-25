# 029 canopy patches — depth-slab patch stacks

## Idea

A billboard is one flat card at the plant's axis: every pixel of the plant sits
at the same depth, so when the camera moves the whole plant slides rigidly and
the silhouette merely rotates. This experiment keeps everything that makes the
billboard baseline cheap and replaces the ONE card with a small **stack of
baked canopy patches**, each of which is a real surface in the world:

- **2 depth slabs.** The bake splits the plant along each of 8 azimuth view
  axes at the depth MEDIAN of its foliage and captures the front half and the
  back half separately, each framed tightly on the foliage it contains. At
  runtime the two slabs are two quads perpendicular to the *bake azimuth* —
  **not** to the camera — placed at the coverage-weighted mean depth of their
  own content (±0.08…0.21 m for calamagrostis, ±0.3 m for elymus). Because the
  planes are fixed in the plant's frame, everything is real: the front slab
  slides across the back slab as the camera moves inside the ±22.5° bin (true
  motion parallax inside the silhouette), the union outline changes shape
  instead of rigidly rotating, and the front slab depth-occludes the back one
  through the ordinary depth test. No `frag_depth` anywhere, so early-z lives.
- **2 crown patches.** Two horizontal quads carrying the straight-down capture
  of the upper (>60% height) and lower height bands, each at the mean height of
  its band (≈0.95 m / ≈0.45 m for calamagrostis). They give real vertical
  parallax from above and they are what fills a steep view. They are gated on
  *elevation over their own height* (upper 0.50→0.72, lower 0.70→0.88), which is
  what stops the baseline's "floating pale pancake" artifact.
- **The distance LOD is the collapse of the stack, not a crossfade.** As a
  plant approaches `patchDist` (10 m), `blend` slides the slab depths to zero
  and rotates the planes to face the camera; at the switch distance the slabs
  are coincident camera-facing quads whose union IS the composite image, so the
  handover to the single composite card is continuous. Beyond `patchDist` a
  plant is exactly a billboard — one camera-facing card + one crown card — out
  of a deliberately tiny 176² atlas (a plant at 10 m is ~95 px tall, so that is
  already more texels than the screen can use). Cost per far plant is therefore
  the baseline's cost, and only the near band pays for layers.
- **The same volume rule at the other end.** The multiplane approximation is
  only valid outside the plant's bounding volume, so the slab spread also
  collapses over `1.2r → 2.6r` of the plant radius: a plant the camera is about
  to walk into degrades into exactly the baseline's flat card before the
  camera-inside fade erodes it. (Without this the front slab is pushed toward a
  camera 0.4 m away and magnifies into soft blobs — very visible in the first
  build.)

Hard alpha test + depth write everywhere; no dither, no blending. Fades (camera
inside a plant, region rim, crown elevation) erode the alpha reference, so
coverage dissolves from the thinnest texels inward, and a quad whose fade drops
below the reference is emitted behind the near plane instead of rasterized.

Draw order is **quad-major**: all front slabs of all species, then all back
slabs, then the crowns, then the far cards. The front slabs prime the depth
buffer before any deeper quad is rasterized, which is what keeps the extra
layer nearly free at grazing angles.

Plant-count independence is the baseline's: one compute pass evaluates the
shared scatter WGSL twin over a camera-centered cell region (CPU-clamped to the
stand's cell range), rejects whole cells against the region circle and the
frustum before touching the scatter, frustum-culls survivors and compacts them
into two contiguous ranges (patch-stack / composite-card) of one instance array
(12 B per plant: xz as 2×i16 around a per-frame origin, y as f32, yaw/scale/
phase packed into one u32). A single 64-thread dispatch then fans the two
counters out into the 6 indirect draw slots per stand entry. Verified: the
default stand (557k plants) and `scaling-100m` (134.2M plants) render at the
same cost (Σp50 5.89 vs 6.00 ms, same VRAM, same frame shape).

Wind is the shared `wind_sway` weighted by each vertex's own height fraction,
so slabs and crowns lean together and stay attached. Lighting is the baked
mesh-frame oct normal rotated by the plant yaw *plus* however far the plane was
turned away from its bake azimuth, flipped toward the viewer (two-sided
foliage), through the shared `light_surface` + fog. All debug views are routed
through `debug_shade()`.

## VRAM budget math

Per species, HUD-verified (25 MiB budget):

| item | bytes |
|---|---|
| near albedo array, 448², 18 layers (8 az × 2 slabs + 2 crowns), 5 mips, rgba8 | 19.25 MB (18.4 MiB) |
| near normal array, 224², 18 layers, 4 mips, rg8 (oct) | 2.40 MB (2.3 MiB) |
| far albedo array, 176², 9 layers (8 az + 1 crown), 7 mips, rgba8 | 1.49 MB (1.4 MiB) |
| far normal array, 88², 9 layers, 6 mips, rg8 | 0.19 MB (0.2 MiB) |
| culled instances, 12 B × (18 366 near + 181 700 far) at density 3 | 2.40 MB (2.3 MiB) |
| info uniform (1 072 B) + shared counters/args (~0.4 kB) | noise |

Totals on the default stand: calamagrostis **24.5/25 MiB**, elymus 24.2,
poa 24.5. Normals are stored at HALF the albedo resolution deliberately —
coverage/silhouette crispness is what the eye reads on grass, normals are
low-frequency by comparison — and that is what pays for 448² slabs (380
texels/m vertically, against the baseline's 445) instead of ~300.

On the heaviest defined stand (`dense-mixed`, density 5) the instance buffer
grows to 3.96 MB and the species row lands at ~26.0 MiB, i.e. marginally over
budget at a non-default setting; the default, `close-quality`, `calamagrostis-pure`
and `scaling-100m` stands all sit at 24.2–24.5 MiB. Trimming the far capacity to
the frustum wedge (it is currently the full region circle) would fix that at the
cost of silently dropping plants under a very wide fov, which is not a trade I
wanted to make by default.

Instance capacity is sized for the *region*, never the stand:
`π·R²·density·1.15` at the region cap (128 m) for the far bucket plus
`π·patchDist_max²·density·1.15` for the near one. Overflow is clamped in the
cull (the count stays at capacity), so a pathological camera thins the far rim
instead of corrupting anything.

Bake-only transient allocations (raw vertex/index buffers up to ~175 MB for
poa, two 3584² MRT targets + depth, two readbacks) go through `ctx.res` tagged
`bake-scratch` and are destroyed at the end of the bake.

## Bake

`bake.ts` → `mesh/baked/029-canopy-patches/patches-v2-<species>.bin`,
17.5 MB each (128 B header + 27 slice records + four image blocks).

Per species, in-browser, once (harness `bakedArtifact`/`commitBake`, OPFS
cached, magic+size validated because the dev server answers a missing
`/mesh/baked` file with `index.html` at status 200):

1. **CPU analysis** over the decoded vertices: exact horizontal support radius;
   per azimuth, the depth of every vertex along that view axis, a 2048-bin
   histogram for the median cut, then per slab the tight (u, v) bounding rect
   and the mean depth. Crowns split the height range geometrically at 0.6 — a
   *count* quantile puts both crowns up in the panicles (grass meshes carry
   most of their triangles there) and leaves no vertical parallax between them.
2. **27 orthographic captures** (18 near + 9 far) at 2× supersampling into
   3584² MRT atlases in chunks, one viewport + dynamic-offset uniform per
   slice. The fragment shader discards outside the slab BEFORE the depth test,
   so each slice records the frontmost surface *inside its own slab* — the
   slabs are a partition of the plant, not depth peels (peels would put
   far-away surfaces on a near plane).
3. **CPU reduction**: coverage-weighted 2× downsample for albedo, 4× for
   normals, 5 dilation passes into empty texels (alpha stays 0) so filtering
   and mip generation never pull in background black, oct-encode, pack.
4. Mip chains are generated on the GPU at load, one pass per (array layer, mip
   level). Every slice is its own array layer, so there is no neighbouring tile
   to bleed into and no UV-inset hack is needed.

## Status

**working** — verified by headless screenshots at `grazing`, `topdown`,
`inside-plant`, `far-horizon` on the default stand, plus `scaling-100m`,
`close-quality`, `calamagrostis-pure`, `dense-mixed`, `t=3` vs `t=7` (wind), all
five debug views, and the A/B page against 001. Zero console errors or warnings.
Contended solo Σp50: 4.0 ms on `calamagrostis-pure`, 5.9 on `default`, 6.0 on
`scaling-100m`, 8.0 on `dense-mixed` (7.7M plants).

## Findings

**Speed (contended — 15 sibling agents were rendering on this GPU, so only
ratios measured in the SAME A/B frame are quoted, never absolute ms).**
`#/ab/001-billboard-smoke/029-canopy-patches?cam=…`, comparing
(A/cull + A/cards) with (B/cull + B/patches):

| cam | ratio B/A (two independent runs) |
|---|---|
| grazing | 1.29× / 1.34× |
| far-horizon | 1.39× / 1.36× |
| topdown | 1.15× / 1.15× |
| inside-plant | 1.36× / 1.33× |

So ~1.3× the baseline's own passes — inside the "modestly slower is acceptable"
band and under the 1.5× ceiling. Of that, ~1.2× is the far path itself
(measured with `b.patchDist=0`, which turns B into a pure billboard renderer:
still 1.21×, from the two extra flat varyings and 18 draw slots instead of 6),
and only ~0.1× is the actual layering. Solo Σp50 including the harness
base+composite passes: grazing 5.9 ms, topdown 4.2, far-horizon 7.7,
inside-plant 8.4 — under the 9 ms bar even while contended. `patchDist` is the
one perf knob: 13 m costs 1.41× at grazing, 10 m (default) 1.29×, 9 m 1.21×
(i.e. the layering becomes free).

No `results/` bench JSON is claimed: benching against a GPU shared with 15
other agents would be fiction. Rerun `#/bench/029-canopy-patches?stand=default`
on an idle GPU before quoting numbers.

**Looks — honest verdict.** Better, but not by a landslide, and the win is
mostly in motion and up close:

- **It genuinely has 3D structure.** With `patchTint` on you can watch the
  front slab (red) and back slab (blue) of neighbouring plants interleave in
  depth, each occluding the other through the depth test; the depth debug view
  shows per-slab depth *inside* a single plant instead of one flat sheet. The
  parallax is geometric, not a shader trick: front-slab content is genuinely
  0.1–0.3 m closer to the camera than back-slab content, so it moves faster on
  screen, and the union silhouette changes shape as the view azimuth sweeps
  through a bin.
- **Close range (1–4 m) is where it clearly beats the baseline**: plants read
  as tufts with foliage in front of and behind other foliage, and ground
  contact is layered instead of a single card edge. At the same range the
  baseline is visibly a set of posters.
- **At the standard cams the two are close in a still.** Mean RGB over the
  far-horizon frame: baseline (76, 70, 40) vs this (66, 63, 35) — this build is
  ~10% darker, about half of that the per-slab canopy darkening (`slabShade`,
  default 0.10) and the rest lighting from half-resolution normals. Panicle
  density, silhouette character and horizon coverage match the baseline closely
  (they did NOT until bug 1 below was fixed).
- **Where the baseline still wins:** its 512² card is 1.2× denser in texels
  than the 448² slabs, so nose-to-plant it is slightly crisper; and 8 azimuth
  bins still pop per plant when orbiting, exactly as the baseline does. The
  slab parallax shrinks the pop (both neighbouring reconstructions have already
  moved their content toward the true view) but does not remove it.

**Bugs worth recording** (all found by looking at pixels, not by reasoning):

1. *The indirect-args fan-out silently drew the wrong things.* `args.wgsl` kept
   a stale `QUADS = 7` after the quad layout dropped to 6, so slot→(entry,
   quad) decoding drifted by one per entry: the far side card was issued with
   the NEAR instance count (a handful of plants) and every far plant instead
   drew its crown card as a vertical quad. The far field turned into a fine
   green lawn texture (top-down imagery on side cards) and lost every pink
   panicle — and it made the method look ~20% *cheaper* than it is. The quad
   count is shared by three files; it is now named and cross-referenced in all
   of them.
2. *Uniform array sizes must match the data, not just be "big enough".* The
   WGSL `slice_rect`/`slice_depth` arrays were left at 35 entries after SLICES
   became 27, which silently shifted every slab depth by 8 slices (plausible
   magnitudes, wrong values) and put the crown planes at ground level.
3. *Never push a slab toward a camera that is inside the plant* — see the
   volume collapse above.
4. *Mip/alpha-test coverage loss was a red herring here.* Measured directly on
   the artifacts: this 176² far slice and the baseline's 512² tile have the same
   panicle-band pass fraction at alpha 0.4 (0.095 vs 0.094), the same surviving
   colour (164,121,118 vs 163,120,116), and their coverage decays identically
   per halving. The far field's colour problem was bug 1, not resolution.
5. *Crown patches are the artifact factory.* With the baseline's single
   0.35→0.6 elevation gate, a *pair* of horizontal patches reads as brown
   pancakes lying through nearby plants at eye level, and from below (camera
   inside the canopy) it renders top-down imagery as dark radial stars. Gating
   on signed elevation, harder for the lower crown, removed both.

**Interpretation of the seed ("tileable canopy patches that cover many
plants").** A patch covering many *stand plants* cannot reproduce the stand's
exact per-plant placement, and the one previous experiment that tried it (009,
an aggregate canopy field) is the worst-rated in the repo. So "patch" here
means what the source assets actually are: two of the three species are
*periodic community tiles*, so one stand instance already is a patch of dozens
of blades. The patches are tileable in the sense that matters at runtime — one
6-vertex quad topology instanced everywhere, whose slabs abut seamlessly and
whose union reconstructs the plant — and the "covers many plants" amortisation
shows up as the LOD: beyond `patchDist` the whole stack has collapsed into a
single card sampling an atlas 6× smaller.

**Harness wishlist:** none. `bakedArtifact` + texture arrays covered
everything; a dev-server 404 (instead of the SPA fallback) for missing
`/mesh/baked` files would let experiments drop the validation shim.
