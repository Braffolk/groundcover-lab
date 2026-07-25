# slice prisms

## Idea

A billboard is one flat card at the plant's axis. **Slice prisms** cut the
plant into three vertical slabs and give each slab its own card, placed where
that slab's geometry actually is.

Per baked azimuth the bake projects the mesh orthographically and splits it by
**equal-mass depth quantiles** into three slices — so each slice carries a
third of the geometry, not a third of the bounding box — and records each
slice's **centroid depth** along the bake axis. At draw time, prism *k* is a
camera-facing quad whose centre is

```
axis + azimuth_dir(view_az, plant_yaw) * slab_centroid_depth[k] * scale
```

The offset is along the **baked azimuth direction**, not along the current view
ray. That single decision is what makes the plant read as a solid object:

* **Rotational parallax.** Whenever the view is off the baked azimuth by δ, the
  offset has a lateral component `c_k·sin δ`, so the three layers *slide across
  each other* as the camera orbits. Camera-facing offsets along the view ray
  would give zero lateral motion.
* **Smaller view snap.** The residual impostor error for a point at depth `c`
  becomes `(c − c_k)·sin δ` instead of `c·sin δ`. With three equal-mass slabs
  that is about a **third** of a billboard's error, so the 45° azimuth switch
  pops correspondingly less.
* **Perspective opening.** The front prism is genuinely nearer, so it magnifies
  faster than the rear one as you approach — the plant opens up.
* **Real depth, real occlusion, real contact.** Each prism writes depth at its
  own distance. Prisms occlude each other, neighbouring plants interleave
  *between* one plant's layers, and the plant touches the ground along three
  lines spread across its footprint instead of one line.
* **Canopy shading.** A front-to-back occlusion term (`canopyShade`, centred on
  1.0 so mean luminance is unchanged) makes the front layer read hotter than
  the sky-starved interior. This is the cue that sells volume in stills.

**Exclusive ownership** is the non-obvious part. A hard alpha test is
non-linear: if a texel's coverage is split over two prisms, both halves can
fail the test that the merged texel would have passed, and the plant comes out
*thinner and darker* than the billboard. First version did exactly that and
looked worse than the baseline. So each final texel is assigned to exactly ONE
prism (the one owning most of its front-most-covered subsamples) and given the
**merged** colour and coverage. The three prisms are then a perfect partition
of the billboard image — identical silhouette, identical colours, every texel
drawn once — but each texel at its own depth in the world. That is also why the
LOD switch is invisible.

Top-down views get the same treatment rotated 90°: three **horizontal** height
slabs, each a card at its own baked centroid height. They are gated on the
camera being above the *canopy top* (not above the plant's middle), so an
eye-level camera never sees edge-on lids.

**LOD.** Inside `slabDist` (16 m at scale 1): 3 vertical + 3 horizontal prisms.
Outside: one merged card + one merged top card — which is *exactly* a
billboard, at the same cost. The prism separation ramps to zero over the last
30% of the range, so at the boundary the three prisms are coplanar and their
union is bit-for-bit the merged tile. Nothing pops.

**Why it is O(1) in plant count.** One compute pass evaluates the shared
scatter over a camera-centred cell region (cell-level region + frustum rejects
are workgroup-uniform), frustum-culls, and compacts survivors into **two**
indirect draws. The expensive prism list is bounded by a disc of radius
`slabDist`, so it does not grow with the region *or* the stand: `scaling-100m`
(134 M plants) measures *cheaper in absolute ms* than `default` (557 k) and at
the same A/B ratio. No per-fragment loops, no marching, exactly **two texture
taps** per fragment, hard alpha test with depth write, front prism first so
early-z rejects what it covers, no `frag_depth`, no dither anywhere.

## VRAM budget math

Atlas: **1536×1536**, laid out as an 8×8 grid of 192 px cells so texels go
where the eye is — deliberately non-uniform:

| tiles | size | what |
|---|---|---|
| 12 | 384 px | 4 baked azimuths × 3 vertical depth slabs |
| 3 | 192 px | 3 horizontal height slabs (top-down) |
| 8 | 192 px | merged card per **view** azimuth (far LOD) |
| 1 | 192 px | merged top card |

24 tiles, 60 of 64 cells used.

*Mirroring* is what pays for the 384 px prism tiles. View azimuth *a*+180°
sees the same slabs with u flipped, slab order reversed and depth negated —
exact except for which surface wins inside one thin slab. So 4 azimuths of
storage serve 8 view directions, i.e. the same 45° switch interval as the
billboard baseline at 1.5× its linear tile resolution for the prisms. The
mesh-frame normal is mirrored with `n.xz = -n.xz` (the geometric mirror
composed with re-facing the normal at the opposite camera), which deliberately
leaves `n.y` alone so the hemisphere ambient does not step across the 180° seam.

Per species:

```
albedo rgba8  1536² × 4 B × 1.333 (7 mips)  = 12.58 MB
normals rg8   1536² × 2 B × 1.333           =  6.29 MB
far instance buffer  160,583 × 16 B         =  2.57 MB   (default stand)
prism instance buffer  8,682 × 16 B         =  0.14 MB
info uniform 1 KB + indirect 32 B           =  ~0 MB
                                              --------
                                              21.6 MB  / 25 MB
```

Measured in the HUD and the bench JSON: **22.5 / 21.9 / 22.5 MB** for
calamagrostis / elymus / poa (the baseline billboard sits at 22.3 / 21.8 /
22.3 MB — the same footprint). Worst stand is `dense-mixed`, 23.8 MB, still
under. 7 mip levels is not an accident: 192 = 64·3, so every tile boundary
stays texel-aligned through level 6 and the box-filtered mip chain **never**
mixes two tiles.

## Bake

`bake.ts` → `mesh/baked/024-slice-prisms/prisms-v3-<species>.bin` ("SLP2"
format, version 3, 13.5 MB each, 41 MB total). Reads the raw GCMESH1 mesh via
`ctx.meshes`, then:

1. Exact horizontal support radius from the vertices; capture box `y0..y1`.
2. Per baked azimuth, a 512-bin depth histogram over subsampled vertices →
   equal-mass cut planes + slice centroid depths. Same for height slices.
3. **15 orthographic draws** (12 depth slices + 3 height slices) at 2× into one
   3072² pair of targets, each slice trimmed per fragment against its own cut
   planes so triangles straddling a plane are cut exactly.
4. CPU resolve: exclusive per-texel ownership for the prism tiles; front-most
   covered subsample for the 9 merged tiles (that *is* the z-buffered result,
   so 9 extra render passes are saved and the merged tiles are guaranteed to
   equal the union of the prisms).
5. Dilate 5 passes into empty texels (tile-clamped) so filtering never pulls in
   background black; per-tile tight coverage box stored twice, as atlas UV
   (what to sample) and tile-local UV (what geometry it maps to), so a prism
   only rasterizes where its slice exists.
6. Mip chain generated on the GPU at load (`mipgen.wgsl`), 7 levels.

## Status

**working** — seen working headless at all four standard cameras, all five
debug views, and on `default`, `close-quality`, `dense-mixed` and
`scaling-100m`, with zero console errors.

## Findings

### Speed — A/B ratio (contended; absolute ms deliberately not quoted)

`#/ab/001-billboard-smoke/024-slice-prisms?stand=default&seed=42` — both sides
render the same stand in the same frame, so only the ratio is meaningful. Sum
of `B/*` p50 over sum of `A/*` p50, several samples each:

| cam | ratio (mine / billboards) |
|---|---|
| grazing | 1.19×, 1.23× |
| far-horizon | 1.13×, 1.13× |
| topdown | 1.14× |
| inside-plant | 1.13× |
| `dense-mixed` grazing | 1.39× |
| `close-quality` grazing | 1.45× (whole stand is inside slabDist — no far LOD at all) |
| `scaling-100m` grazing | 1.33×, and *lower absolute* ms than the 557 k default stand |

So **~1.15–1.25× the champion on the standard stand**, worst case 1.45× on a
±24 m plot where every plant is a prism. Comfortably inside the 1.5× bar and
nowhere near the 9 ms ceiling. Cost is flat in `slabDist` from ~9 m upward
(measured 1.34× at 9/14/20 m before the exclusive-ownership fix), which is why
the default is 16 m rather than 9 — the range is free, so take it.

Solo bench (`#/bench/024-slice-prisms?stand=default&spline=orbit-low`), run
while 15 sibling agents shared the GPU, so treat these as an order of
magnitude only:

* `results/024-slice-prisms__default__p-6cbe0a1b__apple-metal-3__2026-07-25T03-22-07-366Z.json` — Σp50 **4.54 ms**
* `results/024-slice-prisms__default__p-6cbe0a1b__apple-metal-3__2026-07-25T03-36-03-845Z.json` — Σp50 **7.64 ms** (much heavier contention)
* `results/001-billboard-smoke__default__p-86e4907a__apple-metal-3__2026-07-25T03-23-45-088Z.json` — Σp50 **3.85 ms** (baseline, paired with the first)
* `results/001-billboard-smoke__default__p-86e4907a__apple-metal-3__2026-07-25T03-36-45-587Z.json` — Σp50 **5.19 ms** (baseline, paired with the second)

Both stay under the 9 ms ceiling even in the contended pair.

### Looks — honest verdict

**Yes, it beats the billboard baseline, clearly at some cameras and mildly at
others.** Evidence from the A/B wipe and matched crops at identical poses:

* **`far-horizon` is the decisive one.** The baseline is a uniform pink-and-green
  mat of flat strips; individual plants are not separable. With prisms the front
  plants stand out lit against darker plants behind them, green shows through
  the canopy, and the skyline silhouette is far more varied. It reads as a
  meadow rather than as wallpaper.
* **Near field at `grazing`** (matched 500×180 crops): same density and mean
  brightness as the baseline (that is the exclusive-ownership fix working), but
  clear front/back separation — bright foreground spikes over dim interiors.
  The baseline is uniformly lit and flat.
* **Parallax is real but subtle at stand density.** Stepping the camera 0.35 m
  sideways at 3 m and comparing frames, layers do reshuffle inside silhouettes
  and the front prisms shift against the rear ones. At 8.5 plants/m² the effect
  is mostly perceived as canopy depth rather than as an individually trackable
  shift; it would read much more strongly on a sparse stand.
* **Silhouette** changes with view direction on the same 45° azimuth ladder as
  the baseline, but the switch is visibly gentler because the residual error is
  ~3× smaller and the layer arrangement stays continuous through it.
* **Ground contact** is better: three contact lines across the footprint.
* **`topdown` is roughly a draw** — slightly blobbier than the baseline (192 px
  merged top tile against its 512 px) but with better clump separation.
* **Sharpness is the one place the baseline still wins**, marginally. Prism
  tiles are 384 px against the baseline's 512 px, so plants closer than ~2.5 m
  are a touch softer. This was much worse at 256 px in the first version; the
  mirrored 4-azimuth layout bought the resolution back inside the same VRAM.

### What went wrong on the way (worth not repeating)

1. **Naive slab alpha testing lost coverage.** Splitting a texel's coverage
   across prisms made both halves fail an alpha test the merged texel passed →
   thinner, darker, worse than the baseline. Fixed by exclusive per-texel
   ownership with merged colour/coverage; this *also* dropped the cost from
   1.34× to ~1.16× because each texel is now shaded once.
2. **Canopy shading must be centred on 1.0.** Making the rear prism darker
   without brightening the front made the whole scene 20% dark and muddy, and
   put a luminance step at the LOD boundary.
3. **Top cards must be gated on the canopy top, not the plant middle.** Gated
   on the middle, an eye-level camera counts as "above" every plant within a
   metre and lays pale edge-on lids through the near field.

### Ideas not taken

* 8 baked azimuths at 256 px instead of 4 mirrored at 384 px — measured
  visibly softer at close range for the same VRAM.
* A 2-prism mid LOD — would need a fourth "rear two slabs merged" tile per
  azimuth, and the cost curve is already flat in `slabDist`, so there is
  nothing to buy.
* Blending two neighbouring azimuths to kill the switch entirely — doubles the
  taps per fragment for an artefact the prism placement already reduces ~3×.
