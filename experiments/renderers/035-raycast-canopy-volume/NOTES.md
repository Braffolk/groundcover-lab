# 035-raycast-canopy-volume — baked ray answers over a periodic canopy tile

## Idea

**No plant primitive is rasterized anywhere in this renderer.** The grass exists
only as the answer to a per-pixel ray query.

The canopy of one stand entry is baked over a periodic L×L tile as a 4D
**ray-answer table**: for a ray that crosses the tile's canopy-top plane at
tile coordinate (u,v) travelling in a quantized direction, the table stores what
that ray hits *first*:

| channel | meaning |
|---|---|
| R | `q` — vertical drop from the entry plane to the hit, / canopy height |
| G | `cov` — fraction of the texel's 4 sub-rays that hit canopy at all |
| B,A | octahedral surface normal at the hit (tile frame) |

Everything else is closed form from `q` plus the ray: hit position, along-ray
distance (`drop / |d_y|` — the article's `|OB| = |OA| / cos α` in disguise),
height fraction, AO and the albedo curve. So **one `textureSampleGrad` per
elevation-band tap resolves a pixel** — no marching, no iteration, no search.

Directions are quantized elevation-adaptively, because a near-vertical ray
hardly cares about azimuth (at exactly vertical, not at all) while a grazing one
cares a lot:

| band | zenith | azimuth bins |
|---|---|---|
| 0 | 0° | 1 |
| 1 | 26° | 12 |
| 2 | 48° | 20 |
| 3 | 66° | 30 |
| 4 | 81° | 48 |

= 111 layers of a 192² 2D-array texture per stand entry.

Per pixel, per stand entry, all arithmetic:

1. eye ray → local canopy frame. The terrain is removed as an **affine change of
   variables** (`d_y -= grad · d_xz`), so the ray stays a straight line and the
   canopy stays ground-conformal.
2. **wind = inverse shear of the query.** Sway is a linear-in-height shear of the
   canopy; lines stay lines under a shear, so shearing the *ray* is exact rather
   than approximate: `s_table = k·s_world + W/H`, entry `-= W`. Per-species sway
   comes from the stand table.
3. entry-plane crossing, referenced to the rasterized carrier fragment.
4. band + azimuth bin, azimuth-only parallax correction, **1 fetch per tap**.
5. reconstruct the hit, `frag_depth` from it, AO from canopy depth, albedo from
   a baked height→colour curve with two colour clusters per height bin.

### What is rasterized (the no-geometry proof)

ONE camera-centred **polar carrier shell**, conformal to `terrain + a local
height`, generated procedurally in the vertex shader from a ring-radius buffer:
144 spokes × 143 rings = **41,184 triangles per instance**, 1–2 instances,
**one `draw()` call**. Two instances exist only when the camera is *inside* the
canopy: shell A above the whole canopy (catches every ray entering from above,
plus the tips that stick into the sky), shell B just under the eye (catches
descending rays — the `grazing` camera sits 1.35 m up in a 1.5 m canopy).

The shell's tessellation depends on the stand *radius* only (and is capped at
420 m, past which the canopy is fog): **it is identical for 557k plants and for
134.2M** — verified, `stand=scaling-100m` renders the same shell and the same
20.8 MB/species, with the frame cost differing only by visible screen area.
There is not a single card, quad, ribbon, prism, fin or per-plant shell.

### Texture fetches per pixel (constant)

| what | fetches |
|---|---|
| terrain heightmap → local ground plane | 1 |
| ray answer, 2 elevation-band taps × 3 stand entries | 6 |
| terrain heightmap again → "not floating above the ground" gate on the winner | 1 |
| **total, default 3-species stand** | **8** |

With `bandBlend=false` it is 5. Nothing scales with plant count, canopy depth or
distance; the mip chain does the distance work.

## VRAM budget math

Per stand entry (HUD-verified: 20.8 / 25 MB for each of the three species):

- ray-answer table 192 × 192 × 111 layers, rgba8, full mip chain
  = 192²·111·4 B · 1.333 = **19.7 MiB**
- baked palette (32 height bins × 2 clusters × vec4, 3 entries share one buffer)
  = 3 KB, carrier ring radii = 780 B, params = 256 B → noise

No per-plant buffer of any kind exists — no instance buffer, no cull buffer,
nothing that grows with the stand. Total for the default stand: 71.9 MB
(3 × 20.8 MB + noise).

Bake-time transient GPU memory (tagged `bake-scratch`, destroyed at the end):
5 mm occupancy bitmask (10.6 MB for calamagrostis), 4 cm skip mask (30 KB),
1.5 cm normal-accumulation volume (36.7 MB), the raw vertex buffer (33–135 MB),
the 16.4 MB output + readback. Peak ~200 MB for poa (8.46M vertices).

## Bake

`bake.ts` → `mesh/baked/035-raycast-canopy-volume/table-v3-<species>-d<density>-s<min>-<max>-sd<seed>.bin`,
**16.4 MB each** (1280 B header + palette, then 192²·111·4 B of answers). The key
includes the stand entry's density and scale range and the seed, because those
change the canopy that is baked; the three committed artifacts are the default
stand at seed 42. Other stands bake in-browser in a few seconds and land in the
OPFS cache. Every load is magic/size-validated (the dev server answers missing
`/mesh/baked` files with `index.html` at HTTP 200, which would otherwise poison
both the cache and the committed path).

Per entry:

1. **Tile content = the stand's own plants.** `scatter.region()` over the
   `[0,L)²` window, keeping each plant's exact position, yaw and scale, placed on
   a flat local ground. calamagrostis 16 plants / 2.60 m, elymus 8 / 2.15 m,
   poa 13 / 1.90 m. Periods are deliberately incommensurate (and each species
   gets its own lattice rotation ψ) so the three periodic canopies never repeat
   in step.
2. **Voxelize** (`voxelize.wgsl`): one compute dispatch splats every source-mesh
   vertex of every instance (2–8.5M verts × 8–16 instances) into a wrapping 5 mm
   occupancy bitmask + a 4 cm "any" skip mask, and every 4th vertex adds its
   rotated normal to a 1.5 cm atomic normal volume. Source vertices are ~0.5 mm
   apart, so 5 mm voxels fill solidly.
3. **Resolve the rays** (`rayanswer.wgsl`): one thread per (texel, azimuth) per
   band, 4 jittered sub-rays each = 16.4M rays per entry, marched through the
   wrapping volume with a 4 cm coarse skip. This is the only place a ray is ever
   marched. ~2 s per species on this machine (5 dispatches, one per band).
4. **Palette**: mesh colours binned by height, each bin split along the
   red-vs-green axis into two cluster means — for calamagrostis that separates
   the pink panicles from the green leaves *at the same height*, so the runtime
   tuft hash mixes colours that actually occur up there instead of inventing a
   tint.
5. Mips are generated at load (`mipgen.wgsl`, one dispatch per level):
   coverage-weighted for `q`, decode→average→re-encode for the normal, wrapping
   taps because the table is periodic. **This mip chain is the antialiasing of
   the whole technique.**

## Status

**working** — verified by headless screenshots at all four standard cameras on
the default stand, all five debug views (albedo / normals / lighting / coverage /
depth all meaningful and honest), `stand=scaling-100m` (134.2M plants), and the
same-frame A/B page against 001-billboard-smoke. Zero console errors.
`npx tsc --noEmit | grep 035-raycast-canopy-volume` is silent.

## Findings

### Performance

Same-frame A/B (contended GPU — ratios only, never absolute ms):

| run | A billboards (cull + cards) | B this method (canopy-rays) | B / A |
|---|---|---|---|
| grazing | 1.89 ms | 3.14 ms | 1.66× |
| far-horizon | 2.06 ms | 3.35 ms | 1.63× |
| grazing (repeat, heavier contention) | 1.74 ms | 3.91 ms | 2.25× |

So **1.6–2.3× the champion's cost** depending on how contended the GPU was — inside the "parity to modestly slower" bar, and
the cost is bounded by *screen pixels*: the `scaling-100m` stand (240× the
plants) costs the same shell and the same table. No `results/` bench JSON is
claimed: several sibling agents were rendering on this GPU throughout, so bench
numbers would be contaminated. Re-run
`#/bench/035-raycast-canopy-volume?stand=default&spline=orbit-low` on an idle GPU
before quoting numbers.

### Looks, honestly

**It does not beat billboard cards.** What it wins:

- **Real per-pixel depth.** `frag_depth` is the reconstructed hit, so the canopy
  inter-occludes correctly, contacts the terrain correctly (no cardboard
  intersection), and the depth debug view is a clean near→far ramp with visible
  per-plant structure.
- **Silhouette that changes with view direction** — 111 baked directions vs the
  baseline's 8 azimuths + top, and no card ever turns edge-on.
- **Parallax inside the canopy** and believable self-occlusion: the AO term is
  the hit's depth below the canopy top, so the interior genuinely darkens.
- Cost independent of plant count *and* of canopy depth.

What it loses:

- **Detail.** A table texel is 1.0–1.35 cm of tile, so near-field grass resolves
  to ~1 cm blobs where a 512 px card gives ~2 mm. Individual plants read as
  vertical strands rather than as plants with stems and tips.
- **Grazing softness.** At grazing angles the lookup footprint is enormously
  anisotropic (the entry point sweeps *along* the ray hundreds of times faster
  than across it), so the honest prefiltered answer is a smeared canopy. Side by
  side at `far-horizon`, billboards show crisp plants and this shows melted
  strands. `lodBias` (default −0.55) trades that softness against shimmer.
- The above-eye canopy at `inside-plant` is blotchy — that camera is the
  documented breakdown case, and it is the sparse top few centimetres of the
  canopy being answered by a table baked for entry at the top plane.
- Two or three dark lens-shaped smudges near the eye line at `grazing`: grass
  tips 0.3–2 m in front of the eye, magnified far past the table's resolution.

### The stand contract — the honest deviation

Densities, species mix, scale ranges, sway and region are all exactly the
stand's, and the plants inside the baked tile are *real scatter plants* (exact
positions, yaws, scales from `scatter.region()`). But the rendered field is the
**periodic repetition of that window**, so a plant at world position P is
answered by the equivalent plant at P mod L. A single-lookup ray answer cannot do
better: resolving arbitrary per-plant positions would require the table to be
indexed by world position at blade resolution. Three incommensurate periods
(2.60 / 2.15 / 1.90 m), three lattice rotations and a tuft-scale colour hash hide
the repeat well at eye level; at `topdown` the tiling is visible if you look for
it. Plant *counts* per tile land within ~±30 % of density·L² (16 vs 20 for
calamagrostis, 8 vs 11.6 for elymus) — a fairer window search would fix that and
is the first thing I would change.

### Wind

Closed form and exact for the model used: the canopy is the baked tile sheared
linearly in height, and a shear maps lines to lines, so the query ray is
inverse-sheared instead of the geometry being moved (`s_table = k·s_world + W/H`,
entry `-= W`). Per-species amplitude comes from `stand_table.sway`; the shared
`wind_sway()` field supplies the travelling gust. What is *lost*: the per-plant
phase offset (a smooth shear field cannot decorrelate neighbouring plants), and
the shared model's small vertical dip. So the canopy sways coherently over ~1 m
patches rather than plant by plant.

### Camera inside the canopy

The `grazing` camera is 1.35 m up in a 1.5 m canopy, so the "camera above the
slab" assumption of an entry-plane table fails for exactly the most important
view. Handled by dropping the entry plane to just under the eye and reading the
table as a vertically compressed canopy (`k = y_start / H`, `k = 1` and therefore
exact whenever the camera is above the canopy top). The compression is a
per-frame constant, so it is spatially smooth; its cost is that the canopy
"breathes" slightly if the camera moves vertically *through* the canopy top — a
static screenshot cannot show that, so it is written down here instead.

### Things that were structurally wrong and what fixed them

Recorded because they were expensive to find:

1. **A fat uniform struct indexed dynamically cost 62× the frame time.** The
   per-entry uniform originally carried its 64-vec4 palette, and
   `params.entries[ei]` with a runtime `ei` made the compiler copy all 1168 bytes
   per access — six times per pixel. 242 ms → 3.9 ms by shrinking `Entry` to
   4 vec4 and moving the palette to its own binding. Nothing else changed.
2. **Anchoring the parallax correction at a fixed *drop* is wrong at grazing.**
   The shear error between the true ray and the baked band reaches Δs ≈ 14, so a
   hit at a different depth than the anchor is misplaced by *metres*. Anchoring
   at a fixed *horizontal* distance instead (the band's mean free path) makes the
   correction azimuth-only and centimetre-scale, and re-expressing the baked drop
   for the true shear (`shearFix`, `q ∝ (tan_bake/tan_true)^β`) keeps grazing
   hits near the canopy top where they belong.
3. **Anchoring at the *measured* hit depth (the classic impostor refinement tap)
   is worse still**: the anchor distance carries a `1/|d_y|` factor, so at
   grazing a 0.01 wobble in `q` moves the lookup 0.7 m and the image turns to
   noise. Fixed per-band anchors are stable by construction.
4. **Per-pixel decisions must not flip regimes.** Deriving the entry height (and
   thus the canopy's vertical compression `k`) from the *local* ground plane made
   neighbouring pixels flip between the compressed and exact regimes — pure
   speckle. Both are now per-frame constants from the camera's own terrain
   height; only the ground plane stays local.
5. **Reference the rasterized fragment, not the camera.** Computing the entry
   crossing from the camera means extrapolating a local plane tens of metres;
   on slopes that put the canopy under the terrain (a bright band of bare ground
   across the screen) and floated grass into the sky. The carrier fragment lies
   exactly on a known surface, so `t_entry = t_frag + (y_start − carrier_h)/d_y`
   is a short, well-conditioned step. Plus a one-fetch gate on the winning hit
   ("cannot be above terrain + canopy height") kills the remaining sky artifacts.
6. **A fixed hard alpha threshold coin-flips against mip-averaged coverage.**
   At high mips coverage tends to the canopy mean, which sat right at the 0.42
   reference → speckle. The reference now halves with LOD, so a distant canopy
   reads as the opaque prefiltered surface it actually is. No dithering anywhere:
   the edge is a hard alpha test at mip 0 and honest coverage averaging beyond.

### What I would try next

- Store the **horizontal free path** normalised by a per-band cap instead of the
  drop fraction. At band 4 the useful drop range is ~0.6 m out of 1.5 m, so today
  grazing answers live in ~20 of the 256 quantisation levels; storing `g/G_max`
  would spend all 8 bits where grazing rays actually hit. Needs a re-bake.
- A fairer tile window (search offsets for the one whose plant count matches
  density·L² best) to fix the ±30 % per-species density error.
- A second, small, high-resolution table for hits within ~0.6 m to fix the
  near-field blobbiness, blended in by distance.

### Harness wishlist

- A way to report bake progress to the runner UI (currently `console.info`).
- A 404 (rather than SPA `index.html`) for missing `/mesh/baked` files would let
  experiments drop the magic-validation shim.
