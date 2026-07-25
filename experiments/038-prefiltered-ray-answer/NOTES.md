# prefiltered ray answer

## Idea

**No plant primitive is ever rasterized.** The entire three-species canopy is a
baked table of *ray answers*, and a pixel's grass appears because its eye ray
asks the table a question.

The parameterisation is the whole experiment. For a ray of direction `d`, define
its **entry point** as where the ray crosses the canopy-top plane `y = H`:

    E(X) = X.xz - d.xz * (X.y - H) / d.y

`E` is invariant along the ray, so it is a *label for the line*. Bake, for every
quantised direction, a 2D image over `E` of what that line meets first:

| channel | meaning |
|---|---|
| `geom.r` | hit height / H (premultiplied by coverage) |
| `geom.gb` | octahedral normal of the hit surface |
| `geom.a` | coverage |
| `surf.rgb` | authored mesh albedo |
| `surf.a` | baked AO (depth below the local canopy top) |

Two facts make this cheap and exact:

1. Along a descending ray `y` decreases monotonically, so **"first hit" is
   literally "greatest y"**. The bake therefore composites an arbitrary number
   of plants with a plain depth test, and the runtime stores a *height*, not a
   distance — 8 bits over 1.4 m is 5.5 mm, and the distance is reconstructed at
   runtime from the pixel's TRUE direction, so hit geometry stays continuous even
   though the pattern is quantised.
2. Prefiltering is free and honest: every channel is stored premultiplied by
   coverage, so the mip chain is a plain box filter. Mip *n* is the true mean hit
   height, mean albedo, mean AO and **true areal coverage** of the canopy over a
   2^n-texel footprint of entry points. A distant pixel is one fetch of an
   honestly averaged canopy — which is exactly the seed of this experiment.

Per pixel, everything is closed form — no marching, no loop over the ray, no
iteration toward the surface:

1. one heightmap fetch → local ground gradient at the crossing;
2. world → patch space: the terrain gradient is sheared out (ground becomes
   flat) and the wind shear is inverted. Both are linear, so the ray stays a ray
   and its parameter `t` is preserved — bending the ray instead of the canopy is
   *exact*, not an approximation;
3. one division → `E`, and `atan2`/`log` → the direction bin;
4. one geom fetch → a probe height, which anchors **one closed-form parallax
   correction**: the lookup is re-indexed so that the quantised bin's answer
   lands on the true eye ray. This is what makes 15° azimuth bins usable;
5. geom+surf at the corrected entry (×2 azimuth bins, blended);
6. one heightmap fetch at the hit → snap the hit onto the true terrain-following
   canopy, and **the snap residual is the confidence**: a large residual means
   the ray really flew over a dip and left the canopy, so coverage fades and
   whatever is really behind (sky, or the next carrier crossing) shows through.

### What is rasterized (the no-geometry proof)

**One draw. One camera-centred, terrain-conformal grid: 240×240 cells =
115,200 triangles, a compile-time constant.** It is snapped to its own cell size
so it never crawls, and it carries no plant information whatsoever — it exists
only to hand the fragment shader a point on the eye ray near the canopy top.
`default` (557k plants), `close-quality` (20k) and `scaling-100m` (134.2M) all
rasterize *the same 115,200 triangles* and read *the same table*: the runtime
never touches the scatter, so plant count is not merely amortised, it is absent.
Measured: `scaling-100m` 2.17 ms vs `default` 1.9–2.4 ms in the same pass, same
18.0 MB/species (screenshots taken at both).

### Fetches per pixel

**7, constant, at any distance and any plant count**: 5 from the answer table
(1 probe + geom/surf × 2 blended azimuth bins) and 2 from the shared terrain
heightmap. With `azBlend` off it is 5 (3 + 2); with `correct` off, 4. Hardware
trilinear + 16× anisotropic filtering happens inside those fetches — the
anisotropy is capped to 16 in the shader so the mip level follows the *short*
axis of the footprint (letting the long grazing axis drive it over-blurred
everything, an early bug).

### Wind

The sway is a horizontal displacement proportional to height, i.e. a **shear of
space**, which is linear and therefore exactly invertible: the ray is unsheared
into patch space instead of the canopy being sheared, and lines stay lines. The
phase comes from a smooth value-noise field, so gusts travel across the meadow
instead of every plant swaying in lockstep. Honest limitation: one table serves
all three species, so there is one shear, using the stand's density-weighted
mean sway (0.70 for `default` vs the per-species 0.6/0.65/0.85). Tip
displacement error is ≤18% of ~10 cm, i.e. under 2 cm — below one texel of the
table. Per-species sway would need three tables and 3× the fetches.

## VRAM budget math

The table is **shared by the whole canopy** (one texel's answer can be any of the
three species, including their mutual occlusion, which is baked in). It is
allocated as three (surf, geom) pairs split by azimuth band, one pair per species
budget row, so the HUD meter stays meaningful:

| item | per species |
|---|---|
| `answer-surf` 192² × 48 layers rgba8 + 8 mips | 9.0 MiB |
| `answer-geom` 192² × 48 layers rgba8 + 8 mips | 9.0 MiB |
| cfg uniform (64 B) | noise |
| **total** | **18.0 / 25 MiB** |

Whole-canopy total 54.0 MiB (HUD reads 63.4 MB including the harness's own scene
allocations). Identical on every stand — nothing scales with plant count.

Resolution follows from that budget: 144 slices (24 azimuths × 6 elevations) at
192² over a 6 m patch = **3.125 cm per texel**. Elevation bins are uniform in
`ln(cot θ)` (centres 3.5°, 9.6°, 25°, 52°, 74°, 84°) — fine where the shear
changes fast (grazing), coarse where it barely changes (steep). Bake scratch
peaks around 300–350 MB (raw mesh buffers + a 2496² × 3-layer plant atlas), all
tagged `bake-scratch` and destroyed at the end.

## Bake

`bake.ts`, two GPU stages, via the `bakedArtifact`/`commitBake` flow.

**Stage 1 — per species, 145 orthographic renders of the raw GCMESH1 mesh** (one
per direction plus one straight down) into a 2496² × 3-layer atlas. An
orthographic render with a depth test along the view axis *is* a first-hit ray
solve — the rasterizer answers the whole ray bundle at once, offline. ~1.9 G
triangles total.

**Stage 2 — the patch answer.** The patch is a 6 m × 6 m periodic window of the
**active stand's own scatter** (`ctx.scene.scatter.cell()`): its real cells, real
positions, real scales, real species mix, real densities, yaws snapped to the 15°
azimuth grid. For each of the 144 directions, every plant's stage-1 tile is drawn
as *one sheared quad* into E space (the map is affine because `E` is invariant
along `d`), with the baked height as depth, rendered at 2× and resolved
coverage-weighted. ~280 k quads total, and the depth test makes the composite of
the plant set *exact*.

Artifact: `mesh/baked/038-prefiltered-ray-answer/answer-v1-mix<hash>-s42.bin`,
40.5 MiB (64 B header + 144 × 192² rgba8 × 2). The key hashes the stand's
density/scale mix + seed, so `default` and `close-quality` share one file;
`scaling-100m` and `dense-mixed` re-bake in-browser (~3 min, verified) and are
deliberately **not** committed, to keep the deploy payload lean. Mips are
generated at load on the GPU.

Because the key hashes the stand's own numbers, an edit to `src/scene/stands.ts`
(shared code, edited by others during this session) invalidates the committed
file and the table re-bakes in-browser. That is the price of a stand-dependent
table; 001's per-species card atlas has no such coupling. The committed file
matches the stand definitions as of this session — verified by watching the
runtime fetch it rather than re-bake.

## Status

**working** — verified by headless screenshots at all four standard cams, all
five debug views, three stands (`default`, `close-quality`, `scaling-100m`),
frozen-time `det=1&t=3` vs `t=7` for deterministic wind, and the A/B page against
001. Zero console errors.

## Findings

### Honest verdict on looks vs billboards (001)

**Mid and far field: mine wins.** From ~10 m out the canopy is a continuous,
honestly averaged, non-shimmering meadow that covers the *whole* terrain to the
horizon, with real per-pixel depth, real baked normals and correct hill
occlusion. 001 erodes out at its `regionRadius` (110 m default) and aliases at
distance because it magnifies/minifies 512 px cards per plant. The `topdown` cam
is the clearest win: dense tuft structure, no floating pancakes, no popping.

**Near field (0–8 m): 001 wins, clearly.** This is the honest headline finding
and it is a *resolution* limit, not a bug — worth stating quantitatively because
it bounds the whole idea:

> A 4D table (2D entry × 2D direction) at 25 MB/species affords ~3 cm texels.
> Billboards spend the same memory on 2D + 8 views *per species* and afford
> 2.3 mm. Whenever the canopy top is within ~10 m of the eye at grazing
> incidence, one table texel covers many pixels and the answer is magnified into
> smooth bilinear blobs. Grass blades are 3–5 mm wide; they are simply not
> representable at 3 cm, at any filter quality.

`detail` (default 0.85) adds a deterministic, world-anchored blade-scale normal +
albedo modulation *only* in that magnified regime (amplitude → 0 the moment the
footprint reaches one texel). It invents no coverage — there is no dithering
anywhere in this renderer — and it makes the near field read as foliage rather
than as interpolated paint, but it does not put blades back. The fix that would
is a second, small-period high-resolution table for the near ring; that is the
obvious next experiment and does not fit in this one's budget.

### Cost (A/B, same frame, contended — never quote absolute ms)

`#/ab/001-billboard-smoke/038-prefiltered-ray-answer?cam=grazing&seed=42`
3 repetitions per cam; B = my pass, A = 001's cull + cards:

| cam | B / A |
|---|---|
| grazing | 1.04, 1.15, 0.81 |
| topdown | 1.06, 1.09, 1.27 |
| far-horizon | 0.95, 0.83, 1.01 |
| inside-plant | 0.91, 1.01, 0.98 |

**Parity with the champion (~0.8–1.3×)** while being O(1) in plant count, which
is the trade this method exists to make. Under contention from sibling agents
these move ±30 %; the *pairing inside one frame* is what is trustworthy. No
`results/` bench JSON is claimed — the GPU was shared during this session
(CLAUDE.md rule). Total GPU Σp50 sat at 2.9–5.3 ms at 1280×800 including the
harness base pass, well inside the 9 ms ceiling.

### The camera-inside-canopy limit (the real structural one)

A first-hit-from-outside table cannot answer a ray that starts *inside* the
canopy: its first hit is behind the eye, and the next hit is not in the table.
So the eye must stay above the canopy top, and the carrier surface is dropped
below the eye when the eye descends into the canopy (the canopy itself keeps its
full height — only the surface the rasterizer intersects moves, and the entry
point slides back up to the true canopy top analytically). Consequences:

- `grazing` (eye 1.35 m, canopy 1.40 m — deliberately *at* the canopy top): the
  visible canopy tops out at the dropped carrier height, ~9–18 % short, and you
  look *down onto* the canopy instead of standing in it. 001 shows the sides of
  tall panicled stems there; mine shows their tops. This is the biggest visual
  gap and it is structural, not tuning.
- `inside-plant` (eye 0.55 m, deep inside): degrades to a low field with a soft
  near zone — the sanctioned "fade it out when the camera is inside a plant"
  behaviour, implemented as a carrier drop + a distance fade, never as a dither.
- Hits that still resolve behind the eye are pinned to the near-fade distance
  rather than dropped, so there are no holes; their colour is still a real canopy
  sample, only their distance is unknowable from a first-hit table.

### Stand contract — what is exact and what is not

Exact: species set, per-species density, scale range, yaw distribution, plant
*identity* (the real GCMESH1 meshes), and region extent (the canopy stops at the
stand radius with a 3 m soft edge; `close-quality` ±24 m verified). Approximated,
and this is the honest cost of a table: the field is a **6 m periodic window of
the stand's own scatter**, so a plant at cell (37, −12) shows cell (0..1, 0..1)'s
arrangement rather than its own. The lattice is broken by a smooth 34 m-wavelength
lookup warp (`warpAmp`, 1.4 m) — a constant offset per pixel is exactly a
translated copy of a periodic patch, so it is geometrically free and seamless,
unlike per-cell variant selection which would cut plants at cell borders. Habitat
bands (the `wetCenter`/`wetWidth` zoning behind the `bog` stand) cannot be
expressed by a single tiled patch at all: the patch is composed at one wetness
value.

### Bugs worth remembering (all found by screenshot, all structural)

1. **Never extrapolate the local ground plane back to the camera.** The first
   version anchored patch space at the eye, so a tangent plane fitted 100 m away
   decided the answer; the fix was to anchor everything at the carrier crossing,
   which lies on the ray by construction.
2. **At grazing incidence, any mismatch between the carrier's interpolated height
   and the fragment's true terrain gets divided by `ray · plane normal`** (~1/16
   here) — a 10 cm tessellation error became metres of entry-point jitter and
   rendered as a faceted swirl. Fix: define the ground plane *by the carrier
   itself* (`base_y = p.y - carrier_h`) and let the terrain snap put the hit back
   on the real ground. The carrier can then be coarse for free.
3. **Keep the snap local.** Letting one crossing snap its hit tens of metres
   forward let a far crossing win the depth test over a near one — a patchwork.
   Bounded snap + the confidence fade means each crossing answers its own
   neighbourhood and the nearest one always wins.
4. A ray flatter than the flattest baked bin sends the entry point to infinity;
   clamping the descent rate to that bin's cotangent removed a band of holes that
   tracked the terrain slope.
5. Azimuth bins showed up as hard vertical bands near the camera (~8 blocks
   across the screen). Blending the two neighbouring bins fixes it cleanly
   *because* the parallax correction lands both bins' hits on the same true ray —
   without the correction the blend would ghost.

### Harness wishlist

- A dev-server 404 (instead of the SPA `index.html` at 200) for missing
  `/mesh/baked` files would let experiments drop the magic-validation shim.
- `ctx.res` cannot attribute one allocation across several species; a genuinely
  shared table has to be split into per-species pieces purely for the meter.
