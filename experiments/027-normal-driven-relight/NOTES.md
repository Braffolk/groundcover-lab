# 027-normal-driven-relight — relief-lit cards

## Idea

Keep the billboard baseline's geometry *exactly* — one camera-facing side quad
plus one horizontal canopy quad per plant, 12 vertices, compute-culled into an
indirect draw — and let the **fragment** do the 3D, from a baked field that says
where the surface actually is.

Per species one 3x3 atlas of 448px tiles is baked from the raw GCMESH1 mesh:
tiles 0..7 are side views at 45° azimuth steps, tile 8 looks straight down.
Every tile is a capture in a generic orthonormal frame `(R, V, F)` with
half-extents `(ru, rv, rw)`, and it stores two planes:

| plane | rgba8 channels |
|---|---|
| **A** | albedo rgb, coverage |
| **G** | octahedral mesh-frame normal (rg), **heightfield s** (b), **canopy occlusion** (a) |

`s` is the front surface's coordinate along the capture axis `F`. So each tile is
a *depth-augmented* image, and at runtime a fragment knows

1. where its card point sits in that tile's frame — an affine varying, exact
   under perspective interpolation, and
2. its own eye ray.

Intersecting ray with heightfield takes **one analytic step**, no marching:

```
t   = (z_card - z_surface) / dot(e, F)
uv' = uv - t * (dot(e,R)/2ru, dot(e,V)/2rv)
```

With the card point already foreshortened (`x_card = x·cos β`) this reduces to
`x/cos β + z·tan β` — the *exact* reprojection of the height it sampled, not a
fudge factor. Consequences:

- **parallax inside the silhouette**: near blades slide across far ones as the
  camera moves, continuously, within one view bin;
- **the silhouette deforms** with view direction instead of only rotating,
  because the coverage that gets alpha-tested is the warped coverage;
- **less bin popping** than the baseline's raw image switch: both neighbouring
  bins reconstruct the *same* surface, so their reprojections mostly agree.

Then lighting carries the rest of the 3D read (the "normals do the work" half of
the brief):

- per-texel baked normals, yaw-rotated (and x-negated for mirrored plants);
- **baked canopy occlusion**, computed at bake time by treating tile 8 as a
  canopy-top heightfield, max-filtering it into a "lid", and asking every texel
  of every tile how deep under that lid its own 3D position sits;
- occlusion is applied as a *sky-visibility* term — full on the hemisphere
  ambient, partial on the sun — which is what keeps sunlit tips crisp while clump
  interiors go deep. With occ = 1 the expression is bit-for-bit
  `light_surface()`, so groundcover and terrain sit in the same light and
  `debug=lighting` stays exact (everything is multiplicative in albedo);
- forward scatter for backlit blades (near tiers only);
- optional 1-tap canopy self-shadow along the sun direction — **off by default,
  see Findings**.

Two more free-ish wins: per-plant **mirroring** (one hash bit flips the plant
about its own x=0 plane — doubles apparent variety, never moves/rescales/
respecies an instance) and a **terrain-following buried skirt** so card bottoms
never hover over a slope.

### Why it is O(1) in plant count

Identical mechanism to the baseline: a compute pass evaluates the shared scatter
(`src/wgsl/scatter.wgsl` twin) over a camera-centred cell region clamped to the
stand, frustum-culls, and compacts survivors into an indirect draw. 557k plants
(`default`) and 134.2M plants (`scaling-100m`) issue the same work — measured
3.00 ms vs 3.00 ms on the relief-cards pass, same VRAM, same everything.

### LOD is structural, not a branch

The cull sorts survivors into **four distance rings** (7 / 24 / 55 / region m),
each with its own sub-range of the instance buffer and its own indirect draw.
The ring index travels in `firstVertex` (`firstInstance` is unusable in an
indirect draw without the `indirect-first-instance` feature), and **each ring is
drawn by its own pipeline specialized on an `override LOD_TIER`**:

| ring | taps | shading |
|---|---|---|
| 0 (< 7 m) | 2 (3 with `reliefSteps=2`) | relief, per-texel normal/occlusion, glow; the only ring that can afford the Newton refinement or the self-shadow tap |
| 1 (< 24 m) | 2 | relief (1 probe), per-texel normal/occlusion, glow |
| 2 (< 55 m) | 2 | no warp, per-texel normal/occlusion |
| 3 (rest) | 1 | no warp, no geometry plane: canopy-ish normal + mean occlusion |

Rings are submitted **near → far**, so filled near cards become depth that
early-z uses to reject the deep layers of a grazing view. No `frag_depth`
anywhere (early-z intact), hard alpha test, depth write, **no dithering**.

## VRAM budget math (per species)

| item | bytes |
|---|---|
| A atlas 1344² rgba8, 7-level per-tile mip chain | 9.63 MB |
| G atlas 1344² rgba8, same chain | 9.63 MB |
| culled instances, 4 rings sized by annulus area at R=128 (density 3) | 2.87 MB |
| entry-info UBO + indirect args | < 1 KB |
| **total** | **22.1 MB** ✓ (HUD reports 21.1 / 20.7 / 21.1 MiB) |

Worst stand is `dense-mixed` (density 5): instances grow to 4.79 MB → 24.0 MB,
still inside the 25 MB budget. Bake-time resources (mesh upload up to ~230 MB for
poa, 2688² targets, readbacks) are transient and destroyed immediately.

## Bake

`bake.ts` renders the source mesh 9 times (one ortho draw per view, viewport +
dynamic-offset uniform, depth32 resolving the front surface) at 2× supersample
into two rgba8 targets, then does everything else once on the CPU:

1. coverage-weighted 2× downsample to 1344²;
2. **canopy occlusion** from tile 8's heightfield (separable max filter → lid,
   then `ao = 0.3 + 0.7·exp(-1.3·depth_under_lid)`);
3. **heightfield envelope blur** (coverage-weighted, radius 10 texels) — see
   Findings, this one matters a lot;
4. dilation 6 rings into empty texels (colour/normal/height/occlusion grow,
   coverage stays 0) so bilinear and warped taps never read background;
   heightfield keeps `s = 0` as a "no data" **sentinel**, which is how the
   runtime switches the warp off where there is no surface;
5. a **per-tile** mip chain (7 levels, down to a 7px tile) — no level ever blends
   one view into another, which is the atlas bleed that makes distant impostors
   shift colour.

The chain ships inside the artifact, so a normal load is fetch + `writeTexture`,
with no GPU mipgen pass at all. Standard harness flow (`bakedArtifact()` → OPFS →
committed file → in-browser bake → `commitBake()`), magic+size validated, so the
dev server's SPA-fallback HTML can only cause a rebake, never a bad atlas.

Artifacts (18.4 MiB each, only v2 is loaded — the v1 files were deleted):
`mesh/baked/027-normal-driven-relight/relief-v2-t448-{calamagrostis-canescens,elymus-repens,poa-pratensis}.bin`

## Status

**working** — verified by headless screenshots at `grazing`, `topdown`,
`inside-plant`, `far-horizon` on `default`, plus `close-quality`,
`scaling-100m` (134.2M plants), all five debug views, the LOD-tier view, and the
A/B page against the baseline. Typecheck clean, zero console errors.

## Findings

### Performance (A/B ratios, GPU was CONTENDED by 15 sibling agents — never quote the ms)

`#/ab/001-billboard-smoke/027-normal-driven-relight?stand=default&cam=grazing&seed=42`

| cam | A (cull+cards) | B (cull+relief-cards) | ratio |
|---|---|---|---|
| grazing | 2.26 | 2.90 | **1.28x** |
| topdown | 1.75 | 2.00 | **1.14x** |
| far-horizon | 2.19 | 2.71 | **1.24x** |

Solo Σp50 at 1280×800 stayed between 3.6 ms (topdown) and 6.7 ms (grazing) even
under that contention, i.e. inside the 9 ms ceiling. Bench JSON (also contended,
kept for provenance):
`results/027-normal-driven-relight__default__p-29681b0f__apple-metal-3__2026-07-25T06-33-13-584Z.json`

**Four things cost real time, and none of them were the ones I expected.** The
first version measured **2.2x** the baseline, and every param probe (relief off,
shadow off, AO off, LOD radii shrunk) changed it by less than the noise — the
cost was structural, not featural:

1. **Interpolant traffic.** 11 varying slots (nine of them just the tile's axis
   vectors) → 4 slots by rebuilding the frame in both stages from two cos/sin
   pairs. 2.2x → 1.9x.
2. **One monolithic fragment shader.** Peak register allocation is set by the
   worst path, not the path taken, so far cards paid for relief and shadow code
   they never ran. Per-ring pipelines specialized on `override LOD_TIER` →
   1.9x → ~1.6x.
3. **Per-fragment mip machinery.** `dpdx/dpdy + length + log2` (and
   `normalize(eye)`) in every fragment cost ~0.2x on their own. A card's
   uv-to-pixel ratio is a *per-plant constant*, so the mip level is now computed
   once in the vertex shader from `proj[1][1]`, the card's metre width and the
   plant distance (canopy cards fold in their own foreshortening). ~1.45x → 1.28x.
4. **Vertex texture fetches.** The exact per-corner terrain height for the
   contact skirt was ~13% of the pass when every card paid it; now only the two
   near rings do (a plant 30 m out cannot show a base that hovers by a
   centimetre).

Also measured: `textureSampleGrad` with the same gradients keeps anisotropy alive
but costs ~15% of the pass for no visible difference here, so all taps use
`textureSampleLevel` with the per-card level.

### Looks — honest verdict

**Not clearly better than the baseline at the grazing camera; clearly better from
above; better in motion.** Details:

- **Mid/steep elevation (6 m up, 54° down) is a clear win.** The baseline is a
  bright flat carpet; the relief-lit version has canopy volume — dark gaps
  between clumps, lit tops, layered heads. This is the baked canopy occlusion plus
  the warp on the canopy card doing exactly what the brief asked for.
- **Grazing is a wash.** Side by side in the A/B wipe the two are nearly
  indistinguishable in character; mine has a little more depth contrast, the
  baseline a little more uniform brightness.
- **Eye level / inside-plant** favour mine slightly: more depth layering, and the
  canopy card no longer shows its (wrong) top face from below because its fade
  now uses *signed* elevation.
- **Parallax is real and measurable**, not a claim: turning `relief` off changes
  25% of the image energy (RMSE 16199/65535 over a 900×700 frame at a close
  pose), and the Newton refinement another 9% (`reliefSteps=2`, which is off by
  default because it moves the pass from 1.28x to 1.42x of the baseline). In motion the interior slides
  correctly against the silhouette; in a still frame it mostly reads as a
  slightly softer image, which is why the still comparison is a wash.
- **Grass is not a heightfield**, and that is the single biggest trap here.
  Neighbouring texels can be a blade at the front of the clump and bare ground
  30 cm behind it; a one-step solve against that raw field warps colour to places
  no surface ever was, and the result was *visibly muddier than a plain
  billboard*. Blurring the heightfield to the canopy **envelope** at bake time
  (radius 10 texels — coarser than a blade, finer than the clump) plus clamping
  the warp to 14% of a tile is what turned the parallax from damage into depth.
- **Occlusion must not multiply the whole light term.** The first version did,
  and the meadow went to mud. Splitting it (full on ambient, 0.4 + 0.6·occ on the
  sun) recovered the baseline's brightness while keeping the depth cue.
- **The 1-tap canopy self-shadow is a measured failure at this sun**, and it is
  off by default: the harness sun sits 49° up, so a sun ray escapes the canopy
  lid almost immediately and the whole term changes the image by RMSE
  27/65535 — 0.04%. The code path stays (runtime-gated, so it costs nothing when
  off) because at a low sun it is the right mechanism; claiming it as a feature
  at this sun angle would be dishonest. Same story for `translucency`: invisible
  at these camera/sun angles, kept because it is nearly free and does matter when
  the sun is behind the plant.
- Honest artifacts: (1) inter-*plant* occlusion is still card-plane depth, same as
  the baseline — the relief buys intra-plant depth only, because writing
  `frag_depth` would have cost early-z (it sank two methods in the previous
  round); (2) 448px tiles are 12% coarser than the baseline's 512, marginally
  softer up close; (3) azimuth bins still switch — the warp shrinks the pop, it
  does not remove it; (4) calamagrostis/elymus source meshes are 0.52 m periodic
  community tiles, so each "plant" is a baked clump repeated per scatter point,
  which reads busier than ground truth (poa, a true specimen, is cleanest).

### Verification links

- run: `#/run/027-normal-driven-relight?stand=default&cam=grazing&seed=42`
- LOD tiers: add `&p.showLod=true` (red/yellow/green/blue = ring 0..3)
- A/B vs baseline: `#/ab/001-billboard-smoke/027-normal-driven-relight?stand=default&cam=grazing&seed=42`
- A/B vs ground truth: `#/ab/027-normal-driven-relight/000-ground-truth?stand=default&cam=grazing&seed=42`
- plant-count independence: `#/run/027-normal-driven-relight?stand=scaling-100m&cam=grazing&seed=42`

## Debug views

The fragment shader routes its final colour through the shared `debug_shade()`,
so `debug=albedo|normals|lighting|coverage|depth` all work; fog is applied only
in `DEBUG_OFF`.

- **albedo** — the baked tile colour times the per-plant tint, i.e. exactly the
  multiplicative base of the light term (occlusion is NOT folded in, so
  **lighting** divides out exactly).
- **normals** — the baked per-texel mesh-frame normal at the *warped* hit,
  x-negated for mirrored plants and yaw-rotated. Genuinely per fragment and
  genuinely high frequency; the far ring shows its canopy-ish substitute normal.
- **lighting** — sun+ambient with the occlusion split; saturates white on sunlit
  blades exactly like the baseline does (the shared `sun_color` exceeds 1).
- **coverage** — the baked alpha that was alpha-tested, so it is bounded below by
  `alphaRef` by construction: white = solid interior, grey = silhouette texels
  sitting on the threshold.
- **depth** — from the card plane (this method deliberately does not write
  `frag_depth`), so the relief is visible in shading but not in the depth ramp.
