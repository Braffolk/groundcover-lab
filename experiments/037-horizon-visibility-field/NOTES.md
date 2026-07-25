# horizon visibility field

## Idea

**No plant geometry exists at any point in the frame.** The grass is the answer to
a per-pixel ray query, read out of a baked table.

What is rasterized, exactly: ONE instanced draw of a camera-centred polar grid
(128 rings x 192 sectors x 6 verts = 147,456 vertices = 49,152 triangles per
instance) ground-conformalised onto `LEVELS` iso-clearance shells (1 when the eye
is above the canopy, 6 when it is inside it) — so **49k triangles at best and
295k at worst, identically for 557k plants (default) and 134M (scaling-100m)**.
There is no per-plant, per-clump, per-blade or per-cluster primitive, no card, no
quad, no ribbon. The shells are only a conservative mask plus a carrier of the
local ground plane; the silhouette comes entirely out of the table.

Per pixel, in a constant number of fetches and with no loop over the ray:

1. The shell fragment hands over the exact point where the eye ray's *clearance
   above the terrain* equals that shell's height — i.e. the rasterizer solves the
   ray/offset-surface crossing in hardware. Each fragment answers only if its own
   height brackets the local canopy top (two-stage test), so whichever shell sits
   at the canopy's height owns the pixel. This is what makes the eye-level
   cameras work at all: a single shell above the canopy would never be crossed
   when the eye is level with the canopy top.
2. The **canopy-hull field** (world-space, 25 cm, built once from the stand's real
   scatter) gives the local canopy height, which species entry owns the column,
   and that plant's wind phase. Bare ground where no plant covers the column.
3. One closed-form walk from the shell crossing to the canopy-top crossing
   (clamped, see Findings), then the entry is snapped onto the top plane.
4. The query is un-sheared by the **exact inverse of the harness wind shear**. The
   harness displaces a point by `D * (height fraction)`, which is linear in
   height, i.e. a shear matrix `M = I + D⊗e_y`; applying `M⁻¹` (Sherman-Morrison,
   3 mul-adds) to the entry point and direction, looking up the un-sheared table,
   then mapping the hit back through `M` is exact for that wind model. No wind
   states are baked and nothing is approximated away.
5. **1-4 taps of the 4D visibility table** return where the canopy first blocks
   that ray: `1/(1+distance)`, oct normal, coverage — plus 1 tap for albedo + sky
   occlusion.

Per-pixel fetch count (constant, no loops anywhere):

| purpose | fetches |
|---|---|
| canopy-hull field at the shell footprint (bilinear) | 1 |
| terrain heightmap re-anchor at the entry (`terrain_sample`) | 4 texel loads |
| canopy-hull field at the entry (bilinear) + species (`textureLoad`) | 2 |
| ray table A (dist/normal/coverage): 1 nearest, 2 anisotropic, 4 with θ-lerp | 1-4 |
| ray table B (albedo/AO) | 1 |

**= 5-8 filtered fetches + 4 heightmap loads per pixel, whatever the stand.**
Cost is bounded by screen pixels: the shading is identical for 557k and 134M
plants, and even the field build is a fixed-size region, done once (not per
frame) unless the camera moves off it by 32 m.

## VRAM budget math

Per species (HUD-verified, MiB):

- `rayA` 128x128 x 160 layers (16 azimuth x 10 elevation) rgba8 + 8 mips: 14.0
- `rayB` 64x64 x 160 layers rgba8 + 7 mips: 2.7
- **total 16.7 / 25 MiB per species** (all three species identical)

Shared, charged to no species (documented rather than hidden):

- canopy-hull field rgba16float 1024² (256 m at 25 cm): 8.0 MiB
- atomic pack buffer r32u 1024²: 4.0 MiB
- uniforms: noise

Whole experiment: 71.5 MiB total for the default stand's three species
(16.7 x 3 + 12 shared + render targets), i.e. **20.7 MiB/species** if the shared
field is divided three ways — inside the 25 MiB budget without an exception.

Bake-time transient (tagged `bake-scratch`, destroyed at the end of each bake):
3 atomic accumulation buffers 28 MiB each, three 3D volumes (128x107x128:
rgba16f 14 MiB + 2x rgba8 7 MiB), the repacked source mesh (12 B/vertex +
12 B/triangle: 78 MiB for poa's 6.5M verts), 2 readbacks of 10.5 MiB. Peak
~250 MiB, same ballpark as 001. The raw 16 B records had to be repacked because
poa's vertex buffer alone (135 MB) exceeds `maxStorageBufferBindingSize`.

## Bake

`mesh/baked/037-horizon-visibility-field/hvf-v6-<species>-d8.50-s<min>-<max>.bin`,
16.7 MiB each (three files, 50 MiB total). Keyed by species + density + scale
range, so a different stand re-bakes rather than silently reusing a wrong table.
Every load is magic-validated (the dev server answers missing `/mesh/baked` files
with `index.html` at HTTP 200, which poisons both the OPFS cache and the
committed-file path); a poisoned entry is re-baked and repaired in place.

Three GPU stages, all offline, ~25 s per species on this machine:

1. **voxelize** — a periodic tile of the entry's real GCMESH1 plants (1.2 canopy
   heights across = 1.77 m for calamagrostis, 27 plants at the stand's total
   density, random yaw/scale/offset from the shared PCG hash, wrapping at the
   tile edge) is splatted by one thread per (triangle x instance) — 58M threads
   for calamagrostis, 78M for poa — into a 128x107x128 volume as **projected
   surface area per axis** (`area*|n.x|`, `area*|n.y|`, `area*|n.z|`), plus total
   area, area-weighted mean normal and mean authored colour, in u32/i32 fixed
   point with atomics.
   Per-axis projected area is the single most important thing in this bake. A
   scalar density with a mean normal is *wrong* for grass in two ways: the mean
   normal cancels between the two faces of a blade, and a canopy of near-vertical
   blades is strongly anisotropic — it stops a horizontal ray dead and barely
   touches a vertical one. Measured on the source mesh: calamagrostis has
   0.698 m² of leaf area per plant but only 0.243 m² projected downward. With the
   mean-normal model the baked canopy came out ~3.6x too transparent (vertical
   coverage 0.17); with `dot(abs(dir), A)` it is 0.47 vertical / 0.996 grazing,
   which is what a real grass canopy does.
2. **resolve** — one thread per column turns the accumulators into three
   filterable 3D textures and sweeps vertical transmittance top-down in the same
   pass, so sky openness (AO) comes out for free and exactly (it is the y
   component of the projected area).
3. **trace** — the precomputed raycast: for every (entry point on the canopy top)
   x (azimuth, elevation) — 128² x 160 = 2.6M rays — the volume is integrated
   once and the ANSWER stored. Coverage is `1 - transmittance`; the distance is
   where accumulated coverage crosses one half (a real silhouette depth), falling
   back to the coverage-weighted mean; normal/albedo/AO are coverage-weighted
   means, with the normal flipped toward the incoming ray (two-sided foliage) and
   blended toward the anisotropy axis where the voxel's normals cancel. This is
   the ONLY place a ray is ever marched.

Mip chains are built on the CPU, coverage-weighted (normals averaged as decoded
vectors, not as oct bytes). The coarse levels are additionally blurred across the
**direction** dimension — a genuine 4D prefilter — because at those levels the
answer depends only on (azimuth, elevation) and neighbouring bins differ enough
to paint the direction grid across a distant view.

## Status

wip — verified by headless screenshots at all four standard cams plus
`debug=normals/depth/lighting/albedo/coverage` and the A/B page, zero console
errors, typecheck clean. It renders, it is fast, all five debug views are
meaningful, and it is a genuine per-pixel ray answer with real depth. It does
**not** beat the billboard baseline on looks at every camera, so it stays `wip`
rather than `working` — see the honest verdict below.

## Findings

### Speed — parity with billboards, sometimes better

Same-frame A/B (`#/ab/001-billboard-smoke/037-horizon-visibility-field`),
GPU heavily contended by sibling agents, so only the RATIO is quoted:

| cam | A (cull+cards) | B (canopy) | B/A |
|---|---|---|---|
| grazing | 0.41 + 2.55 | 3.53 | 1.19 |
| inside-plant | 0.41 + 2.71 | 2.25 | 0.72 |
| topdown | 0.26 + 1.75 | 2.52 | 1.25 |
| far-horizon | 0.15 + 1.27 | 1.27 | 0.89 |

Across the whole session's A/B runs the ratio sat between 0.72 and 1.25, i.e.
**parity**, with solo canopy-pass p50 measured between 0.8 and 3.5 ms at
1280x800 depending on camera. The field build is a one-off 1-3 ms on the frame
where the camera moves 32 m, not a per-frame cost. No `results/` bench JSONs are
claimed: other agents were rendering throughout, so bench numbers would be
contaminated (CLAUDE.md rule). Re-run `#/bench/037-horizon-visibility-field` on
an idle GPU before quoting absolute milliseconds.

### Looks — better than billboards at moderate elevation, worse at grazing

What genuinely works, and what a billboard cannot do:

- Real per-pixel depth (`frag_depth` from the reconstructed hit), so the canopy
  depth-composites against terrain and against itself. The depth view shows
  metre-scale variation *within* the canopy, not one plane per plant.
- Real parallax and self-occlusion: the hit point moves through the canopy as the
  camera moves, and the elevation dimension of the table makes coverage rise from
  0.47 (straight down, you see soil between blades) to 0.996 (grazing, a wall of
  blades) — a directional silhouette change that flat cards cannot express.
- Clumping, height variation and species mosaic come from the real scatter
  through the hull field, so per-plant scale scales the canopy pattern too.
- Ground contact is baked: where foliage does not block the ray, the answer is
  the ground with the canopy's own AO, and the contact darkening follows the
  actual local canopy height.
- Wind is exact for the harness model (inverse shear), not approximated.
- Prefiltering is honest: distance blurs the *answer* (mip + direction blur), so
  the far field is an averaged canopy. No dithering anywhere — the only alpha
  decision is a hard test on baked coverage, and the region rim / inside-canopy
  fades erode that coverage through the same hard test.

Where it loses to billboards, plainly:

- **Near-horizontal views (the `grazing` and `far-horizon` bookmarks, 3.4° below
  horizontal) are the method's weak point.** A pixel's footprint on the
  canopy-top plane is *metres* long at that elevation, so the honest prefiltered
  answer is low-frequency: the carpet reads as smeared grain and soft blobs
  rather than blades. Three separate mitigations went in (anisotropic 2-tap
  filtering along the ray with the anisotropy capped at 8x, a two-octave
  continuous warp of the tile lookup to break the lattice, and an explicit
  convergence onto the table's mean canopy below 4° so the residue reads as a
  distant mat instead of marbling) and it is still visibly worse than billboard
  cards there. This is inherent to parameterising the table by the canopy-top
  entry point: at grazing the entry sweeps a metre per pixel row.
- **The eye below the canopy top** (`inside-plant`) is outside the
  parameterisation: the canopy-top crossing is behind the eye. Clamping the entry
  to just in front of the eye keeps something on screen (a dark wall of grass a
  couple of metres out, which is what being inside a canopy looks like) but it is
  blocky. The README's camera-inside allowance covers it; it is not pretty.
- **Near-field sharpness is capped by table resolution**: 128 texels over a 1.77 m
  tile = 1.4 cm/texel, so within ~3 m a texel covers several pixels and blades go
  soft. A 4D table cannot be given blade-scale (5 mm) resolution inside 25 MiB;
  the honest fix is a 5th dimension (entry height) or a near-field detail layer,
  neither of which fits this round.
- Tiling is the known weakness of any baked ray table (the source article hides
  it with texture bombing). The warp hides the lattice at moderate angles; at
  grazing it is still occasionally readable.

### Stand fidelity — what is exact and what is not

Exact: which species entry owns each 25 cm column, the canopy top height there,
the per-plant scale (it scales the canopy pattern vertically), the per-plant wind
phase, and bare ground — all stamped from `scatter_candidate()` (the shared WGSL
twin, including the wetness/habitat and carpet branches) for every plant of every
stand entry, with atomicMax picking the tallest plant per column.

Approximated, and stated plainly: the sub-25 cm blade arrangement is a periodic
1.77 m tile of the same species at the stand's total density, not the exact
plants. An O(1) table cannot store the whole region's blade arrangement — that is
information-theoretically impossible at 17 MiB — so what a ray table can honour
is *which plant is where and how big it is*, which is what the field does. The
tile is populated at the stand's TOTAL density (8.5/m² on the default stand)
rather than the entry's own, because a column the scatter gives to calamagrostis
also has the other entries' plants beneath it.

### Things worth trying next

1. Add entry height as a 5th table dimension (4 bins) and let each shell use the
   layer matching its own height. That directly attacks both the grazing smear
   (the entry would be near the hit) and the inside-plant case. Cost: 4x memory,
   so it needs 64² spatial or 12x8 directions.
2. A near-field detail layer: a small tileable blade texture modulating albedo and
   normal below ~3 m, where the table is out of resolution anyway.
3. Anisotropic filtering with 4 taps along the ray instead of 2, only below 10°.
4. Toroidal (clipmap) update of the hull field instead of a full rebuild on the
   32 m snap — currently a 1-3 ms hitch when the camera crosses a boundary.
