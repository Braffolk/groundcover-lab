# dissolving canopy

## Idea

One rule drives the whole method: **how continuous a canopy looks depends on the
viewing elevation, not just on distance.** Looking along a meadow you resolve
individual plants a hundred metres out; looking down at the same meadow you see
one closed surface almost immediately. So the renderer keeps three
representations of the *same* stand and dissolves between them with a field
`d = smoothstep(0.5r, r, dist)`, where the radius `r` itself interpolates from
`dissolveGrazing` (106 m) to `dissolveOverhead` (16 m) by the viewing elevation
of the point being drawn:

1. **Near — a plant is a cloud of 4 sub-tuft splats.** The bake partitions the
   mesh's *triangles* into the four xz quadrants of the plant and captures each
   quadrant from 8 azimuths. At draw time each quadrant is a camera-facing quad
   anchored at *its own* centre, so the four sit at four different world
   positions and depths. That buys what one card cannot have:
   - parallax INSIDE one plant (near quadrants slide against far ones),
   - a silhouette that changes shape with view direction instead of rotating
     rigidly, because the quadrants occlude each other differently,
   - real self-occlusion (they depth-test against each other) plus a shading
     term that darkens the quadrant pointing away from the camera, because it
     is deeper inside the volume,
   - reprojection error that scales with a node's radius, so a quarter-plant
     node is ~2x more faithful off its baked azimuth than a whole-plant card,
   - wind evaluated at each quadrant's own position, so a plant twists instead
     of shearing as one flat card.
2. **Mid — one whole-plant splat** (atlas node 0). Past ~12 m the intra-plant
   parallax is a couple of pixels, so it is not worth paying for.
3. **Far — one continuous canopy shell.** A bufferless two-level snapped grid
   (0.75 m out to ±36 m, 3 m out to ±144 m, clipped to the stand) displaced by an
   analytic canopy relief and textured with ONE tileable canopy texture,
   composited at init from the baked per-species top-down captures at the active
   stand's densities. Its coverage threshold starts above 1.0 near the camera —
   nothing can pass, and those quads are collapsed in the vertex stage — and
   falls to `shellFloor` far away, so the same texture reads as "islands of dense
   clumps" through the hand-off band and as a closed canopy beyond it. This is
   the LOD collapse with teeth: thousands of distant plants become a few thousand
   triangles and 2 texture taps, and from a high camera the splat set empties out
   completely (the cull rejects every plant whose fade reached zero), leaving
   only the surface.

Two physical corrections make the shell agree with the splats instead of looking
like a different biome:

- **Path-length coverage.** The baked coverage is what the canopy hides looking
  straight *down*; a ray at elevation θ travels 1/sin θ as far through the same
  layer, so its coverage is `1-(1-c)^(1/sinθ)`. One `pow()` closes the far field
  at grazing (a meadow is see-through from above and opaque edge-on) and, since
  it can never exceed 1, it never resurrects the near field the dissolve
  threshold killed.
- **View-dependent albedo and normal.** From above, the visible surface *is* the
  canopy top: use the composited top-view colour and the surface normal. At
  grazing you see the sides of blades, which statistically face the viewer, so
  the shading normal rotates toward the camera (the same "flip toward the bake
  camera" convention the splats' baked normals use) and the albedo becomes the
  baked *side-view* mean hue — split into a flower-head band and everything
  below it, picked per texel by the baked canopy height, and darkened by a
  canopy-interior factor because an opaque shell has to stand for the whole
  self-shadowed volume behind the first blade. Without these the far field came
  out as a bright olive-tan carpet against a pink-green near field.

O(1) in plant count: nothing is ever materialized for the whole stand. Per frame
the cost is the region area (dissolve radius) plus the shell's screen area —
identical for 557k plants (default) and 134.2M (scaling-100m), verified below.

## VRAM budget math

Per species (HUD-verified on the default stand: 16.4 MiB):

- coverage atlas r8 1280x4096 + 6 mips — 6.96 MiB. This is the alpha-tested
  silhouette, so it gets full resolution: ~500 texels/m, i.e. billboard parity.
- albedo atlas rgba8 640x2048 (half res) + 5 mips — 6.96 MiB (a = coverage, so
  the mip chain stays coverage-weighted)
- oct normals rg8 320x1024 (quarter res) + 4 mips — 0.87 MiB
- culled instances, sized for the param maxima (36 m tuft radius, 112 m region)
  at the entry's density, 16 B/instance — 2.30 MiB at density 3, 3.5 MiB at 5
- entry info uniform (336 B) + indirect args (32 B) — noise

Shared by all species (tagged `canopy-shell`, deliberately NOT charged to a
species row): the canopy tile, 2 x rgba8 768x768 + 9 mips = **6.29 MiB**, i.e.
2.1 MiB per species if amortized. Splitting the channels by the resolution each
actually needs is what pays for it: a plain rgba8 atlas at the same silhouette
resolution would be 21 MiB per species.

Totals: default stand 16.4 / 16.0 / 16.4 MiB per species + 6.3 shared → ~18.5
MiB effective per species. Worst defined stand (dense-mixed, density 5): 17.8 /
17.1 / 17.1 + 6.3 → ~20 MiB. Inside the 25 MiB budget in every stand.

The bake uses ~150 MB of transient GPU memory (raw vertex/index buffers, a
2560x2048 supersampled MRT chunk, readbacks), allocated via `ctx.res` tagged
`bake-scratch` and destroyed at the end.

## Bake

`bake.ts` → `mesh/baked/026-dissolving-canopy/canopy-v2-<species>.bin`,
11.35 MiB each (34 MiB for the three species; no stale variants).

Per species, in one render pass per two azimuth rows (chunked to cap transient
VRAM):

- **Node atlas**, 5 nodes x 8 azimuths, 256x512 px tiles. Node 0 is the whole
  plant; nodes 1..4 are the four xz quadrants of the *triangle* set, partitioned
  by centroid, so the four tiles reassemble the plant exactly. Each node's box is
  its own content bounds (a cylinder radius for the width, so a camera-facing
  quad is valid from every azimuth) with a 3% margin. Rendered at 2x and reduced
  three ways: coverage 2x2 → full res, colour 4x4 coverage-weighted → half res,
  normals 8x8 coverage-weighted → quarter res; colour and normals are then
  dilated a few texels into empty space so filtering never pulls in background
  black. Normals are flipped toward the bake camera (two-sided foliage). All
  nodes share one azimuth grid (see the findings: rotating them per node broke
  blades that cross a quadrant boundary).
- **Top-down capture** of the whole plant, 160 px: albedo+coverage, oct normal,
  and the height fraction of the topmost surface. `canopy.ts` composites these
  into the shell's tileable canopy texture at init — not at bake time, because
  the species mix and densities belong to the stand, not to a species. Stamps use
  the shared PCG hash (deterministic), one of 8 free orientations (no resampling
  blur), torus wrapping, coverage accumulated as independent occluders
  `1-Π(1-c)`, and colour/normal/height composited front-to-back (pass 1 finds the
  canopy top per texel, pass 2 weights every copy by its coverage attenuated by
  how far below that top it sits).

Mip chains are generated on the GPU at load; tile sizes stay even at every
generated level, so a 2x2 box never crosses an atlas tile. The oct-normal variant
decodes, averages and re-encodes rather than box-filtering oct pairs. Every load
is magic+size validated, because the dev server answers missing /mesh/baked files
with index.html at 200.

## Status

working — verified by headless screenshots at grazing, topdown, inside-plant and
far-horizon on the default stand, the same four on scaling-100m (134.2M plants),
dense-mixed, all five debug views, and A/B wipes against 001-billboard-smoke.
Zero console errors or warnings.

## Findings

### Performance (A/B, in-frame, CONTENDED — ratios only, never the ms)

`#/ab/001-billboard-smoke/026-dissolving-canopy?cam=grazing&seed=42` at
1280x800, summing `A/cull + A/cards` vs `B/cull + B/canopy`:

| cam | ratio B/A |
|---|---|
| grazing | 1.22x |
| topdown | 0.66x |
| far-horizon | 1.07x |
| inside-plant | 1.08x |

~1.0x on average, worst case 1.22x, and 1.5x *faster* than the baseline from
above.
15 sibling agents were rendering on this GPU throughout, so absolute numbers are
worthless — the `composite` pass (a fullscreen blit) alone reported 2.5-3.5 ms.
No `results/` bench JSON is claimed for the same reason; rerun
`#/bench/026-dissolving-canopy?stand=default` on an idle GPU before quoting ms.

Sweeps at grazing (`b.shell=0`, `b.tuftRadius=6..20`) moved the ratio by less
than the run-to-run noise (1.21-1.24x), i.e. neither the shell nor the near band
is where the time goes — the far band (one splat per plant out to 106 m) is, and
that is the same work the baseline does with its cards.

Plant-count independence: default (557k) vs scaling-100m (134.2M) at the four
standard cams — 6.5/2.8/6.2/6.5 vs 6.3/3.3/8.4/6.2 ms Σp50, i.e. equal within
the contention noise, same VRAM.

### Looks — does it beat the baseline?

Yes at every camera, with one honest caveat.

- **Near field (grazing, close-down, inside-plant): clearly better.** The splat
  cloud gives visible depth layering — panicles occlude other panicles, stems
  read at different depths, and the back-quadrant shading makes plants look like
  volumes. The baseline's worst artifact is simply gone: no pale floating top
  cards (light beige discs all over its close-down view). Both methods soften
  under magnification (<1 m), and there the baseline is the sharper of the two:
  its silhouette carries ~690 texels/m of one unbroken capture against my ~500
  split across four, so from *inside* the canopy its panicles resolve finer
  spikes than mine. Everywhere further out my depth structure wins.
- **Topdown: better, and 1.5x cheaper.** The baseline's top cards read as
  discrete bright discs; the shell is a continuous clumped canopy with
  height-driven occlusion and a one-tap heightfield sun shadow, and the splats
  are gone entirely (the elevation-driven dissolve rejects them in the cull).
- **Far field: better in coverage, weaker in character.** The baseline stops at
  its region radius and shows bare terrain past it; the shell carries the canopy
  to the stand edge. But a horizontal surface cannot show the *vertical*
  structure that gives a distant meadow its texture, so past ~106 m mine reads as
  a well-textured mat rather than as grass. That is the one place the baseline's
  cards look more alive, and it is exactly why `dissolveGrazing` ended up at
  106 m: the splats are better at grazing, so they keep the field, and the shell
  takes the deep distance and the high-elevation views, where it is better than
  they are.

### Things that were wrong along the way (all fixed, all instructive)

- **Two render passes cost a full pass worth of contention noise.** The first
  version timed `tufts` and `shell` separately and measured 2.2x; merging them
  into one pass — which is also the right ordering: near band, far band, then the
  surface behind them, so early-z does the culling — brought it to 1.2x. The
  radial band sort is what makes that ordering free: the cull writes each band
  into its own segment of the instance buffer and each band binds its own view of
  it, because WebGPU forbids a non-zero `firstInstance` in indirect draws without
  an optional feature.
- **A top-down capture of grass is only 5-15% covered.** Blades are edge-on from
  above, so the composited canopy tile is ~49% covered and the shell was
  invisible at grazing (its threshold sat above the coverage) — for a while I was
  looking at bare terrain and calling it the shell. The path-length correction is
  the physically correct fix and costs one `pow()`.
- **The mean of that capture's height channel is a third of the plant height**
  (the wide lower leaves dominate the view from above), which sank the shell into
  the ground. It now uses the 85th percentile of the per-texel canopy top for the
  surface height and the 97th for the height normalization.
- **Two hues, not one.** A single mean side colour makes the far field either
  khaki (flower band only) or green (whole plant). It now mixes the two bands per
  texel by the baked canopy height, with the gain chosen so a fully mip-averaged
  sample lands on a flower-dominant mix — because at grazing the heads are the
  outermost layer a horizontal ray hits, which is also why alpha-tested cards
  keep their heads and lose their stems at distance.
- **Rotating each node's azimuth grid was a bad trade.** Offsetting node j's 8
  views by j/5 of a step spreads the view-switch pop beautifully — but it also
  means two adjacent quadrants reproject their *shared* blades from azimuths up
  to 9 deg apart, and at 1 m that displaces the two halves of one panicle by
  ~30 px. Close up, plants looked like disconnected cauliflower lumps. All nodes
  now share one azimuth grid (bake v2): blades stay connected, and the pop is
  coherent, i.e. no worse than the baseline's single-card pop.
- `macro` is a WGSL reserved keyword (worth adding to CLAUDE.md's list).

### Known limits, left as honest behaviour

- Beyond `dissolveGrazing` (106 m) the far field is the shell, which has no
  vertical structure. With fog it reads acceptably; without fog it would not.
- The shell's relief is analytic value noise, not the actual per-plant canopy, so
  its lumps do not line up with the texture's clumps. Invisible past ~45 m, and
  it is what makes the two grid levels agree exactly on the surface (no crack, no
  swimming) — a texture-driven displacement at two different mip levels would
  not.
- The 12 m canopy tile repeats. A slow macro tint variation and the relief noise
  break it up; at grazing it is invisible, from 42 m straight down you can find
  it if you look for it.
- Magnification below ~1 m goes soft, like every impostor method here.
- The dissolve frontier is camera-relative, so walking forward makes distant
  clumps dissolve in and out. It is spread over ~50 m at grazing, so the
  per-frame change is small, but it is there.
- The per-entry info uniform (336 B x 3) is rewritten every frame although only
  the planes, region rect and params change; the node table never does. Not worth
  the field-order churn.
- The shell's fine grid level (±36 m) is entirely skipped at grazing (the
  dissolve kills everything inside 47 m there) but its 9216 instances still run
  the vertex stage. It is needed for high-elevation views, and a CPU-side test
  would have to know the per-cell elevation, so it stays.

### Harness wishlist

None. `bakedArtifact` + `commitBake` + the shared scatter twin covered
everything; the only shim is the magic validation for the SPA-fallback 200.
