# statistical assemblage

## Idea

A deterministic marked Cox process authors a physical 0.42 m, 2048² Sphagnum
tile. Each accepted event carries correlated size, posture, openness, vitality,
pigment, and elevation. Capitula are coherent phyllotactic crowns plus spreading
branch packets; repeated lower fascicles add spreading and pendent cohorts at
continuously descending heights. The top four distinct surfaces are depth-ranked
per texel, so the same structural events drive height, tissue age, pigmentation,
and aperture/AO instead of combining unrelated noises.

An irregular compact-support cushion field coordinates local shoot density,
compactness, posture, size, and baseline height. Its geometry correlation is
deliberately stronger than its pigment correlation: the existing light/dark
mosaic reinforces cushion state without becoming a one-to-one colour copy.
Within strong cushions, per-shoot baseline jitter is reduced while the shared
cushion offset is increased; the height relief inside every individual crown is
unchanged.

All population work is authored offline into one descriptor PNG. Runtime work is
constant in plant count: one sampled descriptor subtree feeds ordinary POM,
albedo, height-derived normals, AO, and the WGSL BRDF. A small C1 per-tile
interior warp decorrelates repeats while leaving periodic boundaries fixed.

## VRAM budget math

The material report measures **22,369,620 bytes (21.33 MiB)**. This is one
2048² RGBA8 descriptor with its mip chain; graph-derived height, albedo, normal,
and AO nodes remain live and allocate no additional resident material maps.
There are no experiment-owned GPU buffers. The renderer-oriented 25 MB budget
is not material-calibrated, but this experiment remains below it anyway.

## Bake

No source mesh bake. `author.ts` runs the deterministic TypeScript structural
generator and writes `assets/descriptor.png` plus diagnostic captures. Seed 2507
currently authors 7,747 structural events (6,966 capitula and 781 exposed
fascicles), corresponding to 43,917 events/m² and a mean nominal capitulum width
of 12.08 mm.

The final raster has 100% causal coverage and no artificial zero-height pixels.
Only 4,017 texels (0.096%) required leaf-contact closure, which converged in
three texel steps (about 0.62 mm) by extending the deepest adjacent real packet's
surface id, age, pigment, and depth. This operation is bounded below a typical
branch-leaf scale and cannot become a hidden substrate layer.

## Status

Working. Ordinary POM with pixel depth offset and an eight-step sun march is
active; microshadow defaults on. The descriptor, graph validation, and all
required channel views pass without console errors, WebGPU errors, or toasts.
The experiment deliberately does not attempt SPOM geometry joins or outward
silhouettes; that is the separate prism track.

## Findings

- Current depth overlap fractions are 100.0%, 99.41%, 98.10%, and 95.57% for
  layers one through four. The lower cohorts therefore replace catastrophic POM
  wells with dark, low, branch-shaped biomass while retaining visible apertures.
- Cushion state has 0.62 Pearson correlation with event baseline height and
  0.30 with pigment. This is strong enough to form coherent microtopography but
  weak enough that red/ochre regions do not simply trace bump contours.
- QA captures cover plane macro, sphere and cube at tileScale 1, and sphere and
  cube at tileScale 8 / reliefGain 2 / 2048 px. Raw height, albedo, normal, and
  AO captures are also present under `captures/`.
- A standard grazing comparison at 36 versus 56 POM steps showed no meaningful
  visual difference. The default stays at 36 for performance. Very highly
  magnified, almost-sideways inspection can still expose relief-march terracing;
  it is recorded rather than hidden by reducing relief depth.
- Reference architecture: Flora of North America describes successive fascicles
  with typically two spreading and one or two pendent branches; Clymo measured a
  2–6 cm green canopy below capitula depending on species. Modern morphology and
  water-retention work likewise identifies interleaved branches and colony
  packing—not an empty floor—as the structure beneath the visible surface:
  <https://bryophyteportal.org/portal/taxa/index.php?tid=157188>,
  <https://www.fs.usda.gov/nrs/pubs/jrnl/2018/nrs_2018_weston_001.pdf>,
  <https://doi.org/10.3390/plants13081061>, and
  <https://research.sbcs.qmul.ac.uk/r.clymo/Clymo-article-PDFs/10-Clymo-1970-Sphagnum-measurement.pdf>.
