# clump impostors

## Idea

A billboard is flat because a plant is drawn as **one** quad. So don't draw one:
cut the plant into K_SUB = 4 **sub-clumps** and give each its own camera-facing
card, standing at that sub-clump's real offset inside the plant.

Four cards ~0.2 m apart in world space buy every depth cue a single card cannot:

- **parallax inside the silhouette** — the cards are at genuinely different
  depths and positions, so moving the camera slides the near sub-clump across
  the far one. It is not an effect, it is real 3D placement.
- **a silhouette that changes with view direction** — the union of four offset
  shapes re-arranges as you orbit (a sub-clump that was hidden behind another
  swings out to the side). A single card can only rotate.
- **self-occlusion and interleaving** — each card writes depth, so sub-clumps
  occlude each other *and* weave into the neighbouring plants' sub-clumps
  instead of stacking like cutouts. This is the cue that makes a meadow read as
  a volume rather than a deck of cards.
- **ground contact** — four bases touching the soil at four places instead of
  one card standing on a line.
- sub-clumps sway slightly out of phase with each other (`swaySpread`), which is
  what sells them as separate masses in motion.
- as a bonus, the 8-azimuth snap 001 pops on is spread over four cards that
  switch view at four different camera angles, so each pop is a quarter the size.

The cut has to be invisible, and that is where the source meshes cooperate: the
generators emit **one blade at a time**, so a whole blade is exactly a maximal
run of consecutive vertices with no positional jump. Segmenting on a 6 cm jump
threshold yields 158 / 851 / 221 strands for calamagrostis / elymus / poa, with
0.05–0.26 % of triangles straddling a run (those follow their first vertex).
Cutting on **strand** boundaries — never on triangles — means no blade is ever
sliced in half between two cards. Strands are then k-means clustered (weighted by
vertex count, deterministic angular seeding, no RNG) into 4 spatial clumps.

Three more things carry their weight:

1. **Per-view tight crops.** After the capture, every tile is scanned for its
   exact alpha bounding box and the runtime quad spans only that. No empty margin
   is ever rasterized — this is what pays for most of the extra cards, and the
   merged card benefits too (001 rasterizes its margin).
2. **Coverage-preserving alpha, on every mip level.** Splitting a clump splits
   each texel's partial coverage 4 ways, and a hard alpha test is non-linear: two
   half-covered sub-clump texels can both vanish where the one merged texel
   survived. So each tile's true geometric coverage is measured from the
   supersampled capture, and each mip level's alpha is scaled by one scalar (a
   histogram quantile) so the fraction of texels passing the alpha reference
   equals that truth. Sub-clump and merged cards then have the same density, the
   LOD switch does not pop, and grass stops thinning out with distance — with no
   dithering anywhere.
3. **Front-to-back rings.** The cull bins every survivor by distance into
   equal-area rings (3 near + 4 far) and each ring is its own indirect draw,
   issued nearest first, so hard alpha test + depth write let the near rings
   occlude and early-z reject the deeper ones. (Honesty note: the 25 % win I
   first measured for this was an artifact of the `firstInstance` bug below. Its
   real benefit is unmeasured — see Findings.)

Per frame: one compute pass evaluates the shared scatter WGSL twin over a
camera-centered cell region (cell rect CPU-clamped to the stand), rejects whole
cells against the region circle and the frustum before touching the scatter,
frustum-tests each plant, and compacts survivors into the ring lists. Then one
render pass, 7 indirect draws per stand entry. Cost is O(visible region), never
O(plants): the `default` stand (557 k plants) and `scaling-100m` (134 M) measure
identically.

**LOD.** `lodDistance` (default 14 m) is the collapse: inside it a plant is 4
sub-clump cards + 4 top cards; outside it a plant is exactly one merged
whole-plant card + one top card, i.e. a plain billboard. Distant plants therefore
cost a quarter of the vertices and a smaller quad, and the far field is *cheaper*
than 001's because of the tight crop. `lodDistance = 0` collapses the method to a
pure billboard renderer, which makes an honest self-A/B control.

Fragment shader: 2 texture taps, hard alpha test, depth write, no `frag_depth`,
no loops, no marching, no dither. Camera-inside erosion is per **card** (a
sub-clump card can stand 0.25 m off the plant axis, so a per-plant test happily
leaves one card 0.1 m from the eye as a screen-filling smear); top cards erode
from twice that distance and use a **signed** elevation test, which removes the
"pale pancake overhead" artifact 001 shows at eye level.

## VRAM budget math

Per species, atlas 1280×2464 with (K_SUB+1)×9 = 45 tiles: sub-clump side tiles
160×512, merged side tiles 160×256, top tiles 160×160. Every tile origin and size
is a multiple of 32, so a 6-level mip chain never mixes two tiles into one texel —
the atlas is bleed-free by construction.

- albedo rgba8 (rgb + coverage), 6 levels: 4 204 200 texels × 4 B = **16.04 MiB**
- normals oct rg8 at **half** resolution (640×1232), 5 levels: 1 050 280 × 2 B =
  **2.00 MiB** (halving the normals is what buys the sub-tiles their resolution;
  coverage stays full-res, only lighting is filtered)
- far instance rings, 4 × 46 848 × 16 B = **2.86 MiB** (circle bound at
  regionRadius 128 and density 3, +20 % slack, split over equal-area rings —
  tighter than the region's square bound; ring capacities are rounded to 16
  instances so each ring's byte offset is 256-aligned and bindable)
- near instance rings, 3 × 7 552 × 16 B = **0.35 MiB**
- two uniforms per stand entry (192 B dynamic + 1 616 B static atlas layout):
  noise

**Total 21.25 MiB / 25 MiB** per species on the `default` stand (bench-verified:
21.25 / 20.72 / 21.25). Worst defined stand `dense-mixed` (density 5) fits at
**23.3 MiB**. The static uniform (atlas layout + per-unit capture boxes + 45 tile
rects) is written **once**, never per frame.

Texel density vs 001: a sub-clump tile resolves ~0.30 m across 160 px (533 px/m)
and 1.15 m across 512 px (445 px/m); 001's card is 731 × 445 px/m. So per-blade
sharpness at sub-metre range is ~0.73× 001's horizontally — the honest price of 5
units instead of 1 in the same 25 MiB. See Findings.

Bake transient: ~300–500 MB of GPU scratch (raw vertex+index buffers, two
2560×4928 rgba8 targets + depth, one readback at a time), all tagged
`bake-scratch` and destroyed.

## Bake

`bake.ts` → `mesh/baked/018-clump-impostors/clumps-v2-<species>.bin`, 17.5 MiB
each (1 KB header + the whole albedo mip chain + half-res oct normals). Steps:

1. Segment vertices into strands (6 cm jump in vertex order).
2. Weighted k-means over strand centroids → 4 sub-clumps; build a cluster-major
   index buffer so each unit is one `drawIndexed` range (the merged unit is the
   whole buffer, no extra memory).
3. Per unit: bbox centre in xz, exact max horizontal radius, y range.
4. Render 45 tiles orthographically at 2× (one command submission per unit — 45
   viewport draws over 6.5 M triangles in one buffer is a watchdog risk),
   albedo+coverage and view-flipped mesh normals.
5. Coverage-weighted downsample; normals straight from 2× to half resolution;
   per-tile true coverage; per-tile tight alpha rect; colour dilation clamped to
   each tile.
6. CPU mip chain with the per-level coverage calibration described above (this is
   why the albedo chain ships baked instead of being generated at load — a box
   filter would undo it). Normal mips are still generated on the GPU, but by
   decoding, averaging as vectors and re-encoding, so a distant tile keeps a real
   mean direction instead of an average of oct coordinates.

Artifacts are magic+size validated on every load because the dev server answers
missing `/mesh/baked` files with `index.html` at HTTP 200; a poisoned OPFS entry
is rebaked and repaired in place. First bake of all three species ≈ 60 s.

## Status

**working** — verified by headless screenshots at `grazing`, `topdown`,
`inside-plant`, `far-horizon`, plus `debug=normals`, `debug=lighting`,
`debug=coverage`, plus the A/B page against 001 at four cameras and hand-placed
in-canopy close-ups, on `default`, `scaling-100m` and `dense-mixed`. Zero console
errors AND zero harness toasts (see the bug below — toasts are where WebGPU
validation errors surface, and my verification script now scrapes them).
`npx tsc --noEmit` clean.

## Findings

### The bug that invalidated my first round of numbers

The ring draws pass each ring's slice base as `firstInstance` in the indirect
args. **WebGPU rejects a non-zero `firstInstance` in an indirect draw unless the
optional `indirect-first-instance` feature is enabled**, and the harness does not
request it. So six of my seven draws per stand entry were silently dropped: only
near ring 0 and far ring 0 ever rendered. The renderer looked plausible — a
continuous meadow — because the two surviving rings covered 0–10 m and 14–57 m;
what it actually had was a missing annulus at 10–14 m and nothing beyond 57 m.

It hid well because the validation error goes to `device.onuncapturederror` →
the harness's `onError` → an on-screen **toast**, not the console: my screenshot
script was reporting "0 console errors" the whole time. It also produced a
plausible-looking *speedup*, which is how I nearly shipped "front-to-back rings
cut grazing cost 25 %".

What found it: painting near-LOD cards red and sweeping `lodDistance`. At
`lodDistance = 30` the red band stopped at ~17 m and everything from 17–30 m was
bare terrain — geometry that must be there was simply absent. Fix: bind the
instance buffer at the ring's byte offset in a per-ring bind group (ring
capacities rounded to 16 instances so the offsets are 256-aligned) and keep
`firstInstance = 0`. Everything below is measured *after* that fix.

Two earlier conclusions died with it: the "open band inside lodDistance" I blamed
on split coverage was the missing annulus (so `nearAlphaBias`, added to
compensate, now defaults to 1 = no-op and is just a density knob), and the ring
ordering's real benefit is **unmeasured** — I attempted a 1-ring control build to
quantify it, it failed to come up, and I reverted rather than ship a broken
config. Rings cost nothing measurable to keep, so they stayed.

### Speed — modestly slower than 001, worst case 1.35×

The GPU went quiet late in the session, so these are stable, repeatable numbers
(4 consecutive samples within ±0.02 ms), not contended guesses.

**Bench** (authoritative), `?stand=default&spline=orbit-low`, 1600×900, both run
2 minutes apart on the same quiet GPU:

| | cull | cards | renderer | total Σp50 |
|---|---|---|---|---|
| `results/001-billboard-smoke__default__p-86e4907a__apple-metal-3__2026-07-25T01-50-34-363Z.json` | 0.380 | 2.501 | 2.881 | 5.868 |
| `results/018-clump-impostors__default__p-08c1f77a__apple-metal-3__2026-07-25T01-48-48-766Z.json` | 0.491 | 3.398 | 3.889 | 7.244 |

→ **1.35× on the renderer, 1.23× on the total**, and 7.24 ms is comfortably under
the 9 ms bar. `orbit-low` is a grazing-height orbit, i.e. the expensive case.

**Interleaved solo runs at the standard cameras** (cull+cards p50, medians of 3
alternating samples, 1280×800, shipped defaults):

| cam | 001 | 018 | ratio |
|---|---|---|---|
| grazing | 2.54 | 3.37 | 1.33× |
| in-canopy close-up | 2.50 | 2.60 | 1.04× |
| far-horizon | 2.39 | 2.55 | 1.07× |
| topdown | 1.73 | 1.94 | 1.12× |
| inside-plant | 2.69 | 2.68 | 1.00× |

So: parity everywhere except grazing-type views, where the sub-clump band costs
~1/3 more. `lodDistance` is the dial — 10 m measures 1.25× at grazing, 14 m
(shipped) 1.33×, 18 m 1.39×.

**A/B same-frame** ratios are quoted only as a sanity check, because the B slot
of that page is systematically 26–35 % slower for *identical* work (running 001
against itself gives A/A = 1.26–1.35×). Raw B/A for 018 in the final run was 1.26×
(grazing), 1.05× (far-horizon), 0.98× (inside-plant), 0.97× (topdown) — i.e.
consistent with, or flattering relative to, the solo table.

**Plant-count independence:** `default` (557 k) cull 0.45 / cards 1.8–2.9,
`scaling-100m` (134 M) identical, same VRAM. `dense-mixed` (7.7 M, density 5)
rises with density, not count, and stays inside the VRAM budget.

### Looks — better than 001 in the near and mid field, level at distance

Judged on same-camera, same-frozen-time pairs plus the A/B wipe, after the fix.

- **In-canopy (eye at 0.85 m, fov 30) is the clearest win.** 001 is a flat pink
  wall with a handful of giant elymus spikes pasted in front of it — the
  calamagrostis field has no internal structure at all. 018 shows panicles at
  several depths and scales, dark gaps between clumps, green blades crossing in
  front of pink heads, and the spikes sit *inside* the stand at their real
  positions. It reads as standing in a volume.
- **`inside-plant`:** 001 fades out the plant you are inside and leaves a
  backdrop of near-identical spikes, plus a large pale disc overhead (its
  top card seen from below). 018 keeps sub-clumps at their offsets, so blades
  pass at several depths with sky through the gaps, and the signed elevation test
  removes the overhead-disc artifact entirely.
- **`grazing`** (the judging camera): 001 is a fairly uniform pink haze; 018 has
  green stems and shadowed gaps between the panicles and a visible near→far depth
  gradient — more chromatic variety, more layering.
- **Parallax:** a 0.30 m lateral camera step at 14° fov moves near panicles much
  further across the screen than the material behind them and re-orders what is
  visible inside single plants. This is structural, not a trick: the cards really
  are at different world positions.
- **`far-horizon`:** a wash. 001 is slightly sharper per panicle (bigger tiles),
  018 slightly deeper. Past `lodDistance` both draw one card per plant.
- **`topdown`:** identical by construction (everything is past `lodDistance`), at
  1.12× the cost.
- **Where 001 still wins:** per-blade crispness inside ~1 m. Five units share the
  same 25 MiB, so a sub-clump tile gets ~0.73× 001's horizontal texel density and
  the very-close view is grainier. Fixing it needs a bigger budget or fewer
  sub-clumps, and both trade away the thing that makes the method work — even
  halving the normals again only buys 1.10× linear.

### Notes for whoever reads this next

- **Scrape the toasts, not just the console.** Every WebGPU validation error in
  this harness lands in a `.toast` element and disappears after a few seconds; a
  screenshot-only check will report a clean run while half your draws are being
  rejected. `page.locator('.toast').allInnerTexts()` after each capture is two
  lines and would have saved me an hour.
- The coverage calibration is tuned for `alphaRef = 0.4` (the default); moving
  that slider trades part of the LOD density match back. The `coverage` debug view
  shows what it costs, and it is worth stealing for any alpha-tested foliage
  method in this repo — it also stops distant grass from thinning out.
- `lodDistance = 0` renders the method as plain billboards over the same atlas,
  which is the cleanest way to isolate what the sub-clumps are actually buying.
- Harness wishlist: (1) `indirect-first-instance` in `requiredFeatures` (it is
  widely supported and the workaround costs a bind group per bucket); (2) the A/B
  page's B slot is systematically 26–35 % slower for identical work — a built-in
  self-control, or alternating the A/B encode order between frames, would remove
  a real footgun.
