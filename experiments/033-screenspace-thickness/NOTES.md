# depth-shell impostors (033-screenspace-thickness)

Seed: *"screen-space depth and thickness cues to make cheap proxies read as
volumetric."* What came out of it: the **screen-space depth of a baked view,
promoted to real geometry**.

## Idea

A billboard is a flat card, so every texel of a plant sits at exactly one
depth: the plant's centre plane. Everything that reads as "3D" is missing for
that one reason — no parallax inside the silhouette, no interpenetration with
neighbours, no honest contact with the soil, and a depth buffer that lies about
where the plant is.

So bake the missing coordinate. For each of 13 orthographic views the bake
keeps, besides albedo/coverage/normals, the **front-depth shell**: the
coverage-weighted depth of the first surface the view ray hits, sampled at a
9x9 lattice over the card. At runtime the card is still ONE camera-facing quad
per plant, but it is drawn as that 9x9 lattice with every vertex pushed along
the view axis by its shell value. Three consequences, all free:

* the card is a **real 3D surface** in the depth buffer — no `frag_depth`
  anywhere, so early-z stays intact; depth comes out of the rasterizer exactly
  as it does for a mesh;
* **parallax inside the silhouette** — a vertex 0.3 m nearer projects with a
  different perspective divide, so the bulge slides against the rim as the
  camera moves, and the plant's front reaches the camera where the geometry
  actually is;
* **self-occlusion, interpenetration and ground contact** — plants cut into
  each other and into the terrain along the curved front surface instead of
  along the straight line where two flat cards cross.

The lateral axes of the card still follow the camera (like a billboard), so the
silhouette never foreshortens or "breathes" at azimuth-cell edges; only the
tilt follows the baked ring. That keeps the one thing billboards do best.

The other half of the seed is the **thickness cue**, baked per view at two
scales:

* `ao` = *burial*: how far this texel sits behind the front shell of its own
  neighbourhood, weighted by the local canopy density and maxed with a
  grounding gradient. This is a screen-space (image-space) thickness measure
  computed at bake time — it costs nothing at runtime and is what makes the
  interior of a clump read as volume.
* `thickness` = (back surface − front surface) along the view ray, from a
  second reverse-depth pass. It drives forward-scatter transmission, so thin
  tips glow when you look toward the sun and the dense core stays dark.

**LOD** is index ranges over one vertex program: the lattice holds 85 vertices
(81 shell corners + 4 flat-card corners) and each LOD is a slice of one shared
index buffer — 8x8 shell (< `lodNear`), 4x4 shell (< `lodFar`, every 2nd
corner), single flat quad beyond. The cull pass buckets plants into three
per-LOD lists and fills three `drawIndexedIndirect` arg sets. A distant plant
is therefore *exactly* the billboard baseline: 2 triangles, one texture tap
pair, no shell lookup.

Plant-count independence is the same construction as the baseline: the cull
compute pass walks the scatter cells of a camera-centred region (clamped to the
stand on the CPU) and compacts survivors; nothing in the frame scales with the
stand's plant count. The region rect, not the stand, sets the dispatch size and
every instance list is sized by area x density.

## VRAM budget math

Per species, 13 views (8 azimuths at elevation 0, 4 at 45 deg, 1 straight
down):

| item | size | bytes |
|---|---|---|
| `texA` albedo+coverage | 480² x 13, rgba8, 9 mips | 15.97 MB |
| `texB` octN.xy + AO + thickness | 240² x 13, rgba8, 8 mips | 3.99 MB |
| front-depth shell | 13 x 9 x 9 f32 | 4.2 KB |
| culled instance lists (LOD0/1/2, density 3) | 180 250 x 16 B | 2.88 MB |
| entry uniform + indirect args | | < 1 KB |
| **total (density 3)** | | **≈ 22.9 MB** |

Measured by the HUD tracker on the default stand: **21.8 / 21.3 / 21.8 MB of
25** (calamagrostis / elymus / poa — the spread is instance capacity, which
scales with the entry's density). Under budget, and within ~0.5 MB of the
billboard baseline's own footprint (21.3 / 20.7 / 21.3) while carrying depth,
AO and thickness on top of the same imagery.

The attribute texture is deliberately half-resolution: normals, occlusion and
thickness are low-frequency next to the coverage edges, and that one decision
is what buys the extra views (a 45-degree ring) inside the budget.

The screen-space canopy pass, when enabled, adds one full-res r8 target
(~1.0 MB at 1280x800, not species-scoped).

## Bake

`bake.ts` renders the raw GCMESH1 mesh orthographically, once per view, into a
2x supersampled 960² tile, twice:

* front pass (`depthCompare: less`) → albedo+coverage, mesh normal (flipped
  toward the bake camera), and normalized front depth;
* back pass (`depthCompare: greater`, depth cleared to 0) → normalized back
  depth, i.e. the far side of the same ray.

CPU post-process per view: coverage-weighted 2x downsample to 480², dilation
into empty texels (so filtering never pulls in background black), a second
coverage-weighted reduction to 240², the 9x9 shell (coverage-weighted mean
front depth over each corner's window, holes filled from filled neighbours),
then burial AO / thickness — both blurred back to the clump scale, because
per-texel burial swings by centimetres between neighbouring blades and reads as
dirt on the fluffy heads if you leave it in (it did; that was v1).

Framing is analytic and azimuth-independent: treating the plant's support as a
cylinder (radius `rXZ`, height `H = y1-y0`), a view at elevation `e` needs
half-extents `su = rXZ`, `sv = rXZ·sin e + H/2·cos e`, `sd = rXZ·cos e +
H/2·sin e`. Every vertex lands inside `[-1,1]³` exactly with no slack wasted,
and only the *ring* changes the card size — never the azimuth, so a plant
rotating through azimuth cells never pops in size.

Artifacts: `mesh/baked/033-screenspace-thickness/dome-v3-<species>.bin`,
14.98 MB each (45 MB for three species). One variant only; v1/v2 deleted.

## Status

**working.** Verified by headless screenshot at `grazing`, `topdown`,
`inside-plant`, `far-horizon`, plus `debug=albedo|normals|lighting|coverage|depth`
and the A/B page against the baseline. Console clean, `npx tsc --noEmit` clean.

## Findings

### Speed — parity, measured in the same frame

Sibling agents were hammering the GPU throughout, so absolute milliseconds are
worthless here: the harness's own `composite` blit — a trivial fullscreen pass,
identical work in both views — read anywhere between 1.5 ms and 5.7 ms across
runs. Only same-frame A/B ratios mean anything.

`#/ab/001-billboard-smoke/033-screenspace-thickness?stand=default&seed=42`,
experiment passes only (`cull` + main pass), **contended**:

| cam | A billboards | B depth shell | ratio |
|---|---|---|---|
| topdown | 2.11 / 1.71 | 2.21 / 2.11 | **1.05x / 1.23x** |
| far-horizon | 1.63 | 2.06 | **1.26x** |
| inside-plant | 2.51 | 3.30 | **1.31x** |
| grazing | 1.26 / 2.47 | 1.86 / 2.96 | **1.48x / 1.20x** |

(two independent A/B sessions where I have both — the spread between them is the
noise floor of this measurement, so read the ratio as "≈1.2–1.3x, worst case
1.5x", not as four exact numbers.)

Parity to +48 %, worst at grazing, ~1.3x on average — inside the "must not
exceed ~1.5x" allowance and nowhere near the 3x that sinks a method. Total GPU
Σp50 solo at 1280x800, across several runs: **4.8–6.8 ms (grazing)**,
**3.8 ms (topdown)**, **5.5–6.7 ms (inside-plant)**, **4.9 ms (far-horizon)**,
against the baseline's **5.43 ms (inside-plant)** / **3.58 ms (topdown)** in the
same sessions. Every reading is under the 9 ms ceiling; the spread inside one
camera is contention, not the method.

Bench (`#/bench/<id>?stand=default&spline=orbit-low`, 1600x900, also contended):
* `results/033-screenspace-thickness__default__p-90f83bd3__apple-metal-3__2026-07-25T07-59-17-863Z.json`
  — Σp50 6.38 ms (shell 2.93).
* `results/001-billboard-smoke__default__p-86e4907a__apple-metal-3__2026-07-25T07-59-28-876Z.json`
  — Σp50 9.17 ms (cards 3.57), run a minute later under heavier contention.
  That is **not** evidence the shell is faster than billboards; it is evidence
  that solo benches on this machine right now cannot resolve a 30 % gap.
* `results/033-screenspace-thickness__default__p-9e9ec9e7__apple-metal-3__2026-07-25T07-46-46-468Z.json`
  — the earlier 3-extra-pass version, Σp50 8.90 ms. Kept as the record of why
  the pass count came down.

Two structural findings worth writing down:

1. **Pass count costs more than pass work here.** The first version had three
   extra passes (two mask reductions + a composite). Each measured ≈ the
   harness blit — ~2–2.5 ms of *measured* time for microseconds of real work —
   and Σp50 went to 11–18 ms. Folding the blur into a single 13-tap composite,
   then defaulting it off, took the structure back to the baseline's two passes.
2. **An extra render target is not free on a tile GPU.** Writing the canopy mask
   as a second attachment of the main pass cost ~15 % of that pass (2.56 →
   2.19 ms) even when nothing consumed it. There are now two fragment entry
   points and two pipelines; the default path binds one colour target, exactly
   like the baseline.

The shell itself is nearly free: the same-frame ablation `domeScale` 0 vs 1
(both sides this experiment) puts the shell draw at **1.06–1.16x** the flat
draw. Vertex load is *below* the baseline's, which draws two quads (12
unindexed vertices) per plant where a distant plant here draws 4 indexed ones;
only near plants pay 81.

### Looks — better volume, honest about the trade

Ablation, same camera, same frame, `mode=diff`, `a.domeScale=0` vs
`b.domeScale=1`: **mean 23/255 over 44.7 % of pixels**, concentrated in the near
and mid field and vanishing at the horizon where the LOD is flat by
construction. That is the shell doing geometric work, not a shading tweak.

What the A/B wipe and flicker against `001-billboard-smoke` actually show:

* **Parallax inside the silhouette: yes, and it is geometric.** The near part of
  a plant's front surface sits up to ~0.3 m closer than its rim (measured shell
  range -0.19…+0.28 m for calamagrostis at elevation 0), so it projects larger
  and slides against the rim as the camera translates. In flicker at `grazing`
  the baseline's plants translate rigidly; the shell's plants shift internally.
* **Silhouette changes with view direction: yes.** 13 views instead of 9, and
  the outline is a curved surface's projection rather than a rotating rectangle.
  The 45-degree ring also removes the baseline's worst artifact — a pale flat
  top card fading in and out over the side card. Here the top view *is* a canopy
  height field, so from above you get relief instead of cutouts.
* **Self-occlusion / depth layering: clearly better.** At `inside-plant` near
  blades occlude far ones along curved intersections and the frame reads as a
  canopy you are standing in. Billboard cards cross in straight lines, which is
  the most impostor-looking thing in the baseline.
* **Contact with the ground: better, for free.** The shell's base is where the
  plant's base actually is, so the plant meets the soil at the right depth
  instead of at a plane through its middle; the baked grounding gradient
  darkens the last few centimetres.
* **Honest downsides.** (a) Very near plants (< ~1.5 m) are *softer* than the
  baseline's, because the shell correctly brings their front surface toward the
  camera and the 480 px imagery gets magnified. The baseline hides this by
  keeping everything on the centre plane — sharper and wronger. (b) The mid
  field is darker: burial AO opens real gaps between plants where the baseline
  is uniformly lit. I read that as the volumetric win, but it is a taste call
  and the one place someone could reasonably prefer the baseline's brighter,
  flatter mat. (c) Elevation ring boundaries (22.5 / 67.5 deg) pop; azimuth
  cells within a ring do not, because the card's lateral frame is
  camera-derived, so only the imagery snaps — exactly as in the baseline.
* **The one place the baseline was better, now fixed.** At `topdown` the
  straight-down view is genuinely sparser than the 45-degree one (blades seen
  end-on present almost no area), so the elevation cut between the two rings
  drew a hard circle of tone change on the ground — obvious and artificial.
  `pick_view` now wobbles both elevation cuts by ±7 deg using the plant's own
  hash, which scatters that circle into a band you cannot pick out. The centre
  of a top-down frame still reads a touch sparser than the baseline's (which
  always draws a side card *and* a top card there), but it also shows the pink
  heads the baseline's flat top card washes out.

Verdict against the bar: **yes on volume, parallax, occlusion and grounding; not
a win on foreground crispness.** Clearly better in motion and at
`inside-plant` / `topdown`; a close call on a static `grazing` screenshot.

### What the extra pass buys (and why it ships off)

`canopyOcclusion > 0` enables the screen-space half of the seed: the main pass
writes a fog-attenuated `coverage x thickness` weight into an r8 target, and one
fullscreen pass estimates local canopy density from a 13-tap two-ring kernel and
darkens by it with a multiply blend (`srcFactor: zero`, `dstFactor: src`) — no
copy of the colour target, no mip chain, sky rejected by a depth load, and bare
soil between plants darkened at a lower weight, which is a free contact shadow.

It works, and it is the right idea, but measured honestly it changes the image by
**mean 5.4/255** and costs a whole extra pass plus the second render target
(≈1.7 ms of *measured* time, about one harness-blit's worth) for something a
tone curve mostly fakes. The volumetric read it was meant to provide is already
carried by the baked burial AO, which is free. So it ships at 0 and stays one
slider away — `p.canopyOcclusion=0.24` on any run/AB URL. That decision is the
whole reason this method sits at billboard cost instead of 1.7x it.

### No dithering

Coverage is hard alpha-tested with depth write throughout. The only stochastic
thing anywhere is the ±7 deg wobble of the two elevation cuts in `pick_view`,
and it is per-PLANT: each plant still picks exactly one view and draws solid,
depth-writing, hard-edged coverage. No screen-door, no holes punched in the
depth buffer, nothing for early-z to lose, and nothing that shimmers in motion —
it only decides which of two baked views a given plant uses. Both fades (camera
inside a plant, region rim) erode *the alpha reference*, and a card whose
reference exceeds 1 is dropped in the cull pass — the same set the fragment
shader would have discarded, which is exactly the set that covers the most
pixels. The reference also relaxes with distance (x0.62 by 95 m) so mipped
coverage does not thin the far field out; that costs some fragments and is the
main reason `far-horizon` sits at 1.26x rather than parity.
