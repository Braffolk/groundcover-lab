# 005 octahedral view-set impostors

Rebuilt from scratch (the previous attempt under this id — 144-view atlas,
4-view per-fragment blend, plants rendered upside down — was discarded whole).

## Idea

One plant becomes **121 pre-rendered views** covering the entire upper
hemisphere of viewing directions, and every plant on screen is **one quad** that
picks and blends between those views as the camera moves.

**The view set.** Directions are parameterised hemi-octahedrally: a unit
direction `d` (plant → camera, `d.y ≥ 0`) maps to
`uv = (n.x + n.z, n.z - n.x)·0.5 + 0.5` with `n = d / |d|₁`. The square's
boundary is the horizon, its centre is straight down, its corners are the ±X/±Z
horizon views. An 11×11 grid of nodes over that square is the view set: 32
distinct horizon azimuths (11.25° apart) up to an exact top-down view. Each
node is one orthographic capture, stored as **one layer of a texture array**,
not a tile of an atlas — layers cannot bleed into each other under minification,
so the whole thing is mipped with no gutters and no atlas-edge rules.

**Each view has its own extents.** A view's ortho box is the plant's local AABB
*projected onto that view's basis* (`ext = |axis|·half`), so a tall thin side
view and a squat top view each fill their tile completely. Nothing is wasted on
the empty margin a shared bounding sphere would force (≈2.5× on these plants),
and the runtime knows every view's exact basis, which is what makes step 3
possible.

**Selection is per plant, in the vertex shader.** A plant subtends a fraction of
a degree, so its viewing direction is effectively constant across its own
silhouette — computing the octahedral cell, its 3 surrounding nodes and their
barycentric weights once per plant (not once per fragment) costs nothing and
makes the layer index primitive-uniform.

**Blending is a reprojection, not a cross-fade.** The card is the AABB projected
along the *actual* view direction. In the fragment shader the card position is
turned back into a local-frame offset and projected into **each selected view's
own orthographic basis** to get that view's uv. The three images therefore line
up on the card plane instead of sliding against each other, which is what keeps
the blend sharp and is also the parallax that makes an off-axis view usable at
all. Colour is accumulated coverage-weighted; the normal is taken from the
dominant view only (1 tap, not 3 — neighbouring views disagree far less about
normals than about silhouettes). 4 texture taps per fragment, not 8.

`viewBlend=nearest` switches to a single view with a hard switch — the built-in
A/B for what the blend actually buys (ghost vs. pop).

**Per-frame cost is O(visible region).** One compute pass per stand entry
evaluates the shared scatter over the camera-centred cell region (clamped on the
CPU to the stand's cell range), one workgroup per cell so the region and frustum
rejects are workgroup-uniform and a cell that cannot contribute exits before a
single hash round or heightmap fetch. Survivors are compacted into **four
distance buckets** drawn near-to-far: impostors are hard alpha-tested with depth
write, so the near bucket lays down solid depth and hi-z rejects most of the far
buckets' fragments — exactly the overdraw that killed the previous attempt at
grazing angles. No dithering anywhere; the near/region fades erode the alpha
reference instead (CLAUDE.md taste rule).

Wind: the shared `wind_sway` shears the card by height fraction squared,
matching 000-ground-truth's weighting, with each species' stand `sway`.

## VRAM budget math

**Tile size is 256 px (`TILE` in `bake.ts`), chosen by the project owner: 128 px
was visibly too blurry up close.** That deliberately spends well over the 25 MB
soft budget — the README allows exceeding it when the result justifies it, and
sharpness is the whole point of a view-set impostor. The 128 px numbers are
kept below for comparison.

Per species, 121 views × 256 px tiles, 5 mip levels (256 → 16 px):

| item | bytes |
|---|---|
| albedo+coverage rgba8 array, mipped | 121 × 87 296 texels × 4 B = **42.2 MiB** |
| oct normals rg8 array, mipped | 121 × 87 296 × 2 B = **21.1 MiB** |
| per-view basis table (right/up + extents) | 121 × 32 B = 3.8 KiB |
| culled instances (default stand, density 3, 4 buckets, R=128 m) | 185 824 × 16 B = **2.84 MiB** |
| entry uniform + indirect args | 288 B |
| **total** | **≈ 66 MiB** (2.6× the 25 MB soft budget) |

At the original 128 px the same table totalled **≈ 17.9 MiB / 25 MiB**
(albedo 10.07 MiB + normals 5.04 MiB + instances 2.84 MiB), and the HUD agreed:
17.9 MiB for calamagrostis and poa, 17.5 MiB for elymus (density
2.5 → smaller instance buffer). The dense-mixed stand (density 5) pushes the
instance buffer to ≈4.7 MiB, still ≈20 MiB total. Instance capacity is a real
bound, not a guess: bucket *j* covers an annulus and scatter density is capped,
so `density × area × 1.15 + 2048` cannot overflow in practice, and it is sized
from `REGION_MAX`, never from the stand's plant count.

Transient bake-only allocations (source vertex/index buffers, 768² chunk
targets, readbacks) are tracked with tag `bake-scratch` and destroyed before the
bake returns, so they never sit in the live budget.

## Bake

`bake.ts` renders the raw GCMESH1 mesh once per view node with an orthographic
projection built from that node's basis — the *same* `view_right()` rule the
runtime uses, so reprojection is exact rather than a reconstruction that could
drift. MRT: albedo+coverage and the local-frame octahedral normal, flipped
toward the bake camera so a 2× supersample of a two-sided blade averages
agreeing normals instead of cancelling. Views are rendered 9 at a time into a
768² target (transient VRAM stays ≈7 MB) and resolved on the CPU:
coverage-weighted downsample, 4 dilation passes (colour spreads into empty
texels, alpha stays 0), oct-encode. Mips are then generated on the GPU, one
render pass per (layer, level), coverage-weighted.

Committed artifacts (45.4 MiB each, `mesh/baked/005-octa-impostors/`):
`views-v2-11x256-{calamagrostis-canescens,elymus-repens,poa-pratensis}.bin`.
A fresh browser profile loads them straight from disk — verified.

The 128 px and 512 px variants exist on disk but are deliberately NOT committed:
the 128 px set is superseded, and each 512 px file is 190 MB, over GitHub's
100 MB per-file limit. Change `TILE` and reload to re-bake any variant.

## Status

working — verified by headless screenshots at `cam=grazing`, `topdown`,
`inside-plant`, `far-horizon`, on the default, close-quality, dense-mixed and
scaling-100m stands, with every debug view. Console clean of WebGPU/shader
errors.

Debug views, and what they showed:

- **albedo** — the authored mesh colours exactly as captured (pink heads, green
  blades), unfogged. No light baked into the atlas, so nothing is shaded twice.
- **normals** — real per-fragment normals decoded from the rg8 view atlas, with
  per-blade variation. Directional structure changes with azimuth at grazing,
  and goes dominantly +Y from `topdown` (blades seen from above face up), which
  is the check that the normal atlas is actually decoded and yaw-rotated rather
  than faked.
- **lighting** — bright but *the same brightness as the harness's own terrain*
  in the same mode (sun 1.15 + ambient with half-lambert saturates); the point
  is that it is neither flat nor double-applied. Lighting goes through the
  shared `light_surface()`.
- **coverage** — blended atlas alpha, dark at silhouettes near the cutoff,
  solid inside blades. Meaningful, and it is the same number the alpha test
  uses.
- **depth** — correct near→far ramp; per-plant flat patches are visible because
  an impostor card *is* flat in depth. Honest.
- **viewTint** (own param) — paints each plant by its dominant view id: a wide
  spread of ids at grazing, and a clean radial pattern from `topdown` (view id
  changes with elevation as you move out from the point directly under the
  camera). That is the view set doing its job, visible.

Known limits / honesty:

- 128 px tiles: plants within ~3 m are visibly soft. 121 directions × 128 px is
  the trade this experiment deliberately makes (the thesis is angular coverage);
  the same budget could buy 64 views at 176 px instead.
- calamagrostis and elymus are periodic *community tiles*, so their impostor is
  a baked ~0.5–0.6 m clump repeated per scatter point (the mesh bounds overhang
  the tile, so the card is ~0.75/0.96 m wide). Density therefore reads busier
  than the tiled reference. poa is a true single specimen and is the clean case.
- Views from *below* the horizon are not baked; they clamp to the horizon ring.
  Only the camera-inside case really produces them and that is faded out.
- The 3-view blend ghosts slightly on fast azimuth changes; `viewBlend=nearest`
  trades it for a pop. Judge in motion, not in a still.
- Mip alpha is compensated by relaxing the alpha reference with an estimated
  LOD (one line in the VS), not by a proper coverage-preserving mip rescale.
  Distant coverage is therefore approximate.
- Per-plant maths is recomputed by all 6 vertices of the quad rather than stored
  by the cull pass. That is deliberate: ~80 ALU ops × 6 is cheaper than widening
  the instance record from 16 B to 32 B and re-reading it per vertex.

## Findings

**No bench numbers.** Up to four agents shared this GPU while the experiment was
built, so every frame time available to me is contaminated; nothing here is
tuned against measured time and no `results/` JSON is claimed. The structural
claims (region-bounded enumeration, 4 taps/fragment, near-to-far buckets,
count-independence) are from the code, and count-independence is confirmed
visually: the `scaling-100m` stand (134.2 M plants, ±2048 m) renders the same
image with the same VRAM as the default stand (557 k plants).

A/B links:

- vs ground truth: `#/ab/005-octa-impostors/000-ground-truth?stand=default&cam=grazing&seed=42`
- vs the billboard baseline: `#/ab/005-octa-impostors/001-billboard-smoke?stand=default&cam=topdown&seed=42`
  — top-down is where the view set earns its keep: 001 has a single top card,
  this has a whole neighbourhood of directions around straight-down.
- blend vs hard switch: `#/run/005-octa-impostors?cam=grazing&p.viewBlend=nearest`
