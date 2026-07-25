# Moss round — agent prompt (for review, not yet dispatched)

One agent per renderer. `{{ID}}` is substituted with the experiment directory
name (e.g. `003-shell-slices`). Scope: every experiment except
`000-ground-truth`, which is stand-independent by design.

Everything below the line is the verbatim prompt.

---

You are making the SIMPLEST fix or fixes that meaningfully improve how MOSS
looks in exactly one renderer: `experiments/{{ID}}/` in
`/Users/sebastian/IdeaProjects/groundcover-experiments`.

## This is not a rewrite

Prefer the smallest change that buys the most visible improvement, and work in
order of visual impact per unit of effort. Some renderers are fundamentally
ill-suited to low, wide, densely intricate geometry like this. If, after honest
effort, your conclusion is *"this representation cannot render this kind of
geometry well; here is the least-bad version I could reach, and here is why"*, that is a perfectly good and useful outcome —
write it plainly in `NOTES.md` and in your report. Do not thrash trying to make
it perfect, and do not oversell a marginal improvement.

## The subject

Three new species were added: `spaghnum-palustre-wet-vigorous`,
`spaghnum-palustre-late-season`, `spaghnum-palustre-sun-exposed` — *Sphagnum
palustre* in three micro-habitat states. They are nothing like the grasses:

- a **periodic community tile only 0.18 m across and 0.07–0.09 m tall**. Note
  carefully what this is *not*: it is neither an upright plant nor a flat
  surface. It is dense, intricate cushion geometry — a mass of individual
  capitula — that simply happens to be low, wide and periodic. In the ideal
  render it reads as the intricate 3D it actually is. Treating it as a flat
  plane is the single most common way this goes wrong, and produces the
  featureless blobs seen so far;
- ~19.8 M triangles per source mesh (~479 MB), so a fresh bake of one moss
  species takes roughly two minutes;
- laid out by the new `bog` stand in **carpet mode**: an 11×11 grid of tiles per
  4 m scatter cell, grid-snapped at a constant scale (2.02, so a tile exactly
  fills its 0.3636 m step and overlaps its neighbours), rotated **only in 90°
  steps**, with the three states zoned across a wetness field. Calamagrostis and
  a trace of Poa are scattered through it normally.

Your test scene: `http://localhost:5175/#/run/{{ID}}?stand=bog`

## Already established — do not re-derive

- **The scatter's grid is correct, but small holes in the tiling are a KNOWN
  HARNESS BUG, now fixed.** The carpet zone jitter used to be hashed together
  with the entry index, so the three moss states evaluated different wetness at
  the same grid node: 2.55% of nodes were claimed by no entry (bare peat holes)
  and 2.63% by two (double-stacked tiles). If you still see isolated tile-sized
  holes along zone boundaries after that fix, report it rather than papering
  over it — but do not assume every gap is yours.
- `001-billboard-smoke` is the useful reference point: it covers the ground without gaps,
  but renders as flat blocky squares with no visible moss structure, because its
  atlas texel density for a 0.36 m tile is low. Beating that is a low bar. A/B against it: `#/ab/001-billboard-smoke/{{ID}}?stand=bog`.

## A worked example exists — read it

`001-billboard-smoke` has already been through this task. Read its carpet path
(`main.ts`, `shaders/cull.wgsl`, `shaders/cards.wgsl`) and its `NOTES.md` before
starting. You are not obliged to copy it — a different representation may want a
different answer — but these findings transferred:

- A carpet entry drew **one ground-parallel, tile-sized quad** and skipped the
  camera-facing card entirely (6 verts instead of 12).
- The top-view texture was **cropped to the tile's own square** in the mesh
  frame, which multiplied on-screen texel density ~4x. Tile origin is `(0,0)` in
  the mesh frame for every current source mesh — treat that as a guarantee.
- The grass alpha reference (0.4) **punched holes in the mat**: tile alpha is
  ~80% solid up close, but the mip chain pulls it toward the whole tile's mean
  (27–37% covered), so distant tiles failed the test entirely. A carpet-specific
  reference (0.06) made the mat a solid depth-writing occluder.
- **No camera-inside fade for a carpet** — a mat you are standing on must not
  open a hole under you.
- Overlapping coplanar tiles z-fight; a sub-millimetre phase-based depth bias
  fixed it. Note this only arises if you overscale.

Its honest verdict, worth knowing before you start: a single flat quad reads as
very good moss *texture on the ground* down to ~0.4m, but a card has no
thickness — no silhouette, no cushion parallax — so the mesh's 3.3cm of
capitulum relief is entirely lost. If your representation can express thickness,
that is where it can beat the reference.

## Failure modes observed across renderers — check for these first

1. **Cards stood upright.** Upright cards for a low cushion slice through the
   ground and through each other, and seen edge-on they show almost nothing.
   Moss geometry lies along the ground rather than standing off it — which is
   not the same as being flat: it is intricate, just low.
2. **Breaking the tiling** by overriding the yaw or the scale the scatter gave
   you — camera-facing billboarding, continuous per-tile yaw, per-plant scale
   jitter, or scale that shrinks with distance for LOD. See the lattice rule
   below.
3. **Sizing width from `height_scale`.** Moss is 0.07 m tall and 0.24 m wide, so
   this makes it about 3.5× too small and opens gaps in ground that should be
   fully covered. Use `stand_table[i].footprint_m`, or extents you measured yourself.
4. **Ignoring the slope.** The scatter gives position and yaw only, so a naive
   renderer stands everything bolt upright; a horizontal tile on a slope buries
   one edge and floats the other.
5. **Budget spent in the wrong place.** Not an overrun — a megabyte or two over
   the 25 MB soft budget is fine and not worth chasing. The question is
   *allocation*: a representation tuned for an upright plant may spend most of
   its storage on views or structure that a low cushion barely shows, while
   starving the detail you actually look at. If your budget split assumes an
   upright plant, reconsider it for this species.

Diagnose which of these actually apply to *your* renderer before changing
anything — they differ a lot between techniques.

## Carpet slots exceed the scatter budget

A carpet entry has exactly `carpetDiv²` slots per 4 m cell — **484** for the bog
moss, deliberately over `SCATTER_MAX_PER_CELL` (128). That is because div 22 is
what puts a 0.18 m tile at **life size** (scale 1.01); div 11 would fit the
budget but render 0.36 m tiles of 18 cm moss. Drive enumeration and buffer
capacity from `standEntrySlots(entry)` (exported from `@harness`) or from
`stand_table[i].carpet_div²` — **never** from `SCATTER_MAX_PER_CELL`. Hardcoding
128 renders about a quarter of the mat and leaves holes that look exactly like a
placement bug. Life size also means ~4x as many moss tiles as an oversized mat
would need (~1.13M on the bog stand), so if your renderer has no distance
aggregation, this is where it will show.

**Do not conflate "slots to evaluate" with "instances to store".** This bit the
pilot hard. Many renderers carry a density-scaled slot count
(`SCATTER_MAX_PER_CELL * density/8`) and use it for buffer capacity, which is
correct — but if you also drive your cull dispatch from it, you visit only that
fraction of the slots and the rest of the plants silently vanish (in the pilot,
22 of 128 slots for calamagrostis, so ~83% of the grass disappeared). Keep two
numbers: **every** slot must be evaluated, while capacity only needs the
expected survivors — for a zone-partitioned carpet that is roughly `wetWidth` of
the slots, so sizing capacity for all 484 wastes about 3x the memory.

## The lattice invariant

Tiled species read as one continuous stand only while every tile still agrees
with its neighbours — same grid spacing, same rotation set. Break that and the
field visibly falls apart into separate rotated patches. So, when
`stand_table[i].carpet_div > 0`:

- rotation **only in 90° steps**;
- scale **identical for every tile** of that species.

Within that invariant, a tiling trick is permitted **only where it is actually
needed and demonstrably artifact-free**. A *uniform* overscale of every tile is
allowed in principle — grid spacing is untouched and each tile simply covers
more than its cell, which is how the source mesh's own overflow already works
(0.24 m of geometry inside a 0.18 m period). It is **not** a default, and it is
not assumed to help: the pilot measured ×1.15 and ×1.35 as clearly *worse*,
because a single flat plane has no thickness to hide an overlap in, so every
overlapped edge painted a dark line and the field became a chicken-wire lattice
at grazing. If you reach for overscale, prove at both grazing and
`cam=carpet-close` that it removes more artifact than it introduces, and leave
it at 1.0 otherwise. The invariant test remains **"do neighbouring tiles still
agree"**.

## Fitting to the terrain — your choice of fidelity

`stand_table[i].slope_align` says **how much** a species should conform (1 for
carpets, ~0.3 for the bog's calamagrostis, 0 elsewhere). It deliberately does
not say how — and it may have no natural insertion point in your
representation, which is fine; `carpet_div` already tells you the species is a
mat, and a mat wants full conformance regardless. Primitives in `src/wgsl/terrain.wgsl`, mirrored on
`ctx.scene.terrain`: `terrain_height`, `terrain_normal`,
`terrain_plane_fit(xz, radius)`, `plant_basis_from_up(up, yaw)`,
`plant_basis(xz, yaw, align)`. For per-vertex conforming, `terrain_sample(xz)` is
usually the one you want: height **and** (nx, nz) from a single bilinear fetch,
where `terrain_height` + `terrain_normal` (or `plant_basis`, which re-fetches
internally) pays for the same taps twice.

`CLAUDE.md` documents a ladder — point-normal tilt, plane fit over the
footprint, per-vertex conforming, or warping the query itself for ray and volume
methods. **All four are valid; picking the rung is your decision.** Do not
assume the cheapest suffices because it happens to look fine on the current
terrain — that terrain is one arbitrary sample, and these methods are meant to
generalise across regimes. Note one finding from the pilot: for a *tiled*
species a per-tile plane fit (rungs 1–2) is not merely cheaper but **wrong**,
because neighbouring tiles fit different planes and crack apart at their shared
edge. Only per-vertex conforming keeps the surface continuous. If you deliberately choose a cheap rung, say why in
`NOTES.md`.

## Do not regress the grass

The `default` stand must look and perform as it does now. Verify it explicitly
with before/after screenshots at `cam=grazing` on `default` — not by reasoning.
Keep total GPU Σp50 on `default` under 9 ms and preserve your renderer's
existing performance character. This matters most for `001-billboard-smoke`,
which is the baseline everything else is measured against, but it applies to
every renderer.

## Rules that bind you

Read `README.md` and `CLAUDE.md` in full first. In particular: the stand
contract (you are a pure renderer of `ctx.stand` and never a placer), the
mandatory global debug views via `debug_shade()`, the no-gratuitous-dither taste
rule, WGSL reserved words, `assetUrl()` for asset fetches, the VRAM budget, and
the **silent-failure traps** section — non-zero `firstInstance` in indirect
draws, validation errors arriving as on-screen toasts rather than exceptions,
premultiply/mip conventions, octahedral normals not being mip-averageable, and
measuring per distance band rather than per frame. Several of those cost earlier
experiments a wrong result.

- Edit **only** `experiments/{{ID}}/` and `mesh/baked/{{ID}}/`. Never touch
  `src/**`, other experiments, `package.json`, or git — make **no commits**.
  Never create or edit `rating.json` (the owner's verdict).
- Delete stale bake variants you no longer load; never produce a file over
  100 MB. Avoid gratuitous bake-version bumps given the two-minute moss bakes.
- The dev server is **already running** at `http://localhost:5175`. Never start,
  stop, or restart it, and never run `npm run dev`.
- Other agents work in parallel in other folders. Ignore their files and their
  typecheck errors.

## Verification

- `npx tsc --noEmit 2>&1 | grep {{ID}}` prints nothing.
- Headless screenshots, script **outside** the repo (e.g. `/tmp/verify-moss-{{ID}}.ts`),
  run with `NODE_PATH=/Users/sebastian/IdeaProjects/groundcover-experiments/node_modules npx tsx ...`
  and `chromium.launch({ headless: true, channel: 'chrome', args: ['--enable-unsafe-webgpu','--enable-features=WebGPU'] })`.
  Capture **before and after**:
  - `?stand=bog` at `cam=grazing`, `cam=topdown`, `cam=inside-plant`;
  - `cam=carpet-close`, a named bookmark 1 m straight down, so tile-level
    detail and any seams are visible. Use the bookmark, not a hand-written
    pose: URL `cam=x,y,z,...` poses are ABSOLUTE, and the terrain at the origin
    sits at about −7.7 m, so "y=2" is roughly 10 m up, where a tile is ~16 px of
    mush. The standard bookmarks are all terrain-relative;
  - a view across the ridged slopes, to check terrain conformance;
  - `?stand=default` at `cam=grazing`, to prove no regression;
  - `&debug=normals`, `&debug=lighting`, `&debug=coverage` on `bog`.
- **Read every screenshot.** Do not report an improvement you have not looked
  at. Remember that "0 console errors" is not proof of correctness — validation
  errors also surface as on-screen toasts, and a plausible-looking image can
  hide entire missing draws.
- Timings are unreliable while sibling agents share the GPU. For any perf claim
  use the same-frame A/B page and correct for the fact that the **B slot
  measures ~1.26–1.35× slower than A for identical work**. Mostly you only need
  to show `default` did not get worse.
- Update `NOTES.md`: what you changed (smallest first), which rung of the
  terrain-fitting ladder you chose and why, what improved, what is still bad,
  and whether your representation is fundamentally suited to geometry of this
  shape.

## Report back

1. The fixes you made, smallest first.
2. Which harness information you used and how — `carpet_div`, `footprint_m`,
   `slope_align`, the terrain primitives.
3. **Interface feedback:** anywhere that information was missing, awkward, or
   insufficient for what you actually needed. Say so explicitly rather than
   working around it silently; this shapes the harness for later rounds.
4. Before/after description of the moss, from screenshots you actually looked
   at, including the sloped view.
5. Confirmation that the `default` stand is unchanged.
6. Your honest verdict on whether this representation can render moss decently
   at all.
