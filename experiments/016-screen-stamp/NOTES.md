# 016 screen-stamp — tile-frustum binning + per-pixel stamp resolve

## Idea

Invert the loop: iterate over SCREEN TILES, not plants. Per frame:

1. **near pass** (render): the constant 6x6 ring of scatter cells around the
   camera (< ~8 m) drawn as classic alpha-tested hemi-octa impostor cards
   with depth write. Rationale: a plant 1-3 m away angularly overlaps almost
   every tile, so per-tile lists would drown in near plants (verified — the
   first build had every list full of 1-3 m plants and a sparse mid-field).
   Region-bounded, plant-count independent, and it primes the depth buffer.
2. **bin pass** (compute, one 128-thread workgroup per 16x8 screen tile):
   - reduces the tile's scene-depth texels into a world footprint (visible
     ground cell bbox, max ground distance, sky presence) — the terrain base
     pass already made the depth buffer THE tile-to-world map;
   - picks a per-tile strategy: **enumerate** the few footprint cells
     (close-up / top-down), **column-march** cells front-to-back along the
     center ray's ground track (grazing wedge / above-horizon), or
     **tint-only** (footprint too large for individual plants);
   - evaluates the procedural scatter twin over those cells only (no plant
     arrays at any count), sphere-tests candidates against the tile's four
     frustum planes, applies wind sway + fades, snaps each plant to its
     nearest baked hemi-octa view, packs 32 B per plant;
   - hard work bound per tile: <=32 surviving cells x entries x 128 slot
     evals, list capped at K=64; thread 0 sorts front-to-back.
3. **stamp pass** (fullscreen fragment): each pixel walks its tile's sorted
   list, ray-intersects each impostor card, samples baked albedo/normal
   (mipmapped, premultiplied), and composites over-operator front-to-back
   with two early exits (transmittance saturated; entry behind the visible
   surface — valid because the list is depth-sorted). Beyond the per-tile
   stamp horizon a statistical meadow tint (species average albedos mixed by
   world-anchored value noise, lit by the terrain normal, fogged) takes over
   with a crossfade.

Cost is tiles x bounded constants + screen pixels x bounded walk — per-frame
work is independent of total plant count AND stand area (verified visually on
`scaling-100m`, ~134M plants, same frame cost as the default stand). No
source geometry per frame, no per-pixel marching: each pixel does at most K
precomputed lookups; the per-tile column walk is a bounded tiled-culling
walk (<=32 cells), not a raymarch.

Key trick for top-down: a per-plant **view-angle card clip** (cards shrink to
0.45R when seen from above, mirrored bit-exactly in bin+resolve+near). From
overhead, full-radius community-tile cards overlap ~40x — no bounded list
could hold them, and arbitrary overflow choices showed as tile seams. The
clip bounds overlap below K and the seams disappear; clipped fringes are
covered by neighbouring plants and the under-tint.

## VRAM budget math

Per species (HUD-confirmed 16.6 / 25 MB each):
- albedo atlas 1280x1280 rgba8, 4 mips: 6.25 MiB x 1.328 = 8.3 MiB
- normal atlas same: 8.3 MiB
- => 16.6 MiB / species for all three species.

Screen-scoped (not per species, scales with resolution, NOT with plants):
- tile lists: ceil(W/16) x ceil(H/8) x (8 + 64*8) u32 = 2080 B/tile;
  16.6 MB at 1280x800, ~66 MB at 4K-retina — flagged as a known cost, K or
  the packing could shrink it 2x if needed.
- 160 B uniform.

## Bake

`bake.ts` renders each species' GCMESH1 mesh into a 10x10 hemi-octahedral
grid of 128px orthographic captures (MRT albedo+coverage / local-frame
normal), reads back, then builds a CPU box-filtered mip chain (albedo
premultiplied — empty texels are black, so plain averaging is the correct
filter; this fixed dark silhouette fringes) and the coverage-weighted average
albedo used by the far tint. ~2-4 s for all three species, baked fresh per
session. The harness bakedArtifact/commitBake flow is intentionally bypassed
for the same reason 005 documents (dev server answers missing bake files with
200 index.html, poisoning the cache). Capture pipeline adapted from
005-octa-impostors; everything after the readback is new.

## Status

working — verified by headless screenshots at grazing, topdown,
inside-plant (fades via coverage erosion, no dither), far-horizon, and
far-horizon on `scaling-100m` (134M plants). Wind verified by diffing
det=1 frames at t=2 vs t=6 (foreground moves; per-species sway from the
shared model, computed once per plant per tile in the bin pass, sheared
per-pixel in resolve). Covers all three species.

Honesty / known issues:
- All frame timings this session were contended (many agents on one GPU):
  observed bin 3.5-17 ms, stamp 4.7-21 ms p50 at 1280x800 across runs of the
  SAME code. No bench recorded — needs a quiet GPU; the structure is bounded
  but the constants deserve tuning (candidates: half-res resolve, smaller K,
  tighter cell budget).
- Grazing: lists overflow at ~35-45 m and the scene crossfades into the
  aggregate tint field earlier than ground truth would; the tint carries
  noise octaves but is visibly flatter than stamped plants.
- Single nearest hemi-octa view per plant (no 4-view blend): possible view
  popping in motion (not verified in a moving capture).
- Close-up (< 1 m) impostors go blobby — 128px views magnified; inherent.
- calamagrostis/elymus are community tiles baked as one "plant", so density
  reads busier than ground truth (same caveat as 005).
- Stamps write no depth (nothing draws after them in the frame).
- Enumerate-mode footprints come from visible ground only: plants whose base
  is hidden behind a close ridge but whose tops should peek over can be
  missed at close range (column mode handles the common grazing case).

## Findings

- The depth buffer as tile->world footprint map works remarkably well: sky
  tiles cost almost nothing, ridges bound the walk automatically, and the
  near-quad depth primes the sorted-occlusion early-out so pixels behind
  near plants walk ~1 entry.
- Slot exhaustion is THE failure mode of screen-tile plant lists — three
  separate mechanisms were needed (near-field ownership pass, view-angle
  card clip, per-tile overflow horizon + crossfade) before all four standard
  cameras held up.
- Impostor orientation gotchas found the hard way: atlas v-flip (plants
  rendered upside down — dense root line floating at the skyline) and uv
  clamp-vs-reject at atlas tile borders (clamping smears cropped
  community-tile edges into hollow card outlines).
- No bench JSONs yet (contended GPU). A/B vs ground truth:
  `#/ab/016-screen-stamp/000-ground-truth?stand=default&cam=grazing&seed=42`.
