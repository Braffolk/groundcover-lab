# ribbon skeleton

## Idea

Distill a whole plant into a **handful of curved ribbons** whose entire shape
lives in two tiny textures, then unroll each ribbon into a triangle strip in
the vertex shader. Plant identity from almost no geometry.

Offline distiller (`bake.ts`), per species:

1. Gather the point cloud of one tuft (for periodic community tiles, points are
   folded through their nearest periodic copy so foliage wrapping a tile edge
   stays with the tuft; single specimens use the whole mesh). Subsampling is
   **density-equalizing** — at most N points per 2cm cell — so sparse blade
   surfaces keep detail while dense panicle fluff is decimated.
2. **Classify** each cell as *thin* (blade/stem — locally 1D/2D: the smallest
   covariance eigenvalue's share of the trace is near zero) or *fluff* (panicle
   — volumetric).
3. **Trace strands** through thin points: seed at a thin point, estimate the
   local axis by PCA, and walk a curve both ways (toward root and tip) with a
   direction-cone neighbour filter. Stamp a tube around each polyline as
   covered so strands don't overlap.
4. For each strand, bin the stamped points into K=12 cross-sections along the
   centerline; per bin store the centerline point, a **half-width and lateral
   azimuth** from the 2D covariance in the tangent-perpendicular plane, and the
   mean authored colour. Widths/azimuths are smoothed and the azimuth is
   unwrapped so the strip never twists 180° between samples.
5. **Plumes**: flood-fill the remaining fluff into clusters (~individual
   panicles; oversized blobs are split into azimuth sectors), and fit one fat
   *camera-facing* ribbon per cluster (width = radial spread about its axis).
6. One extra **aggregate** ribbon per variant stores the plant's vertical
   silhouette profile (radius + colour vs. height) — the far-LOD card.

4 variants per species × (48 ribbons + 1 aggregate) × 12 samples are packed
into two `rgba16float` textures (12 × 196): T0 = xyz + half-width (sign flags
camera-facing), T1 = rgb + azimuth.

Per frame: placement is the procedural scatter twin evaluated over a **bounded
region of cells around the camera** (two draws — near detail ribbons, far
aggregate cards), so 100 plants and 134M plants issue the identical bounded
draw. No source geometry is touched; no raymarching. The vertex shader reads
12 texels per strip and builds the curved strip: centerline scaled+yawed and
planted at the scatter root, lateral from the baked azimuth (oriented blades)
or a camera-facing screen axis (plumes/aggregate), wind lean weighted by height.
LOD cross-fades by **collapsing width to zero** — no stochastic alpha.

## VRAM budget math

Per species: two `rgba16float` textures of 12 × 196 texels.
- 12 × 196 × 8 B = 18.8 KB each → **~37.6 KB / species** for the ribbon atlas.
- 2 tiny 64 B meta uniforms per entry.

Three orders of magnitude under the 25MB/species budget (HUD reads 0.0/25MB —
it rounds; total across all species ≈ 0.1 MB). No per-plant instance buffers
exist (placement is procedural), so cost never scales with plant count.

The committed baked artifacts in `mesh/baked/014-ribbon-skeleton/` are 75 KB
each (f32 on disk; converted to f16 at upload).

## Bake

`bake.ts::distillSpecies` runs in-browser (or under node via `tsx`) from the
raw GCMESH1 vertex cloud — no connectivity used, which matters because the
three meshes range from one welded component (calamagrostis) to 1.7M micro-
fragments (poa). ~0.5–10 s per species (poa slowest). Output serialized by
`serializeAtlas`, committed to `mesh/baked/<id>/` via `commitBake`.

The harness `bakedArtifact()` cache is **intentionally bypassed**: the dev
server answers a missing `/mesh/baked/*.bin` with a 200 `index.html`, which
would poison the OPFS cache. Instead we fetch the committed file directly and
**validate it with `parseAtlas` (magic+version)** before trusting it; on any
failure we re-distill and re-commit. Committed artifacts are real bytes and
load instantly.

## Status

**working** — verified by headless screenshot at all four standard cams
(grazing, topdown, inside-plant, far-horizon) on the default stand, seed 42,
det=1, t=2. Zero console errors. All three species render.

## Findings

- **inside-plant** is the strongest angle: clearly curved 3D blades, straw-
  pink calamagrostis panicle tufts, distinct stems — real geometry unrolled
  from a 37 KB texture per species.
- **grazing / far-horizon**: dense grass meadow to the horizon, green blades
  with brown panicle heads; reads convincingly as groundcover.
- **topdown** is the weakest: camera-facing aggregate/plume cards seen down
  their own axis want to spin; a stability guard blends toward a world-
  horizontal lateral there, which tames most of the pinwheel but a faint radial
  texture and the hard `regionRadius` circle remain. This is the classic
  billboard-from-overhead limitation; the rules permit graceful breakdown.
- Perf not quoted — GPU contended by 15 sibling agents during the run. The
  dominant cost is the detail-ribbon dispatch (detailSide² × 128 slots × 48
  ribbons instances, most degenerate-culled in the VS). A future structural win
  would be a compute prepass that compacts existing plants so the draw stops
  dispatching empty scatter slots × 48.

Bench: none recorded yet (contended GPU makes numbers meaningless during the
parallel wave). A/B vs `000-ground-truth`:
`#/ab/014-ribbon-skeleton/000-ground-truth?stand=calamagrostis-pure&cam=grazing&seed=42`.
