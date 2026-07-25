// Per-stand-entry parameters shared by the cull compute pass and the card
// draw. `ClumpDyn` is rewritten every frame by main.ts (camera-dependent);
// `ClumpAtlas` is written once at startup (bake geometry + atlas layout).
// Keep the field order in sync with the writes in main.ts.

const K_SUB: u32 = 4u;      // sub-clump cards per plant in the near LOD
const N_UNITS: u32 = 5u;    // K_SUB sub-clumps + the merged whole-plant unit
const MERGED: u32 = 4u;
const N_SIDE: f32 = 8.0;    // baked azimuths per unit
const N_VIEWS: u32 = 9u;    // 8 side + 1 top
// Front-to-back draw buckets: the cull bins every survivor by distance into
// equal-AREA rings (so the bins hold roughly equal counts) and each ring is a
// separate indirect draw, issued near ring first. Hard alpha test + depth
// write then let early-z reject most of the deeper rings before they sample.
const NEAR_BINS: u32 = 3u;
const FAR_BINS: u32 = 4u;
const N_BINS: u32 = 7u;

struct ClumpDyn {
  planes: array<vec4f, 6>, // world-space frustum planes (nx,ny,nz,d), normalized
  origin_cell: vec2f,      // south-west scatter cell of the region (CPU-clamped
                           // to the stand's cell range)
  side_x: f32,             // region width in cells
  side_z: f32,             // region depth in cells
  seed: f32,
  entry_index: f32,        // index into stand_table
  region_r: f32,           // region radius (m) around the camera
  near_bin_cap: f32,       // instances per near distance bin
  far_bin_cap: f32,        // instances per far distance bin
  cull_radius: f32,        // bounding-sphere radius at scale 1
  alpha_ref: f32,          // base alpha-test reference
  top_frac: f32,           // top card height as fraction of [y0, y1]
  bottom_shade: f32,       // grounding gradient strength
  lod_dist: f32,           // sub-clump cards inside this radius, merged beyond
  merged_y0: f32,          // merged capture box (unit scale, metres)
  merged_y1: f32,
  merged_r: f32,
  sway_spread: f32,        // per-sub-clump wind phase spread
  // Alpha-reference multiplier for sub-clump cards. A sub-clump tile resolves
  // ~2.3x more world detail per texel than the merged tile, so at the same
  // screen size it samples a coarser mip: features end up thinner and more
  // isolated, and bilinear-then-threshold eats more of them. The bake's
  // coverage calibration matches texel COUNTS, which cannot see that, so
  // sub-clump cards test against a slightly lower reference to land on the
  // same on-screen density as the merged card.
  near_alpha_bias: f32,

  _pad1: f32,
  _pad2: f32,
  _pad3: f32,
  _pad4: f32,
  _pad5: f32,
}

struct ClumpAtlas {
  // x,y = albedo atlas size in texels; z,w unused.
  atlas_dims: vec4f,
  // Per unit: cx, cz, rXZ, y0 (mesh frame, metres at scale 1).
  unit_box: array<vec4f, N_UNITS>,
  // Per unit: y1, 0, 0, 0.
  unit_ext: array<vec4f, N_UNITS>,
  // Per tile, two vec4 each (tile index = unit * N_VIEWS + view):
  //   [2t]   uv rect already narrowed to the baked alpha bbox: u0, v0, du, dv
  //   [2t+1] the same rect in the unit's local card space: lx0, ly0, lw, lh
  tiles: array<vec4f, 90>,
}
