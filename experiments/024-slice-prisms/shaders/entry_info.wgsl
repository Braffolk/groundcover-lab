// Per-stand-entry parameters shared by the cull compute pass (cull.wgsl) and
// the prism draw (prisms.wgsl). Written every frame by main.ts — keep the
// field order in sync with the info[] writes there.

const N_AZ_VIEW: u32 = 8u;
const N_AZ_BAKED: u32 = 4u;
const N_SLAB: u32 = 3u;
const N_TOP: u32 = 3u;
const TILE_TOP_SLAB0: u32 = 12u; // N_AZ_BAKED * N_SLAB
const TILE_MERGED0: u32 = 15u;   // + N_TOP
const TILE_TOP_MERGED: u32 = 23u;

struct EntryInfo {
  planes: array<vec4f, 6>, // world-space frustum planes (nx,ny,nz,d), normalized
  origin_cell: vec2f,      // south-west scatter cell of the region (already
                           // clamped to the stand's cell range on the CPU)
  side_x: f32,             // region width in cells (clamped to the stand)
  side_z: f32,             // region depth in cells (clamped to the stand)

  seed: f32,
  entry_index: f32,        // index into stand_table
  region_r: f32,           // region radius (m) around the camera
  near_capacity: f32,      // prism-LOD instance buffer capacity

  far_capacity: f32,       // merged-card instance buffer capacity
  card_y0: f32,            // baked capture box (unit scale, metres)
  card_y1: f32,
  card_r: f32,             // baked horizontal support radius

  card_cx: f32,            // clump center offset in mesh frame
  card_cz: f32,
  cull_radius: f32,        // bounding-sphere radius at scale 1
  alpha_ref: f32,          // base alpha-test reference

  bottom_shade: f32,       // grounding gradient strength
  canopy_shade: f32,       // front-to-back canopy occlusion spread
  slab_dist: f32,          // prism LOD range (m at scale 1)
  top_elev_lo: f32,        // top cards fade in between these view elevations

  top_elev_hi: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,

  // [azBaked * N_SLAB + slab] centroid depth of the prism along the baked
  // azimuth (metres at unit scale, + = toward the bake camera). 4 per vec4.
  slab_depth: array<vec4f, 3>,
  // xyz: centre height of top prism 0..2; w: merged top card height.
  top_height: vec4f,
  // Per-tile tight coverage box in ATLAS uv (u0, v0, u1, v1).
  uv_rect: array<vec4f, 24>,
  // The same box in TILE-LOCAL uv, i.e. what geometry it maps to.
  geo_rect: array<vec4f, 24>,
}
