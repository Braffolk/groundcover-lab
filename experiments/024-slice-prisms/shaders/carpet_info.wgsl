// Per-stand-entry parameters for the CARPET path (carpet_cull.wgsl +
// carpet.wgsl). Written every frame by main.ts — keep the field order in sync
// with the carpetInfo[] writes there. A carpet entry never uses EntryInfo:
// it has no azimuths, no per-tile coverage boxes and no wind, and it needs the
// tile's own footprint and slice heights instead.

const CARPET_SLICES: u32 = 5u;
const CARPET_COLS: f32 = 2.0;

struct CarpetInfo {
  planes: array<vec4f, 6>, // world-space frustum planes (nx,ny,nz,d), normalized

  origin_cell: vec2f,      // SW scatter cell of the region (clamped to the stand)
  side_x: f32,             // region width in cells
  side_z: f32,             // region depth in cells

  seed: f32,
  entry_index: f32,        // index into stand_table
  region_r: f32,           // region radius (m) around the camera
  near_capacity: f32,      // slice-stack instance capacity

  far_capacity: f32,       // merged-card instance capacity
  slots_per_cell: f32,     // carpet_div^2 rounded UP to a multiple of 64, so the
                           // cell-level rejects stay workgroup-uniform
  slice_dist: f32,         // slice-stack LOD range (m)
  alpha_ref: f32,          // carpet alpha-test reference (a mat must not dissolve)

  tile_m: f32,             // periodic tile footprint at scale 1 (m)
  tile_scale: f32,         // the stand's constant carpet scale
  y0: f32,                 // baked capture box, mesh frame (m at scale 1)
  y1: f32,

  uv_step: vec2f,          // tile size in atlas uv
  uv_inset: f32,           // baked margin, as a fraction of the tile
  n_slice: f32,

  ao: f32,                 // interstitial occlusion spread across the stack
  cull_radius: f32,        // bounding-sphere radius of one tile (m)
  // Camera height above the ground under the camera (CPU-side terrain lookup).
  // The LOD distance is built from this instead of a per-vertex terrain fetch,
  // and the cull uses the SAME formula so the two never disagree about which
  // tiles are stacked.
  cam_height: f32,
  // Alpha reference for a RELIEF SLICE (the base card uses alpha_ref). A slice
  // must own the majority of the footprint it is minified into, or the mip
  // chain lets all five paint everywhere — see carpet.wgsl.
  relief_ref: f32,

  // Scales the stored tangent of the cushion normal map. 1 = as baked, 0 = a
  // perfectly flat-lit mat (which is what every version before v3 drew).
  // (three f32 pads, not a vec3f: a vec3f would demand 16B alignment here and
  // silently shift slice_h.)
  normal_gain: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,

  // [0..n_slice-1] slice centroid heights (m, mesh frame); [n_slice] is the
  // merged card's height, i.e. where the collapsed stack lands.
  slice_h: array<vec4f, 2>,
}
