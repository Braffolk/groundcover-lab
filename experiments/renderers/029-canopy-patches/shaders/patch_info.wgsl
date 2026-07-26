// Per-stand-entry parameters shared by the cull compute pass (cull.wgsl) and
// the patch draw (patches.wgsl). Written every frame by main.ts — the field
// order IS the float layout there (all scalars are f32, all arrays vec4f, so
// every offset is trivially predictable).
//
// Slice tables are indexed by a flat slice id:
//   near side   bin * SLABS + slab        (0 .. BINS*SLABS-1)
//   near crown  BINS*SLABS + crown        (.. NEAR_SLICES-1)
//   far side    NEAR_SLICES + bin
//   far crown   NEAR_SLICES + BINS
// The texture array layer is that id, minus NEAR_SLICES for the far family.

struct PatchInfo {
  planes: array<vec4f, 6>,  // world-space frustum planes (nx,ny,nz,d), normalized
  origin_cell_x: f32,       // south-west scatter cell of the region (CPU-clamped
  origin_cell_z: f32,       // to the stand's cell range)
  side_x: f32,              // region width in cells
  side_z: f32,              // region depth in cells
  seed: f32,
  entry_index: f32,         // index into stand_table
  region_r: f32,            // region radius (m) around the camera
  cap_near: f32,            // patch-stack instance capacity
  cap_far: f32,             // composite-card instance capacity
  far_base: f32,            // first far record index in the shared instance array
  patch_dist: f32,          // stack -> card collapse distance (m)
  alpha_ref: f32,
  slab_spread: f32,         // multiplier on baked slab plane offsets
  slab_shade: f32,          // darkening per slab of canopy depth
  bottom_shade: f32,
  y0: f32,                  // baked capture vertical extent (unit scale, m)
  y1: f32,
  r_xz: f32,                // baked horizontal support radius
  cx: f32,                  // clump centre offset in the mesh frame
  cz: f32,
  crown_h: f32,             // composite crown-card height (unit scale, m)
  cull_radius: f32,         // bounding-sphere radius at scale 1
  pos_origin_x: f32,        // instance position quantisation origin
  pos_origin_z: f32,
  pos_range: f32,           // instance position quantisation half-range (m)
  top_enable: f32,          // 0 disables the crown patches
  tint: f32,                // 1 = tint fragments by patch index
  _pad0: f32,
  // SLICES = BINS*SLABS + CROWNS + BINS + 1 = 27. These MUST match main.ts's
  // sliceTables() layout exactly: rects first, then depths.
  slice_rect: array<vec4f, 27>,  // a0, a1, b0, b1 — tight framing, metres
  slice_depth: array<vec4f, 27>, // x: plane depth (side) / height (crown), m
}
