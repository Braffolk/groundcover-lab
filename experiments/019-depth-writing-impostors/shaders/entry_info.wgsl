// Per-stand-entry parameters shared by the cull compute pass (cull.wgsl) and
// the impostor draw (impostor.wgsl). Written every frame by main.ts — keep the
// field order in sync with the info[] writes there (INFO_FLOATS = 96).

struct EntryInfo {
  planes: array<vec4f, 6>,  // world-space frustum planes (nx,ny,nz,d), normalized
  /**
   * Distance shells, near to far: (outer radius m, base element in the shared
   * instance buffer, capacity, unused). Every shell is its own indirect draw
   * and they are issued in order, so the whole field rasterizes front to back
   * and hi-z rejects the shells behind the ones already laid down.
   */
  shells: array<vec4f, 8>,
  origin_cell_x: f32,       // south-west scatter cell of the region (already
  origin_cell_z: f32,       // clamped to the stand's cell range on the CPU)
  side_x: f32,              // region width in cells
  side_z: f32,              // region depth in cells
  seed: f32,
  entry_index: f32,         // index into stand_table
  region_r: f32,            // region radius (m) around the camera
  n_shells: f32,
  cx: f32,                  // local capture centre (metres at scale 1)
  cy: f32,
  cz: f32,
  y0: f32,                  // vertical span of the plant in its own frame
  y1: f32,
  row_el0: f32,             // elevation of each view row, radians
  row_el1: f32,
  row_el2: f32,
  row_el3: f32,
  row_az0: f32,             // azimuths in each row
  row_az1: f32,
  row_az2: f32,
  row_az3: f32,
  row_off0: f32,            // first layer index of each row
  row_off1: f32,
  row_off2: f32,
  row_off3: f32,
  n_rows: f32,              // rows in use
  top_layer: f32,           // layer index of the straight-down view
  alpha_cut: f32,           // base alpha-test reference
  cull_radius: f32,         // bounding-sphere radius at scale 1
  depth_push: f32,          // metres toward the camera when writing frag depth
  ao_strength: f32,         // 0 = ignore baked AO, 1 = full
  warp_scale: f32,          // 0 = plain billboard, 1 = full depth warp
  warp_fade0: f32,          // distance (m) where the warp starts fading out
  warp_end: f32,            // distance (m) where the warp has faded to nothing
  tile_px: f32,             // atlas layer resolution
  max_lod: f32,             // highest mip level in the chain
  warp_lod: f32,            // floor mip level for the warp's depth estimate
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
}
