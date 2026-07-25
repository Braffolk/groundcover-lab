// Per-stand-entry parameters shared by the cull compute pass (cull.wgsl), the
// impostor draw (impostor.wgsl) and the carpet draw (carpet.wgsl). Written
// every frame by main.ts — keep the field order in sync with the info[] writes
// there (INFO_FLOATS = 112).

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
  /**
   * Candidate slots per scatter cell for THIS entry, rounded up to a multiple
   * of the 64-wide workgroup. A carpet entry has carpet_div^2 of them (484 for
   * the bog moss) — deliberately more than SCATTER_MAX_PER_CELL, which is why
   * nothing here may hardcode 128.
   */
  slots_per_cell: f32,
  // --- carpet (mat) entries only; zero for upright species -------------------
  tile_world: f32,          // footprint_m * carpet scale (m), = the grid step
  carpet_y: f32,            // quad plane height above the ground (m)
  relief_m: f32,            // world metres per unit of the stored depth code
  t_lo_x: f32,              // texcoord of the tile square's (0,0) mesh corner
  t_lo_y: f32,
  t_span_x: f32,            // texcoord extent of one tile period
  t_span_y: f32,
  t_per_m_x: f32,           // texcoord shift per world metre in the tile frame
  t_per_m_y: f32,
  carpet_alpha: f32,        // alpha reference for mat tiles (NOT alpha_cut)
  carpet_layer: f32,        // atlas layer holding the straight-down view
  carpet_push: f32,         // metres toward the camera when writing frag depth
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
  _pad3: f32,
  _pad4: f32,
  _pad5: f32,
}
