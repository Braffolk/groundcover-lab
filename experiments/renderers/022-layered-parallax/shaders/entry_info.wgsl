// Per-stand-entry parameters shared by the cull compute pass (cull.wgsl) and
// the draws (cards.wgsl for upright plants, carpet.wgsl for mats). Written
// every frame by main.ts — keep the field order in sync with the info[] writes
// there (INFO_FLOATS = 64). `carpet_drops` is a vec4 and therefore 16-byte
// aligned: the three trailing pads before it are load bearing.

struct EntryInfo {
  planes: array<vec4f, 6>, // world-space frustum planes (nx,ny,nz,d), normalized
  origin_cell: vec2f,      // south-west scatter cell of the region (already
                           // clamped to the stand's cell range on the CPU)
  side_x: f32,             // region width in cells (clamped to the stand)
  side_z: f32,             // region depth in cells (clamped to the stand)
  seed: f32,
  entry_index: f32,        // index into stand_table
  region_r: f32,           // region radius (m) around the camera
  capacity: f32,           // far instance buffer capacity
  card_y0: f32,            // baked capture box (unit scale, metres)
  card_y1: f32,
  card_r: f32,             // baked horizontal support radius
  card_cx: f32,            // clump center offset in mesh frame
  card_cz: f32,
  cull_radius: f32,        // bounding-sphere radius at scale 1
  alpha_ref: f32,          // base alpha-test reference
  lod_dist: f32,           // layered -> merged switch distance (m)
  side_span: f32,          // d(front slab plane) - d(back slab plane), unit scale
  top_span: f32,           // h(top band) - h(bottom band), unit scale
  layer_shade: f32,        // extra darkening per slab behind the front one
  bottom_shade: f32,       // grounding gradient strength
  near_capacity: f32,      // near instance buffer capacity
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
  // --- carpet entries (stand_table[i].carpet_div > 0) ----------------------
  /** Metres the eye ray descends from the proxy plane to bands 1..3 and to
      the merged tile, at scale 1. Band 0 IS the proxy plane. */
  carpet_drops: vec4f,
  slots_per_cell: f32,     // carpet_div^2 for a mat, SCATTER_MAX_PER_CELL else
  carpet: f32,             // 1 = this entry is a carpet
  carpet_h0: f32,          // proxy plane height above the ground, scale 1
  carpet_alpha: f32,       // carpet-specific alpha reference (a mat is closed)
  carpet_max_slope: f32,   // cap on horizontal travel per metre of descent
  carpet_merge_lod: f32,   // mip level where bands start dissolving to merged
  carpet_merge_span: f32,  // mip levels the dissolve takes
  carpet_tex: f32,         // band texels across the tile (for the lod)
  carpet_band_ref: f32,    // coverage a band needs to CLAIM a fragment
  carpet_depth_shade: f32, // canopy occlusion at the bottom of the cushion
  _cpad0: f32,
  _cpad1: f32,
}
