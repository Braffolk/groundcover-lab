// Per-stand-entry parameters shared by the cull pass and both draw paths.
// Written every frame by main.ts — keep the field order in sync with the
// info[] writes there (INFO_FLOATS = 76; float index in the comments).

struct EntryInfo {
  planes: array<vec4f, 6>, // 0..23  world-space frustum planes, normalized
  origin_cell: vec2f,      // 24,25  south-west scatter cell of the region
                           //        (CPU-clamped to the stand's cell range)
  side_x: f32,             // 26     region width in cells
  side_z: f32,             // 27     region depth in cells

  seed: f32,               // 28
  entry_index: f32,        // 29     index into stand_table
  region_r: f32,           // 30     region radius (m) around the camera
  slots_per_cell: f32,     // 31     candidate slots per cell for THIS entry:
                           //        carpet_div^2 for a mat (484 at life size,
                           //        deliberately over SCATTER_MAX_PER_CELL),
                           //        else 128

  cap0: f32,               // 32     instance capacity, LOD 0 (patches / cards)
  cap1: f32,               // 33     LOD 1 (flat per-tile quads)
  cap2: f32,               // 34     LOD 2 (2x2 merged quads)
  cull_radius: f32,        // 35     bounding-sphere radius at scale 1

  patch_dist: f32,         // 36     nearer than this: displaced patch
  merge_dist: f32,         // 37     further than this: 2x2 merged quads
  flags: f32,              // 38     bit0 micro shadow, bit1 parallax shadow,
                           //        bit2 parallax refinement step
  tile_span: f32,          // 39     world size of one periodic tile

  plane_h: f32,            // 40     mean cushion surface, m above the ground
  relief_h: f32,           // 41     world height of the baked capture range
  base_y: f32,             // 42     world y of height 0, relative to the ground
  gap_ref: f32,            // 43     height below which a texel is a real gap

  parallax: f32,           // 44
  displace: f32,           // 45
  patch_div: f32,          // 46     NxN quads per near patch
  patch_mip: f32,          // 47     mip the patch's vertex displacement reads

  ao_strength: f32,        // 48
  light_wrap: f32,         // 49
  fuzz: f32,               // 50
  ao_sat: f32,             // 51

  wetness: f32,            // 52     this state's habitat wetness x the param
  tile_scale: f32,         // 53     the carpet's constant scale, from the stand
  ao_fresnel: f32,         // 54     how far a glancing angle fades the AO out
  _pad1: f32,              // 55

  // Upright-plant card path (baked 3x3 view atlas).
  card_y0: f32,            // 56
  card_y1: f32,            // 57
  card_r: f32,             // 58
  card_cx: f32,            // 59

  card_cz: f32,            // 60
  alpha_ref: f32,          // 61
  top_frac: f32,           // 62
  bottom_shade: f32,       // 63

  tip_color: vec4f,        // 64..67 measured capitulum-tip albedo (fuzz tint)
  base_color: vec4f,       // 68..71 measured deep-cushion albedo (wrap tint)
  mean_color: vec4f,       // 72..75
}
