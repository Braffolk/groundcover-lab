// Per-stand-entry constants shared by the cull compute pass (cull.wgsl) and
// the splat draw (splats.wgsl). Written every frame by main.ts — keep the
// field order in sync with the info[] writes there (88 f32 = 352 B).

struct EntryInfo {
  planes: array<vec4f, 6>,          // f32 0..23   world frustum planes, normalized
  lod_inst_base: array<vec4f, 4>,   // f32 24..39  first instance slot of each LOD bucket
  lod_cap: array<vec4f, 4>,         // f32 40..55  capacity of each LOD bucket
  origin_cell: vec2f,               // f32 56,57   SW scatter cell of the region
  side_xz: vec2f,                   // f32 58,59   region size in cells
  seed: f32,                        // 60
  entry_index: f32,                 // 61
  region_r: f32,                    // 62  region radius (m)
  px_scale: f32,                    // 63  pixels per metre at 1 m
  splat_y0: f32,                    // 64  baked support box (unit scale, m)
  splat_y1: f32,                    // 65
  splat_rxz: f32,                   // 66
  cull_radius: f32,                 // 67  bounding-sphere radius at scale 1
  lod_t0: f32,                      // 68  plant pixel height that still gets LOD 0
  inv_log_ratio: f32,               // 69  1 / ln(LOD distance ratio)
  lod_count: f32,                   // 70
  splat_scale: f32,                 // 71
  min_px: f32,                      // 72
  ao_strength: f32,                 // 73
  bulge: f32,                       // 74
  alpha_ref: f32,                   // 75
  bmin: vec4f,                      // f32 76..79  splat dequantization box
  bext: vec4f,                      // f32 80..83
  /**
   * x: 1 if this entry is a tiled CARPET (stand_table[i].carpet_div > 0).
   *    A carpet is ground-parallel, terrain-conforming, never camera-facing,
   *    and never faded out from under the camera.
   * y: length (m, at scale 1) whose projected size drives the LOD ladder —
   *    plant height for an upright plant, footprint_m for a carpet tile.
   * z: candidate slots per scatter cell for this entry: carpet_div² (484 for
   *    the bog moss) or SCATTER_MAX_PER_CELL. Hardcoding 128 renders a
   *    quarter of the mat.
   */
  carpet_info: vec4f,               // f32 84..87
}

/// Per-LOD constants, bound with a dynamic offset (256 B stride).
struct LodInfo {
  splat_base: f32,   // first splat of this level in the species splat buffer
  inst_base: f32,    // first instance slot of this level's bucket
  size_a: f32,       // decode scale for the major half-extent
  size_b: f32,       // decode scale for the minor half-extent
  solid: f32,        // 0 = stamp-textured, 1 = solid ellipse (far field)
  face_cam: f32,     // 0 = baked minor axis, 1 = minor axis rolled toward the eye
  ao_lift: f32,      // 1 = lift the baked occlusion back toward unoccluded
  /**
   * 1 = drop the stamp's colour modulation and use the splat's own mean.
   * Only ever set where `solid` already discards the stamp's alpha cut: the
   * atlas stores colour as a modulation around a neutral 0.5 and only dilates
   * it 4 texels into empty space, so a solid ellipse that covers a large
   * uncovered part of its stamp tile would otherwise read black.
   */
  neutral: f32,
  /**
   * How much of the solid-ellipse AREA SHRINK to apply (0..1). A grass canopy
   * at 2px really is only `rho` covered, so its solid stand-in shrinks to that
   * area. A carpet is the opposite: its tiles must still abut, and shrinking
   * the ellipse below the grid step opens the mat into a stipple pattern —
   * the ellipse is the tile's stand-in, not a sample of it.
   */
  shrink_mix: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
}
