// Per-stand-entry parameters shared by the cull compute pass (cull.wgsl) and
// the part draw (parts.wgsl). Written every frame by main.ts — keep the field
// order in sync with the info[] writes there (INFO_FLOATS = 168).
//
// Two copies exist per entry, identical except for `draw_level`: one bound for
// the near (part-assembled) draw, one for the far (whole-plant card) draw.

struct EntryInfo {
  planes: array<vec4f, 6>, // world-space frustum planes (nx,ny,nz,d), normalized
  origin_cell: vec2f,      // south-west scatter cell of the region (CPU-clamped
                           // to the stand's cell range)
  side_x: f32,             // region width in cells
  side_z: f32,             // region depth in cells
  seed: f32,
  entry_index: f32,        // index into stand_table
  region_r: f32,           // region radius (m) around the camera
  capacity_near: f32,      // near instance buffer capacity
  capacity_far: f32,       // far instance buffer capacity
  part_r: f32,             // LOD radius (m): inside it a plant is assembled
                           // from parts, outside it it is one billboard pair
  alpha_ref: f32,          // base alpha-test reference
  self_shade: f32,         // per-part sun-side occlusion strength
  bottom_shade: f32,       // vertical canopy-depth gradient strength
  plant_y0: f32,           // whole-plant capture box (unit scale, metres)
  plant_y1: f32,
  plant_rxz: f32,
  clump_cx: f32,           // clump center offset in the mesh frame
  clump_cz: f32,
  cull_radius: f32,        // bounding-sphere radius at scale 1
  draw_level: f32,         // 1 = near/parts draw, 0 = far/whole-plant draw
  top_frac: f32,           // far top card height as a fraction of [y0, y1]
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
  // Per part (p = band * SECTORS + sector): [0] center xyz (relative to the
  // clump axis, mesh frame), [1] half extents xyz. Already inflated by the
  // bake's tile margin, so the runtime card matches the baked box exactly.
  part_box: array<vec4f, 24>,
  // Per band: [0] = (center x, center z, half x, half z), [1] = (card height,
  // 0, 0, 0) — the horizontal fill card.
  band_box: array<vec4f, 6>,
}
