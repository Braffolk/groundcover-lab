// Per-stand-entry parameters shared by the cull compute pass (cull.wgsl) and
// the card draw (cards.wgsl). Written every frame by main.ts — keep the field
// order in sync with the info[] writes there (INFO_FLOATS = 48).

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
}
