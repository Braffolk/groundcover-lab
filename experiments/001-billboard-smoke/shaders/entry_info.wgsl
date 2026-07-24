// Per-stand-entry parameters shared by the cull compute pass (cull.wgsl) and
// the card draw (cards.wgsl). Written every frame by main.ts — keep the field
// order in sync with writeEntryInfo() there.

struct EntryInfo {
  planes: array<vec4f, 6>, // world-space frustum planes (nx,ny,nz,d), normalized
  origin_cell: vec2f,      // south-west scatter cell of the region
  side: f32,               // region size in cells
  seed: f32,
  entry_index: f32,        // index into stand_table
  region_r: f32,           // region radius (m) around the camera
  capacity: f32,           // instance buffer capacity
  stand_cell_min: f32,     // stand region clamp, in cell indices (inclusive)
  stand_cell_max: f32,
  card_y0: f32,            // baked capture box (unit scale, metres)
  card_y1: f32,
  card_r: f32,             // baked horizontal support radius
  card_cx: f32,            // clump center offset in mesh frame
  card_cz: f32,
  cull_radius: f32,        // bounding-sphere radius at scale 1
  alpha_ref: f32,          // base alpha-test reference
  top_frac: f32,           // top card height as fraction of [y0, y1]
  bottom_shade: f32,       // grounding gradient strength
  _pad0: f32,
  _pad1: f32,
}
