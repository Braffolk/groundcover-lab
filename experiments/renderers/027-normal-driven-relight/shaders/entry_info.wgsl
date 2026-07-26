// Per-stand-entry parameters shared by the cull compute pass (cull.wgsl) and
// the card draw (relight.wgsl). Written every frame by main.ts — keep the
// field order in sync with the info[] writes there (64 floats / 256 bytes).
//
// The LOD tier is NOT in here: it is the distance ring the cull sorted the plant
// into, and each ring is drawn by its own pipeline specialized on an override
// constant, so the tier is a compile-time value in the shader.

struct EntryInfo {
  planes: array<vec4f, 6>, // world-space frustum planes (nx,ny,nz,d), normalized
  origin_cell: vec2f,      // south-west scatter cell of the region (already
                           // clamped to the stand's cell range on the CPU)
  side_x: f32,             // region width in cells (clamped to the stand)
  side_z: f32,             // region depth in cells
  seed: f32,
  entry_index: f32,        // index into stand_table
  region_r: f32,           // region radius (m) around the camera
  cull_radius: f32,        // bounding-sphere radius at scale 1
  // Distance rings: the cull sorts survivors into 4 rings and the draw submits
  // them near->far, so early-z can reject the deep layers. Ring == LOD tier.
  bucket_base: vec4f,      // instance-buffer offset of each ring
  bucket_cap: vec4f,       // capacity of each ring
  bucket_r: vec4f,         // outer radius of rings 0..3
  card_y0: f32,            // baked capture box (unit scale, metres)
  card_y1: f32,
  card_r: f32,             // baked horizontal support radius
  card_cx: f32,            // clump center offset in mesh frame
  card_cz: f32,
  alpha_ref: f32,          // base alpha-test reference
  top_plane: f32,          // canopy card height as fraction of [y0, y1]
  relief: f32,             // heightfield warp strength
  relief_clamp: f32,       // max warp offset in tile-uv units
  self_shadow: f32,
  shadow_step: f32,        // sun-ray step, fraction of card_r
  ao_strength: f32,
  translucency: f32,
  tile_inset: f32,         // uv guard band inside each atlas tile
  mirror: f32,             // 1 = per-plant mirrored variants enabled
  contact_skirt: f32,      // card skirt below terrain, fraction of height
  top_card: f32,           // 1 = canopy card enabled
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
}
