// Per-stand-entry parameters shared by the cull compute pass (cull.wgsl) and
// the card draw (cards.wgsl). Written every frame by main.ts — keep the field
// order in sync with the info[] writes there (INFO_FLOATS = 72).
//
// The vec4 arrays come first so every 16-byte-aligned member starts where the
// TypeScript side thinks it does.

const NB: u32 = 6u; // distance buckets (front-to-back draw order)

struct EntryInfo {
  planes: array<vec4f, 6>,      // world-space frustum planes (nx,ny,nz,d), normalized
  bucket_edge: array<vec4f, 2>, // outer radius (m) of bucket i, 8 slots (6 used)
  bucket_base: array<vec4f, 2>, // firstInstance of bucket i in the instance buffer
  bucket_cap: array<vec4f, 2>,  // instance capacity of bucket i
  origin_cell: vec2f,           // SW scatter cell of the region (CPU-clamped to the stand)
  side_x: f32,                  // region width in cells
  side_z: f32,                  // region depth in cells
  seed: f32,
  entry_index: f32,             // index into stand_table
  region_r: f32,                // region radius (m) around the camera
  card_y0: f32,                 // baked capture box (unit scale, metres)
  card_y1: f32,
  card_r: f32,                  // baked horizontal support radius
  card_cx: f32,                 // clump centre offset in the mesh frame
  card_cz: f32,
  cull_radius: f32,             // bounding-sphere radius at scale 1
  alpha_ref: f32,               // base alpha-test reference
  top_frac: f32,                // top card height as a fraction of [y0, y1]
  ao_strength: f32,
  translucency: f32,
  relief_scale: f32,
  relief_fade0: f32,            // distance where parallax starts fading out
  relief_fade1: f32,            // distance where parallax is gone (= far tier edge)
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
}

/// Bucket index for a horizontal camera distance — six static compares,
/// evaluated once per surviving plant in the cull pass, never per fragment.
fn bucket_of(d: f32, e0: vec4f, e1: vec4f) -> u32 {
  var b = 0u;
  if (d > e0.x) { b = 1u; }
  if (d > e0.y) { b = 2u; }
  if (d > e0.z) { b = 3u; }
  if (d > e0.w) { b = 4u; }
  if (d > e1.x) { b = 5u; }
  return b;
}

fn pick8(a: vec4f, b: vec4f, i: u32) -> f32 {
  let v = select(a, b, i >= 4u);
  let j = i & 3u;
  var out = v.x;
  if (j == 1u) { out = v.y; }
  if (j == 2u) { out = v.z; }
  if (j == 3u) { out = v.w; }
  return out;
}
