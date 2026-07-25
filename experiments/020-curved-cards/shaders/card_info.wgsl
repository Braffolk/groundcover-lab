// Per-(stand entry, LOD) parameters shared by the cull compute pass
// (cull.wgsl) and the curved-card draw (curved.wgsl). Exactly 256 bytes so the
// three LOD blocks of one entry live in ONE uniform buffer addressed by a
// dynamic offset; the cull pass binds block 0, which carries the LOD-invariant
// fields plus every LOD's ring base/capacity. Written every frame by main.ts —
// keep the field order in sync with the info[] writes there.

struct CardInfo {
  planes: array<vec4f, 6>, // world-space frustum planes (nx,ny,nz,d), normalized
  origin_cell: vec2f,      // south-west scatter cell of the region (already
                           // clamped to the stand's cell range on the CPU)
  side_x: f32,             // region width in cells (clamped to the stand)
  side_z: f32,             // region depth in cells (clamped to the stand)
  seed: f32,
  entry_index: f32,        // index into stand_table
  region_r: f32,           // region radius (m) around the camera
  alpha_ref: f32,          // base alpha-test reference
  card_y0: f32,            // baked capture box (unit scale, metres)
  card_y1: f32,
  card_r: f32,             // baked horizontal support radius
  card_cx: f32,            // clump center offset in mesh frame
  card_cz: f32,
  cull_radius: f32,        // bounding-sphere radius at scale 1
  top_frac: f32,           // flat-canopy height as fraction of [y0, y1]
  ao_strength: f32,        // how much of the baked sky occlusion to apply
  lod_base: vec4f,         // first instance slot of each LOD ring
  lod_cap: vec4f,          // capacity of each LOD ring
  lod_dist: vec4f,         // x,y: scale-normalized ring radii (m)
  grid: vec4f,             // side grid (x,y) and top grid (z,w) of THIS LOD
  shape: vec4f,            // x: carpet height-field decode SPAN,
                           // y: occlusion normalization gain,
                           // z: depth shading, w: card curvature scale
  ring: vec4f,             // x: this LOD's instance base, y: its capacity,
                           // z: scatter slots per cell to ENUMERATE (128 for a
                           //    scattered entry, carpet_div^2 for a mat — the
                           //    two are not the same number, and the mat's is
                           //    deliberately over the scatter budget),
                           // w: carpet height-field decode BASE. A mat's DISP
                           //    texture is its own tile-square field rescaled
                           //    to the full 8-bit range, so a height fraction
                           //    is ring.w + texel * shape.x. top_frac carries
                           //    its mean, i.e. where a flat tile sits.
}
