// Per-(stand entry, LOD bucket) parameters shared by cull.wgsl and
// strips.wgsl. One 512B slot per bucket, addressed with a dynamic uniform
// offset; the cull pass reads slot 0 (its bucket-specific fields are unused,
// the bases/caps/thresholds arrays cover all buckets). Field order must match
// the info[] writes in main.ts.

const N_BUCKETS: u32 = 5u;

struct EntryInfo {
  planes: array<vec4f, 6>, // world-space frustum planes (nx,ny,nz,d), normalized
  // origin_cell.xy = south-west scatter cell of the region (already clamped to
  // the stand's cell range on the CPU), zw = region size in cells
  region: vec4f,
  ids: vec4f,       // seed, entry_index, region_r (m), cull_radius (m at scale 1)
  box: vec4f,       // strip_y0, strip_y1, strip_r (baked support radius), px_scale
  clump: vec4f,     // clump_cx, clump_cz, alpha_ref, face_cam (this bucket)
  shade: vec4f,     // bottom_shade, top_card (0/1), top_frac, tint_jitter
  bucket: vec4f,    // level_base (strip slot), n_strips, inst_base, flags
                    // flags bit 0: skip the normal-map tap (flat far bucket)
  thresholds: vec4f, // projected-diameter (px) thresholds between the 5 buckets
  bases: array<vec4f, 2>, // instance-buffer base slot per bucket (5 used)
  caps: array<vec4f, 2>,  // instance capacity per bucket (5 used)
}

struct PlantInst {
  pos: vec3f,
  packed_bits: u32, // yaw 10b | scale 12b (0..4 m) | phase 10b
}
