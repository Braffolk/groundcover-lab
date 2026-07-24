// Shared declarations for the splat-reconstruct passes.
// Must stay in sync with the constants in ../bake.ts.

const LOD_LEVELS: u32 = 5u;
const SR_LOD_COUNTS = array<u32, 5>(2048u, 384u, 96u, 24u, 6u);
const SR_BUCKET_CAPS = array<u32, 5>(1024u, 4096u, 16384u, 32768u, 131072u);
const SR_BUCKET_BASES = array<u32, 5>(0u, 1024u, 5120u, 21504u, 54272u);

struct SrParams {
  base_cell: vec2i,     // window origin cell (clamped to the stand region)
  seed: u32,
  num_entries: u32,
  rings: vec4f,         // LOD ring distances 0..3 (m), detail-scaled
  fade_start: f32,
  fade_end: f32,
  splat_scale: f32,
  var_amt: f32,
  surface_tol: f32,     // depth affinity tolerance (m)
  gap_fill: f32,
  debug_mode: u32,      // 0 off, 1 coverage, 2 normals, 3 depth
  flags: u32,           // bit0 = reconstruct enabled
}

// Plant record produced by the cull pass (16B).
//   xz  : world position
//   ys  : pack2x16float(yaw, scale)
//   pfy : y-height f16 | phase u8 << 16 | fade u8 << 24
struct PlantRec {
  xz: vec2f,
  ys: u32,
  pfy: u32,
}

// Octahedral decode, y-up — matches octEncode8 in bake.ts.
fn sr_oct_decode(u: f32, v: f32) -> vec3f {
  let f = vec2f(u, v) * 2.0 - 1.0;
  var n = vec3f(f.x, 1.0 - abs(f.x) - abs(f.y), f.y);
  if (n.y < 0.0) {
    let ox = n.x;
    n.x = (1.0 - abs(f.y)) * select(-1.0, 1.0, ox >= 0.0);
    n.z = (1.0 - abs(f.x)) * select(-1.0, 1.0, f.y >= 0.0);
  }
  return normalize(n);
}
