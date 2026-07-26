// Shared declarations for the splat-reconstruct passes.
// Must stay in sync with the constants in ../bake.ts.

// Six distance buckets: the five baked LOD levels plus a synthetic far level of
// one splat per plant, which only a carpet ever reaches (see BUCKET_SPLATS in
// ../bake.ts).
const SR_BUCKETS: u32 = 6u;
const SR_BUCKET_SPLATS = array<u32, 6>(2048u, 384u, 96u, 24u, 6u, 1u);
// Two record layouts over the same 185344-record buffer — see BUCKET_CAPS_* in
// ../bake.ts. A scattered entry's cap for bucket 5 is 0, which is what stops it
// from ever writing there.
const SR_CAPS_SCATTER = array<u32, 6>(1024u, 4096u, 16384u, 32768u, 131072u, 0u);
const SR_BASES_SCATTER = array<u32, 6>(0u, 1024u, 5120u, 21504u, 54272u, 185344u);
const SR_CAPS_CARPET = array<u32, 6>(256u, 1024u, 4096u, 16384u, 32768u, 131072u);
const SR_BASES_CARPET = array<u32, 6>(0u, 256u, 1280u, 5376u, 21760u, 54528u);

/**
 * Height fraction the far carpet disc reports, i.e. where the visible top of a
 * cushion sits inside the mesh's height range. Chosen so the far mat's height
 * AO matches the near field's, verified in debug=lighting across distance bands.
 */
const CARPET_TOP_HF: f32 = 0.92;

struct SrParams {
  base_cell: vec2i,     // window origin cell (clamped to the stand region)
  seed: u32,
  num_entries: u32,
  rings: vec4f,         // bucket ring distances 0..3 (m), detail-scaled
  fade_start: f32,
  fade_end: f32,
  splat_scale: f32,
  var_amt: f32,
  surface_tol: f32,     // depth affinity tolerance (m)
  gap_fill: f32,
  cell_pad: f32,        // conservative whole-cell cull margin (m), stand-derived
  flags: u32,           // bit0 = reconstruct enabled
  ring4: f32,           // bucket 4 -> 5 ring; only a carpet ever crosses it
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
