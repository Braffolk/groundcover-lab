// Shared between cull.wgsl (compute) and ldi.wgsl (render): the per-stand-entry
// uniform block. Field packing must match the Float32Array layout in main.ts.
//
// The block is split into a STATIC half (bake + stand constants, written once
// at create time) and a DYNAMIC half (`region` + `tune`, floats 8..15, the only
// two vec4s main.ts rewrites per frame). Keep the dynamic pair contiguous — the
// per-frame writeBuffer covers exactly byte range [32, 64).

const LDI_DIRS: u32 = 5u;      // 4 side captures + 1 top capture
const LDI_LAYERS: u32 = 4u;    // depth-peeled layers per capture
const LDI_STACKS: u32 = 3u;    // drawn per plant: 2 azimuth-nearest sides + top
const LDI_QUAD_VERTS: u32 = LDI_STACKS * LDI_LAYERS * 6u;  // 72

struct DirRec {
  right: vec4f,   // xyz = axis (plant-local, unit scale), w = ortho half width (m)
  up: vec4f,      // xyz = axis, w = ortho half height (m)
  fwd: vec4f,     // xyz = axis toward capture camera, w = sMax (m)
  mean_d: vec4f,  // per-layer mean capture depth (m); < 0 = layer empty
}

struct SpeciesU {
  // --- static: written once at create time ---------------------------------
  center: vec4f,   // xyz = mesh bbox center (unit scale), w = bounding radius (m)
  atlas: vec4f,    // tile px, atlas w px, atlas h px, layer count
  // --- dynamic: rewritten every frame (floats 8..15) ------------------------
  region: vec4f,   // origin cell x, origin cell z, side (cells), region radius (m)
  tune: vec4f,     // layer cull dist (m), parallax 0/1, coverage threshold, inspect mode
  // --- static ---------------------------------------------------------------
  ids: vec4f,      // seed, stand entry index, instance capacity, stand radius (m)
  dirs: array<DirRec, 5>,
}

fn rot_yaw(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}
