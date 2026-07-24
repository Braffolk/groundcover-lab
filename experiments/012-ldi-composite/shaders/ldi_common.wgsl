// Shared between cull.wgsl (compute) and ldi.wgsl (render): the per-stand-entry
// uniform block, filled by main.ts every frame. Field packing must match the
// Float32Array layout in main.ts (writeSpeciesUniform).

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
  center: vec4f,   // xyz = mesh bbox center (unit scale), w = bounding radius (m)
  atlas: vec4f,    // tile px, atlas w px, atlas h px, layer count
  region: vec4f,   // origin cell x, origin cell z, side (cells), seed
  fade: vec4f,     // region radius (m), layer cull dist (m), parallax 0/1, entry index
  misc: vec4f,     // coverage threshold, instance capacity, stand radius (m), unused
  dirs: array<DirRec, 5>,
}

fn rot_yaw(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}
