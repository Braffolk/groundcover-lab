// Standalone shader for the raw GCMESH1 inspector (no scene includes — the
// inspector is tooling, not an experiment). The one include is the shared
// source-normal decode: this view is where "the convention was validated
// visually" was claimed, so it above all must not carry a private copy.
#include "src/wgsl/gcmesh.wgsl"

struct U {
  view_proj: mat4x4f,
  bounds_min: vec3f,
  mode: f32,        // 0 = vertex color, 1 = normals, 2 = lit
  bounds_max: vec3f,
  repeat: f32,      // 0 = single, 1 = 3x3 periodic tiles
  tile: vec2f,
  _pad: vec2f,
}
@group(0) @binding(0) var<uniform> u: U;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) normal: vec3f,
}

@vertex
fn vs_main(
  @location(0) a0: vec4<u32>, // x, y, z, r  (quantized u16)
  @location(1) a1: vec4<u32>, // g, b, octU, octV
  @builtin(instance_index) ii: u32,
) -> VOut {
  let q = vec3f(f32(a0.x), f32(a0.y), f32(a0.z)) / 65535.0;
  var pos = u.bounds_min + q * (u.bounds_max - u.bounds_min);
  if (u.repeat > 0.5) {
    let gx = f32(i32(ii % 3u) - 1);
    let gz = f32(i32(ii / 3u) - 1);
    pos += vec3f(gx * u.tile.x, 0.0, gz * u.tile.y);
  }
  var out: VOut;
  out.pos = u.view_proj * vec4f(pos, 1.0);
  out.color = vec3f(f32(a0.w), f32(a1.x), f32(a1.y)) / 65535.0;
  out.normal = gcmesh_normal_decode_u16(a1.z, a1.w);
  return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let n = normalize(in.normal);
  if (u.mode < 0.5) {
    return vec4f(in.color, 1.0);
  }
  if (u.mode < 1.5) {
    return vec4f(n * 0.5 + 0.5, 1.0);
  }
  let sun = normalize(vec3f(0.35, 0.75, 0.3));
  let ndl = dot(n, sun) * 0.5 + 0.5;
  return vec4f(in.color * (ndl * ndl * 1.15 + 0.25), 1.0);
}
