// Stage A of the bake: the STAMP LIBRARY.
//
// One source mesh is rasterized with a SHEARED orthographic projection: the
// image is parameterised by where the view ray crosses the GROUND plane
//     uv = p.xz + phi * s * p.y          (s = cot(elevation), phi = (1,0))
// and the depth test keeps the HIGHEST surface on each ray (the first thing an
// eye coming down that ray meets). That single affine map is the whole trick —
// no marching anywhere, the rasterizer resolves the ray bundle.
//
// The plant is pre-rotated by a quantised yaw, so stage B can place any plant
// of that yaw at any azimuth by rotating the stamp rect (see composite.wgsl).

#include "src/wgsl/gcmesh.wgsl"

struct StampCfg {
  // Sheared-ortho rect in the plant's local frame (metres, origin = plant).
  rect: vec4f,          // u0, u1, v0, v1
  shear_yaw: vec4f,     // shear s, yaw (rad), cos yaw, sin yaw
  height: vec4f,        // topH, y0, 1/topH, unused
  qmin: vec4f,          // mesh bounds min xyz, unused
  qsize: vec4f,         // mesh bounds size xyz, unused
  centre: vec4f,        // mesh centre x, z, unused, unused
}
@group(0) @binding(0) var<uniform> cfg: StampCfg;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) albedo: vec3f,
  @location(1) normal: vec3f,
  @location(2) h01: f32,
}

@vertex
fn vs(@location(0) a: vec4u, @location(1) b: vec4u) -> VOut {
  let q = vec3f(f32(a.x), f32(a.y), f32(a.z)) / 65535.0;
  let local = cfg.qmin.xyz + q * cfg.qsize.xyz;
  // Centre horizontally on the plant origin, then apply the quantised yaw.
  let cx = local.x - cfg.centre.x;
  let cz = local.z - cfg.centre.y;
  let cy = cfg.shear_yaw.z;
  let sy = cfg.shear_yaw.w;
  let rx = cx * cy - cz * sy;
  let rz = cx * sy + cz * cy;
  let py = local.y;

  // Sheared-ortho: u along the ray's horizontal travel, v across.
  let u = rx + cfg.shear_yaw.x * py;
  let v = rz;
  let r = cfg.rect;
  let ndc_x = (u - r.x) / (r.y - r.x) * 2.0 - 1.0;
  let ndc_y = -((v - r.z) / (r.w - r.z) * 2.0 - 1.0);
  // Depth = "how far down the ray": the highest surface wins (depthCompare less).
  let depth = clamp((cfg.height.x - py) / max(cfg.height.x - cfg.height.y, 1e-4), 0.0, 1.0);

  var out: VOut;
  out.pos = vec4f(ndc_x, ndc_y, depth, 1.0);
  out.albedo = vec3f(f32(a.w), f32(b.x), f32(b.y)) / 65535.0;
  let n = gcmesh_normal_decode_u16(b.z, b.w);
  out.normal = vec3f(n.x * cy - n.z * sy, n.y, n.x * sy + n.z * cy);
  out.h01 = clamp(py * cfg.height.z, 0.0, 1.0);
  return out;
}

struct FOut {
  @location(0) surf: vec4f,
  @location(1) geom: vec4f,
}

@fragment
fn fs(in: VOut) -> FOut {
  // Two-sided foliage: flip the normal toward the eye of THIS bake direction.
  // Eye direction (toward viewer) = -d = (-s, 1, 0) / |..|.
  let s = cfg.shear_yaw.x;
  let eye = normalize(vec3f(-s, 1.0, 0.0));
  var n = normalize(in.normal);
  if (dot(n, eye) < 0.0) { n = -n; }
  var out: FOut;
  out.surf = vec4f(in.albedo, 1.0);
  out.geom = vec4f(n.x * 0.5 + 0.5, n.z * 0.5 + 0.5, in.h01, 1.0);
  return out;
}
