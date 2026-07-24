// Orthographic ray-answer baker. One pass per view direction: rasterizes the
// raw GCMESH1 source mesh with an ortho projection along dv — every texel of
// the supersampled target IS the answer of one precomputed raycast (nearest
// hit along dv through that slab offset). Rasterization here is exactly the
// "precomputed raycast" loophole: the depth test resolves each ray bundle to
// its first hit, offline, once.

struct BakeView {
  bounds_min: vec3f,
  radius: f32,
  bounds_max: vec3f,
  _p0: f32,
  r_ax: vec3f,
  _p1: f32,
  u_ax: vec3f,
  _p2: f32,
  dv: vec3f,
  _p3: f32,
  center: vec3f,
  _p4: f32,
  anchor: vec2f,
  _p5: vec2f,
}
@group(0) @binding(0) var<uniform> bv: BakeView;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) oct: vec2f,
  @location(2) height01: f32,
}

@vertex
fn vs_bake(@location(0) a0: vec4<u32>, @location(1) a1: vec4<u32>) -> VOut {
  let q = vec3f(f32(a0.x), f32(a0.y), f32(a0.z)) / 65535.0;
  let local = bv.bounds_min + q * (bv.bounds_max - bv.bounds_min);
  // Anchored space: xz origin at the plant/tile anchor (instance pivot).
  let a = local - vec3f(bv.anchor.x, 0.0, bv.anchor.y);
  let rel = a - bv.center;
  let z01 = (dot(rel, bv.dv) + bv.radius) / (2.0 * bv.radius);
  var out: VOut;
  // Clip y is negated so texture v grows with +u_ax slab coordinate — the
  // runtime sampler then needs no flip.
  out.pos = vec4f(dot(rel, bv.r_ax) / bv.radius, -dot(rel, bv.u_ax) / bv.radius, z01, 1.0);
  out.color = vec3f(f32(a0.w), f32(a1.x), f32(a1.y)) / 65535.0;
  out.oct = vec2f(f32(a1.z), f32(a1.w)) / 65535.0;
  out.height01 = clamp(
    (local.y - bv.bounds_min.y) / max(bv.bounds_max.y - bv.bounds_min.y, 1e-5),
    0.0,
    1.0,
  );
  return out;
}

struct FOut {
  @location(0) surf: vec4f, // albedo rgb, hit flag
  @location(1) geom: vec4f, // depth01 along dv, oct normal, height01
}

@fragment
fn fs_bake(in: VOut) -> FOut {
  var out: FOut;
  out.surf = vec4f(in.color, 1.0);
  out.geom = vec4f(in.pos.z, in.oct, in.height01);
  return out;
}
