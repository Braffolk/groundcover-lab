// Bake stage 1: orthographic G-buffer captures of the raw GCMESH1 mesh from
// one full-sphere octahedral direction per tile (viewport-selected).
// `fwd` is the direction RAYS TRAVEL (camera looks along +fwd), so depth
// 'less' keeps the first surface a ray entering the proxy would hit.
// MRT: target0 = albedo rgb + hit mask, target1 = oct normal (rg8);
// hit distance comes from the depth attachment itself (read in the gather).

struct ViewTile {
  right: vec3f, rb: f32,
  up: vec3f, _p0: f32,
  fwd: vec3f, _p1: f32,
  cb: vec3f, _p2: f32,          // frustum bounding-sphere center (proxy frame)
  bmin: vec3f, _p3: f32,
  bmax: vec3f, _p4: f32,
  axis: vec3f, _p5: f32,        // mesh->proxy translation (axisX, axisY, axisZ)
}
@group(0) @binding(0) var<uniform> vt: ViewTile;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) oct: vec2f,
}

@vertex
fn vs(@location(0) a0: vec4<u32>, @location(1) a1: vec4<u32>) -> VOut {
  let q = vec3f(f32(a0.x), f32(a0.y), f32(a0.z)) / 65535.0;
  let p = vt.bmin + q * (vt.bmax - vt.bmin) - vt.axis; // proxy frame
  let rel = p - vt.cb;
  let cx = dot(rel, vt.right) / vt.rb;
  let cy = dot(rel, vt.up) / vt.rb;
  let depth01 = dot(rel, vt.fwd) / (2.0 * vt.rb) + 0.5;

  var o: VOut;
  o.pos = vec4f(cx, cy, depth01, 1.0);
  o.color = vec3f(f32(a0.w), f32(a1.x), f32(a1.y)) / 65535.0;
  o.oct = vec2f(f32(a1.z), f32(a1.w)) / 65535.0; // keep encoded, avg later
  return o;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) oct: vec2f,
}

@fragment
fn fs(i: VOut) -> FOut {
  var o: FOut;
  o.albedo = vec4f(i.color, 1.0);
  o.oct = i.oct;
  return o;
}
