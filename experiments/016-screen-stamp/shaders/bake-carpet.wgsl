// CARPET BAKE — one straight-down orthographic capture of a periodic community
// tile, cropped to the tile's OWN square in the mesh frame (tile origin/size
// come from the GCMESH1 header). A mat is only ever seen from above-ish, so
// spending the whole texture budget on this one view is what buys texel
// density: 1024px across a 0.18m tile = 0.18mm/texel, vs 128px across a
// 0.35m sphere-fitted card (2.7mm) in the hemi-octa atlas.
//
// The mesh overflows its period (0.24m of geometry inside a 0.18m tile), so the
// draw is INSTANCED 9x at (-1,0,+1) x (-1,0,+1) periods: the neighbours' hangover
// fills the parts of the window this tile's own geometry left empty. Without it
// the window has holes along every edge and the mat shows a grid of gaps.
//
// MRT:
//   target0 = albedo.rgb + coverage
//   target1 = (nx, nz) of the mesh-frame normal flipped up + height + coverage
// ny is reconstructed as sqrt(1-nx^2-nz^2) at runtime: normals are flipped
// toward the capture camera (two-sided foliage), so ny >= 0 by construction,
// and rg/b stay plain linear quantities that a box filter may average (unlike
// octahedral, which is not mip-averageable).

struct Capture {
  // xz of the tile square's min corner (mesh frame), 1/tile size, periodic step
  tile_min: vec2f, inv_tile: f32, period: f32,
  // cushion base y and 1/(y extent) — height is stored normalised
  y_min: f32, inv_y_range: f32, _p0: f32, _p1: f32,
  bmin: vec3f, _p2: f32,
  bmax: vec3f, _p3: f32,
}
@group(0) @binding(0) var<uniform> cap: Capture;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) normal: vec3f,
  @location(2) h01: f32,
}

// Standard octahedral normal decode (y up) — mirrors GcMesh.normalAt().
fn oct_decode(e: vec2f) -> vec3f {
  let y = 1.0 - abs(e.x) - abs(e.y);
  var x = e.x;
  var z = e.y;
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * sign(e.x);
    z = (1.0 - abs(e.x)) * sign(e.y);
  }
  return normalize(vec3f(x, y, z));
}

@vertex
fn vs(@builtin(instance_index) ii: u32, @location(0) a0: vec4<u32>, @location(1) a1: vec4<u32>) -> VOut {
  let q = vec3f(f32(a0.x), f32(a0.y), f32(a0.z)) / 65535.0;
  var p = cap.bmin + q * (cap.bmax - cap.bmin);
  // Periodic neighbour copy (3x3 ring, instance 4 = this tile itself).
  let off = vec2f(f32(ii % 3u) - 1.0, f32(ii / 3u) - 1.0) * cap.period;
  p.x += off.x;
  p.z += off.y;

  let u = (p.x - cap.tile_min.x) * cap.inv_tile;
  let v = (p.z - cap.tile_min.y) * cap.inv_tile;
  let h01 = clamp((p.y - cap.y_min) * cap.inv_y_range, 0.0, 1.0);

  var o: VOut;
  // v = 0 at the +1 NDC row, so texture row 0 is tile-frame z_min: sampling
  // with v = (z - z0)/tile needs no flip at runtime.
  // Depth: nearer the (above) camera = higher y = smaller z, depthCompare 'less'.
  o.pos = vec4f(u * 2.0 - 1.0, 1.0 - v * 2.0, 1.0 - h01, 1.0);
  o.color = vec3f(f32(a0.w), f32(a1.x), f32(a1.y)) / 65535.0;
  let e = vec2f(f32(a1.z), f32(a1.w)) / 65535.0 * 2.0 - 1.0;
  o.normal = oct_decode(e);
  o.h01 = h01;
  return o;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) nh: vec4f,
}

@fragment
fn fs(i: VOut) -> FOut {
  var n = normalize(i.normal);
  // Two-sided foliage: flip toward the capture camera (straight down => +y).
  if (n.y < 0.0) { n = -n; }
  var o: FOut;
  o.albedo = vec4f(i.color, 1.0);
  o.nh = vec4f(n.x * 0.5 + 0.5, n.z * 0.5 + 0.5, i.h01, 1.0);
  return o;
}
