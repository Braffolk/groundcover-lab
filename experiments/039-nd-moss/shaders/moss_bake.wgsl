// Offline (in-browser, once) bake of the Uncharted-4 moss material set for ONE
// periodic Sphagnum tile, straight down over exactly the tile square
// [0, tileM]^2 of the mesh frame (tile origin is (0,0) for every source mesh).
//
// The mesh is drawn NINE times, offset by -tileM/0/+tileM in x and z: the source
// geometry overflows its own period (0.24m of moss inside a 0.18m tile), so
// without the eight wrapped copies the crop would be missing every neighbour's
// overhang and the baked tile would not be periodic — that is exactly the
// "1-texel dark line along a tile boundary" the card baseline reports.
//
// Outputs (rgba8, later downsampled coverage-weighted):
//   target 0: rgb = authored vertex colour, a = coverage
//   target 1: rgb = mesh-frame normal * 0.5 + 0.5, flipped into the upper
//             hemisphere (two-sided leaves), a = HEIGHT of the topmost surface,
//             normalised over the capture range [y0, y1]. The depth test picks
//             the topmost surface per texel, so target 1's alpha IS the
//             heightmap the paper's parallax/AO stack wants.
// No @group(0) frame include — the bake is self-contained.

struct BakeInfo {
  b_min: vec4f,   // mesh bounds min (dequantisation)
  b_range: vec4f, // bounds max - min
  tile: vec4f,    // x: tile period (m), y: y0, z: y1
}
@group(0) @binding(0) var<uniform> bake: BakeInfo;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) nrm: vec3f,
  @location(2) height: f32,
}

// Octahedral decode, y-primary — mirrors GcMesh.normalAt() in src/mesh/gcmesh.ts.
fn oct_decode_mesh(e: vec2f) -> vec3f {
  var x = e.x;
  var z = e.y;
  let y = 1.0 - abs(e.x) - abs(e.y);
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0);
    z = (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0);
  }
  return normalize(vec3f(x, y, z));
}

@vertex
fn vs(
  @builtin(instance_index) ii: u32,
  @location(0) q_pos: vec4<u32>,
  @location(1) q_attr: vec4<u32>,
) -> VOut {
  // Vertex record: [x y z r] [g b octU octV], all u16 UNORM against bounds.
  let p = bake.b_min.xyz + (vec3f(vec3<u32>(q_pos.xyz)) / 65535.0) * bake.b_range.xyz;
  let color = vec3f(f32(q_pos.w), f32(q_attr.x), f32(q_attr.y)) / 65535.0;
  let oct = (vec2f(f32(q_attr.z), f32(q_attr.w)) / 65535.0) * 2.0 - 1.0;

  let tile_m = bake.tile.x;
  let y0 = bake.tile.y;
  let y1 = bake.tile.z;

  // Periodic copy: 3x3 lattice centred on the tile being captured.
  let off = vec2f(f32(i32(ii % 3u) - 1), f32(i32(ii / 3u) - 1)) * tile_m;
  let u = (p.x + off.x) / tile_m;
  let v = (p.z + off.y) / tile_m;
  // Row 0 of the texture is v = 0, i.e. mesh-frame z = 0.
  let d = (y1 - p.y) / (y1 - y0);

  var n = oct_decode_mesh(oct);
  if (n.y < 0.0) {
    n = -n;
  }

  var out: VOut;
  out.pos = vec4f(u * 2.0 - 1.0, 1.0 - v * 2.0, clamp(d, 0.0, 1.0), 1.0);
  out.color = color;
  out.nrm = n;
  out.height = clamp((p.y - y0) / (y1 - y0), 0.0, 1.0);
  return out;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) nrm_height: vec4f,
}

@fragment
fn fs(in: VOut) -> FOut {
  var out: FOut;
  out.albedo = vec4f(in.color, 1.0);
  out.nrm_height = vec4f(normalize(in.nrm) * 0.5 + 0.5, in.height);
  return out;
}
