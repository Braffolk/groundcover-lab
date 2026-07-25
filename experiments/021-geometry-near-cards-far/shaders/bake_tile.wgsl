// Offline (in-browser, once) ortho TOP-DOWN bake of a periodic community tile
// (Sphagnum) into ONE square capture of exactly [0, tile_m]^2 in the mesh
// frame — the tile's own period, not the mesh bounds.
//
// The mesh overflows its period (0.24m of geometry inside a 0.18m tile), so the
// mesh is drawn 3x3 times, offset by the period: every neighbour's overhang
// that falls inside OUR square is captured, and by periodicity the values on
// opposite edges match exactly. Laid side by side the tiles then reproduce the
// full mat with no missing overhang and no seam.
//
// Outputs (both rgba8):
//   target 0: rgb = authored albedo, a = coverage
//   target 1: rgb = mesh-frame normal * 0.5 + 0.5 (flipped UP: a top view only
//             ever sees the upper face of a leaf), a = SURFACE HEIGHT,
//             normalized over the capture box [y0, y1].
// The height channel is what turns one flat quad into a relief: at draw time
// the near LOD stacks N ground-parallel shells and shell k keeps only the
// texels whose surface is at or above its own height, so the terraced stack
// reconstructs the cushion the flat card cannot express.

struct TileView {
  b_min: vec4f,   // mesh bounds min (dequantization)
  b_range: vec4f, // mesh bounds max - min
  info: vec4f,    // tile_m, y0, y1, unused
}
@group(0) @binding(0) var<uniform> tile_view: TileView;

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
  @location(0) q_pos: vec4<u32>,
  @location(1) q_attr: vec4<u32>,
  @builtin(instance_index) ii: u32,
) -> VOut {
  // Vertex record: [x y z r] [g b octU octV], all u16 UNORM against bounds.
  let p0 = tile_view.b_min.xyz + (vec3f(vec3<u32>(q_pos.xyz)) / 65535.0) * tile_view.b_range.xyz;
  let color = vec3f(f32(q_pos.w), f32(q_attr.x), f32(q_attr.y)) / 65535.0;
  let oct = (vec2f(f32(q_attr.z), f32(q_attr.w)) / 65535.0) * 2.0 - 1.0;

  let t = tile_view.info.x;
  let shift = vec2f(f32(ii % 3u), f32(ii / 3u)) - vec2f(1.0);
  let p = p0 + vec3f(shift.x * t, 0.0, shift.y * t);

  let y0 = tile_view.info.y;
  let y1 = tile_view.info.z;
  let hn = clamp((p.y - y0) / max(y1 - y0, 1e-6), 0.0, 1.0);

  var n = oct_decode_mesh(oct);
  if (n.y < 0.0) {
    n = -n; // two-sided foliage, seen from straight above
  }

  var out: VOut;
  // x -> u (column), z -> v (row): clip y is +1 at row 0, so v grows with z.
  out.pos = vec4f((p.x / t) * 2.0 - 1.0, 1.0 - (p.z / t) * 2.0, 1.0 - hn, 1.0);
  out.color = color;
  out.nrm = n;
  out.height = hn;
  return out;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) nrmh: vec4f,
}

@fragment
fn fs(in: VOut) -> FOut {
  var out: FOut;
  out.albedo = vec4f(in.color, 1.0);
  out.nrmh = vec4f(normalize(in.nrm) * 0.5 + 0.5, clamp(in.height, 0.0, 1.0));
  return out;
}
