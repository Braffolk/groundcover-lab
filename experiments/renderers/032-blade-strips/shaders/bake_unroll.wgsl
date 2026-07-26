#include "src/wgsl/gcmesh.wgsl"

// Offline (in-browser, once) UNROLLED bundle capture.
//
// One atlas tile per baked ribbon. The ribbon is a 5-node polyline in the
// clump frame with a per-node half-width and a fixed horizontal lateral axis
// L. Every triangle of the bundle this ribbon replaces is rendered with a
// curvilinear vertex transform:
//
//   tile.u = (lateral offset along L) / half_width(s)      -> clip x
//   tile.v = s = (segment index + t) / 4                   -> clip y
//   depth  = residual along the ribbon face normal N       -> clip z
//
// i.e. the real geometry is flattened onto the ribbon surface and STRAIGHTENED
// along the ribbon's own arc. Re-rolling the tile onto the same curve at draw
// time reproduces the bundle's real blade silhouettes, colours and normals on
// a curved 3D strip. Normals are stored in the ribbon's tangent frame
// (T2, L, N) so wind bending and the runtime twist stay correct.
//
// Outputs: target 0 rgb = authored albedo, a = coverage.
//          target 1 rg  = tangent-frame normal (T2, L) * 0.5 + 0.5, N >= 0
//                         is implied and rebuilt at draw time.
// No @group(0) frame include — the bake is self-contained.

const NODES: u32 = 5u;
const SEGS: f32 = 4.0;

struct Tile {
  nodes: array<vec4f, 5>, // xyz = node position (clump frame), w = half-width
  lat: vec4f,             // xy = lateral axis (unit, horizontal)
  cfg: vec4f,             // x = depth range (m), y = is_top, z = y0, w = y1
  b_min: vec4f,           // mesh bounds min (dequantization)
  b_range: vec4f,         // mesh bounds max - min
  capture: vec4f,         // x = cx, y = cz, z = rXZ (top view extent)
}
@group(0) @binding(0) var<uniform> tile: Tile;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) nrm: vec3f, // already in the ribbon tangent frame
}

@vertex
fn vs(@location(0) q_pos: vec4<u32>, @location(1) q_attr: vec4<u32>) -> VOut {
  // Vertex record: [x y z r] [g b octU octV], all u16 UNORM against bounds.
  let p_mesh = tile.b_min.xyz + (vec3f(vec3<u32>(q_pos.xyz)) / 65535.0) * tile.b_range.xyz;
  let color = vec3f(f32(q_pos.w), f32(q_attr.x), f32(q_attr.y)) / 65535.0;
  let n_mesh = gcmesh_normal_decode_u16(q_attr.z, q_attr.w);
  // Clump frame: horizontally centred on the capture centre, y unchanged.
  let p = p_mesh - vec3f(tile.capture.x, 0.0, tile.capture.y);

  var out: VOut;
  if (tile.cfg.y > 0.5) {
    // --- straight-down top view (one tile, used by the canopy card) --------
    let r = tile.capture.z;
    let y0 = tile.cfg.z;
    let y1 = tile.cfg.w;
    let u = p.x / r;
    let v = -p.z / r;
    let d = (y1 - p.y) / max(y1 - y0, 1e-4);
    // Tangent frame of the top card: T2 = +X, L = -Z, N = +Y.
    var n = n_mesh;
    if (n.y < 0.0) {
      n = -n;
    }
    out.pos = vec4f(u, v, clamp(d, 0.0, 1.0), 1.0);
    out.color = color;
    out.nrm = vec3f(n.x, -n.z, n.y);
    return out;
  }

  let lat = vec3f(tile.lat.x, 0.0, tile.lat.y);

  // --- closest ribbon segment, measured perpendicular to the lateral axis --
  var best_d2 = 1.0e30;
  var best_i = 0u;
  var best_t = 0.0;
  var best_q = tile.nodes[0].xyz;
  for (var i = 0u; i < NODES - 1u; i++) {
    let a = tile.nodes[i].xyz;
    let b = tile.nodes[i + 1u].xyz;
    let d = b - a;
    let t = clamp(dot(p - a, d) / max(dot(d, d), 1e-8), 0.0, 1.0);
    let q = a + d * t;
    let r = p - q;
    // Lateral offset is legitimate, so it must not influence the choice.
    let d2 = dot(r, r) - pow(dot(r, lat), 2.0);
    if (d2 < best_d2) {
      best_d2 = d2;
      best_i = i;
      best_t = t;
      best_q = q;
    }
  }

  let a = tile.nodes[best_i];
  let b = tile.nodes[best_i + 1u];
  let seg = b.xyz - a.xyz;
  let half_w = max(mix(a.w, b.w, best_t), 1e-4);
  let r = p - best_q;

  let tangent = normalize(seg + vec3f(0.0, 1e-6, 0.0));
  let face_n = normalize(cross(lat, tangent));
  let t2 = cross(face_n, lat);

  let u = dot(r, lat) / half_w;
  let s = (f32(best_i) + best_t) / SEGS;
  let depth = 0.5 - 0.5 * clamp(dot(r, face_n) / max(tile.cfg.x, 1e-4), -1.0, 1.0);

  // Flip toward the capture side so both faces of a blade light like a front.
  var n = n_mesh;
  if (dot(n, face_n) < 0.0) {
    n = -n;
  }

  out.pos = vec4f(u, s * 2.0 - 1.0, clamp(depth, 0.0, 1.0), 1.0);
  out.color = color;
  out.nrm = vec3f(dot(n, t2), dot(n, lat), dot(n, face_n));
  return out;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) nrm: vec4f,
}

@fragment
fn fs(in: VOut) -> FOut {
  let n = normalize(in.nrm);
  var out: FOut;
  out.albedo = vec4f(in.color, 1.0);
  // rg = tangent/lateral components; the face component is implied (>= 0).
  out.nrm = vec4f(n.x * 0.5 + 0.5, n.y * 0.5 + 0.5, 0.0, 1.0);
  return out;
}
