// Offline (in-browser, once) CARPET tile capture: one orthographic straight-down
// render of exactly the species' periodic tile square, in the mesh frame.
//
// The mesh is drawn NINE times, offset by (-1,0,+1) x (-1,0,+1) tile steps, and
// the rasterizer clips everything outside the tile square. That is what makes
// the captured texture genuinely periodic: the geometry that overflows the tile
// (0.24m of Sphagnum inside a 0.18m period) is replaced by the same overflow
// arriving from the neighbouring copy, exactly as the infinite mat would look.
// Cropping a single copy — the obvious thing to do — instead cuts every
// overhanging capitulum in half at the tile border and leaves the neighbour's
// contribution missing, which reads as a faint grid.
//
// Outputs:
//   target 0: rgb = authored albedo, a = 1 (coverage comes from the resolve)
//   target 1: rgb = mesh-frame normal * 0.5 + 0.5, flipped into the upper
//             hemisphere; a = APEX HEIGHT of this fragment as a fraction of
//             [y0, y1]. Depth test 'less' with 1 - height as depth keeps the
//             topmost surface, so the alpha channel of the winner is the
//             cushion's height field — the relief the shells are built from.

#include "src/wgsl/gcmesh.wgsl"

struct CarpetView {
  tile: vec4f,   // origin_x, origin_z, tile_m, 0
  yrange: vec4f, // y0, y1, 0, 0
  b_min: vec4f,  // mesh bounds min (dequantization)
  b_range: vec4f,// mesh bounds max - min
  offset: vec4f, // wrap offset (metres) for this copy
}
@group(0) @binding(0) var<uniform> view: CarpetView;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) nrm: vec3f,
  @location(2) height: f32,
}

@vertex
fn vs(@location(0) q_pos: vec4<u32>, @location(1) q_attr: vec4<u32>) -> VOut {
  // Vertex record: [x y z r] [g b octU octV], all u16 UNORM against bounds.
  let p = view.b_min.xyz + (vec3f(vec3<u32>(q_pos.xyz)) / 65535.0) * view.b_range.xyz;
  let color = vec3f(f32(q_pos.w), f32(q_attr.x), f32(q_attr.y)) / 65535.0;

  let w = p + view.offset.xyz;
  let tile_m = view.tile.z;
  let u = ((w.x - view.tile.x) / tile_m) * 2.0 - 1.0;
  // Framebuffer row 0 is clip y = +1, and the runtime maps v to +Z, so flip.
  let v = 1.0 - ((w.z - view.tile.y) / tile_m) * 2.0;
  let y0 = view.yrange.x;
  let y1 = view.yrange.y;
  let hf = clamp((p.y - y0) / max(y1 - y0, 1e-5), 0.0, 1.0);

  var n = gcmesh_normal_decode_u16(q_attr.z, q_attr.w);
  if (n.y < 0.0) {
    n = -n; // capture is straight down: every visible face lights as a top face
  }

  var out: VOut;
  out.pos = vec4f(u, v, 1.0 - hf, 1.0); // depth 'less' keeps the highest surface
  out.color = color;
  out.nrm = n;
  out.height = hf;
  return out;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) aux: vec4f,
}

@fragment
fn fs(in: VOut) -> FOut {
  var out: FOut;
  out.albedo = vec4f(in.color, 1.0);
  out.aux = vec4f(normalize(in.nrm) * 0.5 + 0.5, in.height);
  return out;
}
