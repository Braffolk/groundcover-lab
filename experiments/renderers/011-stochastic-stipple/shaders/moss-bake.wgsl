// One-time bake for a PERIODIC CARPET tile (Sphagnum): a single orthographic
// top-down capture of exactly the tile's own square in the mesh frame, storing
// albedo + coverage + a HEIGHT FIELD of the cushion top surface.
//
// Two things make this different from the 9-view card bake:
//
//  - The frame is the tile's period, not the mesh bounds. All 512 texels land
//    on 0.18m, so the mat gets ~2800 px/m instead of the ~750 px/m a card's
//    diagonal-sized frame would leave, and none of the budget is spent on
//    azimuth views a 7cm-tall mat never shows.
//  - The mesh is drawn 3x3 times at +-one period. The tile's geometry
//    overflows its own period (0.24m of cushion inside a 0.18m tile), and the
//    mat is periodic, so the overhang of the neighbouring tiles is exactly
//    what must wrap in. Cropping without the wrap would shave every capitulum
//    that leans over a seam.
//
// MRT (both premultiplied by coverage, so the runtime mip chain averages
// statistics and not background):
//   0: rgb = albedo * cov, a = cov
//   1: rgb = (n*0.5+0.5) * cov, a = height_norm * cov
// The height channel is what turns one flat quad into a shell stack at
// runtime: shell k keeps the texels whose cushion top reaches its band.

struct MossView {
  bmin: vec3f,
  tile: f32,      // periodic tile size (m), mesh frame
  bmax: vec3f,
  h_min: f32,     // height range mapped into the height channel
  origin: vec2f,  // tile origin (mesh frame); (0,0) for every current mesh
  h_range: f32,
  _pad: f32,
}
@group(0) @binding(0) var<uniform> moss_uni: MossView;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) nrm: vec3f,
  @location(2) hn: f32,
}

fn oct_decode(e: vec2f) -> vec3f {
  let u = e.x * 2.0 - 1.0;
  let v = e.y * 2.0 - 1.0;
  var x = u;
  var z = v;
  let y = 1.0 - abs(u) - abs(v);
  if (y < 0.0) {
    x = (1.0 - abs(v)) * sign(u);
    z = (1.0 - abs(u)) * sign(v);
  }
  return normalize(vec3f(x, y, z));
}

@vertex
fn vs(@location(0) q0: vec4<u32>, @location(1) q1: vec4<u32>, @builtin(instance_index) ii: u32) -> VOut {
  let qp = vec3f(vec3<u32>(q0.xyz)) / 65535.0;
  let pos = moss_uni.bmin + qp * (moss_uni.bmax - moss_uni.bmin);
  let color = vec3f(f32(q0.w), f32(q1.x), f32(q1.y)) / 65535.0;
  let n = oct_decode(vec2f(f32(q1.z), f32(q1.w)) / 65535.0);

  // 3x3 periodic wrap copies.
  let wrap = (vec2f(f32(ii % 3u), f32(ii / 3u)) - 1.0) * moss_uni.tile;
  let t = (pos.xz + wrap - moss_uni.origin) / moss_uni.tile; // [0,1] over the tile
  // v runs with +Z (row 0 = z_min), so uv = (x, z) / tile.
  let hn = (pos.y - moss_uni.h_min) / moss_uni.h_range;

  var out: VOut;
  out.pos = vec4f(t.x * 2.0 - 1.0, 1.0 - t.y * 2.0, 1.0 - clamp(hn, 0.0, 1.0), 1.0);
  out.color = color;
  // Two-sided foliage: a leaf underside seen from above lights as a top face.
  out.nrm = select(n, -n, n.y < 0.0);
  out.hn = hn;
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
  out.nrmh = vec4f(normalize(in.nrm) * 0.5 + 0.5, clamp(in.hn, 0.0, 1.0));
  return out;
}
