#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./entry_info.wgsl"

// Depth-reprojected cards ("near detail injection").
//
// Geometry is exactly the billboard baseline: 6 verts for a cylindrical
// camera-facing side card + 6 verts for a horizontal top card. What is
// different is that the bake stored, next to the albedo, the SIGNED DEPTH of
// the first surface along each capture axis (plus a per-texel normal and a
// volumetric sky-visibility term). Each fragment therefore knows where the real
// 3D surface is, and can walk the TRUE eye ray onto it analytically:
//
//   the three view axes (u, v, f) are linear functionals of world position, so
//   a straight eye ray stays straight in (u,v,f) space. Tap the depth field at
//   the card point, slide along the ray to that depth plane (t = (zb - f)/df),
//   and re-read colour there. One step is the classic parallax offset; a second
//   step refines it with an exact second sample. No loop, no march, no
//   frag_depth (early-z stays alive), no dither.
//
// The result: the card parallaxes inside its own silhouette, the silhouette
// morphs continuously between the 15 baked azimuths instead of snapping, and
// rising above the horizon views makes the clump lean over like real geometry.
//
// Three fragment entry points share one vertex stage — the cull pass puts each
// plant in a distance bucket and the bucket picks the pipeline:
//   fs_near  2 relief steps, 3 taps, transmission   (< nearDist)
//   fs_mid   1 relief step,  2 taps                 (< midDist)
//   fs_far   no relief,      2 taps, no dependent fetch (rest of the region)

struct PlantInst {
  pos: vec3f,
  packed_bits: u32,
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read> insts: array<PlantInst>;
@group(1) @binding(2) var card_albedo: texture_2d<f32>;
@group(1) @binding(3) var card_geo: texture_2d<f32>;
@group(1) @binding(4) var card_sampler: sampler;

const GRID: f32 = 4.0;            // 4x4 tiles; tile 15 is the top view
const TOP_TILE: u32 = 15u;
const N_SIDE_VIEWS: f32 = 15.0;   // 24deg azimuth steps
const TILE_ALB: f32 = 384.0;      // albedo tile size in texels
const ALB_INSET: f32 = 0.0104167; // 4 texels of a 384 tile
const GEO_INSET: f32 = 0.0208333; // 4 texels of a 192 tile
const GEO_LOD_BIAS: f32 = -1.0;   // geometry tiles are half the albedo's size
const MAX_OFFSET: f32 = 0.28;     // uv units — a reprojection never crosses the card
const MAX_OFFSET_PX: f32 = 14.0;  // ...and never moves more than this on screen

fn rot_y(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn tile_uv(tile: u32, local: vec2f, inset: f32) -> vec2f {
  let col = f32(tile % 4u);
  let row = f32(tile / 4u);
  let c = clamp(local, vec2f(inset), vec2f(1.0 - inset));
  return (vec2f(col, row) + c) / GRID;
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  // (u, v, f) of this card point in the selected view's normalized axes.
  // All three are affine in world position, so ordinary perspective-correct
  // interpolation reproduces them exactly.
  @location(1) axc: vec3f,
  @location(2) @interpolate(flat) cam_axc: vec3f,
  @location(3) @interpolate(flat) yaw_cs: vec2f,
  @location(4) @interpolate(flat) erode: f32,
  @location(5) @interpolate(flat) relief: f32,
  @location(6) @interpolate(flat) tile: u32,
  // Mip level for the albedo atlas, from the CARD's projected size. Cards are
  // camera-facing, so their screen footprint is isotropic and one level per
  // plant is exactly right — which lets every tap be a plain
  // textureSampleLevel: no derivative hardware, no anisotropy, and no sparkle
  // when a reprojected uv jumps across a silhouette.
  @location(7) @interpolate(flat) lod: f32,
  // Screen-space error budget for the reprojection, in uv units. A single depth
  // layer cannot resolve what the capture never saw, so the residual
  // disocclusion error scales with the offset; capping the offset at a fixed
  // number of PIXELS keeps that error imperceptible at every distance and lets
  // the parallax run at full geometric strength as soon as the plant is small
  // enough for it to be exact-looking.
  @location(8) @interpolate(flat) uv_limit: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let inst = insts[ii];
  let bits = inst.packed_bits;
  let yaw = f32(bits & 1023u) * (6.2831853 / 1024.0);
  let scale = f32((bits >> 10u) & 4095u) * (4.0 / 4095.0);
  let phase = f32((bits >> 22u) & 1023u) * (6.2831853 / 1024.0);
  let base = inst.pos;
  let entry = stand_table[u32(info.entry_index)];

  let r = info.card_r * scale;
  let h0 = info.card_y0 * scale;
  let h1 = info.card_y1 * scale;
  let h_half = (h1 - h0) * 0.5;
  // The baked capture is centred on the clump, which sits off the mesh origin —
  // reproduce that offset so imagery lands where the geometry was.
  let cw = base + rot_y(vec3f(info.card_cx, 0.0, info.card_cz) * scale, yaw) + vec3f(0.0, (h0 + h1) * 0.5, 0.0);

  var corners = array<vec2f, 6>(
    vec2f(-1.0, 0.0), vec2f(1.0, 0.0), vec2f(-1.0, 1.0),
    vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = corners[vi % 6u];
  let is_top = vi >= 6u;

  let to_cam = frame.camera_pos - cw;
  let fwd_xz = to_cam.xz / max(length(to_cam.xz), 1e-4);
  let right = vec3f(fwd_xz.y, 0.0, -fwd_xz.x);
  let sway = wind_sway(base, frame.time, entry.sway, phase);

  // --- view selection + that view's axis frame, rotated into world ----------
  // Axis vectors carry 1/extent and 1/scale, so dot(world_offset, axis) is the
  // baked view's normalized coordinate directly.
  var tile = TOP_TILE;
  var au_w: vec3f;
  var av_w: vec3f;
  var af_w: vec3f;
  if (is_top) {
    au_w = rot_y(vec3f(1.0, 0.0, 0.0), yaw) / r;
    av_w = rot_y(vec3f(0.0, 0.0, 1.0), yaw) / r;
    af_w = vec3f(0.0, 1.0 / h_half, 0.0);
  } else {
    let local_dir = rot_y(vec3f(fwd_xz.x, 0.0, fwd_xz.y), -yaw);
    let ang = atan2(local_dir.x, local_dir.z);
    let k = u32((i32(round(ang * (N_SIDE_VIEWS / 6.2831853))) + i32(N_SIDE_VIEWS)) % i32(N_SIDE_VIEWS));
    tile = k;
    let a = f32(k) * (6.2831853 / N_SIDE_VIEWS);
    au_w = rot_y(vec3f(cos(a), 0.0, -sin(a)), yaw) / r;
    av_w = vec3f(0.0, -1.0 / h_half, 0.0);
    af_w = rot_y(vec3f(sin(a), 0.0, cos(a)), yaw) / r;
  }

  // --- card geometry (rest frame; wind is added afterwards) -----------------
  var rest: vec3f;
  var height_frac: f32;
  if (is_top) {
    let c2 = vec2f(c.x, c.y * 2.0 - 1.0); // top card spans [-1,1] on both axes
    rest = cw + rot_y(vec3f(c2.x, 0.0, c2.y) * r, yaw) + vec3f(0.0, (info.top_frac * 2.0 - 1.0) * h_half, 0.0);
    height_frac = info.top_frac;
  } else {
    rest = cw + right * (c.x * r) + vec3f(0.0, (c.y * 2.0 - 1.0) * h_half, 0.0);
    height_frac = c.y;
  }
  let world = rest + sway * height_frac;

  // --- fades (erosion of the hard alpha test, never dither) -----------------
  let dxz = frame.camera_pos.xz - cw.xz;
  let d_xz = length(dxz);
  let seg_y = clamp(frame.camera_pos.y, base.y + h0, base.y + h1);
  let d_core = length(vec3f(dxz.x, frame.camera_pos.y - seg_y, dxz.y));
  let near_fade = smoothstep(r * 0.35, r * 1.05, d_core);
  let edge_fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, d_xz);
  var fade = near_fade * edge_fade;
  if (is_top) {
    // The top card only makes sense from well above; at flatter views it reads
    // as a pale floating cutout AND its depth axis (+Y) is nearly perpendicular
    // to the eye ray, which is where the reprojection guard has to clamp. Erode
    // it away below ~30deg elevation.
    let dy = frame.camera_pos.y - (base.y + mix(h0, h1, info.top_frac));
    let elev = abs(dy) / max(length(vec3f(dxz.x, dy, dxz.y)), 1e-4);
    fade *= smoothstep(0.5, 0.72, elev);
  }

  var out: VOut;
  // A card whose fade dropped below the alpha reference has erode > 1, so every
  // one of its fragments discards — emit it behind the near plane instead of
  // rasterizing a guaranteed-empty quad (that set covers the most screen area:
  // top cards at eye level, cards the camera stands inside, the region rim).
  if (fade < info.alpha_ref) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.world = world;
  let off = rest - cw;
  out.axc = vec3f(dot(off, au_w), dot(off, av_w), dot(off, af_w));
  let cam_off = frame.camera_pos - cw;
  out.cam_axc = vec3f(dot(cam_off, au_w), dot(cam_off, av_w), dot(cam_off, af_w));
  out.yaw_cs = vec2f(cos(yaw), sin(yaw));
  out.erode = info.alpha_ref / max(fade, 1e-3);
  // Same 3D distance the cull bucketed on, so the ramp reaches 0 exactly where
  // the far (no-parallax) pipeline takes over.
  out.relief = info.relief_scale * (1.0 - smoothstep(info.relief_fade0, info.relief_fade1, length(to_cam)));
  out.tile = tile;
  // Texels per metre of the coarser card axis vs pixels per metre at the card
  // centre's view depth. Computed at the centre (never per corner) so both
  // triangles of the quad agree — a flat varying takes one provoking vertex.
  let depth_v = max((frame.view_proj * vec4f(cw, 1.0)).w, 0.05);
  let px_per_m = frame.proj[1][1] * frame.viewport.y * 0.5 / depth_v;
  let card_m = select(min(2.0 * r, h1 - h0), 2.0 * r, is_top);
  out.lod = log2(max(TILE_ALB / card_m / px_per_m, 1e-6));
  out.uv_limit = clamp(MAX_OFFSET_PX / max(px_per_m * card_m, 1e-3), 0.01, MAX_OFFSET);
  return out;
}

// Octahedral decode, y-primary — same convention the bake stored.
fn oct_decode_card(e: vec2f) -> vec3f {
  var x = e.x;
  var z = e.y;
  let y = 1.0 - abs(e.x) - abs(e.y);
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0);
    z = (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0);
  }
  return normalize(vec3f(x, y, z));
}

// Every tap uses the per-plant mip level from the vertex stage. Implicit
// derivatives would be wrong anyway for the reprojected taps: a dependent uv is
// discontinuous wherever the ray crosses a silhouette, and the derivative spike
// picks a far too coarse mip (sparkle). It is also the cheapest sampling path —
// no derivative hardware, no anisotropy.
fn geo_tap(in: VOut, local: vec2f) -> vec4f {
  return textureSampleLevel(card_geo, card_sampler, tile_uv(in.tile, local, GEO_INSET), in.lod + GEO_LOD_BIAS);
}

fn alb_tap(in: VOut, local: vec2f) -> vec4f {
  return textureSampleLevel(card_albedo, card_sampler, tile_uv(in.tile, local, ALB_INSET), in.lod);
}

struct Relief {
  denom: f32,
  t: f32,
  uv: vec2f,
}

/// One analytic step of the eye ray onto the depth field.
/// `depth_n == 0` is the bake's "this view saw no surface here" sentinel: past
/// the dilated ring around a silhouette there is genuinely nothing to walk onto,
/// and stepping anyway drags unrelated foliage into what should stay a gap.
fn relief_step(in: VOut, uv0: vec2f, depth_n: f32, t_in: f32, denom_in: f32, first: bool) -> Relief {
  let dir = in.cam_axc - in.axc; // toward the camera, in view-axis coords
  // The selected capture is within 12deg of the eye ray in azimuth, so dir.z
  // dominates; the guard only matters for near-grazing top cards, whose depth
  // axis is +Y and therefore nearly perpendicular to the ray.
  let denom = select(denom_in, max(dir.z, 0.30 * length(dir.xy) + 1e-4), first);
  let ray_z = in.axc.z + select(t_in * denom, 0.0, first);
  let t_prev = select(t_in, 0.0, first);
  let t = select(t_prev, t_prev + (depth_n * 2.0 - 1.0 - ray_z) / denom, depth_n > 0.002);
  var out: Relief;
  out.denom = denom;
  out.t = t;
  out.uv = uv0 + clamp(t * dir.xy * (0.5 * in.relief), vec2f(-in.uv_limit), vec2f(in.uv_limit));
  return out;
}

/// A reprojection that walks off its tile has no imagery to read — the clamp in
/// tile_uv would smear the border texel into a stripe, so drop the fragment.
fn outside_tile(uv: vec2f) -> bool {
  return any(uv < vec2f(0.0)) || any(uv > vec2f(1.0));
}

fn shade_common(in: VOut, alb: vec4f, g: vec4f) -> vec4f {
  let n_mesh = oct_decode_card(g.rg * 2.0 - 1.0);
  let n = vec3f(
    in.yaw_cs.x * n_mesh.x + in.yaw_cs.y * n_mesh.z,
    n_mesh.y,
    -in.yaw_cs.y * n_mesh.x + in.yaw_cs.x * n_mesh.z,
  );
  // Baked sky visibility is fake occlusion, so it belongs to the LIGHT term:
  // the albedo view then shows the baked atlas colour exactly as captured and
  // the lighting view shows sun+ambient x self-occlusion.
  // sqrt lifts the deep shadows (a canopy interior is dark, not black) while
  // keeping the contrast that makes the clump read as a volume; one instruction.
  let sky = clamp(1.0 - info.ao_strength * (1.0 - sqrt(g.a)), 0.0, 1.0);
  var color = light_surface(alb.rgb * sky, n, in.world);
  // Sun transmitted through the blade: lit from behind, strongest looking toward
  // the sun, and only where the canopy is thin (high sky visibility). Fades out
  // with the parallax, so the far tier (relief = 0) skips the branch entirely.
  let trans = info.translucency * in.relief;
  if (trans > 0.0) {
    let vdir = normalize(frame.camera_pos - in.world);
    let back = max(0.0, -dot(n, frame.sun_dir));
    let fwd = max(0.0, dot(vdir, -frame.sun_dir));
    color += alb.rgb * frame.sun_color * (trans * g.a * back * (0.35 + 0.65 * fwd * fwd));
  }
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, alb.rgb, n, alb.a, in.world), 1.0);
}

/// Two relief steps + a re-read of normals/occlusion at the refined hit.
@fragment
fn fs_near(in: VOut) -> @location(0) vec4f {
  let uv0 = vec2f(0.5, 0.5) + 0.5 * in.axc.xy;
  let g0 = geo_tap(in, uv0);
  let s1 = relief_step(in, uv0, g0.b, 0.0, 0.0, true);
  let g1 = geo_tap(in, s1.uv);
  let s2 = relief_step(in, uv0, g1.b, s1.t, s1.denom, false);
  let alb = alb_tap(in, s2.uv);
  if (alb.a < in.erode || outside_tile(s2.uv)) {
    discard;
  }
  return shade_common(in, alb, g1);
}

/// One relief step: colour and silhouette parallax, shading normal from the card
/// point (a fraction of a card off, invisible at these distances).
@fragment
fn fs_mid(in: VOut) -> @location(0) vec4f {
  let uv0 = vec2f(0.5, 0.5) + 0.5 * in.axc.xy;
  let g0 = geo_tap(in, uv0);
  let s1 = relief_step(in, uv0, g0.b, 0.0, 0.0, true);
  let alb = alb_tap(in, s1.uv);
  if (alb.a < in.erode || outside_tile(s1.uv)) {
    discard;
  }
  return shade_common(in, alb, g0);
}

/// Plain billboard: two independent taps, no dependent fetch, no parallax ALU.
@fragment
fn fs_far(in: VOut) -> @location(0) vec4f {
  let uv0 = vec2f(0.5, 0.5) + 0.5 * in.axc.xy;
  let alb = alb_tap(in, uv0);
  if (alb.a < in.erode) {
    discard;
  }
  return shade_common(in, alb, geo_tap(in, uv0));
}
