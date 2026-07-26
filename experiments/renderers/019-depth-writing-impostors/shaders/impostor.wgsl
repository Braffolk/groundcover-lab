#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./entry_info.wgsl"

// TRUE-DEPTH IMPOSTORS
// ====================
// One quad per plant, but every texel of that quad is a real 3D point.
//
// The bake stores, per baked view k, an orthographic image with its own
// orthonormal basis (R_k, U_k, F_k) and half-extents (eu, ev, ew). A texel at
// image coordinate (u, v) carrying stored depth d is the surface point
//
//     Q = C + u*R_k + v*U_k + d*F_k                                    (1)
//
// in the plant's own frame — a genuine sample of the source mesh, not paint on
// a flat card.
//
// The card is a screen-aligned quad in the basis (rs, us) built from the view
// ray to THIS plant. Under local-orthographic projection (a plant subtends a
// fraction of a degree) the screen offset of Q is
//
//     s = M * (u, v) + d * g,   M = [[R_k.rs, U_k.rs],[R_k.us, U_k.us]],
//                               g = (F_k.rs, F_k.us)                   (2)
//
// M, its inverse, g and the whole card rectangle are constant per plant, so
// they are computed ONCE in the vertex shader. The fragment inverts (2) in a
// single Newton step:
//
//     t0 = M'^-1 * card + 0.5           (naive uv, ignores depth)
//     d0 = depth(t0) at a COARSE mip    (tap 1 — see below)
//     t1 = t0 - d0 * gw                 (first-order exact)
//
// and shades at t1 (taps 2 and 3, the second only if coverage passed). Three
// taps, no loop, no marching. When the camera sits on a baked view, g = 0 and
// M = I, so the warp is exact and free; between views it is what produces real
// parallax — near parts of the plant slide against far parts and the
// silhouette is the warped true silhouette, not a rotating cutout.
//
// The warp's depth estimate deliberately comes from a COARSE mip. Grass depth
// is wildly discontinuous, and one Newton step off a per-texel depth lands
// anywhere — that reads as smeared streaks. The low-frequency canopy depth is
// what the warp actually needs, it is a few KB so the dependent tap stays in
// cache, and the result is smooth volumetric parallax instead of noise. The
// depth WRITTEN as frag_depth is still the sharp per-texel value.
//
// The near LOD writes the projected depth of Q as frag_depth, so plants
// interpenetrate each other and the ground per pixel. That costs early-z, so
// only the near band pays for it; mid and far plants use planar depth and get
// full early-z rejection behind the near band, which is drawn first.
//
// EVERYTHING THE FRAGMENT NEEDS IS PRE-FOLDED IN THE VERTEX STAGE. Flat
// interstage data is a per-fragment load on this class of GPU, and with the
// overdraw a grass field produces it dominates cheap shading, so:
//   - the atlas half-extents, the plant scale and the [0,1] texture remap are
//     baked into inv_m / gw / the three reconstruction axes; none of them is
//     passed separately;
//   - the wind shear is applied to the quad in the vertex stage and only its
//     card-centre value is folded into the plant centre;
//   - the mid and far fragment shaders never touch the reconstruction axes, so
//     they load 16 interstage components instead of 28.
//
// Hard alpha test everywhere, no dithering: coverage fades erode the alpha
// reference instead (CLAUDE.md taste rule).

struct PlantInst {
  pos: vec3f,
  packed_bits: u32,
}

struct ViewRec {
  r_axis: vec4f, // xyz = R_k (mesh frame), w = eu
  u_axis: vec4f, // xyz = U_k, w = ev
  f_axis: vec4f, // xyz = F_k (toward the baked camera), w = ew
  rect: vec4f,   // tight coverage rect in image metres: u0, u1, v0, v1
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read> insts: array<PlantInst>;
@group(1) @binding(2) var<storage, read> views: array<ViewRec>;
@group(1) @binding(3) var albedo_tex: texture_2d_array<f32>;
@group(1) @binding(4) var geo_tex: texture_2d_array<f32>;
@group(1) @binding(5) var atlas_sampler: sampler;

fn rot_y(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

struct VOut {
  @builtin(position) pos: vec4f,
  /** card position in the screen basis, metres at scale 1 */
  @location(0) card: vec2f,
  /** image->texcoord map: t = inv_m * card + 0.5 */
  @location(1) @interpolate(flat) inv_m: vec4f,
  /** warp step in texcoord units, mip level, alpha reference */
  @location(2) @interpolate(flat) gw_lod_cut: vec4f,
  /** plant centre (world, wind included), atlas layer */
  @location(3) @interpolate(flat) c_layer: vec4f,
  /** R_k * 2*eu*scale (world), cos(yaw) — near LOD only */
  @location(4) @interpolate(flat) ax_u: vec4f,
  /** U_k * -2*ev*scale (world), sin(yaw) */
  @location(5) @interpolate(flat) ax_v: vec4f,
  /** F_k * 2*ew*scale (world) — near LOD only */
  @location(6) @interpolate(flat) ax_d: vec4f,
}

/** Nearest baked view for a plant-local view direction. */
fn pick_layer(dl: vec3f) -> u32 {
  var els = array<f32, 4>(info.row_el0, info.row_el1, info.row_el2, info.row_el3);
  var azs = array<f32, 4>(info.row_az0, info.row_az1, info.row_az2, info.row_az3);
  var offs = array<f32, 4>(info.row_off0, info.row_off1, info.row_off2, info.row_off3);
  let el = asin(clamp(dl.y, -1.0, 1.0));

  // At most 5 candidates (rows + top view), once per plant in the vertex stage.
  var best = 0u;
  var best_d = 1.0e9;
  for (var i = 0u; i < u32(info.n_rows); i++) {
    let d = abs(el - els[i]);
    if (d < best_d) {
      best_d = d;
      best = i;
    }
  }
  if (abs(el - 1.5707963) < best_d) {
    return u32(info.top_layer);
  }
  let n_az = azs[best];
  let az = atan2(dl.x, dl.z);
  let k = (i32(round(az * (n_az / 6.2831853))) + i32(n_az) * 4) % i32(n_az);
  return u32(offs[best]) + u32(k);
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

  // Plant centre: the bake captured around this offset inside the mesh, so
  // reproduce it or the imagery lands where the geometry was not.
  let c_local = vec3f(info.cx, info.cy, info.cz) * scale;
  var centre = base + rot_y(c_local, yaw);

  let to_cam = frame.camera_pos - centre;
  let dist = max(length(to_cam), 1.0e-4);
  let dir = to_cam / dist;

  // --- baked view basis, rotated into world --------------------------------
  let layer = pick_layer(rot_y(dir, -yaw));
  let vr = views[layer];
  let rk = rot_y(vr.r_axis.xyz, yaw);
  let uk = rot_y(vr.u_axis.xyz, yaw);
  let fk = rot_y(vr.f_axis.xyz, yaw);
  let eu = vr.r_axis.w;
  let ev = vr.u_axis.w;
  let ew = vr.f_axis.w;

  // --- screen basis about the view ray -------------------------------------
  var rs = cross(vec3f(0.0, 1.0, 0.0), dir);
  let rs_len = length(rs);
  if (rs_len < 1.0e-3) {
    // Looking straight down the plant axis: fall back to the camera's own
    // right vector (row 0 of the view matrix).
    rs = normalize(vec3f(frame.view[0][0], frame.view[1][0], frame.view[2][0]));
  } else {
    rs = rs / rs_len;
  }
  let us = cross(dir, rs);

  // --- image -> screen map and its inverse ---------------------------------
  let m00 = dot(rk, rs);
  let m01 = dot(uk, rs);
  let m10 = dot(rk, us);
  let m11 = dot(uk, us);
  let det = m00 * m11 - m01 * m10;
  let idet = 1.0 / select(det, 1.0e-3, abs(det) < 1.0e-3);
  // Fade the warp out toward the mid/far split so crossing into the no-warp
  // LOD is continuous instead of a silhouette pop, and fade it out again for
  // plants the camera is practically standing in: there the warp displacement
  // spans hundreds of pixels and its unavoidable disocclusion error reads as
  // bent blades, which is worse than the parallax it buys.
  let warp = info.warp_scale
    * (1.0 - smoothstep(info.warp_fade0, info.warp_end, dist))
    * smoothstep(0.5, 1.5, dist);
  let g = vec2f(dot(fk, rs), dot(fk, us)) * warp;

  // Fold the [-e, e] -> [0, 1] texture remap straight into the inverse map, so
  // the fragment never needs eu / ev: t = inv_m * card + 0.5.
  let su = 0.5 / eu;
  let sv = -0.5 / ev;
  let inv_m = vec4f(m11 * idet * su, -m01 * idet * su, -m10 * idet * sv, m00 * idet * sv);
  // Warp step, likewise in texcoord units and already carrying 2*ew.
  let gw = vec2f(inv_m.x * g.x + inv_m.y * g.y, inv_m.z * g.x + inv_m.w * g.y) * (2.0 * ew);

  // Card = the view's TIGHT coverage rectangle mapped through the same linear
  // map, expanded by the depth range along g. Exactly big enough for the
  // warped imagery and not one fragment more.
  let uc = (vr.rect.x + vr.rect.y) * 0.5;
  let vc = (vr.rect.z + vr.rect.w) * 0.5;
  let ru = (vr.rect.y - vr.rect.x) * 0.5;
  let rv = (vr.rect.w - vr.rect.z) * 0.5;
  let sc = vec2f(m00 * uc + m01 * vc, m10 * uc + m11 * vc);
  let ex = (abs(m00) * ru + abs(m01) * rv + abs(g.x) * ew) * 1.02;
  let ey = (abs(m10) * ru + abs(m11) * rv + abs(g.y) * ew) * 1.02;

  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = corners[vi];
  let sx = sc.x + c.x * ex;
  let sy = sc.y + c.y * ey;

  // Wind: shear the quad by height fraction, and fold the card-centre value
  // into the plant centre so the fragment needs no shear term of its own.
  let span = max((info.y1 - info.y0) * scale, 1.0e-4);
  let y_floor = base.y + info.y0 * scale;
  var world = centre + (rs * sx + us * sy) * scale;
  let sway = wind_sway(base, frame.time, entry.sway, phase);
  world += sway * clamp((world.y - y_floor) / span, 0.0, 1.0);
  centre += sway * clamp((centre.y + sc.y * scale * us.y - y_floor) / span, 0.0, 1.0);

  // Coverage fades: camera inside the plant, and the region rim.
  let core_r = max(eu, ew) * scale;
  let seg_y = clamp(frame.camera_pos.y, y_floor, base.y + info.y1 * scale);
  let d_core = length(vec3f(to_cam.x, frame.camera_pos.y - seg_y, to_cam.z));
  let near_fade = smoothstep(core_r * 0.35, core_r * 1.05, d_core);
  let d_xz = length(to_cam.xz);
  let edge_fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, d_xz);
  let fade = near_fade * edge_fade;

  // Mip level from the plant's projected size — primitive-uniform, so every
  // fragment of this quad hits the same level of the same layer, and the
  // coverage test can discard before the second tap is issued.
  let px_per_unit = max(0.5 * frame.viewport.y * frame.proj[1][1] * scale / dist, 1.0e-4);
  let lod = clamp(log2(info.tile_px / (2.0 * min(eu, ev) * px_per_unit)), 0.0, info.max_lod);

  var out: VOut;
  // A quad whose fade dropped below the alpha reference discards EVERY
  // fragment — emit it behind the near plane instead of rasterizing a
  // guaranteed-empty quad (these are the ones covering the most screen area).
  if (fade < info.alpha_cut) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.card = vec2f(sx, sy);
  out.inv_m = inv_m;
  out.gw_lod_cut = vec4f(gw, lod, info.alpha_cut / max(fade, 1.0e-3));
  out.c_layer = vec4f(centre, f32(layer));
  out.ax_u = vec4f(rk * (2.0 * eu * scale), cos(yaw));
  out.ax_v = vec4f(uk * (-2.0 * ev * scale), sin(yaw));
  out.ax_d = vec4f(fk * (2.0 * ew * scale), 0.0);
  return out;
}

// Octahedral decode, y-primary — same convention the bake stored.
fn oct_decode(e: vec2f) -> vec3f {
  var x = e.x;
  var z = e.y;
  let y = 1.0 - abs(e.x) - abs(e.y);
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0);
    z = (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0);
  }
  return normalize(vec3f(x, y, z));
}

struct Resolved {
  color: vec3f,
  /** texcoord actually shaded, and the raw signed depth code at it. */
  t: vec2f,
  depth_code: f32,
}

/**
 * Sample and shade. `world_hint` is the position used for fog and the depth
 * debug view; the near LOD reconstructs the exact surface point afterwards,
 * the others use the plant centre (they write planar depth anyway, so that IS
 * their depth).
 */
fn resolve(in: VOut, do_warp: bool) -> Resolved {
  let layer = i32(in.c_layer.w);
  let lod = in.gw_lod_cut.z;
  let m = in.inv_m;

  var t = vec2f(m.x * in.card.x + m.y * in.card.y, m.z * in.card.x + m.w * in.card.y) + vec2f(0.5);
  if (do_warp) {
    let d0 = textureSampleLevel(geo_tex, atlas_sampler, t, layer, max(lod, info.warp_lod)).b - 0.5;
    t -= d0 * in.gw_lod_cut.xy;
  }
  let alb = textureSampleLevel(albedo_tex, atlas_sampler, t, layer, lod);
  if (alb.a < in.gw_lod_cut.w) {
    discard;
  }
  // Explicit mip level: this second tap is only issued for fragments that
  // actually survived the coverage test.
  let geo = textureSampleLevel(geo_tex, atlas_sampler, t, layer, lod);

  let n_mesh = oct_decode(geo.rg * 2.0 - 1.0);
  let cy = in.ax_u.w;
  let sy = in.ax_v.w;
  let n = vec3f(cy * n_mesh.x + sy * n_mesh.z, n_mesh.y, -sy * n_mesh.x + cy * n_mesh.z);

  // Baked canopy occlusion is a light-side term (the albedo view must stay the
  // captured colour), exactly like the sun and ambient factors.
  let ao = mix(1.0, geo.a, info.ao_strength);
  var color = light_surface(alb.rgb * ao, n, in.c_layer.xyz);
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.c_layer.xyz);
  }

  var out: Resolved;
  out.color = debug_shade(color, alb.rgb, n, alb.a, in.c_layer.xyz);
  out.t = t;
  out.depth_code = geo.b - 0.5;
  return out;
}

struct DepthOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

/// Near band: true per-pixel depth, so plants interpenetrate each other and
/// the terrain instead of stacking as flat cutouts.
@fragment
fn fs_near(in: VOut) -> DepthOut {
  let r = resolve(in, true);
  // Equation (1), with the atlas extents and the plant scale already folded
  // into the three axes.
  let world =
    in.c_layer.xyz +
    in.ax_u.xyz * (r.t.x - 0.5) +
    in.ax_v.xyz * (r.t.y - 0.5) +
    in.ax_d.xyz * r.depth_code;
  // A hair toward the camera: the reconstructed base sits exactly on the
  // terrain the base pass already rendered.
  let p = world + normalize(frame.camera_pos - in.c_layer.xyz) * info.depth_push;
  let clip = frame.view_proj * vec4f(p, 1.0);
  var out: DepthOut;
  out.color = vec4f(r.color, 1.0);
  out.depth = clamp(clip.z / max(clip.w, 1.0e-5), 0.0, 1.0);
  return out;
}

/// Mid band: same warp and shading, planar depth — early-z stays alive.
@fragment
fn fs_mid(in: VOut) -> @location(0) vec4f {
  return vec4f(resolve(in, true).color, 1.0);
}

/// Far band: no warp (one tap saved), planar depth. A distant plant is a few
/// pixels wide; the parallax it would gain is far below one pixel.
@fragment
fn fs_far(in: VOut) -> @location(0) vec4f {
  return vec4f(resolve(in, false).color, 1.0);
}
