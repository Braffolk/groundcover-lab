#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/hash.wgsl"
#include "./frustum_math.wgsl"

// Runtime: each plant is an 8-sided cone-frustum proxy volume (real convex
// geometry, so the silhouette is correct from every angle). The fragment
// shader computes the eye ray's chord through the analytic frustum in closed
// form (one quadratic, no marching) and looks the INTERIOR appearance up in
// the baked chord field: entry surface bin (4-tap bilinear) x hardware
// bilinear over the relative (exit azimuth, exit meridian) chord shape.
// Wind is an inverse shear on the ray (lines stay lines), the reconstructed
// first hit is written as real frag_depth, and the camera fades the plant
// out as it penetrates the proxy.

struct ChordInfo {
  dims: vec4f,     // r0, r1, h, side_len
  caps: vec4f,     // cap_b, cap_t, y_base, max_dist
  ax: vec4f,       // axis_x, axis_z, coverage, entry_smooth
  misc: vec4f,     // entry_index, debug_mode, rb, 0
}

struct PlantInst {
  pos_yaw: vec4f,
  scale_phase: vec4f,
}

@group(1) @binding(0) var<uniform> info: ChordInfo;
@group(1) @binding(1) var<storage, read> instances: array<PlantInst>;
@group(1) @binding(2) var chord_surf: texture_2d<f32>;
@group(1) @binding(3) var chord_geom: texture_2d<f32>;
@group(1) @binding(4) var chord_samp: sampler;

const SIDES: u32 = 8u;
const CIRCUM: f32 = 1.0823922; // 1/cos(pi/8): polygon circumscribes the cone

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world_fade: vec4f,              // xyz world, w far-fade
  @location(1) @interpolate(flat) origin_yaw: vec4f,
  @location(2) @interpolate(flat) wind_scale: vec4f, // xyz wind sway, w scale
}

fn frustum_dims() -> FrustumDims {
  var fd: FrustumDims;
  fd.r0 = info.dims.x; fd.r1 = info.dims.y; fd.h = info.dims.z;
  fd.side_len = info.dims.w; fd.cap_b = info.caps.x; fd.cap_t = info.caps.y;
  return fd;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let inst = instances[ii];
  let fd = frustum_dims();
  let scale = inst.scale_phase.x;
  let yaw = inst.pos_yaw.w;

  // Procedural 8-sided frustum: 48 side verts + 24 top fan + 24 bottom fan,
  // wound CCW-outward; the pipeline culls front faces so only the FAR side
  // rasterizes (no near-plane clipping, camera-inside still gets fragments).
  var ang_i: f32;
  var yf: f32;
  var is_center = false;
  if (vi < 48u) {
    let quad = vi / 6u;
    let c = vi % 6u;
    ang_i = f32(quad) + f32(u32(c == 2u || c == 3u || c == 5u));
    yf = f32(u32(c == 1u || c == 4u || c == 5u));
  } else if (vi < 72u) {
    let tri = (vi - 48u) / 3u;
    let c = (vi - 48u) % 3u;
    yf = 1.0;
    is_center = c == 0u;
    ang_i = f32(tri) + f32(u32(c == 1u)); // (center, fi+1, fi)
  } else {
    let tri = (vi - 72u) / 3u;
    let c = (vi - 72u) % 3u;
    yf = 0.0;
    is_center = c == 0u;
    ang_i = f32(tri) + f32(u32(c == 2u)); // (center, fi, fi+1)
  }
  let ang = ang_i / f32(SIDES) * CF_TWO_PI;
  var rho = mix(fd.r0, fd.r1, yf) * CIRCUM;
  if (is_center) { rho = 0.0; }
  let lp = vec3f(rho * cos(ang), yf * fd.h, rho * sin(ang));

  let origin = inst.pos_yaw.xyz + rot_y(vec3f(info.ax.x, info.caps.z, info.ax.y), yaw) * scale;
  let entry = u32(info.misc.x);
  let sway = wind_sway(inst.pos_yaw.xyz, frame.time, stand_table[entry].sway, inst.scale_phase.y);
  let world = origin + rot_y(lp, yaw) * scale + sway * yf;

  let dcam = distance(frame.camera_pos, origin + vec3f(0.0, fd.h * scale * 0.5, 0.0));
  var fade = 1.0 - smoothstep(info.caps.w * 0.8, info.caps.w, dcam);
  // Gentle near fade: inside ~1 proxy radius the coarse entry bins dominate
  // the screen — thin out before the camera enters (rule: may fade inside).
  let rb_w = info.misc.z * scale;
  fade *= smoothstep(0.45 * rb_w, 1.05 * rb_w, dcam);

  var o: VOut;
  o.pos = frame.view_proj * vec4f(world, 1.0);
  o.world_fade = vec4f(world, fade);
  o.origin_yaw = vec4f(origin, yaw);
  o.wind_scale = vec4f(sway, scale);
  return o;
}

struct ChordSample {
  surf: vec4f,   // premultiplied rgb + coverage, entry-bilinear blended
  t_num: f32,    // coverage-weighted hit fraction numerator
  n_acc: vec3f,  // coverage-weighted normal accumulator
  den: f32,
}

fn tap(bi: i32, bj: i32, dphi01: f32, mx01: f32, w: f32, acc: ptr<function, ChordSample>) {
  let uv = chord_uv(f32(bi), f32(bj), dphi01, mx01);
  let s = textureSampleLevel(chord_surf, chord_samp, uv, 0.0);
  let g = textureSampleLevel(chord_geom, chord_samp, uv, 0.0);
  (*acc).surf += s * w;
  let cw = s.a * w;
  (*acc).t_num += g.x * cw;
  (*acc).n_acc += oct_nrm_decode(g.yz * 2.0 - 1.0) * cw;
  (*acc).den += cw;
}

struct FsOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(in: VOut) -> FsOut {
  let fd = frustum_dims();
  let origin = in.origin_yaw.xyz;
  let yaw = in.origin_yaw.w;
  let scale = max(in.wind_scale.w, 1e-4);
  let inv_scale = 1.0 / scale;

  // Plant-local unit frame, then undo the wind shear on the RAY (a shear is
  // linear in y, so lines stay lines — bending the ray instead of the plant
  // is exact).
  let wl = rot_y(in.wind_scale.xyz, -yaw) * inv_scale;
  let sy = clamp(1.0 + wl.y / fd.h, 0.2, 2.0);
  let cam_l = rot_y(frame.camera_pos - origin, -yaw) * inv_scale;
  let frag_l = rot_y(in.world_fade.xyz - origin, -yaw) * inv_scale;
  let cam_uy = cam_l.y / sy;
  let frag_uy = frag_l.y / sy;
  let o = vec3f(cam_l.xz - wl.xz * (cam_uy / fd.h), cam_uy).xzy;
  let f = vec3f(frag_l.xz - wl.xz * (frag_uy / fd.h), frag_uy).xzy;
  let d = normalize(f - o);

  let ch = frustum_chord(o, d, fd);
  if (!ch.ok) { discard; }

  // Fade out as the camera penetrates the proxy (t0 < 0 = entry behind eye).
  let rmid = (fd.r0 + fd.r1) * 0.5;
  let inside_fade = smoothstep(-0.6 * rmid, 0.0, ch.t0);
  if (inside_fade <= 0.0) { discard; }

  let entry_p = o + d * ch.t0;
  let exit_p = o + d * ch.t1;
  let ce = chart_of(entry_p, ch.face0, fd);
  let cx = chart_of(exit_p, ch.face1, fd);
  let m_lo = chart_m_lo(fd);
  let m_range = chart_m_range(fd);
  let dphi01 = fract((cx.x - ce.x) / CF_TWO_PI);
  let mx01 = clamp((cx.y - m_lo) / m_range, 0.0, 1.0);

  var acc: ChordSample;
  acc.surf = vec4f(0.0); acc.t_num = 0.0; acc.n_acc = vec3f(0.0); acc.den = 0.0;

  let fe = fract(ce.x / CF_TWO_PI) * CF_A_E - 0.5;
  let ge = clamp((ce.y - m_lo) / m_range, 0.0, 1.0) * CF_M_E - 0.5;
  if (info.ax.w > 0.5) {
    // 4-bin bilinear over the entry chart (the inner chord-shape lookup is
    // hardware-bilinear, so together this is full 4D interpolation).
    let b0 = floor(fe);
    let c0 = floor(ge);
    let wx = fe - b0;
    let wy = ge - c0;
    let bi0 = (i32(b0) % i32(CF_A_E) + i32(CF_A_E)) % i32(CF_A_E);
    let bi1 = (bi0 + 1) % i32(CF_A_E);
    let bj0 = clamp(i32(c0), 0, i32(CF_M_E) - 1);
    let bj1 = clamp(i32(c0) + 1, 0, i32(CF_M_E) - 1);
    tap(bi0, bj0, dphi01, mx01, (1.0 - wx) * (1.0 - wy), &acc);
    tap(bi1, bj0, dphi01, mx01, wx * (1.0 - wy), &acc);
    tap(bi0, bj1, dphi01, mx01, (1.0 - wx) * wy, &acc);
    tap(bi1, bj1, dphi01, mx01, wx * wy, &acc);
  } else {
    let bi = (i32(round(fe)) % i32(CF_A_E) + i32(CF_A_E)) % i32(CF_A_E);
    let bj = clamp(i32(round(ge)), 0, i32(CF_M_E) - 1);
    tap(bi, bj, dphi01, mx01, 1.0, &acc);
  }

  let cov = acc.surf.a;
  let alpha = cov * inside_fade * in.world_fade.w;
  let albedo = acc.surf.rgb / max(cov, 1e-4);
  let t01 = acc.t_num / max(acc.den, 1e-4);

  // Reconstruct the hit, re-apply the shear, back to world -> true depth.
  let hit_u = mix(entry_p, exit_p, t01);

  // Hard alpha test against a WORLD-ANCHORED organic pattern: the threshold
  // is jittered by a hash of the plant-local hit cell (~2cm), so mid-alpha
  // chord bins resolve into stable foliage clumps instead of solid ovals.
  // Not screen-space dither: the pattern rides on the reconstructed surface.
  let hp = vec3i(floor(hit_u * 48.0));
  let hn = hash_f32(hash3(bitcast<u32>(hp.x), bitcast<u32>(hp.y), bitcast<u32>(hp.z)));
  let threshold = info.ax.z * (0.55 + 0.9 * hn);
  if (alpha < threshold) { discard; }
  // Same anchored noise re-used as subtle albedo variation: breaks up the
  // bin-mean color smear into foliage-scale detail (no extra hash).
  let albedo_var = 0.82 + 0.36 * hn;
  let hit_b = vec3f(hit_u.x + wl.x * hit_u.y / fd.h, hit_u.y * sy, hit_u.z + wl.z * hit_u.y / fd.h);
  let hit_w = origin + rot_y(hit_b * scale, yaw);
  let clip = frame.view_proj * vec4f(hit_w, 1.0);
  if (clip.w < 1e-4) { discard; }

  var n = acc.n_acc / max(acc.den, 1e-4);
  if (length(n) < 1e-3) { n = vec3f(0.0, 1.0, 0.0); }
  n = rot_y(normalize(n), yaw);
  let vd = normalize(hit_w - frame.camera_pos);
  if (dot(n, vd) > 0.0) { n = -n; } // two-sided foliage

  var color = light_surface(albedo * albedo_var, n, hit_w);
  color = apply_fog(color, hit_w);
  if (info.misc.y > 0.5) { color = vec3f(dphi01, mx01, cov); }

  var out: FsOut;
  out.color = vec4f(color, 1.0);
  out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
  return out;
}
