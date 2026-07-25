#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/hash.wgsl"
#include "src/wgsl/debug.wgsl"
#include "src/wgsl/terrain.wgsl"
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
  misc: vec4f,     // entry_index, debug_chart, rb, carpet normal mix
}

struct PlantInst {
  pos_yaw: vec4f,
  scale_phase: vec4f, // scale, wind phase, terrain-conformed up.x, up.z
}

@group(1) @binding(0) var<uniform> info: ChordInfo;
@group(1) @binding(1) var<storage, read> instances: array<PlantInst>;
@group(1) @binding(2) var chord_surf: texture_2d<f32>;
@group(1) @binding(3) var chord_geom: texture_2d<f32>;
@group(1) @binding(4) var chord_samp: sampler;

const SIDES: u32 = 8u;
const CIRCUM: f32 = 1.0823922; // 1/cos(pi/8): polygon circumscribes the cone

// Everything the fragment stage needs that is CONSTANT over the instance is
// computed once here and passed flat: the plant basis, the wind shear, and
// the eye position already transformed into the plant-local, un-sheared
// frame the chord math runs in. (It used to redo three yaw rotations — six
// transcendentals — per fragment for values that never vary across the
// primitive.)
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world_fade: vec4f,                     // xyz world, w far-fade
  @location(1) @interpolate(flat) origin_scale: vec4f, // xyz proxy origin, w scale
  @location(2) @interpolate(flat) ray_o_shear: vec4f,  // xyz eye in unsheared local frame, w shear_y
  // The plant frame: tangent + up columns of an orthonormal basis (the third
  // column is their cross product). With slope_align = 0 this is exactly the
  // old yaw-only rotation; with 1 the proxy lies into the terrain plane.
  @location(3) @interpolate(flat) basis_t: vec4f,       // t.xyz, wl.x
  @location(4) @interpolate(flat) basis_up: vec4f,      // up.xyz, wl.z
}

// Plant-frame transforms. The basis is orthonormal, so the inverse is the
// transpose — three dots, no matrix inverse, no transcendentals.
fn to_world(v: vec3f, t: vec3f, u: vec3f) -> vec3f { return t * v.x + u * v.y + cross(t, u) * v.z; }
fn to_local(v: vec3f, t: vec3f, u: vec3f) -> vec3f { return vec3f(dot(v, t), dot(v, u), dot(v, cross(t, u))); }

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

  // 8-sided frustum proxy, INDEXED: its 32 triangles have only 18 distinct
  // vertices (8 bottom ring, 8 top ring, 2 fan centers), so the draw is
  // indexed and this stage runs 18x per plant instead of 96x. Winding
  // (CCW-outward) lives in PROXY_INDICES in main.ts; the pipeline culls front
  // faces so only the FAR side rasterizes (no near-plane clipping, and the
  // camera inside a plant still gets fragments).
  var ang_i = 0.0;
  var yf = 0.0;
  var is_center = false;
  if (vi < 8u) {
    ang_i = f32(vi);
  } else if (vi < 16u) {
    ang_i = f32(vi - 8u);
    yf = 1.0;
  } else {
    is_center = true;
    yf = f32(u32(vi == 16u)); // 16 = top-cap center, 17 = bottom-cap center
  }
  let ang = ang_i / f32(SIDES) * CF_TWO_PI;
  var rho = mix(fd.r0, fd.r1, yf) * CIRCUM;
  if (is_center) { rho = 0.0; }
  let lp = vec3f(rho * cos(ang), yf * fd.h, rho * sin(ang));

  // Terrain-conformed plant frame (up from the cull, blended by slope_align).
  let up_xz = inst.scale_phase.zw;
  let up_in = vec3f(up_xz.x, sqrt(max(1.0 - dot(up_xz, up_xz), 0.0)), up_xz.y);
  let bs = plant_basis_from_up(up_in, yaw);
  let bt = bs[0];
  let bu = bs[1];
  let entry = u32(info.misc.x);
  // A LATTICE TILE MUST ROTATE ABOUT ITS OWN CENTRE. The scatter puts the mesh
  // origin at the node, and for these community tiles that origin is the
  // CORNER of the period square (tile origin (0,0), footprint_m across), so
  // rotating there threw each of the 4 quarter turns into a different quadrant:
  // the mat lost ~3/4 of its nodes to holes and double-stacked the rest. Anchor
  // the tile's centre (footprint_m/2) at the node instead and the quarter turns
  // map the square onto itself again — which is the whole point of 90° steps.
  let half_tile = select(0.0, stand_table[entry].footprint_m * 0.5, stand_table[entry].carpet_div > 0.0);
  let anchor = vec3f(info.ax.x - half_tile, info.caps.z, info.ax.y - half_tile);
  let origin = inst.pos_yaw.xyz + to_world(anchor, bt, bu) * scale;
  let sway = wind_sway(inst.pos_yaw.xyz, frame.time, stand_table[entry].sway, inst.scale_phase.y);
  let world = origin + to_world(lp, bt, bu) * scale + sway * yf;

  let dcam = distance(frame.camera_pos, origin + bu * (fd.h * scale * 0.5));
  var fade = 1.0 - smoothstep(info.caps.w * 0.8, info.caps.w, dcam);
  // Gentle near fade: inside ~1 proxy radius the coarse entry bins dominate
  // the screen — thin out before the camera enters (rule: may fade inside).
  // NEVER for a carpet: a mat you stand on must not open a hole under the
  // camera, and its entry bins are ~7mm, not 5cm.
  let rb_w = info.misc.z * scale;
  if (stand_table[entry].carpet_div <= 0.0) {
    fade *= smoothstep(0.45 * rb_w, 1.05 * rb_w, dcam);
  }

  // Per-instance ray setup (constant over the proxy): undo the wind shear on
  // the EYE once here — a shear is linear in y, so bending the ray instead of
  // the plant is exact, and the eye point does not vary per fragment.
  let inv_scale = 1.0 / max(scale, 1e-4);
  let wl = to_local(sway, bt, bu) * inv_scale;
  let shear_y = clamp(1.0 + wl.y / fd.h, 0.2, 2.0);
  let cam_l = to_local(frame.camera_pos - origin, bt, bu) * inv_scale;
  let cam_uy = cam_l.y / shear_y;
  let ray_o = vec3f(cam_l.x - wl.x * (cam_uy / fd.h), cam_uy, cam_l.z - wl.z * (cam_uy / fd.h));

  var o: VOut;
  o.pos = frame.view_proj * vec4f(world, 1.0);
  o.world_fade = vec4f(world, fade);
  o.origin_scale = vec4f(origin, scale);
  o.ray_o_shear = vec4f(ray_o, shear_y);
  o.basis_t = vec4f(bt, wl.x);
  o.basis_up = vec4f(bu, wl.z);
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
  let origin = in.origin_scale.xyz;
  let scale = max(in.origin_scale.w, 1e-4);
  let inv_scale = 1.0 / scale;
  // Flat per-instance ray setup from the vertex stage.
  let bt = in.basis_t.xyz;
  let bu = in.basis_up.xyz;
  let wl_xz = vec2f(in.basis_t.w, in.basis_up.w);
  let shear_y = in.ray_o_shear.w;
  let o = in.ray_o_shear.xyz;

  // Only the FRAGMENT's own point still needs transforming into the
  // plant-local, un-sheared frame the chord math lives in.
  let frag_l = to_local(in.world_fade.xyz - origin, bt, bu) * inv_scale;
  let frag_uy = frag_l.y / shear_y;
  let f = vec3f(frag_l.x - wl_xz.x * (frag_uy / fd.h), frag_uy, frag_l.z - wl_xz.y * (frag_uy / fd.h));
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
  let hit_b = vec3f(hit_u.x + wl_xz.x * hit_u.y / fd.h, hit_u.y * shear_y, hit_u.z + wl_xz.y * hit_u.y / fd.h);
  let hit_w = origin + to_world(hit_b * scale, bt, bu);
  let clip = frame.view_proj * vec4f(hit_w, 1.0);
  if (clip.w < 1e-4) { discard; }

  // Real per-fragment normal: the baked mesh normal of the chord's first hit
  // (coverage-weighted mean of the 4 entry-bin taps), rotated into world by
  // the plant's yaw and flipped towards the eye (foliage is two-sided).
  var n = acc.n_acc / max(acc.den, 1e-4);
  if (length(n) < 1e-3) { n = vec3f(0.0, 1.0, 0.0); }
  n = to_world(normalize(n), bt, bu);
  let to_eye = hit_w - frame.camera_pos;
  let vd = normalize(to_eye);
  if (dot(n, vd) > 0.0) { n = -n; } // two-sided foliage
  // A bin covers ~1.5cm of a cushion made of sub-mm leaves, so the stored
  // first-hit normal is ONE leaf standing in for a whole neighbourhood. Up
  // close that reads as cushion detail; past a couple of metres a tile is a few
  // pixels wide, the per-tile mean is all that survives, and since neighbours
  // differ only by a quarter turn the field broke into a bright/dark
  // CHECKERBOARD at grazing. So filter with distance — the normal equivalent of
  // dropping to a coarser mip — towards the mat's macro normal (the conformed up
  // axis), which is the honest limit normal of a cushion surface.
  let n_mix = info.misc.w * smoothstep(0.7, 1.6, length(to_eye));
  if (n_mix > 0.0) { n = normalize(mix(n, bu, n_mix)); }

  let alb = albedo * albedo_var;
  var color = light_surface(alb, n, hit_w);
  if (info.misc.y > 0.5) { color = vec3f(dphi01, mx01, cov); }
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, hit_w);
  }

  var out: FsOut;
  // Coverage reported to the debug view is the resolved chord coverage after
  // the far/near/inside fades — i.e. exactly what the alpha test judged.
  out.color = vec4f(debug_shade(color, alb, n, alpha, hit_w), 1.0);
  out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
  return out;
}
