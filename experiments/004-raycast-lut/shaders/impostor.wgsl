#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/hash.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./rayfield-common.wgsl"

// Ray-LUT impostor: one camera-facing quad per plant; each fragment builds
// the pixel's actual eye ray, moves it into the plant's unsheared local
// frame, and answers "what does this ray see?" with ~2 lookups into the
// baked ray-answer atlas: pick the nearest baked ray direction, fetch depth
// at the ray's closest-approach slab offset, reproject once along the true
// ray (kills most of the direction-quantization parallax), then take the
// final albedo/depth/normal/height answer. The baked depth is turned back
// into a world-space hit point, so frag_depth gives true inter-plant
// occlusion — no billboard "cardboard" sorting artifacts.
//
// Wind is a linear-in-y shear of plant space: lines stay lines under shear,
// so bending the RAY into unsheared space is exact, not an approximation.

struct EntryParams {
  center: vec3f, // baked bounding-sphere center (anchored mesh space)
  radius: f32,
  top_h: f32,    // mesh-space shear reference height
  max_dist: f32,
  fade_band: f32,
  height_ao: f32,
  entry_index: f32,
  sway_mul: f32,
  show_cells: f32, // method-specific view: tint by baked direction cell
  _pad: f32,
}
@group(1) @binding(0) var<storage, read> plants: array<vec4f>;
@group(1) @binding(1) var<uniform> ep: EntryParams;
@group(1) @binding(2) var atlas_surf: texture_2d<f32>;
@group(1) @binding(3) var atlas_geom: texture_2d<f32>;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) @interpolate(flat) plant_pos: vec3f,
  @location(2) @interpolate(flat) data: vec4f, // cos_yaw, sin_yaw, scale, fade
  @location(3) @interpolate(flat) sway: vec3f,
}

// rot(yaw): local -> world (about +Y). rot_inv is its transpose.
fn rot_fwd(v: vec2f, cy: f32, sy: f32) -> vec2f {
  return vec2f(v.x * cy + v.y * sy, -v.x * sy + v.y * cy);
}
fn rot_inv(v: vec2f, cy: f32, sy: f32) -> vec2f {
  return vec2f(v.x * cy - v.y * sy, v.x * sy + v.y * cy);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let d0 = plants[ii * 2u];
  let d1 = plants[ii * 2u + 1u];
  let pos = d0.xyz;
  let yaw = d0.w;
  let scale = d1.x;
  let phase = d1.z;
  let st = stand_table[u32(ep.entry_index)];
  let k = wind_sway(pos, frame.time, st.sway, phase) * ep.sway_mul;
  let cy = cos(yaw);
  let sy = sin(yaw);

  // Sphere center in world, following the shear at its own height.
  let c_rot = rot_fwd(ep.center.xz, cy, sy);
  let wc = pos + vec3f(c_rot.x, ep.center.y, c_rot.y) * scale + k * (ep.center.y / ep.top_h);
  let rw = ep.radius * scale * 1.06 + length(k) * 0.6;

  let to_c = wc - frame.camera_pos;
  let d = length(to_c);
  let fwd = to_c / max(d, 1e-4);
  var right = vec3f(1.0, 0.0, 0.0);
  if (abs(fwd.y) < 0.999) {
    right = normalize(cross(vec3f(0.0, 1.0, 0.0), fwd));
  }
  let up = normalize(cross(fwd, right));
  // Perspective enlargement so the quad still covers the sphere up close.
  let grow = clamp(d / sqrt(max(d * d - rw * rw, rw * rw * 0.04)), 1.0, 5.0);
  let size = rw * grow;

  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let cn = corners[vi];
  let world = wc + (right * cn.x + up * cn.y) * size;

  // Camera-inside-plant rule: dissolve as the eye enters the sphere.
  let near_fade = clamp((d / rw - 0.55) / 0.65, 0.0, 1.0);
  let dist_fade = clamp((ep.max_dist - d) / max(ep.fade_band, 0.01), 0.0, 1.0);

  var out: VOut;
  out.clip = frame.view_proj * vec4f(world, 1.0);
  out.world = world;
  out.plant_pos = pos;
  out.data = vec4f(cy, sy, scale, near_fade * dist_fade);
  out.sway = k;
  return out;
}

struct FOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(in: VOut) -> FOut {
  let cy = in.data.x;
  let sy = in.data.y;
  let scale = in.data.z;
  let fade = in.data.w;
  let dither = hash_f32(hash2(u32(in.clip.x), u32(in.clip.y)));
  if (fade <= dither * 0.999) {
    discard;
  }
  let k = in.sway;

  // Eye ray -> unsheared plant-local (mesh units) space. The shear map is
  // q = rot(p)*scale + k*(p.y/top_h); invert it for origin and direction.
  let denom = scale + k.y / ep.top_h;
  let rd_w = normalize(in.world - frame.camera_pos);
  let qo = frame.camera_pos - in.plant_pos;
  let oy = qo.y / denom;
  let oxz = rot_inv(qo.xz - k.xz * (oy / ep.top_h), cy, sy) / scale;
  let o_l = vec3f(oxz.x, oy, oxz.y);
  let uy = rd_w.y / denom;
  let uxz = rot_inv(rd_w.xz - k.xz * (uy / ep.top_h), cy, sy) / scale;
  let d_l = normalize(vec3f(uxz.x, uy, uxz.y));

  // Closest approach to the baked sphere — the seed slab offset.
  let m = o_l - ep.center;
  let t0 = -dot(m, d_l);
  let p0 = o_l + d_l * t0;
  if (length(p0 - ep.center) > ep.radius * 1.02) {
    discard;
  }

  // Nearest baked ray direction and its slab basis.
  let e = rf_oct_encode(d_l);
  let cellf = clamp(floor(e * RF_VIEWS), vec2f(0.0), vec2f(RF_VIEWS - 1.0));
  let dv = rf_oct_decode((cellf + vec2f(0.5)) / RF_VIEWS);
  let r_ax = rf_view_right(dv);
  let u_ax = normalize(cross(dv, r_ax));

  // Lookup 1: depth answer at the closest-approach offset.
  var sp = vec2f(dot(p0 - ep.center, r_ax), dot(p0 - ep.center, u_ax));
  var uv = rf_atlas_uv(cellf, sp, ep.radius);
  var surf = textureSampleLevel(atlas_surf, linear_sampler, uv, 0.0);
  var geom = textureSampleLevel(atlas_geom, linear_sampler, uv, 0.0);
  // Beyond ~40m one LUT texel is subpixel — the parallax refinement fetch
  // would be invisible, so spend it only up close.
  let cam_d = length(in.world - frame.camera_pos);
  if (surf.a > 0.01 && cam_d < 40.0) {
    // Reproject the baked hit onto the TRUE eye ray and ask again — one
    // parallax-correction step against direction quantization.
    let depth0 = geom.r / surf.a;
    let hit0 = ep.center + r_ax * sp.x + u_ax * sp.y + dv * ((depth0 * 2.0 - 1.0) * ep.radius);
    let p1 = o_l + d_l * dot(hit0 - o_l, d_l);
    sp = vec2f(dot(p1 - ep.center, r_ax), dot(p1 - ep.center, u_ax));
    uv = rf_atlas_uv(cellf, sp, ep.radius);
    surf = textureSampleLevel(atlas_surf, linear_sampler, uv, 0.0);
    geom = textureSampleLevel(atlas_geom, linear_sampler, uv, 0.0);
  }

  let cov = surf.a;
  let inv_cov = 1.0 / max(cov, 1e-3);
  let depth01 = geom.r * inv_cov;
  // The baked hit in local space — used both for the clutter hash and for the
  // true-depth reprojection below, so reconstruct it exactly once.
  let hitp = ep.center + r_ax * sp.x + u_ax * sp.y + dv * ((depth01 * 2.0 - 1.0) * ep.radius);

  // Sharpened coverage: fuzzy half-covered LUT texels would screen-door the
  // whole plant; remapping keeps interiors solid and dithers only rims.
  var cov_sharp = clamp((cov - 0.26) * 1.9, 0.0, 1.0);
  var albedo = surf.rgb * inv_cov;
  // Plant-local clutter noise: LUT texels are ~2.4cm, pure magnification
  // turns close plants into smooth blobs. A hash keyed to the (stable,
  // unsheared) local hit cell restores leafy high-frequency structure.
  // Its weight is zero past ~8m, so the whole hash is skipped out there
  // instead of being computed and multiplied by 0 for every far fragment.
  let near_w = clamp(1.0 - cam_d * 0.12, 0.0, 1.0);
  if (near_w > 0.0) {
    let cell3 = vec3i(floor(hitp * 44.0)) + vec3i(1024);
    let clutter = hash_f32(hash3(u32(cell3.x), u32(cell3.y), u32(cell3.z)));
    cov_sharp *= 1.0 - near_w * (0.75 - 1.1 * clutter);
    albedo *= 1.0 - near_w * (0.28 - 0.56 * clutter);
  }
  if (cov_sharp * fade <= dither) {
    discard;
  }
  let oct = clamp(geom.gb * inv_cov, vec2f(0.0), vec2f(1.0));
  let height01 = clamp(geom.a * inv_cov, 0.0, 1.0);

  // Reproject the baked hit onto the true eye ray for real world depth.
  let p_l = o_l + d_l * dot(hitp - o_l, d_l);
  let pxz = rot_fwd(p_l.xz, cy, sy);
  let p_w = in.plant_pos + vec3f(pxz.x, p_l.y, pxz.y) * scale + k * (p_l.y / ep.top_h);
  let clip = frame.view_proj * vec4f(p_w, 1.0);
  if (clip.w <= 0.0) {
    discard;
  }

  // Baked mesh normal, yaw-rotated, foliage-style two-sided.
  var n_l = rf_oct_decode(oct);
  let nxz = rot_fwd(n_l.xz, cy, sy);
  var n = vec3f(nxz.x, n_l.y, nxz.y);
  if (dot(n, frame.camera_pos - p_w) < 0.0) {
    n = -n;
  }
  // Method-specific inspection view: flat tint per baked direction cell, so
  // the 24x24 direction quantization (and its popping) is directly visible.
  if (ep.show_cells > 0.5) {
    let ci = u32(cellf.x + cellf.y * RF_VIEWS);
    albedo = vec3f(
      hash_f32(hash2(ci, 7u)),
      hash_f32(hash2(ci, 11u)),
      hash_f32(hash2(ci, 13u)),
    ) * 0.6 + 0.2;
  }

  let ao = mix(1.0 - ep.height_ao, 1.0, height01);
  var color = light_surface(albedo * ao, n, p_w);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, p_w);
  }

  var out: FOut;
  out.color = vec4f(debug_shade(color, albedo, n, cov_sharp * fade, p_w), 1.0);
  out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
  return out;
}
