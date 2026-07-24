#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"

// Factored quantized light field. Each plant is one camera-facing card. The
// 4D light field L(view dir, image offset) is factored as geometry x
// radiance: 576 hemi-oct ortho DEPTH maps (8-bit, one r8 atlas) answer "which
// surface does this ray see"; the reconstructed 3D hit point then indexes
// view-INDEPENDENT 3D albedo/normal volumes. The 4 views nearest the eye ray
// are blended by coverage-weighted hit position — since colour is a function
// of the hit point only, view interpolation can never double-image colours,
// only soften geometry. Placement is the scatter twin over a bounded
// camera-centred region: per-frame cost is independent of total plant count.

struct QlfInfo {
  center: vec3f, radius: f32,      // mesh-local bbox center + bounding radius (m)
  bmin: vec3f, grid: f32,
  bmax: vec3f, tile: f32,
  atlas_px: f32, entry_index: f32, origin_cell: vec2f,
  side: f32, seed: f32, region_r: f32, cov_threshold: f32,
  refine: f32, height_m: f32, _q0: f32, _q1: f32,
}
@group(1) @binding(0) var<uniform> qlf: QlfInfo;
@group(1) @binding(1) var depth_lf: texture_2d<f32>;
@group(1) @binding(2) var vol_albedo: texture_3d<f32>;
@group(1) @binding(3) var vol_normal: texture_3d<f32>;
@group(1) @binding(4) var qlf_samp: sampler;

fn rot_y(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn hemioct_encode(v: vec3f) -> vec2f {
  let s = abs(v.x) + abs(v.y) + abs(v.z);
  let p = vec2f(v.x, v.z) / max(s, 1e-6);
  return vec2f(p.x + p.y, p.x - p.y); // [-1,1]
}

fn hemioct_decode(e: vec2f) -> vec3f {
  let px = (e.x + e.y) * 0.5;
  let pz = (e.x - e.y) * 0.5;
  let y = 1.0 - abs(px) - abs(pz);
  return normalize(vec3f(px, y, pz));
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) center: vec3f,      // plant bbox center, world space
  @location(2) misc: vec4f,        // yaw, R_world, fade, scale
  @location(3) sway: vec3f,        // wind displacement at full lean
  @location(4) bth: vec3f,         // base y (world), height (world), lean t at this vertex
  @location(5) @interpolate(flat) entry: u32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var out: VOut;
  let side = u32(qlf.side);
  let slot = ii % SCATTER_MAX_PER_CELL;
  let cell_lin = ii / SCATTER_MAX_PER_CELL;
  let cxi = i32(qlf.origin_cell.x) + i32(cell_lin % side);
  let czi = i32(qlf.origin_cell.y) + i32(cell_lin / side);
  let entry = u32(qlf.entry_index);
  let sp = scatter_candidate(u32(qlf.seed), entry, vec2i(cxi, czi), slot);

  out.entry = entry;
  if (!sp.exists) {
    out.pos = vec4f(0.0, 0.0, 2.0, 1.0); // degenerate, no fragments
    return out;
  }

  let scale = sp.scale;
  let yaw = sp.yaw;
  let center = sp.pos + rot_y(qlf.center * scale, yaw);
  let rw = qlf.radius * scale;

  // Far fade at the region edge + fade-out when the camera is inside a plant.
  let dcam = distance(frame.camera_pos, center);
  let far_fade = 1.0 - smoothstep(qlf.region_r * 0.72, qlf.region_r * 0.97, dcam);
  let near_fade = smoothstep(rw * 0.45, rw * 1.1, dcam);
  let fade = far_fade * near_fade;
  if (fade < 0.004) {
    out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
    return out;
  }

  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = corners[vi];

  // Spherical (camera-facing) billboard basis.
  let dir = normalize(frame.camera_pos - center);
  var up_ref = vec3f(0.0, 1.0, 0.0);
  if (abs(dir.y) > 0.99) { up_ref = vec3f(1.0, 0.0, 0.0); }
  let right = normalize(cross(up_ref, dir));
  let up = normalize(cross(dir, right));

  // Wind: the card leans with height, and the fragment shader UN-shears its
  // sample coordinates by the interpolated lean, which is exactly what makes
  // the sprite itself lean (unlike leaning only the card window).
  let sway_vec = wind_sway(sp.pos, frame.time, stand_table[entry].sway, sp.phase);
  let base_y = sp.pos.y;
  let height_w = qlf.height_m * scale;
  let corner_y = center.y + c.y * rw;
  let t_lean = clamp((corner_y - base_y) / max(height_w, 1e-4), 0.0, 1.0);
  let world = center + right * (c.x * rw) + up * (c.y * rw) + sway_vec * t_lean;

  out.pos = frame.view_proj * vec4f(world, 1.0);
  out.world = world;
  out.center = center;
  out.misc = vec4f(yaw, rw, fade, scale);
  out.sway = sway_vec;
  out.bth = vec3f(base_y, height_w, t_lean);
  return out;
}

// --- light-field lookup ------------------------------------------------------

struct ViewTap {
  cov: f32,    // bilinear coverage of this view's 2x2 depth footprint
  hit: vec3f,  // reconstructed hit point, unit-sphere local space
}

// One view of the depth light field: project the query point into the view's
// ortho basis, read fractional coverage (bilinear — it is continuous) and
// gather the 2x2 depth footprint (miss sentinel = 1.0), reconstruct the 3D
// hit point on that view's ortho ray.
fn tap_view(node: vec2f, p_unit: vec3f) -> ViewTap {
  var t: ViewTap;
  let gm1 = qlf.grid - 1.0;
  let e = node / gm1 * 2.0 - 1.0;
  let fwd = hemioct_decode(e);
  var up_ref = vec3f(0.0, 1.0, 0.0);
  if (abs(fwd.y) > 0.999) { up_ref = vec3f(0.0, 0.0, 1.0); }
  let bx = normalize(cross(up_ref, fwd));
  let by = normalize(cross(fwd, bx));

  let perp = p_unit - fwd * dot(p_unit, fwd);
  // NDC +y is framebuffer row 0, so v flips.
  var uv = vec2f(dot(p_unit, bx) * 0.5 + 0.5, 0.5 - dot(p_unit, by) * 0.5);
  let inset = 1.5 / qlf.tile;
  uv = clamp(uv, vec2f(inset), vec2f(1.0 - inset));
  let atlas_uv = (node + uv) * qlf.tile / qlf.atlas_px;

  let cov_frac = textureSampleLevel(depth_lf, qlf_samp, atlas_uv, 0.0).g;
  let g = textureGather(0, depth_lf, qlf_samp, atlas_uv);
  // Gather components: x=(0,1) y=(1,1) z=(1,0) w=(0,0) of the footprint.
  let f = fract(atlas_uv * qlf.atlas_px - 0.5);
  let hit_mask = step(g, vec4f(0.998)); // 1 where a surface was captured
  let w4 = vec4f((1.0 - f.x) * f.y, f.x * f.y, f.x * (1.0 - f.y), (1.0 - f.x) * (1.0 - f.y));
  let cw = w4 * hit_mask;
  let wsum = dot(cw, vec4f(1.0));
  let d8 = dot(cw, g) / max(wsum, 1e-5);
  let z01 = d8 * (255.0 / 254.0); // undo the sentinel rescale
  t.cov = select(0.0, cov_frac, wsum > 1e-4);
  t.hit = perp + fwd * (1.0 - 2.0 * z01);
  return t;
}

struct QuadTap {
  cov: f32,
  hit: vec3f, // coverage-weighted sum; divide by cov
}

fn tap_quad(g0: vec2f, wd: vec4f, p_unit: vec3f) -> QuadTap {
  let t00 = tap_view(g0, p_unit);
  let t10 = tap_view(g0 + vec2f(1.0, 0.0), p_unit);
  let t01 = tap_view(g0 + vec2f(0.0, 1.0), p_unit);
  let t11 = tap_view(g0 + vec2f(1.0, 1.0), p_unit);
  var q: QuadTap;
  q.cov = wd.x * t00.cov + wd.y * t10.cov + wd.z * t01.cov + wd.w * t11.cov;
  q.hit = wd.x * t00.cov * t00.hit + wd.y * t10.cov * t10.hit
        + wd.z * t01.cov * t01.hit + wd.w * t11.cov * t11.hit;
  return q;
}

struct FOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(in: VOut) -> FOut {
  let yaw = in.misc.x;
  let rw = in.misc.y;
  let fade = in.misc.z;

  // Un-shear the wind lean: sample coordinates in the plant's rest pose.
  let w_uns = in.world - in.sway * in.bth.z;
  let off_unit = rot_y(w_uns - in.center, -yaw) / max(rw, 1e-4);
  let cam_unit = rot_y(frame.camera_pos - in.center, -yaw) / max(rw, 1e-4);

  // Per-fragment eye direction (toward camera), clamped to the baked
  // hemisphere.
  var vdir = normalize(cam_unit - off_unit);
  if (vdir.y < 0.02) { vdir = normalize(vec3f(vdir.x, 0.02, vdir.z)); }

  let e = hemioct_encode(vdir);
  let gpos = clamp((e * 0.5 + 0.5) * (qlf.grid - 1.0), vec2f(0.0), vec2f(qlf.grid - 1.001));
  let g0 = min(floor(gpos), vec2f(qlf.grid - 2.0));
  let gf = gpos - g0;
  let wd = vec4f((1.0 - gf.x) * (1.0 - gf.y), gf.x * (1.0 - gf.y), (1.0 - gf.x) * gf.y, gf.x * gf.y);

  let q1 = tap_quad(g0, wd, off_unit);
  if (q1.cov < 1e-4) { discard; }
  var hit = q1.hit / q1.cov;
  var cov = q1.cov;

  // One eye-ray reprojection: slide the query to where the first answer says
  // the surface is, along the true eye ray, and ask again. A precomputed
  // lookup refinement, not a march. Beyond ~45m the residual parallax of the
  // 4.3deg view grid is subpixel, so skip the second round of taps.
  if (qlf.refine > 0.5 && distance(frame.camera_pos, in.center) < 45.0) {
    let eray = normalize(off_unit - cam_unit);
    let hp = cam_unit + eray * dot(hit - cam_unit, eray);
    let q2 = tap_quad(g0, wd, hp);
    if (q2.cov > 0.05) {
      hit = q2.hit / q2.cov;
      cov = q2.cov;
    }
  }

  // Hard alpha-test edge (honest coverage falloff via fade, no dithering).
  if (cov * fade < qlf.cov_threshold) { discard; }

  // Radiance factor: the hit point indexes the view-independent volumes.
  let hit_m = qlf.center + hit * qlf.radius; // mesh-local metres
  let uvw = clamp((hit_m - qlf.bmin) / max(qlf.bmax - qlf.bmin, vec3f(1e-4)), vec3f(0.0), vec3f(1.0));
  let alb_tap = textureSampleLevel(vol_albedo, qlf_samp, uvw, 0.0);
  var n_mesh = textureSampleLevel(vol_normal, qlf_samp, uvw, 0.0).xyz * 2.0 - 1.0;
  if (length(n_mesh) < 1e-3) { n_mesh = vec3f(0.0, 1.0, 0.0); }

  // Re-inject the baked per-voxel luminance sigma as deterministic speckle
  // (hash of the quantized mesh-local hit — stable under wind and camera),
  // so voxel-averaged interiors keep sub-voxel texture.
  let spk = vec3i(floor(hit_m * 66.0)); // ~1.5cm cells
  let hn = hash_f32(hash3(bitcast<u32>(spk.x), bitcast<u32>(spk.y), bitcast<u32>(spk.z)));
  let sigma = alb_tap.a * 0.5;
  let albedo = alb_tap.rgb * clamp(1.0 + (hn - 0.5) * sigma * 2.6, 0.55, 1.55);

  // World-space hit: re-yaw, re-scale, re-shear, then true depth.
  var hit_w = in.center + rot_y(hit, yaw) * rw;
  let t_hit = clamp((hit_w.y - in.bth.x) / max(in.bth.y, 1e-4), 0.0, 1.0);
  hit_w += in.sway * t_hit;
  let clip = frame.view_proj * vec4f(hit_w, 1.0);

  var n_ws = rot_y(normalize(n_mesh), yaw);
  let view_ws = normalize(frame.camera_pos - hit_w);
  if (dot(n_ws, view_ws) < 0.0) { n_ws = -n_ws; } // two-sided foliage

  // Height-based baked AO stand-in: canopy tops lit, roots dimmed.
  let ao = mix(0.6, 1.0, clamp(hit_m.y / max(qlf.height_m, 1e-3), 0.0, 1.0));

  var color = light_surface(albedo * ao, n_ws, hit_w);
  color = apply_fog(color, hit_w);

  var o: FOut;
  o.color = vec4f(color, 1.0);
  o.depth = clamp(clip.z / max(clip.w, 1e-4), 0.0, 1.0);
  return o;
}
