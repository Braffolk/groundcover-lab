#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"

// Octahedral impostors. Each plant is a single camera-facing card. Its
// fragments look up the plant's pre-rendered appearance from a hemi-octahedral
// atlas of ~144 orthographic views: the 4 views nearest the current viewing
// direction are projected onto the card and blended. Because the atlas covers
// the whole upper hemisphere (horizon -> straight down), the plant holds up
// from grazing angles AND from directly overhead. Placement is evaluated in
// the vertex shader over a bounded region around the camera, so cost is
// independent of how many plants the stand contains.

struct Meta {
  center: vec3f, radius: f32,        // local bbox center + bounding radius (unit scale)
  grid: f32, tile: f32, atlas_px: f32, entry_index: f32,
  origin_cell: vec2f, side: f32, seed: f32,
  max_dist: f32, coverage: f32, _p0: f32, _p1: f32,
}
@group(1) @binding(0) var<uniform> imp: Meta;
@group(1) @binding(1) var atlas_albedo: texture_2d<f32>;
@group(1) @binding(2) var atlas_normal: texture_2d<f32>;
@group(1) @binding(3) var atlas_samp: sampler;

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
  @location(1) center: vec3f,
  @location(2) params: vec3f,   // yaw, radius(world), fade
  @location(3) @interpolate(flat) entry: u32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var out: VOut;

  let side = u32(imp.side);
  let slot = ii % SCATTER_MAX_PER_CELL;
  let cell_lin = ii / SCATTER_MAX_PER_CELL;
  let cxi = i32(imp.origin_cell.x) + i32(cell_lin % side);
  let czi = i32(imp.origin_cell.y) + i32(cell_lin / side);
  let entry = u32(imp.entry_index);

  let sp = scatter_candidate(u32(imp.seed), entry, vec2i(cxi, czi), slot);

  // Degenerate (zero-area) card for empty slots — cheap cull, no fragments.
  if (!sp.exists) {
    out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
    out.entry = entry;
    return out;
  }

  let scale = sp.scale;
  let yaw = sp.yaw;
  // Plant bbox center placed in world (horizontal offset rotates with yaw).
  let center = sp.pos + rot_y(imp.center * scale, yaw);
  let R = imp.radius * scale;

  // Distance fade: fade the card out far away and when the camera is inside.
  let dcam = distance(frame.camera_pos, center);
  let far_fade = 1.0 - smoothstep(imp.max_dist * 0.75, imp.max_dist, dcam);
  let near_fade = smoothstep(R * 0.45, R * 1.1, dcam);
  let fade = far_fade * near_fade;

  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = corners[vi];

  // Spherical (camera-facing) billboard basis.
  let to_cam = frame.camera_pos - center;
  let dir = normalize(to_cam);
  var up_ref = vec3f(0.0, 1.0, 0.0);
  if (abs(dir.y) > 0.99) { up_ref = vec3f(1.0, 0.0, 0.0); }
  let right = normalize(cross(up_ref, dir));
  let up = normalize(cross(dir, right));

  var world = center + right * (c.x * R) + up * (c.y * R);
  // Wind: lean the top of the card, roots fixed.
  let top = c.y * 0.5 + 0.5;
  world += wind_sway(sp.pos, frame.time, stand_table[entry].sway, sp.phase) * top;

  out.pos = frame.view_proj * vec4f(world, 1.0);
  out.world = world;
  out.center = center;
  out.params = vec3f(yaw, R, fade);
  out.entry = entry;
  return out;
}

struct Sampled {
  color: vec4f,   // rgb premultiplied by coverage, a = coverage
  normal: vec3f,  // local-frame normal (weighted), not normalized
  w: f32,
}

// Sample one atlas node (ni,nj): project the fragment into that view's ortho
// basis to get the tile UV (this is the parallax that makes off-axis frames
// correct), then read albedo + normal.
fn sample_node(ni: f32, nj: f32, local_off: vec3f, weight: f32) -> Sampled {
  var s: Sampled;
  let gm1 = imp.grid - 1.0;
  let e = vec2f(ni / gm1, nj / gm1) * 2.0 - 1.0;
  let node_dir = hemioct_decode(e);

  var up_ref = vec3f(0.0, 1.0, 0.0);
  if (abs(node_dir.y) > 0.999) { up_ref = vec3f(0.0, 0.0, 1.0); }
  let x = normalize(cross(up_ref, node_dir));
  let y = normalize(cross(node_dir, x));

  // local_off spans [-R, R]; radius already divided out (unit space).
  var uv = vec2f(dot(local_off, x), dot(local_off, y)) * 0.5 + 0.5;
  // Keep inside the tile (avoid bilinear bleed into neighbours).
  let inset = 0.5 / imp.tile;
  uv = clamp(uv, vec2f(inset), vec2f(1.0 - inset));
  let atlas_uv = (vec2f(ni, nj) + uv) * imp.tile / imp.atlas_px;

  let alb = textureSampleLevel(atlas_albedo, atlas_samp, atlas_uv, 0.0);
  let nrm = textureSampleLevel(atlas_normal, atlas_samp, atlas_uv, 0.0).xyz * 2.0 - 1.0;
  let cov = alb.a * weight;
  s.color = vec4f(alb.rgb * cov, cov);
  s.normal = nrm * cov;
  s.w = weight;
  return s;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let yaw = in.params.x;
  let R = in.params.y;
  let fade = in.params.z;

  // Everything in the plant's local (unrotated) frame.
  var cam_local = rot_y(normalize(frame.camera_pos - in.center), -yaw);
  cam_local.y = max(cam_local.y, 0.03); // clamp to upper hemisphere
  cam_local = normalize(cam_local);
  let local_off = rot_y(in.world - in.center, -yaw) / max(R, 1e-4);

  // Map view dir to grid coords, pick 4 nearest nodes, bilinear blend.
  let e = hemioct_encode(cam_local);
  let g = (e * 0.5 + 0.5) * (imp.grid - 1.0);
  let g0 = floor(g);
  let f = g - g0;
  let i0 = clamp(g0.x, 0.0, imp.grid - 2.0);
  let j0 = clamp(g0.y, 0.0, imp.grid - 2.0);

  let s00 = sample_node(i0, j0, local_off, (1.0 - f.x) * (1.0 - f.y));
  let s10 = sample_node(i0 + 1.0, j0, local_off, f.x * (1.0 - f.y));
  let s01 = sample_node(i0, j0 + 1.0, local_off, (1.0 - f.x) * f.y);
  let s11 = sample_node(i0 + 1.0, j0 + 1.0, local_off, f.x * f.y);
  let acc_col = s00.color + s10.color + s01.color + s11.color;
  let acc_nrm = s00.normal + s10.normal + s01.normal + s11.normal;

  let coverage = acc_col.a * fade;
  if (coverage < imp.coverage) { discard; }

  let albedo = acc_col.rgb / max(acc_col.a, 1e-4);
  var n_local = acc_nrm / max(acc_col.a, 1e-4);
  if (length(n_local) < 1e-3) { n_local = vec3f(0.0, 1.0, 0.0); }
  let normal = rot_y(normalize(n_local), yaw);

  var color = light_surface(albedo, normal, in.world);
  color = apply_fog(color, in.world);
  return vec4f(color, 1.0);
}
