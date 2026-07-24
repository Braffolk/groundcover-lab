#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"

// Ribbon-skeleton renderer. A whole plant is distilled offline into a handful
// of curved RIBBONS whose shape lives in two tiny textures (T0 = centerline
// xyz + half-width, T1 = albedo + lateral azimuth). At runtime each ribbon is
// unrolled in the vertex shader as a short triangle strip of K cross-sections;
// the strip's width, twist and colour come straight from the texture rows.
//
// Placement is procedural over a bounded region around the camera (scatter
// twin), so per-frame cost is independent of the stand's total plant count.
// Two LODs share one pipeline: lod_mode 0 draws the detail ribbons of near
// plants; lod_mode 1 draws a single camera-facing "aggregate" silhouette card
// for the far ring. Both collapse geometry (width -> 0) to cross-fade and to
// fade out when the camera is inside a plant — no stochastic alpha.

struct RibMeta {
  row_offset: f32, variant_count: f32, rows_per_variant: f32, ribbons: f32,
  origin_cell: vec2f, side: f32, seed: f32,
  entry_index: f32, lod_mode: f32, lod_near: f32, lod_far: f32,
  max_dist: f32, k_cols: f32, width_scale: f32, _pad: f32,
}
@group(1) @binding(0) var<uniform> rib: RibMeta;
@group(1) @binding(1) var tex_posw: texture_2d<f32>;   // rgba16f: xyz + halfWidth
@group(1) @binding(2) var tex_cola: texture_2d<f32>;   // rgba16f: rgb + azimuth

fn rot_y(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn load_posw(k: i32, row: i32) -> vec4f { return textureLoad(tex_posw, vec2i(k, row), 0); }
fn load_cola(k: i32, row: i32) -> vec4f { return textureLoad(tex_cola, vec2i(k, row), 0); }

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) color: vec3f,
  @location(2) normal: vec3f,
  @location(3) side_v: f32,   // signed lateral coordinate for the tapered edge
  @location(4) @interpolate(flat) kill: u32,
}

const VERTS_PER_RIBBON: u32 = 24u; // K_COLS(12) cross-sections * 2

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var out: VOut;
  out.kill = 0u;

  let side = u32(rib.side);
  let K = i32(rib.k_cols);
  let ribbons = u32(rib.ribbons);
  let entry = u32(rib.entry_index);

  // Decode (cell, slot, ribbon) from the flat instance index.
  var slot: u32;
  var cell_lin: u32;
  var ribbon: u32;
  if (rib.lod_mode < 0.5) {
    ribbon = ii % ribbons;
    let plant = ii / ribbons;
    slot = plant % SCATTER_MAX_PER_CELL;
    cell_lin = plant / SCATTER_MAX_PER_CELL;
  } else {
    ribbon = ribbons; // the aggregate row sits just past the detail ribbons
    slot = ii % SCATTER_MAX_PER_CELL;
    cell_lin = ii / SCATTER_MAX_PER_CELL;
  }
  let cxi = i32(rib.origin_cell.x) + i32(cell_lin % side);
  let czi = i32(rib.origin_cell.y) + i32(cell_lin / side);

  let sp = scatter_candidate(u32(rib.seed), entry, vec2i(cxi, czi), slot);
  if (!sp.exists) {
    out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
    out.kill = 1u;
    return out;
  }

  let scale = sp.scale;
  let yaw = sp.yaw;
  let root = sp.pos;

  // Distance to the plant root controls both LODs.
  let dcam = distance(frame.camera_pos, root);
  var lod_alpha = 1.0;
  if (rib.lod_mode < 0.5) {
    // detail: full near, collapses to 0 by lod_far; also fade when camera
    // is basically on top of the plant (inside-plant rule).
    lod_alpha = 1.0 - smoothstep(rib.lod_near, rib.lod_far, dcam);
    lod_alpha *= smoothstep(0.04, 0.22, dcam);
  } else {
    // aggregate: 0 up close, full past lod_far, fades to nothing at max_dist.
    lod_alpha = smoothstep(rib.lod_near, rib.lod_far, dcam);
    lod_alpha *= 1.0 - smoothstep(rib.max_dist * 0.8, rib.max_dist, dcam);
  }
  if (lod_alpha <= 0.001) {
    out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
    out.kill = 1u;
    return out;
  }

  // Pick a per-plant variant (stable across ribbon/LOD) and the atlas row.
  let variant = hash4(u32(rib.seed) ^ 0x9e3779b9u, bitcast<u32>(cxi), bitcast<u32>(czi), slot)
    % u32(rib.variant_count);
  let row = i32(rib.row_offset) + i32(variant) * i32(rib.rows_per_variant) + i32(ribbon);

  // Cross-section index k and which edge of the strip (+/-1).
  let k = i32(vi / 2u);
  let edge = select(-1.0, 1.0, (vi & 1u) == 0u);

  let pw = load_posw(k, row);
  let ca = load_cola(k, row);
  let half_w_raw = pw.w;
  let is_billboard = half_w_raw < 0.0;
  let half_w = abs(half_w_raw) * rib.width_scale;

  // Local tangent from neighbouring samples.
  let ka = max(k - 1, 0);
  let kb = min(k + 1, K - 1);
  let p_a = load_posw(ka, row).xyz;
  let p_b = load_posw(kb, row).xyz;
  var tan_local = p_b - p_a;
  if (length(tan_local) < 1e-5) { tan_local = vec3f(0.0, 1.0, 0.0); }
  tan_local = normalize(tan_local);

  // Centerline position -> world (scaled, yawed, planted at the root).
  let center_world = root + rot_y(pw.xyz * scale, yaw);
  let tan_world = normalize(rot_y(tan_local, yaw));
  let to_cam = frame.camera_pos - center_world;
  let dir_cam = normalize(to_cam);

  var lateral: vec3f;
  var face_normal: vec3f;
  if (is_billboard) {
    // Camera-facing strip (plumes / aggregate card): widen across the screen.
    // Guard the degenerate case where the view looks straight down the ribbon
    // axis (overhead vs. a vertical card) — cross() collapses and the strip
    // spins. Blend toward a stable world-horizontal lateral there.
    let cr = cross(dir_cam, tan_world);
    let stab = length(cr);
    var horiz = normalize(cross(tan_world, vec3f(0.0, 1.0, 0.0)) + vec3f(1e-4, 0.0, 0.0));
    lateral = normalize(mix(horiz, cr / max(stab, 1e-4), smoothstep(0.15, 0.5, stab)));
    face_normal = normalize(cross(tan_world, lateral));
    if (dot(face_normal, dir_cam) < 0.0) { face_normal = -face_normal; }
  } else {
    // Oriented blade: azimuth gives the lateral direction, made perpendicular
    // to the tangent, then rotated into world.
    let phi = ca.w;
    var lat_local = vec3f(cos(phi), 0.0, sin(phi));
    lat_local = lat_local - tan_local * dot(lat_local, tan_local);
    if (length(lat_local) < 1e-4) { lat_local = vec3f(1.0, 0.0, 0.0); }
    lat_local = normalize(lat_local);
    lateral = normalize(rot_y(lat_local, yaw));
    face_normal = normalize(rot_y(normalize(cross(tan_local, lat_local)), yaw));
  }

  // Collapse width smoothly with the LOD alpha (clean geometric fade).
  let w = half_w * scale * lod_alpha;
  var world = center_world + lateral * (edge * w);

  // Wind: lean by normalized height up the plant, roots fixed.
  let hnorm = clamp(pw.y / 1.2, 0.0, 1.0);
  world += wind_sway(root, frame.time, stand_table[entry].sway, sp.phase) * hnorm * hnorm;

  out.pos = frame.view_proj * vec4f(world, 1.0);
  out.world = world;
  out.color = ca.rgb;
  out.normal = face_normal;
  out.side_v = edge;
  return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  if (in.kill == 1u) { discard; }
  // Thin foliage lit from both sides.
  var n = normalize(in.normal);
  let to_cam = frame.camera_pos - in.world;
  if (dot(n, to_cam) < 0.0) { n = -n; }
  var color = light_surface(in.color, n, in.world);
  color = apply_fog(color, in.world);
  return vec4f(color, 1.0);
}
