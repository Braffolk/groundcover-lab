#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/debug.wgsl"

// Crossed-card plant proxies, drawn from a fill.wgsl instance list, in two
// modes sharing all geometry math:
//  - NEAR (vs_near/fs_near): every frame, direct clusters only, current
//    camera, live per-plant wind, fog. Cost bounded by the direct radius.
//  - REFRESH (vs_refresh/fs_refresh): the amortized path — renders one
//    cluster's plants into its 176x176 cache slot with the slot's frozen
//    frustum, canonical pose (NO wind: sway is applied analytically at
//    composite time so cached content never ages), lit but unfogged, and
//    writes packed 16-bit slot depth for later re-projection. Distant levels
//    collapse to one camera-facing card with a minimum-footprint clamp so
//    sub-texel plants stay visible instead of alpha-testing away.

struct CardsU {
  vp: mat4x4f,      // refresh frustum view-proj (identity/unused in near mode)
  cam_pos: vec4f,   // xyz = refresh cam pos, w = mode: 0 near, 1 refresh, 2 refresh-far
  proj_info: vec4f, // x = radians per target pixel, y = min card px, z = instance base
}

struct PlantEntry {
  center: vec4f, // xyz = mesh bbox center (plant-local), w = species atlas layer
  dims: vec4f,   // sideHalfW, sideHalfH, topHalfW, topHalfH (m)
  yrange: vec4f, // x = yMin, y = yMax, z,w unused
}

@group(1) @binding(0) var<uniform> cards: CardsU;
@group(1) @binding(1) var<storage, read> instances: array<vec4f>;
@group(1) @binding(2) var<uniform> plant_table: array<PlantEntry, 4>;
@group(1) @binding(3) var card_albedo: texture_2d_array<f32>;
@group(1) @binding(4) var card_normal: texture_2d_array<f32>;
@group(1) @binding(5) var card_sampler: sampler;

const QUAD_U = array<f32, 6>(0.0, 1.0, 0.0, 1.0, 1.0, 0.0);
const QUAD_V = array<f32, 6>(1.0, 1.0, 0.0, 1.0, 0.0, 0.0);

fn rot_yaw(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn oct_decode(e: vec2f) -> vec3f {
  let y = 1.0 - abs(e.x) - abs(e.y);
  var x = e.x;
  var z = e.y;
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * sign(e.x);
    z = (1.0 - abs(e.x)) * sign(e.y);
  }
  return normalize(vec3f(x, y, z));
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,           // atlas uv (tile-mapped)
  @location(1) world_pos: vec3f,
  @location(2) @interpolate(flat) yaw: f32,
  @location(3) @interpolate(flat) lod: f32,
  @location(4) @interpolate(flat) layer: i32,
  // Extra alpha-test threshold: fades the horizontal crown card when seen
  // edge-on (otherwise it smears into dark streaks at eye level).
  @location(5) @interpolate(flat) thr_bias: f32,
}

fn crown_fade(vi: u32, crown_pos: vec3f, cam: vec3f) -> f32 {
  if (vi / 6u != 2u) { return 0.0; }
  let vdir = normalize(cam - crown_pos);
  // Fully faded below 25 deg elevation, fully present above ~33 deg: the
  // spidery top capture only ever reads well from steep angles.
  return 1.0 - clamp((abs(vdir.y) - 0.42) / 0.13, 0.0, 1.0);
}

struct CardVertex {
  world: vec3f,
  uv: vec2f,
  degenerate: bool,
  hf: f32, // normalized height up the plant (0 root, 1 tip) for wind
}

fn card_vertex(vi: u32, ii: u32, mode: u32) -> CardVertex {
  let i0 = instances[ii * 2u];
  let i1 = instances[ii * 2u + 1u];
  let base = i0.xyz;
  let yaw = i0.w;
  let scale = i1.x;
  let entry = u32(i1.z);
  let pe = plant_table[entry];

  let quad = vi / 6u;
  let corner = vi % 6u;
  let u = QUAD_U[corner];
  let v = QUAD_V[corner];

  var out: CardVertex;
  out.degenerate = mode == 2u && quad > 0u;
  out.uv = vec2f(0.0);
  out.hf = 0.0;

  let center_w = base + rot_yaw(pe.center.xyz, yaw) * scale;

  if (quad == 2u) {
    // Horizontal card at crown height, textured with the top capture.
    let local = vec3f((2.0 * u - 1.0) * pe.dims.z, pe.center.y, -(1.0 - 2.0 * v) * pe.dims.w);
    out.world = base + rot_yaw(local + vec3f(pe.center.x, 0.0, pe.center.z), yaw) * scale;
    out.uv = vec2f(0.5 + u * 0.5, v);
    out.hf = clamp((pe.center.y - pe.yrange.x) / max(pe.yrange.y - pe.yrange.x, 1e-4), 0.0, 1.0);
  } else {
    var hw = pe.dims.x;
    let hh = pe.dims.y;
    var right: vec3f;
    if (mode == 2u) {
      // Far single card: face the (refresh) camera, clamp the footprint to a
      // minimum pixel size so distant plants do not alpha-test into dust.
      let to_cam = cards.cam_pos.xyz - center_w;
      right = normalize(vec3f(-to_cam.z, 0.0, to_cam.x));
      let dist = max(length(to_cam), 0.01);
      let px = (2.0 * hw * scale) / (dist * cards.proj_info.x);
      hw *= max(1.0, cards.proj_info.y / max(px, 1e-4));
    } else {
      right = rot_yaw(vec3f(1.0, 0.0, 0.0), yaw + f32(quad) * 1.5707963);
    }
    let ly = pe.center.y + (1.0 - 2.0 * v) * hh;
    out.world = base
      + rot_yaw(vec3f(pe.center.x, 0.0, pe.center.z), yaw) * scale
      + right * ((2.0 * u - 1.0) * hw * scale)
      + vec3f(0.0, ly * scale, 0.0);
    out.uv = vec2f(u * 0.5, v);
    out.hf = clamp((ly - pe.yrange.x) / max(pe.yrange.y - pe.yrange.x, 1e-4), 0.0, 1.0);
  }
  return out;
}

fn card_lod(world: vec3f, cam: vec3f, hh: f32, scale: f32) -> f32 {
  let dist = max(distance(world, cam), 0.05);
  let px = (2.0 * hh * scale) / (dist * cards.proj_info.x);
  return clamp(log2(256.0 / max(px, 0.5)), 0.0, 5.0);
}

// ---------------------------------------------------------------------------

@vertex
fn vs_near(@builtin(vertex_index) vi: u32, @builtin(instance_index) raw_ii: u32) -> VOut {
  let ii = u32(cards.proj_info.z) + raw_ii;
  let i0 = instances[ii * 2u];
  let i1 = instances[ii * 2u + 1u];
  let cv = card_vertex(vi, ii, 0u);
  let entry = u32(i1.z);
  let sway = stand_table[entry].sway;
  let world = cv.world + wind_sway(i0.xyz, frame.time, sway * cv.hf, i1.y);

  var o: VOut;
  o.pos = frame.view_proj * vec4f(world, 1.0);
  o.uv = cv.uv;
  o.world_pos = world;
  o.yaw = i0.w;
  o.lod = card_lod(i0.xyz, frame.camera_pos, plant_table[entry].dims.y, i1.x);
  o.layer = i32(plant_table[entry].center.w);
  o.thr_bias = crown_fade(vi, i0.xyz + vec3f(0.0, plant_table[entry].center.y * i1.x, 0.0), frame.camera_pos);
  return o;
}

@fragment
fn fs_near(i: VOut) -> @location(0) vec4f {
  let tex = textureSampleLevel(card_albedo, card_sampler, i.uv, i.layer, i.lod);
  let dcam = distance(i.world_pos, frame.camera_pos);
  // Base threshold relaxes with lod (mips dilute coverage), rises to 1 as the
  // camera enters the plant so it fades away instead of clipping.
  var thr = mix(0.5, 0.28, i.lod / 5.0) + smoothstep(0.9, 0.25, dcam) + i.thr_bias;
  if (tex.a < thr) { discard; }
  // Per-fragment normal: baked plant-local octahedral normal, rotated into the
  // world by the instance yaw.
  let ne = textureSampleLevel(card_normal, card_sampler, i.uv, i.layer, i.lod);
  let n = rot_yaw(oct_decode(ne.rg * 2.0 - 1.0), i.yaw);
  let albedo = tex.rgb / max(tex.a, 1e-3);
  var color = light_surface(albedo, n, i.world_pos);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, i.world_pos);
  }
  return vec4f(debug_shade(color, albedo, n, tex.a, i.world_pos), 1.0);
}

// ---------------------------------------------------------------------------

@vertex
fn vs_refresh(@builtin(vertex_index) vi: u32, @builtin(instance_index) raw_ii: u32) -> VOut {
  let ii = u32(cards.proj_info.z) + raw_ii;
  let i0 = instances[ii * 2u];
  let i1 = instances[ii * 2u + 1u];
  let mode = u32(cards.cam_pos.w + 0.5);
  let cv = card_vertex(vi, ii, mode);

  var o: VOut;
  if (cv.degenerate) {
    o.pos = vec4f(0.0, 0.0, 2.0, 1.0); // clipped away
  } else {
    o.pos = cards.vp * vec4f(cv.world, 1.0);
  }
  let entry = u32(i1.z);
  o.uv = cv.uv;
  o.world_pos = cv.world;
  o.yaw = i0.w;
  o.lod = card_lod(i0.xyz, cards.cam_pos.xyz, plant_table[entry].dims.y, i1.x);
  o.layer = i32(plant_table[entry].center.w);
  o.thr_bias = crown_fade(vi, i0.xyz + vec3f(0.0, plant_table[entry].center.y * i1.x, 0.0), cards.cam_pos.xyz);
  return o;
}

struct RefreshOut {
  @location(0) color: vec4f,
  @location(1) aux: vec4f, // rg = packed 16-bit slot depth
}

@fragment
fn fs_refresh(i: VOut) -> RefreshOut {
  let tex = textureSampleLevel(card_albedo, card_sampler, i.uv, i.layer, i.lod);
  let thr = mix(0.5, 0.22, i.lod / 5.0) + i.thr_bias;
  if (tex.a < thr) { discard; }
  let ne = textureSampleLevel(card_normal, card_sampler, i.uv, i.layer, i.lod);
  let n = rot_yaw(oct_decode(ne.rg * 2.0 - 1.0), i.yaw);
  let albedo = tex.rgb / max(tex.a, 1e-3);
  let lit = light_surface(albedo, n, i.world_pos);

  let d16 = floor(clamp(i.pos.z, 0.0, 1.0) * 65535.0 + 0.5);
  let hi = floor(d16 / 256.0);
  let lo = d16 - hi * 256.0;

  var o: RefreshOut;
  // The cache holds SHADED imagery, so the debug view has to be baked in here
  // (main.ts re-reconstructs every slot when the selector changes). Composite
  // overrides the coverage/depth views with live values.
  o.color = vec4f(debug_shade(lit, albedo, n, tex.a, i.world_pos), 1.0);
  o.aux = vec4f(hi / 255.0, lo / 255.0, 0.0, 1.0);
  return o;
}
