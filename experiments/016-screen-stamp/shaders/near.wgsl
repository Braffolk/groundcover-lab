#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"

// NEAR FIELD (< ~8 m): a plant 1-3 m from the camera angularly overlaps
// almost every screen tile, so per-tile lists would drown in near plants
// (slot exhaustion). Instead the constant-size ring of scatter cells around
// the camera (6x6 cells — region-bounded, plant-count independent) is drawn
// as classic alpha-tested impostor cards with depth write, BEFORE the bin
// pass: binning then sees them in the depth buffer and the stamp resolve
// composites behind them. Same atlas, same wind, same card math as the
// stamps, crossfading at 6.5-8 m. Camera-inside-plant fades by coverage
// erosion (alpha threshold ramp), not dither.

const NEAR_SIDE: i32 = 6;            // cells per axis around the camera
const NEAR_RANGE: f32 = 9.5;         // hard candidate cutoff (m)
const FADE_LO: f32 = 6.5;            // stamp crossfade start
const FADE_HI: f32 = 8.0;            // stamp crossfade end
const GRID_F: f32 = 10.0;
const ATLAS_TILE_F: f32 = 128.0;
const ATLAS_F: f32 = 1280.0;
const MAX_LOD: f32 = 3.0;

struct StampParams {
  dims: vec4u,                 // x,y unused, z = seed, w = entry_count
  tuning: vec4f,               // x = max_dist, y = tint_strength, z = tile overlay, w unused
  entry_meta: array<vec4f, 4>, // impostor local center.xyz, bounding radius
  entry_info: array<vec4f, 4>, // species average albedo rgb, pad
}
@group(1) @binding(0) var<uniform> sp: StampParams;
@group(1) @binding(1) var alb0: texture_2d<f32>;
@group(1) @binding(2) var alb1: texture_2d<f32>;
@group(1) @binding(3) var alb2: texture_2d<f32>;
@group(1) @binding(4) var nrm0: texture_2d<f32>;
@group(1) @binding(5) var nrm1: texture_2d<f32>;
@group(1) @binding(6) var nrm2: texture_2d<f32>;
@group(1) @binding(7) var atlas_samp: sampler;

fn rot_y_v(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn hemioct_encode(v: vec3f) -> vec2f {
  let s = abs(v.x) + abs(v.y) + abs(v.z);
  let p = vec2f(v.x, v.z) / max(s, 1e-6);
  return vec2f(p.x + p.y, p.x - p.y);
}

fn hemioct_decode(e: vec2f) -> vec3f {
  let px = (e.x + e.y) * 0.5;
  let pz = (e.x - e.y) * 0.5;
  let y = 1.0 - abs(px) - abs(pz);
  return normalize(vec3f(px, y, pz));
}

fn sample_albedo(entry_index: u32, uv: vec2f, lod: f32) -> vec4f {
  if (entry_index == 0u) { return textureSampleLevel(alb0, atlas_samp, uv, lod); }
  if (entry_index == 1u) { return textureSampleLevel(alb1, atlas_samp, uv, lod); }
  return textureSampleLevel(alb2, atlas_samp, uv, lod);
}

fn sample_normal(entry_index: u32, uv: vec2f, lod: f32) -> vec4f {
  if (entry_index == 0u) { return textureSampleLevel(nrm0, atlas_samp, uv, lod); }
  if (entry_index == 1u) { return textureSampleLevel(nrm1, atlas_samp, uv, lod); }
  return textureSampleLevel(nrm2, atlas_samp, uv, lod);
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) @interpolate(flat) center: vec3f,
  @location(2) @interpolate(flat) info: vec4f, // R, yaw, fade, entry
  @location(3) @interpolate(flat) node: vec4f, // ni, nj, sway.x, sway.z
  @location(4) @interpolate(flat) bh: vec4f,   // base y, plant height, k_clip, 0
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var out: VOut;
  out.pos = vec4f(0.0, 0.0, 2.0, 1.0); // degenerate unless a live plant

  let ec = max(sp.dims.w, 1u);
  let per_cell = SCATTER_MAX_PER_CELL * ec;
  let cell_lin = ii / per_cell;
  let rem = ii % per_cell;
  let entry_index = rem / SCATTER_MAX_PER_CELL;
  let slot = rem % SCATTER_MAX_PER_CELL;
  let cam = frame.camera_pos;
  let origin = vec2i(floor((cam.xz - vec2f(10.0)) / SCATTER_CELL_SIZE));
  let cell = origin + vec2i(i32(cell_lin) % NEAR_SIDE, i32(cell_lin) / NEAR_SIDE);

  let cand = scatter_candidate(sp.dims.z, entry_index, cell, slot);
  if (!cand.exists) { return out; }

  let entry = stand_table[entry_index];
  let im = sp.entry_meta[entry_index];
  let r_world = im.w * cand.scale;
  let swayv = wind_sway(cand.pos, frame.time, entry.sway, cand.phase);
  let center = cand.pos + rot_y_v(im.xyz * cand.scale, cand.yaw) + swayv * 0.6;

  let rel = center - cam;
  let dcam = length(rel);
  if (dcam > NEAR_RANGE) { return out; }

  let far_vis = 1.0 - smoothstep(FADE_LO, FADE_HI, dcam);
  let near_vis = smoothstep(r_world * 0.3, r_world * 0.8, dcam);
  let fade = far_vis * near_vis;
  if (fade < 0.02) { return out; }

  // Nearest baked view for this plant (same snap as the bin pass).
  var cam_local = rot_y_v(-rel / max(dcam, 1e-4), -cand.yaw);
  cam_local.y = max(cam_local.y, 0.03);
  cam_local = normalize(cam_local);
  let e2 = hemioct_encode(cam_local);
  let g = clamp((e2 * 0.5 + 0.5) * (GRID_F - 1.0), vec2f(0.0), vec2f(GRID_F - 1.0));

  // View-angle card clip — mirrors bin.wgsl/resolve.wgsl.
  let k_clip = mix(1.0, 0.45, smoothstep(0.55, 0.9, -rel.y / max(dcam, 1e-4)));

  // Spherical billboard card.
  let dir = -rel / max(dcam, 1e-4);
  var up_ref = vec3f(0.0, 1.0, 0.0);
  if (abs(dir.y) > 0.99) { up_ref = vec3f(1.0, 0.0, 0.0); }
  let right = normalize(cross(up_ref, dir));
  let up = normalize(cross(dir, right));

  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = corners[vi];
  let height = entry.height_scale * cand.scale;
  let r_draw = r_world * k_clip;
  var world = center + right * (c.x * r_draw) + up * (c.y * r_draw);
  let hf = clamp((world.y - cand.pos.y) / max(height, 0.1), 0.0, 1.0);
  world += vec3f(swayv.x, 0.0, swayv.z) * (hf - 0.6);

  out.pos = frame.view_proj * vec4f(world, 1.0);
  out.world = world;
  out.center = center;
  out.info = vec4f(r_world, cand.yaw, fade, f32(entry_index));
  out.node = vec4f(round(g.x), round(g.y), swayv.x, swayv.z);
  out.bh = vec4f(cand.pos.y, height, k_clip, 0.0);
  return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let r_world = in.info.x;
  let yaw = in.info.y;
  let fade = in.info.z;
  let entry_index = u32(in.info.w);

  // Undo the vertex shear to recover the rigid card point, then project into
  // the baked view's orthographic basis — identical math to resolve.wgsl.
  let hf = clamp((in.world.y - in.bh.x) / max(in.bh.y, 0.1), 0.0, 1.0);
  let shear = vec3f(in.node.z, 0.0, in.node.w) * (hf - 0.6);
  let local = rot_y_v(in.world - shear - in.center, -yaw) / max(r_world, 1e-4);
  let kc = in.bh.z;
  if (dot(local, local) > kc * kc * 1.02) { discard; }

  let e2 = vec2f(in.node.xy) / (GRID_F - 1.0) * 2.0 - 1.0;
  let node_dir = hemioct_decode(e2);
  var up_ref = vec3f(0.0, 1.0, 0.0);
  if (abs(node_dir.y) > 0.999) { up_ref = vec3f(0.0, 0.0, 1.0); }
  let bx = normalize(cross(up_ref, node_dir));
  let by = normalize(cross(node_dir, bx));
  var uvp = vec2f(dot(local, bx), dot(local, by)) * 0.5 + 0.5;
  // Texture row 0 is the capture's +up: flip v or plants render upside down.
  uvp.y = 1.0 - uvp.y;

  let t = distance(in.world, frame.camera_pos);
  let pixel_scale = 2.0 / (frame.viewport.y * frame.proj[1][1]);
  let lod = clamp(log2(max(t * pixel_scale * ATLAS_TILE_F / (2.0 * r_world), 1.0)), 0.0, MAX_LOD);
  let inset = (0.5 * exp2(lod) + 0.5) / ATLAS_TILE_F;
  // REJECT outside the safe interior instead of clamping: community-tile
  // meshes are cropped at their bbox, so atlas tile borders carry coverage
  // and clamping smears them into hollow-rectangle card outlines.
  if (uvp.x < inset || uvp.x > 1.0 - inset || uvp.y < inset || uvp.y > 1.0 - inset) { discard; }
  let auv = (in.node.xy + uvp) * (ATLAS_TILE_F / ATLAS_F);

  let alb = sample_albedo(entry_index, auv, lod);
  // Coverage erosion: fading plants lose their thin parts first — no dither,
  // no blending, depth stays honest.
  let threshold = mix(1.01, 0.45, fade);
  if (alb.a < threshold) { discard; }
  // Atlas albedo is coverage-premultiplied (empty texels are black).
  let albedo = alb.rgb / max(alb.a, 0.05);

  var n_local = sample_normal(entry_index, auv, lod).xyz * 2.0 - 1.0;
  if (dot(n_local, n_local) < 1e-4) { n_local = vec3f(0.0, 1.0, 0.0); }
  let n_ws = rot_y_v(normalize(n_local), yaw);
  var col = light_surface(albedo, n_ws, in.world);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    col = apply_fog(col, in.world);
  }
  // Alpha-tested opaque card: this fragment resolved to full coverage.
  return vec4f(debug_shade(col, albedo, n_ws, 1.0, in.world), 1.0);
}
