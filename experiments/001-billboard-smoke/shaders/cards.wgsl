#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./entry_info.wgsl"

// Billboard cards over baked imagery. Each culled instance draws 12 vertices:
//   verts 0..5  — cylindrical camera-facing side card. The fragment samples
//                 the baked side view whose azimuth (in the plant's yawed
//                 frame) is nearest the actual viewing azimuth (8 views).
//   verts 6..11 — horizontal top card at top_frac height, textured with the
//                 baked straight-down view; edge-on (invisible) at grazing
//                 angles, it takes over naturally as the camera rises.
// Hard alpha-test edges with depth write; near/region fades erode coverage
// through the alpha reference instead of dithering or blending.

struct PlantInst {
  pos: vec3f,
  packed_bits: u32,
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read> insts: array<PlantInst>;
@group(1) @binding(2) var card_albedo: texture_2d<f32>;
@group(1) @binding(3) var card_normal: texture_2d<f32>;
@group(1) @binding(4) var card_sampler: sampler;

const GRID: f32 = 3.0;       // 3x3 tiles in the atlas; tile 8 is the top view
const N_SIDE_VIEWS: f32 = 8.0;
const UV_INSET: f32 = 0.0078125; // 4 texels of a 512 tile

fn rot_y(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn tile_uv(view_index: u32, local: vec2f) -> vec2f {
  let col = f32(view_index % 3u);
  let row = f32(view_index / 3u);
  let clamped = clamp(local, vec2f(UV_INSET), vec2f(1.0 - UV_INSET));
  return (vec2f(col, row) + clamped) / GRID;
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) world: vec3f,
  @location(2) @interpolate(flat) yaw_cs: vec2f, // cos(yaw), sin(yaw)
  @location(3) @interpolate(flat) erode: f32,    // effective alpha reference
  @location(4) shade: f32,                       // grounding gradient
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
  // The baked capture is centered on the clump, which sits off the mesh
  // origin — reproduce that offset so imagery lands where the geometry was.
  let clump = rot_y(vec3f(info.card_cx, 0.0, info.card_cz) * scale, yaw);
  let axis = base + clump;

  var corners = array<vec2f, 6>(
    vec2f(-1.0, 0.0), vec2f(1.0, 0.0), vec2f(-1.0, 1.0),
    vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = corners[vi % 6u];
  let is_top = vi >= 6u;

  let to_cam = frame.camera_pos - axis;
  let sway = wind_sway(base, frame.time, entry.sway, phase);

  // Camera-inside fade: 3D distance to the plant's core segment.
  let seg_y = clamp(frame.camera_pos.y, base.y + h0, base.y + h1);
  let d_core = length(vec3f(to_cam.x, frame.camera_pos.y - seg_y, to_cam.z));
  let near_fade = smoothstep(r * 0.35, r * 1.05, d_core);
  // Region edge: erode to nothing exactly at region_r (cull matches).
  let d_xz = length(to_cam.xz);
  let edge_fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, d_xz);
  var fade = near_fade * edge_fade;
  if (is_top) {
    // The top card only makes sense from well above — at flatter views it
    // reads as a pale floating cutout among the side cards, so erode it away
    // unless the card itself is seen steeply (elevation vs the card's height).
    let dy = frame.camera_pos.y - (base.y + mix(h0, h1, info.top_frac));
    let elev = abs(dy) / max(length(vec3f(to_cam.x, dy, to_cam.z)), 1e-4);
    fade *= smoothstep(0.35, 0.6, elev);
  }

  var world: vec3f;
  var uv: vec2f;
  var shade: f32;
  if (is_top) {
    let c2 = vec2f(c.x, c.y * 2.0 - 1.0); // top card spans [-1,1] on both axes
    world = base + rot_y(vec3f(info.card_cx + c2.x * info.card_r, 0.0, info.card_cz + c2.y * info.card_r) * scale, yaw);
    world.y += mix(h0, h1, info.top_frac);
    world += sway * info.top_frac;
    uv = tile_uv(8u, c2 * 0.5 + 0.5);
    shade = 1.0 - info.bottom_shade * 0.3;
  } else {
    let fwd_xz = to_cam.xz / max(length(to_cam.xz), 1e-4);
    let right = vec3f(fwd_xz.y, 0.0, -fwd_xz.x);
    world = axis + right * (c.x * r);
    world.y = base.y + mix(h0, h1, c.y);
    world += sway * c.y;

    // Nearest baked azimuth in the plant's local frame.
    let local_dir = rot_y(vec3f(fwd_xz.x, 0.0, fwd_xz.y), -yaw);
    let ang = atan2(local_dir.x, local_dir.z);
    let k = u32((i32(round(ang * (N_SIDE_VIEWS / 6.2831853))) + i32(N_SIDE_VIEWS)) % i32(N_SIDE_VIEWS));
    uv = tile_uv(k, vec2f(c.x * 0.5 + 0.5, 1.0 - c.y));
    shade = mix(1.0 - info.bottom_shade, 1.0, c.y);
  }

  var out: VOut;
  // A card whose fade dropped below the alpha reference has erode > 1, so
  // EVERY one of its fragments discards — emit it behind the near plane
  // instead of rasterizing a guaranteed-empty quad. This is exactly the set
  // the fragment shader was already throwing away (fully faded top cards at
  // low elevations, cards the camera is standing inside, the region rim), and
  // those are precisely the ones that cover the most screen area.
  if (fade < info.alpha_ref) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.uv = uv;
  out.world = world;
  out.yaw_cs = vec2f(cos(yaw), sin(yaw));
  out.erode = info.alpha_ref / max(fade, 1e-3);
  out.shade = shade;
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

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  // Sample before any non-uniform discard (uniform-control-flow rule).
  let alb = textureSample(card_albedo, card_sampler, in.uv);
  let enc = textureSample(card_normal, card_sampler, in.uv).xy;
  if (alb.a < in.erode) {
    discard;
  }
  let n_mesh = oct_decode_card(enc * 2.0 - 1.0);
  // Rotate the mesh-frame normal by the plant's yaw.
  let n = vec3f(
    in.yaw_cs.x * n_mesh.x + in.yaw_cs.y * n_mesh.z,
    n_mesh.y,
    -in.yaw_cs.y * n_mesh.x + in.yaw_cs.x * n_mesh.z,
  );
  // in.shade is a fake grounding occlusion, so it belongs to the light term:
  // the albedo view then shows the baked atlas colour exactly as captured and
  // the lighting view shows sun+ambient x grounding.
  var color = light_surface(alb.rgb * in.shade, n, in.world);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  // Coverage = the baked alpha this fragment resolved to; with a hard alpha
  // test that is also the alpha-test margin (everything below `erode` is gone).
  return vec4f(debug_shade(color, alb.rgb, n, alb.a, in.world), 1.0);
}
