#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./common.wgsl"

// Sub-tuft splats. A near plant (band 0) is drawn as FOUR splats — one per
// baked xz quadrant of its mesh — anchored at the quadrant's own centre, so
// they sit at four different world positions and depths:
//   * near quadrants shift against far quadrants when the camera moves
//     (parallax INSIDE one plant, which a single card cannot have),
//   * the union silhouette changes shape with view direction instead of
//     rotating rigidly, because the quadrants occlude each other differently,
//   * the depth test between them is real self-occlusion, and the quadrant
//     facing away from the camera is shaded darker (it is deeper in the volume),
//   * wind is evaluated at each quadrant's own position, so the plant twists
//     instead of shearing as one flat card.
// A far plant (band 1) is the whole-plant node: exactly one splat.
//
// Coverage is a full-resolution r8 alpha-test mask; colour is half-res and
// normals quarter-res (3 taps, hard edges, depth write — no dither, no blend).

struct PlantInst {
  pos: vec3f,
  bits: u32,
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read> insts: array<PlantInst>;
@group(1) @binding(2) var tuft_cov: texture_2d<f32>;
@group(1) @binding(3) var tuft_albedo: texture_2d<f32>;
@group(1) @binding(4) var tuft_normal: texture_2d<f32>;
@group(1) @binding(5) var tuft_sampler: sampler;

const UV_INSET_X: f32 = 0.0039; // 1 texel of a 256 px tile
const UV_INSET_Y: f32 = 0.002;  // 1 texel of a 512 px tile

fn tile_uv(node: u32, az: u32, local: vec2f) -> vec2f {
  let l = clamp(local, vec2f(UV_INSET_X, UV_INSET_Y), vec2f(1.0 - UV_INSET_X, 1.0 - UV_INSET_Y));
  return (vec2f(f32(node), f32(az)) + vec2f(l.x, 1.0 - l.y)) / vec2f(NODE_COLS, AZ_COUNT);
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) world: vec3f,
  @location(2) @interpolate(flat) yaw_cs: vec2f,
  @location(3) @interpolate(flat) erode: f32,
  @location(4) shade: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let inst = insts[ii];
  let bits = inst.bits;
  let yaw = f32(bits & 1023u) * (6.2831853 / 1024.0);
  let scale = f32((bits >> 10u) & 2047u) * (4.0 / 2047.0);
  let phase = f32((bits >> 21u) & 1023u) * (6.2831853 / 1024.0);
  let band = bits >> 31u;
  let base = inst.pos;
  let entry = stand_table[u32(info.cfg0.y)];

  // band 0 draws 24 vertices (quadrant nodes 1..4), band 1 draws 6 (node 0).
  let node = select(1u + vi / 6u, 0u, band == 1u);
  var corners = array<vec2f, 6>(
    vec2f(-1.0, 0.0), vec2f(1.0, 0.0), vec2f(-1.0, 1.0),
    vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = corners[vi % 6u];

  let nd0 = info.nodes[node * 2u];      // cx, cy, cz, halfW
  let nd1 = info.nodes[node * 2u + 1u]; // y0, y1, azOffset
  let root = info.nodes[0];
  let plant_h = info.cfg2.z * scale;

  // Node anchor: its mesh-frame offset, yawed and scaled with the plant.
  let off = rot_y(vec3f(nd0.x, 0.0, nd0.z), yaw) * scale;
  let axis = base + off;
  let y0 = base.y + nd1.x * scale;
  let y1 = base.y + nd1.y * scale;
  let anchor = vec3f(axis.x, (y0 + y1) * 0.5, axis.z);

  let to_cam = frame.camera_pos - anchor;
  let fwd = normalize(to_cam.xz + vec2f(1.0e-5, 0.0));
  let right = vec3f(fwd.y, 0.0, -fwd.x);
  let hw = nd0.w * scale;

  let vy = mix(y0, y1, c.y);
  var world = vec3f(axis.x, vy, axis.z) + right * (c.x * hw);
  let hf = clamp((vy - base.y) / max(plant_h, 1.0e-3), 0.0, 1.0);
  world += wind_sway(axis, frame.time, entry.sway, phase) * hf;

  // Baked azimuth nearest the view direction in the plant's own frame. All nodes
  // share one azimuth grid (nd1.z == 0) so that blades crossing a quadrant
  // boundary are reprojected identically on both sides and stay connected.
  let ldir = rot_y(vec3f(fwd.x, 0.0, fwd.y), -yaw);
  let ang = atan2(ldir.x, ldir.z) - nd1.z;
  let k = u32((i32(round(ang * (AZ_COUNT / 6.2831853))) + 8 * i32(AZ_COUNT)) % i32(AZ_COUNT));

  // Coverage erosion: the shell hand-off (same reference point the cull used),
  // plus the camera-inside-plant rule.
  let mid = base + vec3f(0.0, plant_h * 0.5, 0.0);
  var fade = tuft_fade(dissolve_amount(mid, info.cfg2.x, info.cfg2.y));
  let core = base + rot_y(vec3f(root.x, 0.0, root.z), yaw) * scale;
  let core_r = info.cfg2.w * scale;
  let seg_y = clamp(frame.camera_pos.y, base.y, base.y + plant_h);
  let d_core = length(vec3f(core.x - frame.camera_pos.x, seg_y - frame.camera_pos.y, core.z - frame.camera_pos.z));
  fade *= smoothstep(core_r * 0.35, core_r * 1.1, d_core);

  // Fake volumetric occlusion: bottoms are darker, and a quadrant whose offset
  // from the plant axis points away from the camera is deeper inside the
  // volume, so it is darker too.
  let shade_g = info.cfg1.w;
  let rel = rot_y(vec3f(nd0.x - root.x, 0.0, nd0.z - root.z), yaw);
  let back = clamp(-dot(normalize(rel.xz + vec2f(1.0e-6, 0.0)), fwd), 0.0, 1.0) * f32(1u - band);
  let depth_ao = 1.0 - shade_g * 0.45 * back;

  var out: VOut;
  if (fade < info.cfg1.z) {
    // Every fragment would discard — do not rasterize the quad at all.
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.uv = tile_uv(node, k, vec2f(c.x * 0.5 + 0.5, c.y));
  out.world = world;
  out.yaw_cs = vec2f(cos(yaw), sin(yaw));
  out.erode = info.cfg1.z / max(fade, 1.0e-3);
  out.shade = mix(1.0 - shade_g, 1.0, hf) * depth_ao;
  return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  // Sample before any non-uniform discard (uniform-control-flow rule).
  let cov = textureSample(tuft_cov, tuft_sampler, in.uv).r;
  let albedo = textureSample(tuft_albedo, tuft_sampler, in.uv).rgb;
  let enc = textureSample(tuft_normal, tuft_sampler, in.uv).xy;
  if (cov < in.erode) {
    discard;
  }
  let n_mesh = oct_decode_ds(enc * 2.0 - 1.0);
  let n = vec3f(
    in.yaw_cs.x * n_mesh.x + in.yaw_cs.y * n_mesh.z,
    n_mesh.y,
    -in.yaw_cs.y * n_mesh.x + in.yaw_cs.x * n_mesh.z,
  );
  // in.shade is fake occlusion, so it belongs to the light term: the albedo
  // debug view then shows exactly the baked capture.
  var color = light_surface(albedo * in.shade, n, in.world);
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, albedo, n, cov, in.world), 1.0);
}
