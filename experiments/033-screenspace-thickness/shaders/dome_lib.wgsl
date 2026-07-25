#include "src/wgsl/frame.wgsl"

// Shared by the cull compute pass and the shell draw: the per-stand-entry
// uniform, the instance record, view selection and the fade/alpha-erosion
// rule. Both stages MUST agree on view/LOD/erosion — one copy of the code is
// how that stays true.

const N_VIEWS: u32 = 13u;
const N_AZ0: f32 = 8.0;
const N_AZ1: f32 = 4.0;
const GRID_V: u32 = 9u;         // lattice corners per axis
const SHELL_STRIDE: u32 = 81u;  // GRID_V * GRID_V
const RING_MID_MIN: f32 = 0.3926991;  // 22.5 deg
const RING_TOP_MIN: f32 = 1.1780972;  // 67.5 deg
const TAU: f32 = 6.2831853;

struct EntryInfo {
  planes: array<vec4f, 6>,     // world-space frustum planes, normalized
  view_ext: array<vec4f, 13>,  // su, sv, sd, ring  per baked view
  origin_cell: vec2f,          // south-west scatter cell of the region
  side_x: f32,
  side_z: f32,
  seed: f32,
  entry_index: f32,
  region_r: f32,
  cull_radius: f32,            // bounding-sphere radius at scale 1
  cap0: f32,                   // per-LOD instance capacities
  cap1: f32,
  cap2: f32,
  alpha_ref: f32,
  y0: f32,                     // capture box (unit scale, metres)
  y1: f32,
  cx: f32,
  cz: f32,
  lod0_r: f32,
  lod1_r: f32,
  r_xz: f32,
  dome_scale: f32,
  translucency: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
}

struct PlantInst {
  pos: vec3f,
  bits: u32, // yaw 10b | scale 10b (0..4 m) | phase 7b | view 5b
}

@group(1) @binding(0) var<uniform> info: EntryInfo;

fn rot_y(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn pack_inst_bits(yaw: f32, scale: f32, phase: f32, view: u32) -> u32 {
  let yaw_q = u32(yaw * (1024.0 / TAU)) & 1023u;
  let scale_q = u32(clamp(scale * 0.25, 0.0, 1.0) * 1023.0) & 1023u;
  let phase_q = u32(phase * (128.0 / TAU)) & 127u;
  return yaw_q | (scale_q << 10u) | (phase_q << 20u) | ((view & 31u) << 27u);
}

fn inst_yaw(bits: u32) -> f32 { return f32(bits & 1023u) * (TAU / 1024.0); }
fn inst_scale(bits: u32) -> f32 { return f32((bits >> 10u) & 1023u) * (4.0 / 1023.0); }
fn inst_phase(bits: u32) -> f32 { return f32((bits >> 20u) & 127u) * (TAU / 128.0); }
fn inst_view(bits: u32) -> u32 { return (bits >> 27u) & 31u; }

/** Card centre = plant base + yawed clump offset + capture-box mid height. */
fn card_center(base: vec3f, yaw: f32, scale: f32) -> vec3f {
  let clump = rot_y(vec3f(info.cx, 0.0, info.cz) * scale, yaw);
  return base + clump + vec3f(0.0, (info.y0 + info.y1) * 0.5 * scale, 0.0);
}

/**
 * Effective alpha-test reference for this card. Camera-inside and region-rim
 * fades erode coverage through the threshold instead of blending or dithering;
 * distance lowers the reference so mipped coverage keeps a distant stand from
 * thinning out. > 1.0 means every fragment of the card would discard.
 */
fn card_erode(base: vec3f, center: vec3f, scale: f32) -> f32 {
  let to_cam = frame.camera_pos - center;
  let core_r = info.r_xz * scale;
  let seg_y = clamp(frame.camera_pos.y, base.y + info.y0 * scale, base.y + info.y1 * scale);
  let d_core = length(vec3f(to_cam.x, frame.camera_pos.y - seg_y, to_cam.z));
  let near_fade = smoothstep(core_r * 0.35, core_r * 1.05, d_core);
  let d_xz = length(to_cam.xz);
  let edge_fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, d_xz);
  let fade = near_fade * edge_fade;
  let ref_scale = mix(1.0, 0.62, smoothstep(25.0, 95.0, length(to_cam)));
  return info.alpha_ref * ref_scale / max(fade, 1.0e-3);
}

/**
 * Nearest baked view for a plant: 8 azimuths low, 4 at 45 deg, 1 from above.
 *
 * `jitter` in [-0.5, 0.5] comes from the plant's own hash and shifts the two
 * elevation cuts by +/-7 deg. Without it the rings meet along a hard circle on
 * the ground — very visible from above, where the straight-down view is
 * genuinely sparser than the 45 deg one (blades seen end-on). Jittering turns
 * that circle into a scattered band. This is per-PLANT, not per-pixel: no
 * screen-door, no holes in the depth buffer, no temporal shimmer.
 */
fn pick_view(to_cam: vec3f, yaw: f32, jitter: f32) -> u32 {
  let l = rot_y(normalize(to_cam), -yaw);
  let elev = asin(clamp(l.y, -1.0, 1.0));
  let wobble = jitter * 0.24;
  if (elev > RING_TOP_MIN + wobble) {
    return 12u;
  }
  let az = atan2(l.x, l.z);
  if (elev > RING_MID_MIN + wobble) {
    let k = (i32(round(az * (N_AZ1 / TAU))) + i32(N_AZ1)) % i32(N_AZ1);
    return 8u + u32(k);
  }
  let k = (i32(round(az * (N_AZ0 / TAU))) + i32(N_AZ0)) % i32(N_AZ0);
  return u32(k);
}
