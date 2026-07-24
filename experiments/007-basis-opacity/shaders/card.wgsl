#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"

// Fourier appearance cards. Each plant is one camera-facing card; every card
// texel stores the plant's appearance as a truncated Fourier series in view
// AZIMUTH (order 3 for opacity, order 1 for luminance/depth) x E discrete
// elevation rings that are lerped. The fragment shader evaluates the basis in
// CLOSED FORM: cos/sin of the (per-plant constant) view azimuth expanded by
// the double/triple-angle recurrences — no view atlas, no 4-view blending, no
// popping; the angular parallax lives inside the coefficients. Placement is
// the shared scatter twin over a bounded region around the camera, so per
// frame cost is independent of total plant count.

// Must match bake.ts.
const RINGS: f32 = 4.0;
const PHI_MIN: f32 = 0.13962634; //  8 deg
const PHI_MAX: f32 = 1.43116999; // 82 deg

struct Meta {
  center: vec3f, radius: f32,        // local bbox center + bounding radius (unit scale)
  origin_cell: vec2f, side: f32, seed: f32,
  max_dist: f32, alpha_gain: f32, dither: f32, entry_index: f32,
  color_view: f32, _p0: f32, _p1: f32, _p2: f32,
}
@group(1) @binding(0) var<uniform> fc: Meta;
@group(1) @binding(1) var coeff_a: texture_2d_array<f32>;
@group(1) @binding(2) var coeff_b: texture_2d_array<f32>;
@group(1) @binding(3) var coeff_c: texture_2d_array<f32>;
@group(1) @binding(4) var coeff_d: texture_2d_array<f32>;
@group(1) @binding(5) var coeff_samp: sampler;

fn rot_y(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

// Full-sphere octahedral decode — inverse of oct_encode() in fit.wgsl.
fn oct_decode_card(e: vec2f) -> vec3f {
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
  @location(0) uv: vec2f,
  @location(1) world: vec3f,
  // x = cos(theta_local), y = sin(theta_local), z = ring lerp t, w = fade
  @location(2) @interpolate(flat) view: vec4f,
  // x = ring layer 0, y = yaw, z = plant radius (world m)
  @location(3) @interpolate(flat) aux: vec3f,
  @location(4) @interpolate(flat) entry: u32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var out: VOut;

  let side = u32(fc.side);
  let slot = ii % SCATTER_MAX_PER_CELL;
  let cell_lin = ii / SCATTER_MAX_PER_CELL;
  let cxi = i32(fc.origin_cell.x) + i32(cell_lin % side);
  let czi = i32(fc.origin_cell.y) + i32(cell_lin / side);
  let entry = u32(fc.entry_index);

  let sp = scatter_candidate(u32(fc.seed), entry, vec2i(cxi, czi), slot);

  // Degenerate (zero-area) card for empty slots — cheap cull, no fragments.
  if (!sp.exists) {
    out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
    out.entry = entry;
    return out;
  }

  let scale = sp.scale;
  let yaw = sp.yaw;
  let center = sp.pos + rot_y(fc.center * scale, yaw);
  let R = fc.radius * scale;

  // View direction in world, with elevation clamped into the baked ring range
  // (also sidesteps the vertical-basis singularity: the card never tilts past
  // PHI_MAX, at most ~8 deg from horizontal when seen from straight above).
  let to_cam = frame.camera_pos - center;
  let hl = length(to_cam.xz);
  var az = vec2f(1.0, 0.0);
  if (hl > 1e-4) { az = to_cam.xz / hl; }
  let phi = clamp(atan2(to_cam.y, hl), PHI_MIN, PHI_MAX);
  let cp = cos(phi);
  let dir = vec3f(az.x * cp, sin(phi), az.y * cp);
  let right = normalize(cross(vec3f(0.0, 1.0, 0.0), dir));
  let up = cross(dir, right);

  // Ring coordinate + plant-local view azimuth (Fourier evaluation angle).
  let rf = (phi - PHI_MIN) / (PHI_MAX - PHI_MIN) * (RINGS - 1.0);
  let l0 = floor(min(rf, RINGS - 1.001));
  // Mesh-frame azimuth: world -> mesh is rot_y(-yaw), i.e. theta_mesh =
  // theta_world + yaw (see rot_y() sign convention).
  let cy = cos(yaw);
  let sy = sin(yaw);
  let c1 = az.x * cy - az.y * sy;
  let s1 = az.y * cy + az.x * sy;

  // Distance fade: far dissolve + camera-inside dissolve.
  let dcam = length(to_cam);
  let far_fade = 1.0 - smoothstep(fc.max_dist * 0.75, fc.max_dist, dcam);
  let near_fade = smoothstep(R * 0.45, R * 1.1, dcam);
  let fade = far_fade * near_fade;

  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = corners[vi];

  var world = center + right * (c.x * R) + up * (c.y * R);
  // Wind: shared sway model, scaled by normalized height up the plant.
  let plant_h = max(stand_table[entry].height_scale * scale, 0.1);
  let hf = clamp((world.y - sp.pos.y) / plant_h, 0.0, 1.0);
  world += wind_sway(sp.pos, frame.time, stand_table[entry].sway, sp.phase) * hf;

  out.pos = frame.view_proj * vec4f(world, 1.0);
  out.uv = vec2f(c.x, -c.y) * 0.5 + 0.5;
  out.world = world;
  out.view = vec4f(c1, s1, rf - l0, fade);
  out.aux = vec3f(l0, yaw, R);
  out.entry = entry;
  return out;
}

fn sample_ring(tex: texture_2d_array<f32>, uv: vec2f, l0: i32, l1: i32, t: f32) -> vec4f {
  let v0 = textureSampleLevel(tex, coeff_samp, uv, l0, 0.0);
  let v1 = textureSampleLevel(tex, coeff_samp, uv, l1, 0.0);
  return mix(v0, v1, t);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let c1 = in.view.x;
  let s1 = in.view.y;
  let t = in.view.z;
  let fade = in.view.w;
  let l0 = i32(in.aux.x);
  let l1 = min(l0 + 1, i32(RINGS) - 1);

  let A = sample_ring(coeff_a, in.uv, l0, l1, t);
  let B = sample_ring(coeff_b, in.uv, l0, l1, t);

  // Closed-form Fourier evaluation via double/triple angle recurrences.
  let c2 = 2.0 * c1 * c1 - 1.0;
  let s2 = 2.0 * s1 * c1;
  let c3 = c1 * c2 - s1 * s2;
  let s3 = s1 * c2 + c1 * s2;
  var alpha = A.x
    + (A.y * 2.0 - 1.0) * c1 + (A.z * 2.0 - 1.0) * s1
    + (A.w * 2.0 - 1.0) * c2 + (B.x * 2.0 - 1.0) * s2
    + (B.y * 2.0 - 1.0) * c3 + (B.z * 2.0 - 1.0) * s3;

  // Sharpen the truncated (fuzzy) reconstruction, then fade and alpha-test
  // against a screen-stable dither for soft edges and smooth dissolves.
  alpha = clamp((alpha - 0.5) * fc.alpha_gain + 0.5, 0.0, 1.0) * fade;
  let px = vec2u(in.pos.xy);
  let noise = hash_f32(hash2(px.x, px.y));
  let threshold = mix(0.5, clamp(noise, 0.02, 0.98), fc.dither);
  if (alpha < threshold) { discard; }

  let C = sample_ring(coeff_c, in.uv, l0, l1, t);
  let D = sample_ring(coeff_d, in.uv, l0, l1, t);

  // View-dependent luminance (order-1) on top of the DC albedo.
  let m = 1.0 + ((C.a * 2.0 - 1.0) * c1 + (D.z * 2.0 - 1.0) * s1) * fc.color_view;
  let albedo = C.rgb * clamp(m, 0.0, 2.0);

  let n_local = oct_decode_card(D.xy * 2.0 - 1.0);
  let normal = rot_y(n_local, in.aux.y);

  var color = light_surface(albedo, normal, in.world);
  color = apply_fog(color, in.world);
  return vec4f(color, 1.0);
}
