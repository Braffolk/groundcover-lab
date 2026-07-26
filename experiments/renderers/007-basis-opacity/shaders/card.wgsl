#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"

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
  // ext_xz/ext_y: mesh bbox half-extents (horizontal diagonal / vertical) in
  // bounding-radius units — the card crop, see vs_main().
  color_view: f32, ext_xz: f32, ext_y: f32, ring_tint: f32,
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
  // Fourier evaluation angle, constant per card: cos/sin of k*theta_local for
  // k = 1,2 — the recurrences run once in the vertex stage, not per fragment.
  @location(2) @interpolate(flat) harm12: vec4f, // c1, s1, c2, s2
  // c3, s3, ring lerp t, fade
  @location(3) @interpolate(flat) harm3: vec4f,
  // x = ring layer 0, y = ring layer 1, z = plant yaw
  @location(4) @interpolate(flat) aux: vec3f,
}

fn degenerate() -> VOut {
  var out: VOut;
  out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
  return out;
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

  // Cell-level reject BEFORE any placement work: the enumerated region is a
  // square but the fade is a disc, so ~21% of cells are entirely beyond
  // max_dist. XZ distance to the nearest point of the cell lower-bounds the
  // 3D distance to any plant in it, so this can only cull cards that would
  // have faded to zero alpha and discarded every fragment anyway.
  let cell_lo = vec2f(f32(cxi), f32(czi)) * SCATTER_CELL_SIZE;
  let near_xz = clamp(frame.camera_pos.xz, cell_lo, cell_lo + SCATTER_CELL_SIZE);
  if (distance(frame.camera_pos.xz, near_xz) > fc.max_dist) { return degenerate(); }

  let sp = scatter_candidate(u32(fc.seed), entry, vec2i(cxi, czi), slot);

  // Degenerate (zero-area) card for empty slots — cheap cull, no fragments.
  if (!sp.exists) { return degenerate(); }

  let scale = sp.scale;
  let yaw = sp.yaw;
  let center = sp.pos + rot_y(fc.center * scale, yaw);
  let R = fc.radius * scale;

  let to_cam = frame.camera_pos - center;
  let dcam = length(to_cam);

  // Distance fade: far dissolve + camera-inside dissolve. A card at fade 0
  // discards every one of its fragments, so cull it before it rasterizes.
  let far_fade = 1.0 - smoothstep(fc.max_dist * 0.75, fc.max_dist, dcam);
  let near_fade = smoothstep(R * 0.45, R * 1.1, dcam);
  let fade = far_fade * near_fade;
  if (fade <= 0.0) { return degenerate(); }

  // View direction in world, with elevation clamped into the baked ring range
  // (also sidesteps the vertical-basis singularity: the card never tilts past
  // PHI_MAX, at most ~8 deg from horizontal when seen from straight above).
  let hl = length(to_cam.xz);
  var az = vec2f(1.0, 0.0);
  if (hl > 1e-4) { az = to_cam.xz / hl; }
  let phi = clamp(atan2(to_cam.y, hl), PHI_MIN, PHI_MAX);
  let cp = cos(phi);
  let sph = sin(phi);
  let dir = vec3f(az.x * cp, sph, az.y * cp);
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
  // Double/triple-angle recurrences — uniform over the card, so they belong
  // here and not in the (heavily overdrawn) fragment stage.
  let c2 = 2.0 * c1 * c1 - 1.0;
  let s2 = 2.0 * s1 * c1;
  let c3 = c1 * c2 - s1 * s2;
  let s3 = s1 * c2 + c1 * s2;

  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = corners[vi];

  // The baked tile spans +-R (the bounding-sphere radius) along both card
  // axes, but the mesh bbox can only ever project into a sub-rectangle of it:
  // |dot(rel, right)| <= ext_xz*R because `right` is horizontal, and
  // |dot(rel, up)| <= (cos(phi)*ext_y + sin(phi)*ext_xz)*R. Outside that the
  // baked coverage is exactly zero, so crop the quad (and the uv window with
  // it) instead of rasterizing a provably empty margin — ~45% of the card
  // area at grazing elevation.
  let ex = fc.ext_xz;
  let ey = min(cp * fc.ext_y + sph * fc.ext_xz, 1.0);

  var world = center + right * (c.x * R * ex) + up * (c.y * R * ey);
  // Wind: shared sway model, scaled by normalized height up the plant.
  let plant_h = max(stand_table[entry].height_scale * scale, 0.1);
  let hf = clamp((world.y - sp.pos.y) / plant_h, 0.0, 1.0);
  world += wind_sway(sp.pos, frame.time, stand_table[entry].sway, sp.phase) * hf;

  out.pos = frame.view_proj * vec4f(world, 1.0);
  out.uv = vec2f(c.x * ex, -c.y * ey) * 0.5 + 0.5;
  out.world = world;
  out.harm12 = vec4f(c1, s1, c2, s2);
  out.harm3 = vec4f(c3, s3, rf - l0, fade);
  out.aux = vec3f(l0, min(l0 + 1.0, RINGS - 1.0), yaw);
  return out;
}

fn sample_ring(tex: texture_2d_array<f32>, uv: vec2f, l0: i32, l1: i32, t: f32) -> vec4f {
  let v0 = textureSampleLevel(tex, coeff_samp, uv, l0, 0.0);
  let v1 = textureSampleLevel(tex, coeff_samp, uv, l1, 0.0);
  return mix(v0, v1, t);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let c1 = in.harm12.x;
  let s1 = in.harm12.y;
  let c2 = in.harm12.z;
  let s2 = in.harm12.w;
  let c3 = in.harm3.x;
  let s3 = in.harm3.y;
  let t = in.harm3.z;
  let fade = in.harm3.w;
  let l0 = i32(in.aux.x);
  let l1 = i32(in.aux.y);

  let A = sample_ring(coeff_a, in.uv, l0, l1, t);
  let B = sample_ring(coeff_b, in.uv, l0, l1, t);

  // Closed-form Fourier evaluation (harmonics come flat from the vertex stage).
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

  // Per-texel baked mesh-frame normal, rotated into world by the plant yaw.
  let n_local = oct_decode_card(D.xy * 2.0 - 1.0);
  let normal = rot_y(n_local, in.aux.z);

  var color = light_surface(albedo, normal, in.world);
  // Fog (and the optional ring overlay) only in the normal view — debug views
  // stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
    if (fc.ring_tint > 0.0) {
      let r = in.aux.x / (RINGS - 1.0);
      color = mix(color, vec3f(r, t, 1.0 - r), fc.ring_tint);
    }
  }
  return vec4f(debug_shade(color, albedo, normal, alpha, in.world), 1.0);
}
