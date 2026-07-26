#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./dome_lib.wgsl"

// Depth-shell impostors. One camera-facing card per plant — but the card is a
// 9x9 lattice pushed along the view axis by the baked FRONT-DEPTH SHELL of the
// chosen view, so it is a real 3D surface: it writes honest depth (no
// frag_depth, early-z intact), interpenetrates its neighbours and the terrain,
// and the near part of the shell parallaxes against the far part as the camera
// moves. Lattice vertices 0..80 are the shell; 81..84 are the flat-card
// corners used by the far LOD, so all three LODs are index ranges over one
// vertex program.
//
// Fragment cost: two texture taps (albedo+coverage, normal+AO+thickness), hard
// alpha test, depth write. Second target is the screen-space canopy mask.

@group(1) @binding(1) var<storage, read> insts: array<PlantInst>;
@group(1) @binding(2) var<storage, read> shell: array<f32>;
@group(1) @binding(3) var tex_albedo: texture_2d_array<f32>;
@group(1) @binding(4) var tex_attr: texture_2d_array<f32>;
@group(1) @binding(5) var card_sampler: sampler;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) world: vec3f,
  @location(2) @interpolate(flat) view: u32,
  @location(3) @interpolate(flat) yaw_cs: vec2f,
  @location(4) @interpolate(flat) erode: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let inst = insts[ii];
  let bits = inst.bits;
  let yaw = inst_yaw(bits);
  let scale = inst_scale(bits);
  let phase = inst_phase(bits);
  let view = inst_view(bits);
  let ext = info.view_ext[view];
  let entry = stand_table[u32(info.entry_index)];

  // Lattice position. 0..80 = shell corners; 81..84 = flat card corners.
  var gx: u32;
  var gy: u32;
  var d_shell = 0.0;
  if (vi < SHELL_STRIDE) {
    gx = vi % GRID_V;
    gy = vi / GRID_V;
    d_shell = shell[view * SHELL_STRIDE + gy * GRID_V + gx] * info.dome_scale;
  } else {
    let c = vi - SHELL_STRIDE;
    gx = (c & 1u) * (GRID_V - 1u);
    gy = (c >> 1u) * (GRID_V - 1u);
  }
  let u = f32(gx) * 0.25 - 1.0;
  let v = 1.0 - f32(gy) * 0.25;

  let base = inst.pos;
  let center = card_center(base, yaw, scale);
  let to_cam = frame.camera_pos - center;

  // Card frame. The lateral axes follow the CAMERA (so the silhouette never
  // foreshortens or breathes at azimuth-cell edges, exactly like a billboard);
  // the tilt follows the baked ring so the imagery lands on a plausible plane.
  var n: vec3f;
  var right: vec3f;
  var up: vec3f;
  let ring = u32(ext.w + 0.5);
  if (ring == 2u) {
    // Straight-down view: the plant's own frame, else the top image would spin
    // with the camera yaw. Shell depth becomes canopy relief.
    n = vec3f(0.0, 1.0, 0.0);
    right = rot_y(vec3f(1.0, 0.0, 0.0), yaw);
    up = rot_y(vec3f(0.0, 0.0, -1.0), yaw);
  } else {
    let h = normalize(vec2f(to_cam.x, to_cam.z) + vec2f(1.0e-5, 0.0));
    right = vec3f(h.y, 0.0, -h.x);
    if (ring == 1u) {
      let c45 = 0.70710678;
      n = vec3f(h.x * c45, c45, h.y * c45);
      up = cross(n, right);
    } else {
      n = vec3f(h.x, 0.0, h.y);
      up = vec3f(0.0, 1.0, 0.0);
    }
  }

  var world = center
    + right * (u * ext.x * scale)
    + up * (v * ext.y * scale)
    + n * (d_shell * scale);
  // Wind: displacement scales with the vertex's real height up the plant.
  let h_span = max((info.y1 - info.y0) * scale, 1.0e-3);
  let hf = clamp((world.y - base.y) / h_span, 0.0, 1.0);
  world += wind_sway(base, frame.time, entry.sway, phase) * hf;

  var out: VOut;
  let erode = card_erode(base, center, scale);
  if (erode > 1.0) {
    // Every fragment would discard — emit behind the near plane instead of
    // rasterizing a guaranteed-empty card (the cull already drops these; this
    // covers the frame's worth of camera motion between cull and draw).
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.uv = vec2f(f32(gx), f32(gy)) * 0.125;
  out.world = world;
  out.view = view;
  out.yaw_cs = vec2f(cos(yaw), sin(yaw));
  out.erode = erode;
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

struct FOut {
  @location(0) color: vec4f,
  /** Screen-space canopy weight: fog-attenuated coverage x baked thickness. */
  @location(1) canopy: vec4f,
}

struct Shaded {
  color: vec4f,
  canopy: f32,
}

fn shade(in: VOut) -> Shaded {
  // Sample before any non-uniform discard (uniform-control-flow rule).
  let alb = textureSample(tex_albedo, card_sampler, in.uv, in.view);
  let attr = textureSample(tex_attr, card_sampler, in.uv, in.view);
  if (alb.a < in.erode) {
    discard;
  }
  let n_mesh = oct_decode_card(attr.xy * 2.0 - 1.0);
  // Rotate the mesh-frame normal by the plant's yaw.
  let n = vec3f(
    in.yaw_cs.x * n_mesh.x + in.yaw_cs.y * n_mesh.z,
    n_mesh.y,
    -in.yaw_cs.y * n_mesh.x + in.yaw_cs.x * n_mesh.z,
  );
  let ao = attr.z;
  let thick = attr.w;

  // Every light term is multiplicative in albedo, so DEBUG_LIGHTING (which
  // divides albedo back out) stays exact.
  let to_eye = normalize(frame.camera_pos - in.world);
  // Looking toward the sun: thin (low baked thickness) parts transmit light.
  let fwd_scatter = max(0.0, dot(-to_eye, frame.sun_dir));
  let trans = pow(fwd_scatter, 4.0) * (1.0 - thick) * frame.sun_color * info.translucency;
  let light = light_surface(vec3f(1.0), n, in.world) * ao + trans;
  var color = alb.rgb * light;

  let dist = distance(in.world, frame.camera_pos);
  let fog_t = 1.0 - exp(-dist * 0.004);
  let fog_amt = fog_t * fog_t;
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = mix(color, sky_horizon_color(), fog_amt);
  }

  var out: Shaded;
  out.color = vec4f(debug_shade(color, alb.rgb, n, alb.a, in.world), 1.0);
  out.canopy = (1.0 - fog_amt) * (0.35 + 0.65 * thick);
  return out;
}

/** Default path: one colour target, exactly the billboard baseline's shape. */
@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  return shade(in).color;
}

/** Used only when the screen-space canopy pass is enabled. */
@fragment
fn fs_canopy(in: VOut) -> FOut {
  let sh = shade(in);
  var out: FOut;
  out.color = sh.color;
  out.canopy = vec4f(sh.canopy, 0.0, 0.0, 1.0);
  return out;
}
