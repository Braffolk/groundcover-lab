#include "src/wgsl/hash.wgsl"
#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./common.wgsl"

// The literal terrain-conformal shell: a camera-centered annulus draped on the
// terrain at canopy height. Beyond r_outer this ONE surface is the whole
// meadow — species canopy albedos (baked from the strand fields) blended by
// stand density and a world-anchored mottle noise. Verts are camera-relative;
// all texture detail is world-anchored so nothing swims.

@group(1) @binding(0) var<uniform> globals: Globals;

struct VIn {
  @location(0) polar: vec2f, // (angle, ring-row index)
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) ramp: f32,
}

@vertex
fn vs_main(in: VIn) -> VOut {
  let dir = vec2f(cos(in.polar.x), sin(in.polar.x));
  // Row radii grow geometrically from the current inner radius outward. The
  // inner radius shrinks when the camera climbs (the 3D handoff distance is
  // reached at a smaller horizontal radius), so the growth ratio adapts to keep
  // the last row past the stand boundary.
  let radius = globals.shell_r0 * pow(globals.shell_ratio, in.polar.y);
  var xz = frame.camera_pos.xz + dir * radius;
  // The meadow ends at the stand boundary: clamp collapses outside verts.
  xz = clamp(xz, vec2f(-globals.stand_radius), vec2f(globals.stand_radius));
  let ground = terrain_height(xz);
  // Mat stands hand over by 3D distance (see cs_cull_carpet), so the shell has
  // to ramp by the same measure or the two disagree by the camera's altitude.
  let d_h = length(xz - frame.camera_pos.xz);
  let dy = frame.camera_pos.y - ground;
  let r_eff = select(d_h, sqrt(d_h * d_h + dy * dy), globals.shell_3d > 0.5);
  let ramp = smoothstep(globals.shell_in, globals.shell_in + globals.shell_ramp, r_eff);

  var y = ground + 0.03 + globals.shell_h * ramp;
  // A mat shell rides centimetres above the ground, so wherever the annulus
  // interpolates the terrain across a row span the terrain pokes through it.
  // Lift by a fraction of that span (invisible at the distance where the span is
  // large, and zero next to the camera where the rows are dense).
  if (globals.shell_3d > 0.5) {
    y += min(0.02 + radius * (globals.shell_ratio - 1.0) * 0.06, 0.6) * ramp;
  }
  // Gentle canopy bob so the far field is not frozen.
  let bob = sin(frame.time * 1.9 - dot(xz, frame.wind_dir) * 0.35)
    * frame.wind_strength * wind_gust(frame.time) * 0.06 * globals.shell_bob;
  y += bob * ramp;

  var out: VOut;
  out.world = vec3f(xz.x, y, xz.y);
  out.pos = frame.view_proj * vec4f(out.world, 1.0);
  out.ramp = ramp;
  return out;
}

fn vnoise(p: vec2f, salt: u32) -> f32 {
  let i = vec2i(floor(p));
  let f = fract(p);
  let s = f * f * (3.0 - 2.0 * f);
  let h00 = hash_f32(hash3(bitcast<u32>(i.x), bitcast<u32>(i.y), salt));
  let h10 = hash_f32(hash3(bitcast<u32>(i.x + 1), bitcast<u32>(i.y), salt));
  let h01 = hash_f32(hash3(bitcast<u32>(i.x), bitcast<u32>(i.y + 1), salt));
  let h11 = hash_f32(hash3(bitcast<u32>(i.x + 1), bitcast<u32>(i.y + 1), salt));
  return mix(mix(h00, h10, s.x), mix(h01, h11, s.x), s.y);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  // Species mix from the SHARED WETNESS FIELD, exactly the zoning the scatter
  // uses. Without this the shell paints one global density-weighted average and
  // the handoff from the mat shows up as a coloured ring / disc wherever the
  // local micro-habitat differs from the stand mean.
  let wet = scatter_wetness(globals.seed, in.world.xz);
  var color = vec3f(0.0);
  var wsum = 0.0;
  for (var e = 0u; e < globals.entry_count; e++) {
    let entry = stand_table[e];
    let sp = u32(entry.species_index);
    var w: f32;
    if (entry.carpet_div > 0.0) {
      // Carpet entries partition the wetness axis on a half-open interval; a few
      // percent of softening keeps the shell from drawing a hard contour.
      let lo = entry.wet_center - entry.wet_width * 0.5;
      let hi = entry.wet_center + entry.wet_width * 0.5;
      let soft = 0.05;
      let lo_w = select(clamp((wet - lo) / soft, 0.0, 1.0), 1.0, lo <= 0.001);
      let hi_w = select(clamp((hi - wet) / soft, 0.0, 1.0), 1.0, hi >= 0.999);
      w = lo_w * hi_w * (entry.carpet_div * entry.carpet_div / 16.0);
    } else {
      var acc = 1.0;
      if (entry.wet_width > 0.0) {
        acc = clamp(1.0 - abs(wet - entry.wet_center) / entry.wet_width, 0.0, 1.0);
      }
      let n = vnoise(in.world.xz * 0.7 + f32(sp) * 37.31, 11u + sp);
      w = min(entry.density, 8.0) * acc * (0.3 + 0.7 * n * n);
    }
    if (w <= 0.0) { continue; }
    color += globals.canopy[sp].rgb * w;
    wsum += w;
  }
  if (wsum <= 1e-5) { discard; }
  color /= wsum;
  // Broad luminance variation + fine grain to break flatness.
  color *= 0.8 + 0.35 * vnoise(in.world.xz * 0.21, 7u);
  color *= 0.85 + 0.3 * vnoise(in.world.xz * 3.9, 23u);

  // Canopy normal: the real per-fragment terrain normal the shell is draped
  // on, softened toward +Y (a canopy of leaves reads flatter than the ground
  // under it).
  let tn = terrain_normal(in.world.xz);
  let n = normalize(tn + vec3f(0.0, 1.2, 0.0));
  // Canopy self-shadow trim belongs to the albedo, not the light term, so
  // debug_shade's albedo/lighting split stays exact.
  // Trim matched to the geometry it replaces: the mat's own sheets carry relief
  // occlusion AND normals that face the sun less often than a smooth plane, so
  // without this the handoff steps in brightness.
  let albedo = color * globals.shell_tune.x;
  var lit = light_surface(albedo, n, in.world);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    lit = apply_fog(lit, in.world);
  }

  // Fade the shell out at the stand boundary (the meadow's actual edge).
  let b = max(abs(in.world.x), abs(in.world.z));
  let alpha = (1.0 - smoothstep(globals.stand_radius - 3.0, globals.stand_radius, b)) * in.ramp;
  if (alpha < 0.01) { discard; }
  // Debug views paint the shell opaquely — otherwise its normals/albedo would
  // come back blended with whatever the base pass left behind.
  let out_alpha = select(1.0, alpha, debug_mode() == DEBUG_OFF);
  return vec4f(debug_shade(lit, albedo, n, alpha, in.world), out_alpha);
}
