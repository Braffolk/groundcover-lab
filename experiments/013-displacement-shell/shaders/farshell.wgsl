#include "src/wgsl/hash.wgsl"
#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
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
  // Row radii grow geometrically from the current inner radius outward.
  let radius = globals.shell_in * pow(1.285, in.polar.y);
  var xz = frame.camera_pos.xz + dir * radius;
  // The meadow ends at the stand boundary: clamp collapses outside verts.
  xz = clamp(xz, vec2f(-globals.stand_radius), vec2f(globals.stand_radius));
  let r_eff = length(xz - frame.camera_pos.xz);
  let ramp = smoothstep(globals.shell_in, globals.shell_in + 8.0, r_eff);

  var y = terrain_height(xz) + 0.03 + globals.shell_h * ramp;
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
  // Species mottle: density-weighted patches at ~1.4 m scale.
  var color = vec3f(0.0);
  var wsum = 0.0;
  for (var s = 0u; s < 3u; s++) {
    let dens = globals.species_density[s];
    if (dens <= 0.0) { continue; }
    let n = vnoise(in.world.xz * 0.7 + f32(s) * 37.31, 11u + s);
    let w = dens * (0.3 + 0.7 * n * n);
    color += globals.canopy[s].rgb * w;
    wsum += w;
  }
  if (wsum <= 1e-5) { discard; }
  color /= wsum;
  // Broad luminance variation + fine grain to break flatness.
  color *= 0.8 + 0.35 * vnoise(in.world.xz * 0.21, 7u);
  color *= 0.85 + 0.3 * vnoise(in.world.xz * 3.9, 23u);

  let tn = terrain_normal(in.world.xz);
  let n = normalize(tn + vec3f(0.0, 1.2, 0.0));
  var lit = light_surface(color, n, in.world) * 0.92;
  lit = apply_fog(lit, in.world);

  // Fade the shell out at the stand boundary (the meadow's actual edge).
  let b = max(abs(in.world.x), abs(in.world.z));
  let alpha = (1.0 - smoothstep(globals.stand_radius - 3.0, globals.stand_radius, b)) * in.ramp;
  if (alpha < 0.01) { discard; }
  return vec4f(lit, alpha);
}
