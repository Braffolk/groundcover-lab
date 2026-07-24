#include "src/wgsl/frame.wgsl"

// Shared lighting model so every experiment's groundcover sits in the same
// light as the terrain: half-lambert sun + hemisphere ambient, distance fog.

fn light_surface(albedo: vec3f, normal_ws: vec3f, world_pos: vec3f) -> vec3f {
  let ndl = dot(normal_ws, frame.sun_dir) * 0.5 + 0.5;
  let sun = frame.sun_color * (ndl * ndl);
  let hemi = mix(0.6, 1.0, normal_ws.y * 0.5 + 0.5);
  return albedo * (sun + frame.ambient * hemi);
}

fn sky_horizon_color() -> vec3f {
  return vec3f(0.62, 0.72, 0.85);
}

fn apply_fog(color: vec3f, world_pos: vec3f) -> vec3f {
  let dist = distance(world_pos, frame.camera_pos);
  let f = 1.0 - exp(-dist * 0.004);
  return mix(color, sky_horizon_color(), f * f);
}
