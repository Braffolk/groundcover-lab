#include "src/wgsl/fullscreen.wgsl"
#include "src/wgsl/lighting.wgsl"

// Procedural sky, drawn after terrain at far depth (depthCompare less-equal).

@fragment
fn fs_main(in: FullscreenOut) -> @location(0) vec4f {
  let ndc = vec2f(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
  let far = frame.inv_view_proj * vec4f(ndc, 1.0, 1.0);
  let dir = normalize(far.xyz / far.w - frame.camera_pos);
  let up = clamp(dir.y, 0.0, 1.0);
  let zenith = vec3f(0.25, 0.42, 0.72);
  var color = mix(sky_horizon_color(), zenith, pow(up, 0.6));
  // Sun disc + glow.
  let cosSun = dot(dir, frame.sun_dir);
  color += frame.sun_color * (smoothstep(0.9995, 0.9999, cosSun) * 8.0 + pow(max(cosSun, 0.0), 64.0) * 0.12);
  // Below the horizon (visible only off the terrain edge) fade to haze.
  color = mix(color, sky_horizon_color(), smoothstep(0.0, -0.25, dir.y));
  return vec4f(color, 1.0);
}
