#include "src/wgsl/frame.wgsl"

// Standard debug views, driven by the ONE global selector in the frame
// uniform (runner/AB toolbar, URL `debug=`). Every renderer must route its
// final fragment colour through debug_shade() — a method you cannot inspect
// is a method nobody can debug or trust.
//
// Order must match DEBUG_VIEW_MODES in src/harness/debug.ts.

const DEBUG_OFF: u32 = 0u;
const DEBUG_ALBEDO: u32 = 1u;
const DEBUG_NORMALS: u32 = 2u;
const DEBUG_LIGHTING: u32 = 3u;
const DEBUG_COVERAGE: u32 = 4u;
const DEBUG_DEPTH: u32 = 5u;

fn debug_mode() -> u32 {
  return u32(frame.debug_mode + 0.5);
}

/// Route every fragment through this before returning it.
///   shaded    final lit colour you would otherwise output (pre-fog is fine)
///   albedo    base colour before lighting
///   normal_ws world-space shading normal (already flipped/normalised)
///   coverage  alpha/coverage this fragment resolved to (1.0 if opaque)
///   world_pos reconstructed world position of the fragment
///
/// DEBUG_LIGHTING divides the light term back out, which is exact because the
/// shared lighting model in lighting.wgsl is multiplicative in albedo.
fn debug_shade(shaded: vec3f, albedo: vec3f, normal_ws: vec3f, coverage: f32, world_pos: vec3f) -> vec3f {
  let mode = debug_mode();
  if (mode == DEBUG_OFF) {
    return shaded;
  }
  if (mode == DEBUG_ALBEDO) {
    return albedo;
  }
  if (mode == DEBUG_NORMALS) {
    return normalize(normal_ws) * 0.5 + 0.5;
  }
  if (mode == DEBUG_LIGHTING) {
    return shaded / max(albedo, vec3f(1.0e-3));
  }
  if (mode == DEBUG_COVERAGE) {
    return vec3f(clamp(coverage, 0.0, 1.0));
  }
  if (mode == DEBUG_DEPTH) {
    let d = distance(world_pos, frame.camera_pos);
    // Log-ish ramp for range + fine banding so reconstruction errors and
    // depth-precision cliffs are visible rather than smoothly hidden.
    let ramp = 1.0 - exp(-d * 0.03);
    let band = fract(d * 2.0) * 0.12;
    return vec3f(clamp(ramp + band, 0.0, 1.0));
  }
  return shaded;
}
