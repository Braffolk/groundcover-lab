#include "src/wgsl/fullscreen.wgsl"
#include "src/wgsl/frame.wgsl"

// The material-preview backdrop: a neutral studio cyclorama, greys only (this
// lab's palette is five greys plus black and white). Anchored to the WORLD via
// the view ray rather than to the screen, so orbiting the object slides the
// object across the gradient instead of dragging the gradient with it.
//
// Deliberately NOT routed through debug_shade(): the backdrop is not a surface
// and has no albedo, normal or coverage to inspect. Leaving it constant across
// every debug mode is what keeps the object readable in all of them — the same
// choice src/scene/sky.wgsl makes for the sky.

const STUDIO_FLOOR: f32 = 0.086;   // #161616
const STUDIO_CEILING: f32 = 0.333; // #555555

@fragment
fn fs_main(in: FullscreenOut) -> @location(0) vec4f {
  let ndc = vec2f(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
  let far = frame.inv_view_proj * vec4f(ndc, 1.0, 1.0);
  let dir = normalize(far.xyz / far.w - frame.camera_pos);
  let up = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
  let g = mix(STUDIO_FLOOR, STUDIO_CEILING, smoothstep(0.12, 0.92, up));
  return vec4f(vec3f(g), 1.0);
}
