#include "src/wgsl/fullscreen.wgsl"

// Seeds the half-res splat depth buffer from the full-res scene depth
// (max of the 2x2 block = farthest, so groundcover is never over-occluded
// at terrain silhouettes; the composite re-tests at full res).

@group(1) @binding(0) var scene_depth: texture_depth_2d;

@fragment
fn fs_depth_init(in: FullscreenOut) -> @builtin(frag_depth) f32 {
  let dims = vec2i(textureDimensions(scene_depth));
  let p = vec2i(in.pos.xy) * 2;
  var d = 0.0;
  for (var dy = 0; dy < 2; dy++) {
    for (var dx = 0; dx < 2; dx++) {
      let c = clamp(p + vec2i(dx, dy), vec2i(0), dims - 1);
      d = max(d, textureLoad(scene_depth, c, 0));
    }
  }
  return d;
}
