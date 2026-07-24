#include "src/wgsl/fullscreen.wgsl"

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

@fragment
fn fs_main(in: FullscreenOut) -> @location(0) vec4f {
  return textureSample(src_tex, src_sampler, in.uv);
}
