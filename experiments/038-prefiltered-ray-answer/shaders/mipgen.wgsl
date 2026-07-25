// Load-time mip chain for the ray-answer table — THE prefilter.
//
// Every channel is stored premultiplied by coverage, so a plain 2x2 box filter
// is already the honest coverage-weighted average: mip n holds the mean hit
// height, mean albedo, mean AO and true areal coverage of the canopy over a
// 2^n texel footprint of entry points. That is what makes a distant pixel one
// fetch of an averaged canopy instead of a sharp one it cannot resolve.
//
// One dispatch per (texture, level); z = array layer.

@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(dst);
  if (gid.x >= dims.x || gid.y >= dims.y) {
    return;
  }
  let smax = vec2i(textureDimensions(src)) - vec2i(1);
  let base = vec2i(vec2u(gid.xy)) * 2;
  let layer = i32(gid.z);
  var acc = vec4f(0.0);
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      acc += textureLoad(src, min(base + vec2i(i, j), smax), layer, 0);
    }
  }
  textureStore(dst, vec2i(vec2u(gid.xy)), layer, acc * 0.25);
}
