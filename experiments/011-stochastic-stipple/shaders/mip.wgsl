// Mip downsample blit: each level is the linear-filtered half-resolution of
// the previous. On premultiplied atlases this makes every mip texel the exact
// 2^k x 2^k coverage/mean-color statistic of its footprint — the data the
// hashed alpha test realizes at runtime.

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  // Fullscreen triangle.
  let x = f32(i32(vi & 1u) * 4 - 1);
  let y = f32(i32(vi >> 1u) * 4 - 1);
  var out: VOut;
  out.pos = vec4f(x, y, 0.0, 1.0);
  out.uv = vec2f(x * 0.5 + 0.5, 0.5 - y * 0.5);
  return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  return textureSampleLevel(src_tex, src_samp, in.uv, 0.0);
}
