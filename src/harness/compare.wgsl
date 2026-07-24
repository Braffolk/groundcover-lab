#include "src/wgsl/fullscreen.wgsl"

// A/B composite: wipe | flicker | diff heatmap.

@group(0) @binding(0) var tex_a: texture_2d<f32>;
@group(0) @binding(1) var tex_b: texture_2d<f32>;
@group(0) @binding(2) var s: sampler;

struct U {
  mode: f32,   // 0 = wipe, 1 = flicker, 2 = diff
  split: f32,  // wipe divider in [0, 1]
  side: f32,   // flicker: 0 shows A, 1 shows B
  gain: f32,   // diff amplification
}
@group(0) @binding(3) var<uniform> u: U;

fn heat(x: f32) -> vec3f {
  // black -> blue -> yellow -> red
  let a = clamp(x * 3.0, 0.0, 1.0);
  let b = clamp(x * 3.0 - 1.0, 0.0, 1.0);
  let c = clamp(x * 3.0 - 2.0, 0.0, 1.0);
  return mix(mix(mix(vec3f(0.0), vec3f(0.1, 0.2, 0.9), a), vec3f(0.95, 0.9, 0.2), b), vec3f(1.0, 0.15, 0.1), c);
}

@fragment
fn fs_main(in: FullscreenOut) -> @location(0) vec4f {
  let a = textureSample(tex_a, s, in.uv).rgb;
  let b = textureSample(tex_b, s, in.uv).rgb;
  if (u.mode < 0.5) {
    let dims = vec2f(textureDimensions(tex_a));
    if (abs(in.uv.x - u.split) * dims.x < 1.0) {
      return vec4f(1.0, 1.0, 1.0, 1.0);
    }
    return vec4f(select(b, a, in.uv.x < u.split), 1.0);
  }
  if (u.mode < 1.5) {
    return vec4f(select(a, b, u.side > 0.5), 1.0);
  }
  let d = abs(a - b);
  let m = max(d.r, max(d.g, d.b));
  return vec4f(heat(clamp(m * u.gain, 0.0, 1.0)), 1.0);
}
