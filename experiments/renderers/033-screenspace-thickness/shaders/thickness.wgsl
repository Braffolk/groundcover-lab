#include "src/wgsl/fullscreen.wgsl"
#include "src/wgsl/frame.wgsl"
#include "src/wgsl/debug.wgsl"

// Screen-space canopy thickness — ONE extra pass.
//
// The shell pass wrote a canopy weight per visible fragment (fog-attenuated
// coverage x baked thickness) into an r8 target. This pass estimates the local
// canopy density from a 12-tap ring of that mask and darkens by it with a
// MULTIPLY blend, so it needs no copy of the colour target and no mip chain:
//   canopy pixels  -> deep inside the mass reads dark, thin tips stay bright
//   ground pixels  -> contact darkening where plants crowd the soil
//   sky            -> untouched (depth == 1 rejects it)
// Debug views bypass it entirely (factor 1) so they stay honest.

struct SsParams {
  strength: f32,
  ground: f32,
  /** Ring radii in pixels (inner, outer). */
  r_inner: f32,
  r_outer: f32,
}

@group(1) @binding(0) var<uniform> ss: SsParams;
@group(1) @binding(1) var canopy_tex: texture_2d<f32>;
@group(1) @binding(2) var canopy_sampler: sampler;
@group(1) @binding(3) var scene_depth: texture_depth_2d;

// Two rings of 6, rotated against each other — cheap, isotropic enough, and
// every tap is bilinear so it already averages 4 texels.
const RING: array<vec2f, 6> = array<vec2f, 6>(
  vec2f(1.0, 0.0),
  vec2f(0.5, 0.866),
  vec2f(-0.5, 0.866),
  vec2f(-1.0, 0.0),
  vec2f(-0.5, -0.866),
  vec2f(0.5, -0.866),
);

@fragment
fn fs_composite(in: FullscreenOut) -> @location(0) vec4f {
  if (debug_mode() != DEBUG_OFF) {
    return vec4f(1.0);
  }
  let px = vec2i(in.pos.xy);
  if (textureLoad(scene_depth, px, 0) >= 1.0) {
    return vec4f(1.0); // sky
  }
  let texel = 1.0 / vec2f(textureDimensions(canopy_tex, 0));
  let here = textureSampleLevel(canopy_tex, canopy_sampler, in.uv, 0.0).r;
  var acc = here;
  for (var i = 0u; i < 6u; i++) {
    let d = RING[i];
    acc += textureSampleLevel(canopy_tex, canopy_sampler, in.uv + d * texel * ss.r_inner, 0.0).r;
    acc += textureSampleLevel(canopy_tex, canopy_sampler, in.uv - d.yx * texel * ss.r_outer, 0.0).r;
  }
  let occl = clamp(acc * (1.0 / 13.0), 0.0, 1.0);
  // Canopy fragments take the full term; bare soil between plants takes the
  // ground share of it, which is exactly the contact shadow.
  let gate = mix(ss.ground, 1.0, clamp(here * 3.0, 0.0, 1.0));
  let f = 1.0 - ss.strength * gate * occl;
  return vec4f(vec3f(clamp(f, 0.25, 1.0)), 1.0);
}
