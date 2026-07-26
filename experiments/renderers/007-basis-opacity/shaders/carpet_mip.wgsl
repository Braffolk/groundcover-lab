// Mip chain for the carpet textures. A mat is a ground plane, so it is the one
// thing in this experiment that gets genuinely minified with distance — the
// upright cards never sample below level 0 because they shrink with the plant,
// but a tile 60 m away is a handful of pixels of a 512^2 texture and without a
// mip chain it aliases into sparkle.
//
// Every filter is COVERAGE-WEIGHTED by the albedo capture's alpha, so nothing
// bleeds in from the empty texels between capitula:
//   rgb_out = sum(rgb * w) / sum(w)
// The stored colour is therefore already normalised by coverage at every level
// — the runtime must NOT divide by alpha again (the classic premultiply trap).
//
// cs_albedo:  alpha_out = plain mean coverage (so alpha keeps meaning "how much
//             of this footprint is moss", which is what the alpha test needs).
// cs_data:    alpha_out = coverage-weighted mean (height / sky visibility are
//             per-surface quantities, meaningless where there is no surface).
//             Normals are stored as PLAIN xyz, not octahedral, precisely so this
//             box filter is valid (oct encoding is not mip-averageable), and the
//             horizon channels are linear fit coefficients, which likewise are.
//
// The weight floor keeps a fully empty footprint from dividing by zero; there
// the filter degenerates to the plain mean of the cleared values, which are
// deliberately the sane ones (up normal, zero height, unobstructed horizon).

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var wsrc: texture_2d<f32>; // coverage source (albedo level)
@group(0) @binding(2) var dst: texture_storage_2d<rgba8unorm, write>;

const W_FLOOR: f32 = 0.002;

struct Acc {
  rgb: vec3f,
  wa: f32,  // coverage-weighted alpha
  pa: f32,  // plain alpha
  w: f32,
}

fn accumulate(t: vec2u) -> Acc {
  var acc: Acc;
  for (var dy = 0u; dy < 2u; dy++) {
    for (var dx = 0u; dx < 2u; dx++) {
      let c = t + vec2u(dx, dy);
      let v = textureLoad(src, c, 0);
      let w = max(textureLoad(wsrc, c, 0).a, W_FLOOR);
      acc.rgb += v.rgb * w;
      acc.wa += v.a * w;
      acc.pa += v.a;
      acc.w += w;
    }
  }
  return acc;
}

@compute @workgroup_size(8, 8)
fn cs_albedo(@builtin(global_invocation_id) gid: vec3u) {
  let dim = textureDimensions(dst);
  if (gid.x >= dim.x || gid.y >= dim.y) { return; }
  let acc = accumulate(gid.xy * 2u);
  textureStore(dst, gid.xy, vec4f(acc.rgb / acc.w, acc.pa * 0.25));
}

@compute @workgroup_size(8, 8)
fn cs_data(@builtin(global_invocation_id) gid: vec3u) {
  let dim = textureDimensions(dst);
  if (gid.x >= dim.x || gid.y >= dim.y) { return; }
  let acc = accumulate(gid.xy * 2u);
  textureStore(dst, gid.xy, vec4f(acc.rgb / acc.w, acc.wa / acc.w));
}
