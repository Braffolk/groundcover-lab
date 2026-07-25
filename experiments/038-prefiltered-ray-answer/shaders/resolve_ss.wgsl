// BAKE STAGE 2b — coverage-weighted 2x2 resolve of one composited slice into
// the packed answer table, and the ONE place premultiplication happens:
//
//   surf = (albedo * cov, ao * cov)
//   geom = (height01 * cov, oct' * cov, cov)
//
// Everything the runtime reads is therefore premultiplied by coverage, so
// bilinear filtering and the whole mip chain are plain box filters that stay
// honest at silhouettes (no dilation, no background bleed) — the runtime
// divides by geom.a once.

@group(0) @binding(0) var ss_surf: texture_2d<f32>;
@group(0) @binding(1) var ss_geom: texture_2d<f32>;
@group(0) @binding(2) var out_surf: texture_storage_2d_array<rgba8unorm, write>;
@group(0) @binding(3) var out_geom: texture_storage_2d_array<rgba8unorm, write>;

struct Cfg {
  layer: u32,
  res: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(4) var<uniform> cfg: Cfg;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= cfg.res || gid.y >= cfg.res) {
    return;
  }
  let base = vec2i(vec2u(gid.xy)) * 2;
  var w_sum = 0.0;
  var s_sum = vec4f(0.0);
  var g_sum = vec3f(0.0);
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      let c = base + vec2i(i, j);
      let g = textureLoad(ss_geom, c, 0);
      let w = g.a;
      w_sum += w;
      g_sum += g.rgb * w;
      s_sum += textureLoad(ss_surf, c, 0) * w;
    }
  }
  let layer = i32(cfg.layer);
  textureStore(out_surf, vec2i(vec2u(gid.xy)), layer, s_sum * 0.25);
  textureStore(out_geom, vec2i(vec2u(gid.xy)), layer, vec4f(g_sum * 0.25, w_sum * 0.25));
}
