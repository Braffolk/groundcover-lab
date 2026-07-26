#include "./rayfield-common.wgsl"

// Resolves one supersampled ortho view (SS x SS rays per LUT texel) into its
// RF_SLAB x RF_SLAB cell of the atlas. Everything except coverage is stored
// PREMULTIPLIED by coverage, so runtime bilinear filtering can un-premultiply
// (divide by filtered coverage) and get coverage-weighted answers with no
// dark halo from empty texels.

struct DsParams {
  dst_org: vec2u,
  ss: u32,
  _pad: u32,
}
@group(0) @binding(0) var src_surf: texture_2d<f32>;
@group(0) @binding(1) var src_geom: texture_2d<f32>;
@group(0) @binding(2) var dst_surf: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var dst_geom: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> dp: DsParams;

@compute @workgroup_size(8, 8)
fn cs_downsample(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= u32(RF_SLAB) || gid.y >= u32(RF_SLAB)) {
    return;
  }
  var cov = 0.0;
  var color = vec3f(0.0);
  var depth = 0.0;
  var nrm = vec3f(0.0);
  var height = 0.0;
  for (var y = 0u; y < dp.ss; y++) {
    for (var x = 0u; x < dp.ss; x++) {
      let t = vec2u(gid.xy * dp.ss + vec2u(x, y));
      let s = textureLoad(src_surf, t, 0);
      if (s.a > 0.5) {
        let g = textureLoad(src_geom, t, 0);
        cov += 1.0;
        color += s.rgb;
        depth += g.r;
        nrm += rf_oct_decode(g.gb);
        height += g.a;
      }
    }
  }
  let n_tot = f32(dp.ss * dp.ss);
  var out_s = vec4f(0.0);
  var out_g = vec4f(0.0);
  if (cov > 0.0) {
    let c01 = cov / n_tot;
    out_s = vec4f(color / n_tot, c01);
    let enc = rf_oct_encode(normalize(nrm + vec3f(0.0, 1e-4, 0.0)));
    out_g = vec4f(depth / n_tot, enc * c01, height / n_tot);
  }
  textureStore(dst_surf, dp.dst_org + gid.xy, out_s);
  textureStore(dst_geom, dp.dst_org + gid.xy, out_g);
}
