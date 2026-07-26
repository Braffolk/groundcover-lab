// 3D mip generation for the ray-answer slabs, run once at load.
//
// A 3D mip halves the azimuth axis along with the two spatial axes, so distant
// grass is averaged over direction as well as area. That is deliberate: by the
// distance at which those levels are selected, one pixel spans far more than
// one azimuth cell's worth of parallax. Everything is premultiplied by
// coverage, so this plain box filter IS the coverage-weighted prefilter.

@group(0) @binding(0) var mip_src: texture_3d<f32>;
@group(0) @binding(1) var dst_unorm: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(2) var dst_snorm: texture_storage_3d<rgba8snorm, write>;

fn box8(base: vec3i, sdims: vec3i) -> vec4f {
  var acc = vec4f(0.0);
  for (var k = 0; k < 2; k++) {
    for (var j = 0; j < 2; j++) {
      for (var i = 0; i < 2; i++) {
        acc += textureLoad(mip_src, min(base + vec3i(i, j, k), sdims - vec3i(1)), 0);
      }
    }
  }
  return acc * 0.125;
}

@compute @workgroup_size(4, 4, 4)
fn mip_unorm(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(dst_unorm);
  if (any(gid >= dims)) { return; }
  textureStore(dst_unorm, vec3i(gid), box8(vec3i(gid) * 2, vec3i(textureDimensions(mip_src))));
}

@compute @workgroup_size(4, 4, 4)
fn mip_snorm(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(dst_snorm);
  if (any(gid >= dims)) { return; }
  textureStore(dst_snorm, vec3i(gid), box8(vec3i(gid) * 2, vec3i(textureDimensions(mip_src))));
}
