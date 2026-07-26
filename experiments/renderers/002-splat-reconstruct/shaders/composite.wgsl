#include "src/wgsl/fullscreen.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./common.wgsl"

// Pass 3: screen-space reconstruction. The half-res splat buffer is a sparse,
// dithered point-sample cloud; this pass turns it back into continuous
// foliage. For each full-res pixel it gathers a 3x3 half-res neighborhood,
// takes the nearest splat depth as the local surface, and accumulates
// color/normal/height with weights = spatial gaussian x depth affinity
// (surface_tol). Weighted coverage becomes a soft alpha (gap_fill closes the
// dither holes), and lighting + fog are applied ONCE per output pixel from
// the reconstructed normal and depth.
//
// This is also the single place the method produces colour, so it is where
// the global debug views (frame.debug_mode) are answered: albedo / normals /
// lighting / coverage / depth all come out of the reconstructed G-buffer.

@group(1) @binding(0) var<uniform> params: SrParams;
@group(1) @binding(1) var splat_color: texture_2d<f32>;
@group(1) @binding(2) var splat_aux: texture_2d<f32>;
@group(1) @binding(3) var scene_depth: texture_depth_2d;

fn view_forward() -> vec3f {
  return -vec3f(frame.view[0].z, frame.view[1].z, frame.view[2].z);
}

fn ray_dir(ndc: vec2f) -> vec3f {
  let far = frame.inv_view_proj * vec4f(ndc, 1.0, 1.0);
  return normalize(far.xyz / far.w - frame.camera_pos);
}

@fragment
fn fs_main(in: FullscreenOut) -> @location(0) vec4f {
  let dbg = debug_mode();
  // "No groundcover here": blending is premultiplied, so writing a transparent
  // pixel is a no-op that still costs a full-screen ROP read-modify-write —
  // discard instead. DEBUG_COVERAGE is the one mode that wants those pixels
  // written, as the zero end of the coverage map.
  let empty = select(vec4f(0.0), vec4f(0.0, 0.0, 0.0, 1.0), dbg == DEBUG_COVERAGE);

  let dims = vec2i(textureDimensions(splat_aux));
  let q = vec2i(in.pos.xy) / 2;

  var aux: array<vec4f, 9>;
  var col: array<vec4f, 9>;
  var offs: array<vec2i, 9>;
  var dmin = 1e9;
  var k = 0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let t = clamp(q + vec2i(dx, dy), vec2i(0), dims - 1);
      offs[k] = t;
      aux[k] = textureLoad(splat_aux, t, 0);
      col[k] = textureLoad(splat_color, t, 0);
      if (aux[k].w > 0.01 && aux[k].w < dmin) { dmin = aux[k].w; }
      k++;
    }
  }
  if (dmin > 1e8) {
    if (dbg != DEBUG_COVERAGE) { discard; }
    return empty;
  }

  let reconstruct = (params.flags & 1u) != 0u;
  var alpha: f32;
  var albedo: vec3f;
  var normal: vec3f;
  var hf: f32;
  var depth = dmin;

  if (reconstruct) {
    let sub = in.pos.xy * 0.5; // this pixel in half-res texel coords
    var acc_rgb = vec3f(0.0);
    var acc_n = vec3f(0.0);
    var acc_hf = 0.0;
    var acc_w = 0.0;
    var cov_any = 0.0;
    var tot_ws = 0.0;
    var acc_d = 0.0;
    for (var j = 0; j < 9; j++) {
      let d = aux[j].w;
      let o = vec2f(offs[j]) + 0.5 - sub;
      let ws = exp(-dot(o, o) * 0.55);
      tot_ws += ws;
      if (d <= 0.01) { continue; }
      // Alpha comes from coverage at ANY depth: a pixel wedged between a near
      // leaf and the field behind it must not open a sky-colored seam.
      cov_any += ws;
      let affinity = clamp(1.0 - (d - dmin) / params.surface_tol, 0.0, 1.0);
      let w = ws * affinity;
      acc_rgb += col[j].rgb * w;
      acc_n += aux[j].xyz * w;
      acc_hf += col[j].a * w;
      acc_d += d * w;
      acc_w += w;
    }
    let coverage = cov_any / max(tot_ws, 1e-4);
    let a = clamp(coverage * (1.0 + params.gap_fill * 1.3), 0.0, 1.0);
    // Coverage floor: an isolated stray sample never reaches mid alpha, so
    // silhouettes against sky do not sparkle.
    alpha = smoothstep(0.18, 0.78, a);
    albedo = acc_rgb / max(acc_w, 1e-4);
    normal = acc_n / max(acc_w, 1e-4);
    hf = acc_hf / max(acc_w, 1e-4);
    depth = acc_d / max(acc_w, 1e-4);
  } else {
    // Raw mode: nearest half-res sample, no filtering (for A/B-ing the idea).
    let a4 = aux[4];
    if (a4.w <= 0.01) {
      if (dbg != DEBUG_COVERAGE) { discard; }
      return empty;
    }
    alpha = 1.0;
    albedo = col[4].rgb;
    normal = a4.xyz;
    hf = col[4].a;
    depth = a4.w;
  }
  if (alpha < 0.004) {
    if (dbg != DEBUG_COVERAGE) { discard; }
    return empty;
  }

  let nl = length(normal);
  var n = vec3f(0.0, 1.0, 0.0);
  if (nl > 1e-3) { n = normal / nl; }

  // Reconstruct the world position from the surface depth.
  let uv = in.pos.xy / frame.viewport;
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y);
  let dir = ray_dir(ndc);
  let fwd = view_forward();
  let t_hit = depth / max(dot(dir, fwd), 1e-4);
  let world = frame.camera_pos + dir * t_hit;

  // Full-res occlusion re-test against the scene depth buffer.
  let ds = textureLoad(scene_depth, vec2i(in.pos.xy), 0);
  if (ds < 1.0) {
    let sw4 = frame.inv_view_proj * vec4f(ndc, ds, 1.0);
    let svz = dot(sw4.xyz / sw4.w - frame.camera_pos, fwd);
    alpha *= clamp(1.0 - (depth - svz - 0.06) / 0.25, 0.0, 1.0);
    if (alpha < 0.004) {
      if (dbg != DEBUG_COVERAGE) { discard; }
      return empty;
    }
  }

  // Height-fraction ambient occlusion: an occlusion term, so it belongs in the
  // lighting view, not the albedo view.
  let ao = mix(0.5, 1.0, pow(hf, 0.7));
  var color = light_surface(albedo * ao, n, world);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (dbg == DEBUG_OFF) {
    color = apply_fog(color, world);
  }
  color = debug_shade(color, albedo, n, alpha, world);
  // Coverage is drawn opaque: this pass is the only groundcover contribution,
  // so blending a grey coverage value over the terrain's white (coverage = 1)
  // would wash the map out into uniform white.
  if (dbg == DEBUG_COVERAGE) { return vec4f(color, 1.0); }
  return vec4f(color * alpha, alpha);
}
