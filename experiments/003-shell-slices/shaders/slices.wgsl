#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/hash.wgsl"
#include "src/wgsl/debug.wgsl"

// Axis-locked slice-stack volume impostors.
//
// Each plant is a baked voxel volume drawn as K planar slices. The slice
// axis is chosen per plant per frame as the local axis most aligned with
// the view direction: horizontal shells seen from above, vertical slice
// curtains at grazing angles — the shell-texturing grazing failure never
// happens because the stack rotates away from it. K collapses with
// distance (12 -> 4 -> 1); slab opacity is renormalized so fewer, thicker
// slabs stay equally dense, and a mip of the volume integrates the thicker
// slab so content does not fall between slices.

@group(1) @binding(0) var vox_color: texture_3d<f32>;
@group(1) @binding(1) var vox_aux: texture_3d<f32>;

struct SpeciesU {
  lmin: vec3f,
  voxel: f32,
  lsize: vec3f,
  mip_max: f32,
}
@group(1) @binding(2) var<uniform> sp: SpeciesU;

@group(2) @binding(0) var<storage, read> instances: array<vec4f>;

struct BandU {
  k: f32,
  alpha_thresh: f32,
  sigma: f32,
  axis_bias: f32,
  fade_start: f32,
  fade_end: f32,
  inside_near: f32,
  inside_far: f32,
  lod_bias: f32,
  debug_axis: f32,
  /** Meters per pixel at distance 1 (2·tan(fov/2)/heightPx). */
  mpp1: f32,
  pad1: f32,
}
@group(3) @binding(0) var<uniform> band: BandU;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uvw: vec3f,
  @location(1) world: vec3f,
  // x: alpha scale, y: slab optical depth (slabVox^0.7), z: mip level, w: axis id
  @location(2) @interpolate(flat) misc: vec4f,
  // cos/sin of plant yaw for rotating baked normals to world.
  @location(3) @interpolate(flat) rot: vec2f,
}

fn degenerate() -> VOut {
  var out: VOut;
  out.pos = vec4f(2.0, 2.0, 2.0, 1.0);
  out.uvw = vec3f(0.0);
  out.world = vec3f(0.0);
  out.misc = vec4f(0.0);
  out.rot = vec2f(1.0, 0.0);
  return out;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let k = u32(band.k);
  let plant = ii / k;
  let slice = ii % k;
  let d0 = instances[plant * 2u];
  let d1 = instances[plant * 2u + 1u];
  let base = d0.xyz;
  let yaw = d0.w;
  let scale = max(d1.x, 1e-4);
  let entry = stand_table[u32(d1.y)];
  let phase = d1.z;

  let cam = frame.camera_pos;
  let dist = distance(cam, base);

  // Outer region edge: cull whole plants stochastically so the field
  // dissolves into the far canopy shell instead of ending at a wall.
  if (band.fade_end > band.fade_start) {
    let survive = 1.0 - clamp((dist - band.fade_start) / (band.fade_end - band.fade_start), 0.0, 1.0);
    if (fract(phase * 2.3999634) > survive) {
      return degenerate();
    }
  }

  // Camera-inside-plant: dissolve by scaling slab alpha to zero.
  let inside = clamp((dist - band.inside_near) / (band.inside_far - band.inside_near), 0.0, 1.0);
  if (inside <= 0.0) {
    return degenerate();
  }

  let c = cos(yaw);
  let s = sin(yaw);

  // Camera position in plant-local (unrotated, unscaled) space.
  let rel = cam - base;
  let cam_l = vec3f(c * rel.x + s * rel.z, rel.y, -s * rel.x + c * rel.z) / scale;
  let centre = sp.lmin + sp.lsize * 0.5;
  let v = (cam_l - centre) / sp.lsize;
  let av = abs(v) * vec3f(1.0, band.axis_bias, 1.0);
  var axis = 0u;
  if (av.y >= av.x && av.y >= av.z) {
    axis = 1u;
  } else if (av.z > av.x) {
    axis = 2u;
  }

  var t = (f32(slice) + 0.5) / band.k;
  // Slab thickness in voxels along the chosen axis (scale-invariant).
  let slab_vox = max(sp.lsize[axis] / band.k / sp.voxel, 1.0);
  // Very fat Y-slabs (distant plants collapsing toward a single shell)
  // migrate toward canopy-top height: seen from above that is where the
  // visible surface lives, and it matches the far shell's top composite.
  if (axis == 1u) {
    let deep = clamp((slab_vox - 20.0) / 130.0, 0.0, 1.0);
    t = mix(t, 0.7, deep);
  }
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0),
  );
  let q = corners[vi];
  var local: vec3f;
  if (axis == 0u) {
    local = sp.lmin + vec3f(t, q.y, q.x) * sp.lsize;
  } else if (axis == 1u) {
    local = sp.lmin + vec3f(q.x, t, q.y) * sp.lsize;
  } else {
    local = sp.lmin + vec3f(q.x, q.y, t) * sp.lsize;
  }

  var out: VOut;
  out.uvw = (local - sp.lmin) / sp.lsize;
  var world = base + vec3f(c * local.x - s * local.z, local.y, s * local.x + c * local.z) * scale;
  let h_norm = clamp((local.y - sp.lmin.y) / sp.lsize.y, 0.0, 1.0);
  world += wind_sway(base, frame.time, entry.sway, phase) * (h_norm * sqrt(h_norm));

  // Mip level: the slab-integration level (thicker slabs -> deeper mip) but
  // never blurrier than the screen needs — close up we stay at mip 0 and let
  // the dithered alpha threshold stand in for slab integration.
  let lod_slab = log2(slab_vox);
  let lod_dist = log2(max((dist * band.mpp1) / (sp.voxel * scale), 1.0));
  let lod = clamp(min(lod_slab, lod_dist) + band.lod_bias, 0.0, sp.mip_max);

  out.world = world;
  out.pos = frame.view_proj * vec4f(world, 1.0);
  // pow(slab_vox, 0.7) is constant over the whole slice quad — resolve it here
  // (6 vertices) instead of in every fragment the quad covers.
  out.misc = vec4f(inside, pow(slab_vox, 0.7), lod, f32(axis));
  out.rot = vec2f(c, s);
  return out;
}

/**
 * vox_aux.rgb holds `mean_normal * occupancy`, offset-encoded so that 128 is
 * exactly zero (see buildAuxTexture in bake.ts). Linear filtering and mips of
 * that ARE the occupancy-weighted mean normal of the tap footprint — empty
 * voxels contribute a zero vector instead of poisoning the direction — so the
 * decode is a rescale plus a renormalize.
 */
fn decode_normal(e: vec3f) -> vec3f {
  let v = (e * 255.0 - 128.0) / 127.0;
  let l = length(v);
  // All-empty footprint: nothing to average, fall back to canopy-up.
  if (l < 1.0e-4) {
    return vec3f(0.0, 1.0, 0.0);
  }
  return v / l;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let s0 = textureSampleLevel(vox_color, linear_sampler, in.uvw, in.misc.z);
  let cov = s0.a;
  // Slab opacity: coverage integrated over the slab thickness this slice
  // stands in for. Sub-linear in the thickness (pow 0.7, folded in by the
  // vertex stage) — full linear integration turns fluffy low-coverage
  // regions into solid blobs.
  let a_eff = (1.0 - exp(-band.sigma * cov * in.misc.y)) * in.misc.x;
  // Dithered alpha test, keyed in plant volume space (camera-stable): fluffy
  // semi-transparent regions render as sparse pixels instead of solid blobs.
  let dq = vec3u(in.uvw * 512.0);
  let dither = hash_f32(hash3(dq.x, dq.y, dq.z));
  if (a_eff < band.alpha_thresh * (0.35 + 1.3 * dither)) {
    discard;
  }

  let albedo = s0.rgb / max(cov, 0.004);
  let s1 = textureSampleLevel(vox_aux, linear_sampler, in.uvw, in.misc.z);
  let nl = decode_normal(s1.rgb);
  let c = in.rot.x;
  let s = in.rot.y;
  var n = vec3f(c * nl.x - s * nl.z, nl.y, s * nl.x + c * nl.z);
  // Blades are two-sided: flip the mean normal into the camera hemisphere.
  let vdir = normalize(frame.camera_pos - in.world);
  if (dot(n, vdir) < 0.0) {
    n = -n;
  }
  n = normalize(mix(n, vec3f(0.0, 1.0, 0.0), 0.35));

  // Baked sky visibility = canopy self-occlusion (precomputed raycast),
  // plus high-frequency value dither so voxel fluff reads as texture.
  let occ = mix(0.42, 1.0, s1.a) * (0.8 + 0.4 * dither);
  // The occluded albedo is what the shared lighting model multiplies, so it
  // is also what the albedo/lighting debug views must be told about.
  let surf_albedo = albedo * occ;
  var color = light_surface(surf_albedo, n, in.world);

  // Fog and the experiment's own axis tint only in the normal view — debug
  // views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    if (band.debug_axis > 0.5) {
      let ax = u32(in.misc.w);
      var tint = vec3f(1.0, 0.35, 0.35);
      if (ax == 1u) {
        tint = vec3f(0.35, 1.0, 0.35);
      } else if (ax == 2u) {
        tint = vec3f(0.35, 0.45, 1.0);
      }
      color = mix(color, tint, 0.5);
    }
    color = apply_fog(color, in.world);
  }
  // Coverage = the slab opacity this fragment resolved to before the alpha
  // test, i.e. how much canopy this pixel actually stands for.
  return vec4f(debug_shade(color, surf_albedo, n, a_eff, in.world), 1.0);
}
