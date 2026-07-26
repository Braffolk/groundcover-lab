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
  /** Carpet-only HARD alpha reference (no dither — see fs_main). */
  carpet_alpha: f32,
  /** Carpet-only sky-visibility floor: how dark a fully buried voxel gets. */
  carpet_occ: f32,
  pad1: f32,
  pad2: f32,
  pad3: f32,
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
  // 1 = this instance belongs to a carpet entry (stand_table.carpet_div > 0).
  @location(4) @interpolate(flat) carpet: f32,
  // Terrain normal xz under this vertex, pre-scaled by slope_align — free, it
  // comes out of the same bilinear fetch the conforming displacement needs.
  @location(5) tnormal: vec2f,
}

fn degenerate() -> VOut {
  var out: VOut;
  out.pos = vec4f(2.0, 2.0, 2.0, 1.0);
  out.uvw = vec3f(0.0);
  out.world = vec3f(0.0);
  out.misc = vec4f(0.0);
  out.rot = vec2f(1.0, 0.0);
  out.carpet = 0.0;
  out.tnormal = vec2f(0.0);
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
  let entry = stand_table[u32(d1.y)];
  let phase = d1.z;
  // A carpet is a tiled MAT: one grid spacing, 90-degree-only yaw, and ONE
  // constant scale for every tile of the species. stand_table.scale_min carries
  // that constant (grid step / periodic tile size). The per-instance scale in
  // the buffer must NOT be used here: the CPU scatter mirror hands back the
  // stand's placeholder scaleMin for carpet entries (1.4-1.7) instead of the
  // computed carpet scale (1.01), which drew every moss tile ~1.7x life size
  // AND at three different sizes for the three states — see NOTES.md.
  let carpet = entry.carpet_div > 0.0;
  let scale = select(max(d1.x, 1e-4), max(entry.scale_min, 1e-4), carpet);

  let cam = frame.camera_pos;
  let dist = distance(cam, base);

  // Outer region edge: cull whole plants stochastically so the field
  // dissolves into the far canopy shell instead of ending at a wall. Never for
  // a carpet — a mat must not develop tile-sized holes; the far shell simply
  // continues underneath it where the region ends.
  if (band.fade_end > band.fade_start && !carpet) {
    let survive = 1.0 - clamp((dist - band.fade_start) / (band.fade_end - band.fade_start), 0.0, 1.0);
    if (fract(phase * 2.3999634) > survive) {
      return degenerate();
    }
  }

  // Camera-inside-plant: dissolve by scaling slab alpha to zero. Again never
  // for a carpet: fading the mat you are standing on opens a hole under you.
  var inside = 1.0;
  if (!carpet) {
    inside = clamp((dist - band.inside_near) / (band.inside_far - band.inside_near), 0.0, 1.0);
    if (inside <= 0.0) {
      return degenerate();
    }
  }

  let c = cos(yaw);
  let s = sin(yaw);

  // A CARPET is locked to horizontal shells. The axis flip exists because a
  // single plant's horizontal shells collapse edge-on at grazing — but a mat's
  // tiles TILE THE PLANE, so the union of one horizontal slab per tile is a
  // continuous canopy sheet that only foreshortens at grazing, and any hole in
  // one tile's slab is filled by the tile behind it. The flip actively hurts
  // here: a vertical curtain cut through the middle of a 9cm cushion paints the
  // cushion's dark INTERIOR, which is what turned everything past ~30m into a
  // black band (the far field is where a mat is most nearly edge-on, so the
  // flip triggered exactly where it was most damaging).
  var axis = 1u;
  if (!carpet) {
    // Camera position in plant-local (unrotated, unscaled) space: which face of
    // the volume box we see most of. Only the upright path needs it.
    let rel = cam - base;
    let cam_l = vec3f(c * rel.x + s * rel.z, rel.y, -s * rel.x + c * rel.z) / scale;
    let centre = sp.lmin + sp.lsize * 0.5;
    let v = (cam_l - centre) / sp.lsize;
    let av = abs(v) * vec3f(1.0, band.axis_bias, 1.0);
    axis = 0u;
    if (av.y >= av.x && av.y >= av.z) {
      axis = 1u;
    } else if (av.z > av.x) {
      axis = 2u;
    }
  }

  var t = (f32(slice) + 0.5) / band.k;
  // Slab thickness in voxels along the chosen axis (scale-invariant).
  let slab_vox = max(sp.lsize[axis] / band.k / sp.voxel, 1.0);
  // Very fat Y-slabs (distant plants collapsing toward a single shell)
  // migrate toward canopy-top height: seen from above that is where the
  // visible surface lives, and it matches the far shell's top composite.
  //
  // A MAT needs that migration far earlier and far harder. Its Y extent is only
  // 9cm, so a 4-slice band already integrates 2.3cm of cushion and the single
  // far slab integrates all of it — and the middle of a Sphagnum cushion is dark
  // bronze pendent foliage, not the olive capitula you actually see from
  // outside. Un-migrated, band 2 painted a near-black ring around the camera
  // from ~38m out to the region edge, right where a mat covers most of the
  // screen.
  if (axis == 1u) {
    let knee = select(20.0, 5.0, carpet);
    let span = select(130.0, 18.0, carpet);
    let top_h = select(0.7, 0.9, carpet);
    let deep = clamp((slab_vox - knee) / span, 0.0, 1.0);
    t = mix(t, top_h, deep);
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

  // --- Fitting the volume to the ground: ladder rung 3, per vertex ----------
  // Every slice vertex is displaced vertically by the terrain height under
  // ITSELF, relative to the tile's own base height, scaled by the species'
  // slope_align. That is a vertical shear of the volume, and it is a pure
  // function of world xz — so two neighbouring tiles agree wherever they meet
  // and the mat stays continuous. A per-tile plane fit (rungs 1-2) cannot: two
  // neighbours fit two different planes and crack apart at the shared edge.
  // Rendered bolt upright instead, a 0.18m tile turns the mat into a staircase
  // of flat-topped boxes whose step height (up to 0.1m on the bog's 30 deg
  // flanks) is the same order as the 0.09m cushion itself.
  // The heightmap texel is 0.5m, i.e. coarser than a tile, so four corners
  // capture the local ground almost exactly.
  // terrain_sample() gives height AND (nx, nz) from one bilinear fetch, so the
  // shading tilt below costs nothing on top of the displacement.
  var tnormal = vec2f(0.0);
  if (entry.slope_align > 0.0) {
    let ground = terrain_sample(world.xz);
    world.y += (ground.x - base.y) * entry.slope_align;
    tnormal = ground.yz * entry.slope_align;
  }

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
  out.carpet = select(0.0, 1.0, carpet);
  out.tnormal = tnormal;
  return out;
}

/**
 * vox_aux.rgb holds `mean_normal * occupancy`, offset-encoded so that 128 is
 * exactly zero (see buildAuxTexture in bake.ts). Linear filtering and mips of
 * that ARE the occupancy-weighted mean normal of the tap footprint — empty
 * voxels contribute a zero vector instead of poisoning the direction — so the
 * decode is a rescale plus a renormalize.
 */
fn decode_normal(e: vec3f, dead_zone: f32) -> vec3f {
  let v = (e * 255.0 - 128.0) / 127.0;
  let l = length(v);
  // A short mean vector means the footprint has no dominant orientation — an
  // empty region, or (for a moss cushion at a deep mip) leaves facing every way
  // and cancelling. Below a few counts of 8-bit precision the DIRECTION is
  // quantization noise, and normalizing it manufactures a confident unit normal
  // out of rounding error. Carpets pass a real dead zone (~6 counts); the
  // upright path keeps its exact-zero-only guard so the `default` stand is
  // bit-identical.
  if (l < dead_zone) {
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
  let is_carpet = in.carpet > 0.5;
  // Dithered alpha test, keyed in plant volume space (camera-stable): fluffy
  // semi-transparent regions render as sparse pixels instead of solid blobs.
  let dq = vec3u(in.uvw * 512.0);
  let dither = hash_f32(hash3(dq.x, dq.y, dq.z));
  // A CARPET gets a hard edge and a low reference instead. A cushion mat is a
  // closed, near-solid surface: stochastic coverage only fuzzes its silhouette,
  // screen-doors the mid field, and — worse — punches holes in the depth buffer
  // so early-z stops rejecting the slices underneath, exactly where a mat is
  // deepest. Hard alpha keeps it a solid depth-writing occluder.
  let thresh = select(band.alpha_thresh * (0.35 + 1.3 * dither), band.carpet_alpha, is_carpet);
  if (a_eff < thresh) {
    discard;
  }

  let albedo = s0.rgb / max(cov, 0.004);
  let s1 = textureSampleLevel(vox_aux, linear_sampler, in.uvw, in.misc.z);
  let nl = decode_normal(s1.rgb, select(1.0e-4, 0.05, is_carpet));
  let c = in.rot.x;
  let s = in.rot.y;
  var n = vec3f(c * nl.x - s * nl.z, nl.y, s * nl.x + c * nl.z);
  if (is_carpet) {
    // A mat is a SURFACE, not a two-sided blade, so its mean normal belongs in
    // the UPPER hemisphere unconditionally. Flipping it into the camera
    // hemisphere instead (below) makes every fragment above eye level light with
    // a DOWNWARD normal — and dot(up, view) changes sign exactly where the
    // ground crosses the horizon, so that painted a hard black ring around the
    // camera at grazing, from ~38m out to the region edge.
    if (n.y < 0.0) {
      n = -n;
    }
    // slope_align = 1 means the mat lies IN the ground plane, so its baked
    // normals have to be lifted into the terrain's frame; otherwise a cushion on
    // a 30-degree flank lights exactly like a flat one.
    let up = normalize(vec3f(in.tnormal.x, sqrt(max(1.0 - dot(in.tnormal, in.tnormal), 0.0)), in.tnormal.y));
    n = normalize(mix(plant_basis_from_up(up, 0.0) * n, up, 0.35));
  } else {
    // Blades are two-sided: flip the mean normal into the camera hemisphere.
    let vdir = normalize(frame.camera_pos - in.world);
    if (dot(n, vdir) < 0.0) {
      n = -n;
    }
    n = normalize(mix(n, vec3f(0.0, 1.0, 0.0), 0.35));
  }

  // Baked sky visibility = canopy self-occlusion (precomputed raycast),
  // plus high-frequency value dither so voxel fluff reads as texture. For a
  // carpet the value dither is off (it reads as noise on a continuous surface,
  // which has its own texture already) and the floor is a separate dial: the
  // visible surface of a mat is quantized to the nearest slice, so the sky
  // visibility sampled there belongs a few millimetres INSIDE the cushion and
  // systematically under-reads.
  let occ_floor = select(0.42, band.carpet_occ, is_carpet);
  let occ = mix(occ_floor, 1.0, s1.a) * select(0.8 + 0.4 * dither, 1.0, is_carpet);
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
