#include "src/wgsl/scatter.wgsl"

// Pass 2 of the bounded enumeration: one workgroup per visible cell (indirect
// dispatch from pass 1), 128 threads sweeping this entry's scatter slots in
// chunks. Surviving plants are classified by SCREEN SIZE into five LOD buckets
// and compacted into that bucket's instance region; the bucket's instanceCount
// lives directly in the drawIndexedIndirect args, so the draw is issued without
// any CPU readback.
//
// A CARPET entry has carpet_div² slots per cell (484 for the bog moss) — over
// the 128-slot scatter budget — so the slot count comes from the host and the
// workgroup loops. Visiting only 128 slots rendered a quarter of the mat, in
// 4m-periodic stripes that looked exactly like a placement bug.
//
// The LOD metric is the plant's projected screen extent, not distance: looking
// straight down at a tuft it is the XZ footprint that is on screen, not the
// height, so a top-down camera automatically lands on the cheap buckets.

const BUCKETS: u32 = 5u;
const SLOT_THREADS: u32 = 128u;

struct ExpandInfo {
  // x: stand entry index, y: seed, z: region radius, w: px-per-metre at 1 m
  cfg0: vec4f,
  // x: plant radius/height, y: far-fade start (m), z,w: inside-plant fade band
  cfg1: vec4f,
  // screen-size thresholds for buckets 0..3 (descending, px)
  px: vec4f,
  // instance capacity of buckets 0..3
  caps: vec4f,
  // x: capacity of bucket 4, y: morph span factor, z: cell list capacity,
  // w: variant count
  cfg2: vec4f,
  // vec4-unit base offset of buckets 0..3 inside the instance buffer
  bases: vec4f,
  // x: base offset of bucket 4
  bases4: vec4f,
  // x: scatter slots per cell for this entry, yzw: morph gate of buckets 0..2
  cfg3: vec4f,
}

@group(1) @binding(0) var<uniform> expand_info: ExpandInfo;
@group(1) @binding(1) var<storage, read> cell_list: array<vec2i>;
@group(1) @binding(2) var<storage, read_write> instances: array<vec4f>;
@group(1) @binding(3) var<storage, read_write> draw_args: array<atomic<u32>>;

var<workgroup> wg_count: array<atomic<u32>, BUCKETS>;
var<workgroup> wg_base: array<u32, BUCKETS>;

fn bucket_cap(b: u32) -> u32 {
  if (b < 4u) {
    return u32(expand_info.caps[b]);
  }
  return u32(expand_info.cfg2.x);
}

fn bucket_base(b: u32) -> u32 {
  if (b < 4u) {
    return u32(expand_info.bases[b]);
  }
  return u32(expand_info.bases4.x);
}

/// Morph fraction inside a bucket: 0 well above its threshold, exactly 1 at it,
/// where ribbon pairs have converged onto their parent and the next (coarser)
/// bucket takes over with identical geometry.
fn morph_at(px: f32, threshold: f32, span: f32) -> f32 {
  return clamp((threshold * span - px) / max(threshold * (span - 1.0), 1.0e-4), 0.0, 1.0);
}

@compute @workgroup_size(128)
fn cs_expand(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
  let cell_valid = wid.x < u32(expand_info.cfg2.z);
  let cell = cell_list[select(0u, wid.x, cell_valid)];
  let entry_index = u32(expand_info.cfg0.x);
  let seed = u32(expand_info.cfg0.y);
  let e = stand_table[entry_index];
  let is_carpet = e.carpet_div > 0.0;
  let region = expand_info.cfg0.z;
  // Uniform across the workgroup: safe to carry barriers inside the loop.
  let chunks = (u32(expand_info.cfg3.x) + SLOT_THREADS - 1u) / SLOT_THREADS;

  for (var chunk = 0u; chunk < chunks; chunk = chunk + 1u) {
    if (li < BUCKETS) {
      atomicStore(&wg_count[li], 0u);
    }
    workgroupBarrier();

    var bucket = BUCKETS;
    var rec0 = vec4f(0.0);
    var rec1 = vec4f(0.0);
    let slot_index = chunk * SLOT_THREADS + li;

    if (cell_valid) {
      let pt = scatter_candidate(seed, entry_index, cell, slot_index);
      if (pt.exists) {
        let plant_h = e.height_scale * pt.scale;
        let to_cam = pt.pos + vec3f(0.0, plant_h * 0.5, 0.0) - frame.camera_pos;
        let dist = length(to_cam);
        if (dist < region) {
          // Projected extent: height foreshortens as the view goes overhead, the
          // XZ footprint never does.
          let cos_elev = length(to_cam.xz) / max(dist, 1.0e-4);
          let extent = plant_h * cos_elev + 2.0 * expand_info.cfg1.x * plant_h;
          let px = expand_info.cfg0.w * extent / max(dist, 0.05);

          let span = expand_info.cfg2.y;
          var lvl = 4u;
          var morph = 0.0;
          if (px >= expand_info.px.x) {
            lvl = 0u;
            morph = morph_at(px, expand_info.px.x, span) * expand_info.cfg3.y;
          } else if (px >= expand_info.px.y) {
            lvl = 1u;
            morph = morph_at(px, expand_info.px.y, span) * expand_info.cfg3.z;
          } else if (px >= expand_info.px.z) {
            lvl = 2u;
            morph = morph_at(px, expand_info.px.z, span) * expand_info.cfg3.w;
          } else if (px >= expand_info.px.w) {
            lvl = 3u;
          }

          // Camera-inside-plant: fade by collapsing ribbon width, never by
          // stochastic alpha. Distance is to the plant's axis SEGMENT. A CARPET
          // never fades: a mat you are standing on must not open a hole under
          // you, and its region edge is a hard end rather than a dissolve
          // (collapsing the width of a closed mat would punch a lattice of
          // holes through it).
          var fade = 1.0;
          if (!is_carpet) {
            let rel = frame.camera_pos - pt.pos;
            let ty = clamp(rel.y, 0.0, plant_h);
            let axis_dist = length(vec3f(rel.x, rel.y - ty, rel.z));
            let near_fade = smoothstep(expand_info.cfg1.z, expand_info.cfg1.w, axis_dist);
            // Region edge: honest coverage falloff into the (grass-coloured) ground.
            let far_fade = 1.0 - smoothstep(expand_info.cfg1.y, region, dist);
            fade = near_fade * far_fade;
          }

          if (fade > 0.004) {
            bucket = lvl;
            let vh = hash_f32(hash4(seed ^ 0x9e3779b9u, bitcast<u32>(cell.x), bitcast<u32>(cell.y),
                                    (entry_index << 20u) ^ slot_index ^ 0x5bd1e995u));
            let variant = min(floor(vh * expand_info.cfg2.w), expand_info.cfg2.w - 1.0);
            rec0 = vec4f(pt.pos, pt.yaw);
            rec1 = vec4f(pt.scale, pt.phase, variant + min(morph, 0.999), fade);
          }
        }
      }
    }

    // Workgroup-aggregated compaction: one global atomic per bucket per
    // workgroup instead of one per plant.
    var local = 0u;
    if (bucket < BUCKETS) {
      local = atomicAdd(&wg_count[bucket], 1u);
    }
    workgroupBarrier();
    if (li < BUCKETS) {
      let n = atomicLoad(&wg_count[li]);
      var base = 0u;
      if (n > 0u) {
        base = atomicAdd(&draw_args[li * 5u + 1u], n);
      }
      wg_base[li] = base;
    }
    workgroupBarrier();

    if (bucket < BUCKETS) {
      let slot = wg_base[bucket] + local;
      if (slot < bucket_cap(bucket)) {
        let o = bucket_base(bucket) + slot * 2u;
        instances[o] = rec0;
        instances[o + 1u] = rec1;
      }
    }
  }
}
