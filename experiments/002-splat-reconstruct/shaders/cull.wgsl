#include "src/wgsl/scatter.wgsl"
#include "./common.wgsl"

// Pass 1: procedural cull. One workgroup per cell of a fixed camera-centered
// window, one thread per candidate slot. The workgroup first rejects the whole
// cell against the frustum + fade distance (uniform branch, no scatter work at
// all for the ~80% of the window that cannot contribute), then evaluates
// scatter_candidate for every stand entry, frustum-tests it, picks a distance
// LOD and appends a 16B plant record into that (entry, lod) bucket. Work is
// bounded by the window size — never by the stand's total plant count.

@group(1) @binding(0) var<uniform> params: SrParams;
@group(1) @binding(1) var<storage, read_write> counts: array<atomic<u32>, 15>;
@group(1) @binding(2) var<storage, read_write> records0: array<PlantRec>;
@group(1) @binding(3) var<storage, read_write> records1: array<PlantRec>;
@group(1) @binding(4) var<storage, read_write> records2: array<PlantRec>;

@compute @workgroup_size(128)
fn cs_cull(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let cell = params.base_cell + vec2i(i32(wid.x), i32(wid.y));
  let slot = lid.x;

  // Frustum planes (Gribb–Hartmann) from the shared view-proj.
  let m = transpose(frame.view_proj);
  var planes: array<vec4f, 5>;
  planes[0] = m[3] + m[0];
  planes[1] = m[3] - m[0];
  planes[2] = m[3] + m[1];
  planes[3] = m[3] - m[1];
  planes[4] = m[2]; // near (WebGPU z in [0,1])

  // --- Whole-cell reject ---------------------------------------------------
  // Every plant this cell can ever produce lives inside one SCATTER_CELL_SIZE
  // column, bounded in y by the terrain height range (|fbm| <= 1.015) and
  // padded by cell_pad (tallest plant + sway + footprint margin). Testing that
  // one box per workgroup skips 128 x num_entries scatter evaluations for
  // every cell behind the camera / off to the side / past the fade distance —
  // i.e. for most of the square window, which circumscribes a fade_end disc
  // while only the frustum wedge inside it can contribute.
  let pad = params.cell_pad;
  let ty = frame.terrain_height_scale * 1.15;
  let box_lo = vec3f(f32(cell.x) * SCATTER_CELL_SIZE - pad, -ty - pad, f32(cell.y) * SCATTER_CELL_SIZE - pad);
  let box_hi = box_lo + vec3f(SCATTER_CELL_SIZE + 2.0 * pad, 2.0 * (ty + pad), SCATTER_CELL_SIZE + 2.0 * pad);
  let box_c = 0.5 * (box_lo + box_hi);
  let box_e = 0.5 * (box_hi - box_lo);
  if (length(max(abs(frame.camera_pos - box_c) - box_e, vec3f(0.0))) > params.fade_end) { return; }
  for (var p = 0u; p < 5u; p++) {
    let pl = planes[p];
    if (dot(pl.xyz, box_c) + pl.w < -dot(abs(pl.xyz), box_e)) { return; }
  }

  for (var e = 0u; e < params.num_entries; e++) {
    let cand = scatter_candidate(params.seed, e, cell, slot);
    if (!cand.exists) { continue; }
    let entry = stand_table[e];
    let h = entry.height_scale * cand.scale;
    let center = cand.pos + vec3f(0.0, 0.5 * h, 0.0);
    let dist = distance(frame.camera_pos, center);
    if (dist > params.fade_end) { continue; }

    // Camera-inside-plant fade in, far fade out.
    var fade = smoothstep(0.2, 0.7, dist) * (1.0 - smoothstep(params.fade_start, params.fade_end, dist));
    if (fade < 0.01) { continue; }

    let radius = max(h, 0.75 * cand.scale) + 0.35; // sway + tile footprint margin
    var visible = true;
    for (var p = 0u; p < 5u; p++) {
      let pl = planes[p];
      if (dot(pl.xyz, center) + pl.w < -radius * length(pl.xyz)) { visible = false; break; }
    }
    if (!visible) { continue; }

    // Distance LOD with per-plant jitter so ring transitions decorrelate.
    let dj = dist * (0.85 + 0.3 * fract(cand.phase * 0.63661977));
    var lod = 4u;
    if (dj < params.rings.x) { lod = 0u; }
    else if (dj < params.rings.y) { lod = 1u; }
    else if (dj < params.rings.z) { lod = 2u; }
    else if (dj < params.rings.w) { lod = 3u; }

    let idx = atomicAdd(&counts[e * 5u + lod], 1u);
    if (idx >= SR_BUCKET_CAPS[lod]) { continue; }

    var rec: PlantRec;
    rec.xz = cand.pos.xz;
    rec.ys = pack2x16float(vec2f(cand.yaw, cand.scale));
    let y16 = pack2x16float(vec2f(cand.pos.y, 0.0)) & 0xffffu;
    let phase8 = u32(round(fract(cand.phase * 0.15915494) * 255.0));
    let fade8 = u32(round(fade * 255.0));
    rec.pfy = y16 | (phase8 << 16u) | (fade8 << 24u);

    let at = SR_BUCKET_BASES[lod] + idx;
    switch e {
      case 0u: { records0[at] = rec; }
      case 1u: { records1[at] = rec; }
      default: { records2[at] = rec; }
    }
  }
}

// Pass 1b: clamp bucket counts into drawIndexedIndirect args.
// Slot i = entry*5 + lod, 5 u32 each:
//   [indexCount = 6 * splats(lod), instanceCount = min(count, cap),
//    firstIndex, baseVertex, firstInstance].

@group(1) @binding(5) var<storage, read> counts_ro: array<u32, 15>;
@group(1) @binding(6) var<storage, read_write> indirect: array<u32, 75>;

@compute @workgroup_size(16)
fn cs_finalize(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= 15u) { return; }
  let lod = i % 5u;
  let o = i * 5u;
  indirect[o] = 6u * SR_LOD_COUNTS[lod];
  indirect[o + 1u] = min(counts_ro[i], SR_BUCKET_CAPS[lod]);
  indirect[o + 2u] = 0u;
  indirect[o + 3u] = 0u;
  indirect[o + 4u] = 0u;
}
