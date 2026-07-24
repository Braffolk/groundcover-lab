#include "src/wgsl/scatter.wgsl"
#include "./common.wgsl"

// Pass 1: procedural cull. One thread per (cell, slot) inside a fixed
// camera-centered cell window; evaluates scatter_candidate for every stand
// entry, frustum-tests it, picks a distance LOD and appends a 16B plant
// record into that (entry, lod) bucket. Work is bounded by the window size —
// never by the stand's total plant count.

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

// Pass 1b: clamp bucket counts into indirect draw args.
// Indirect slot i = entry*5 + lod: [6 * splats(lod), min(count, cap), 0, 0].

@group(1) @binding(5) var<storage, read> counts_ro: array<u32, 15>;
@group(1) @binding(6) var<storage, read_write> indirect: array<vec4u, 15>;

@compute @workgroup_size(16)
fn cs_finalize(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= 15u) { return; }
  let lod = i % 5u;
  indirect[i] = vec4u(6u * SR_LOD_COUNTS[lod], min(counts_ro[i], SR_BUCKET_CAPS[lod]), 0u, 0u);
}
