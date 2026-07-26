#include "src/wgsl/scatter.wgsl"
#include "./common.wgsl"

// Pass 1: procedural cull. ONE DISPATCH PER STAND ENTRY: (cell.x, cell.z, slot
// group) over a fixed camera-centered window, one thread per candidate slot.
// The workgroup first rejects the whole cell against the frustum + fade
// distance (uniform branch, no scatter work at all for the majority of the
// window that cannot contribute), then evaluates scatter_candidate for its
// entry, frustum-tests it, picks a distance LOD and appends a 16B plant record
// into that (entry, lod) bucket. Work is bounded by the window size — never by
// the stand's total plant count.
//
// Per-entry dispatch, rather than a loop over entries inside one dispatch, for
// two reasons: (1) stands have any number of entries (the bog has five) and a
// fixed set of record bindings does not scale, and (2) the slot count is a
// property of the ENTRY — a carpet has carpet_div^2 slots per cell (484 for the
// bog moss, deliberately over SCATTER_MAX_PER_CELL), while a scattered entry
// has 128. Sizing the z dimension per entry means a carpet enumerates all of
// its slots while the grasses still cost 128.

struct CullInfo {
  entry_index: u32,
  slots: u32,
}

@group(1) @binding(0) var<uniform> params: SrParams;
@group(1) @binding(1) var<storage, read_write> counts: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> records: array<PlantRec>;
@group(1) @binding(3) var<uniform> cull_info: CullInfo;

@compute @workgroup_size(128)
fn cs_cull(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let cell = params.base_cell + vec2i(i32(wid.x), i32(wid.y));
  let e = cull_info.entry_index;
  let entry = stand_table[e];
  let carpet = entry.carpet_div > 0.0;

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
  // one box per workgroup skips 128 scatter evaluations for every cell behind
  // the camera / off to the side / past the fade distance — i.e. for most of
  // the square window, which circumscribes a fade_end disc while only the
  // frustum wedge inside it can contribute.
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

  // One thread per slot. NEVER SCATTER_MAX_PER_CELL: the host sizes the z
  // dimension from this entry's own slot count (carpet_div^2 for a mat), and a
  // carpet has ~3.8x more slots than the scatter budget.
  let slot = wid.z * 128u + lid.x;
  if (slot >= cull_info.slots) { return; }

  let cand = scatter_candidate(params.seed, e, cell, slot);
  if (!cand.exists) { return; }
  let h = entry.height_scale * cand.scale;
  let center = cand.pos + vec3f(0.0, 0.5 * h, 0.0);
  let dist = distance(frame.camera_pos, center);
  if (dist > params.fade_end) { return; }

  // Far fade out, plus a camera-inside fade in — but NOT for a carpet: a mat
  // you are standing on must never open a hole under your feet.
  let near_fade = select(smoothstep(0.2, 0.7, dist), 1.0, carpet);
  let fade = near_fade * (1.0 - smoothstep(params.fade_start, params.fade_end, dist));
  if (fade < 0.01) { return; }

  // Cull sphere: a tile's splats spread over the source mesh's own overflow
  // (0.24m of geometry in a 0.18m period), so 0.75 * footprint covers it —
  // sizing a 0.18m mat tile from height_scale or from a 1m sway margin would
  // accept ~6x too much.
  let radius = select(
    max(h, 0.75 * cand.scale) + 0.35,
    0.75 * entry.footprint_m * cand.scale + 0.1,
    carpet,
  );
  var visible = true;
  for (var p = 0u; p < 5u; p++) {
    let pl = planes[p];
    if (dot(pl.xyz, center) + pl.w < -radius * length(pl.xyz)) { visible = false; break; }
  }
  if (!visible) { return; }

  // Distance LOD. The ring constants are calibrated so a splat lands on ~5
  // half-res pixels for a ~1.2m grass tuft (silhouette ~0.6 m^2, i.e. an
  // effective size of 0.78m); a species whose footprint is 0.18m reaches the
  // same splats-per-pixel at 1/4 the distance, so the rings scale with the
  // species' own size. Without this a 0.18m moss tile gets all 2048 LOD0
  // splats out to 3m — thousands of overlapping splats inside a 40px tile.
  let ring_scale = select(1.0, entry.footprint_m * cand.scale / 0.78, carpet);
  // Per-plant ring jitter so LOD transitions decorrelate spatially instead of
  // drawing a visible arc. A carpet's phase is 0 by construction (moss does not
  // sway), so the jitter comes from the tile position instead.
  let jit = select(
    fract(cand.phase * 0.63661977),
    hash_f32(hash2(bitcast<u32>(cand.pos.x), bitcast<u32>(cand.pos.z))),
    carpet,
  );
  let dj = dist * (0.85 + 0.3 * jit) / ring_scale;
  // Bucket 5 is the one-splat-per-tile far level: only a carpet reaches it,
  // because for an upright plant ring_scale is 1 and ring4 sits past fade_end.
  var lod = 5u;
  if (dj < params.rings.x) { lod = 0u; }
  else if (dj < params.rings.y) { lod = 1u; }
  else if (dj < params.rings.z) { lod = 2u; }
  else if (dj < params.rings.w) { lod = 3u; }
  else if (dj < params.ring4) { lod = 4u; }
  // A scattered entry has no bucket 5 (its cap there is 0, so a record would be
  // silently dropped and the plant would vanish). ring4 is past fade_end for it
  // at every `detail` setting, but the clamp makes that a guarantee rather than
  // an arithmetic coincidence.
  lod = min(lod, select(4u, 5u, carpet));

  let idx = atomicAdd(&counts[e * SR_BUCKETS + lod], 1u);
  if (idx >= select(SR_CAPS_SCATTER[lod], SR_CAPS_CARPET[lod], carpet)) { return; }

  var rec: PlantRec;
  rec.xz = cand.pos.xz;
  rec.ys = pack2x16float(vec2f(cand.yaw, cand.scale));
  let y16 = pack2x16float(vec2f(cand.pos.y, 0.0)) & 0xffffu;
  let phase8 = u32(round(fract(cand.phase * 0.15915494) * 255.0));
  let fade8 = u32(round(fade * 255.0));
  rec.pfy = y16 | (phase8 << 16u) | (fade8 << 24u);
  records[select(SR_BASES_SCATTER[lod], SR_BASES_CARPET[lod], carpet) + idx] = rec;
}

// Pass 1b: clamp bucket counts into drawIndexedIndirect args.
// Slot i = entry*SR_BUCKETS + lod, 5 u32 each:
//   [indexCount = 6 * splats(lod), instanceCount = min(count, cap),
//    firstIndex, baseVertex, firstInstance].

@group(1) @binding(4) var<storage, read> counts_ro: array<u32>;
@group(1) @binding(5) var<storage, read_write> indirect: array<u32>;

@compute @workgroup_size(64)
fn cs_finalize(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.num_entries * SR_BUCKETS) { return; }
  let lod = i % SR_BUCKETS;
  let o = i * 5u;
  indirect[o] = 6u * SR_BUCKET_SPLATS[lod];
  let carpet_e = stand_table[i / SR_BUCKETS].carpet_div > 0.0;
  indirect[o + 1u] = min(counts_ro[i], select(SR_CAPS_SCATTER[lod], SR_CAPS_CARPET[lod], carpet_e));
  indirect[o + 2u] = 0u;
  indirect[o + 3u] = 0u;
  indirect[o + 4u] = 0u;
}
