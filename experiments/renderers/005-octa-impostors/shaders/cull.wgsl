#include "src/wgsl/scatter.wgsl"
#include "./impostor_info.wgsl"

// Per-frame plant selection. One workgroup = one scatter cell (128 candidate
// slots), dispatched as a 2D grid over the camera-centred cell region, which
// main.ts has already clamped to the stand's cell range. The region/frustum
// rejects at the top are workgroup-uniform, so a cell that cannot contribute
// exits before a single hash round or heightmap fetch — the expensive scatter
// only runs for cells that survive.
//
// Survivors are compacted into FOUR distance buckets that the draw submits
// near-to-far. Impostor cards are alpha-tested with depth write, so the near
// bucket lays down solid depth first and hi-z then rejects most of the far
// buckets' fragments at grazing angles, which is exactly where a groundcover
// impostor field is deepest.
//
// Cost is O(region area), never O(plants in the stand).

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read_write> out_insts: array<PlantInst>;
// 4 buckets x DrawIndirect args (vertexCount, instanceCount, first*, first*).
@group(1) @binding(2) var<storage, read_write> draw_args: array<atomic<u32>>;

// |terrain_height| bound: the shared FBM's octave amplitudes sum to 1.015.
const TERRAIN_BOUND_SLACK: f32 = 1.05;

@compute @workgroup_size(128)
fn cs_cull(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) slot: u32) {
  let cx = i32(info.region.x) + i32(wg.x);
  let cz = i32(info.region.y) + i32(wg.y);
  let region_r = info.setup.z;
  let entry_index = u32(info.setup.y);
  let s_max = stand_table[entry_index].scale_max;
  let rad_max = info.shape.x * s_max + 0.35;

  // --- cell-level rejects (workgroup-uniform, no memory traffic) -------------
  let cell_lo = vec2f(f32(cx), f32(cz)) * SCATTER_CELL_SIZE;
  let cell_mid = cell_lo + vec2f(SCATTER_CELL_SIZE * 0.5);
  let d_rect = max(abs(frame.camera_pos.xz - cell_mid) - vec2f(SCATTER_CELL_SIZE * 0.5), vec2f(0.0));
  if (length(d_rect) > region_r) {
    return;
  }
  let mid_y = info.center.y * s_max;
  let h_bound = frame.terrain_height_scale * TERRAIN_BOUND_SLACK;
  let box_c = vec3f(cell_mid.x, mid_y * 0.5, cell_mid.y);
  let box_e = vec3f(
    SCATTER_CELL_SIZE * 0.5,
    h_bound + abs(mid_y) * 0.5 + rad_max,
    SCATTER_CELL_SIZE * 0.5,
  );
  for (var i = 0; i < 6; i++) {
    let pl = info.planes[i];
    if (dot(pl.xyz, box_c) + pl.w + dot(abs(pl.xyz), box_e) < 0.0) {
      return;
    }
  }

  // --- exact per-plant tests --------------------------------------------------
  let sp = scatter_candidate(u32(info.setup.x), entry_index, vec2i(cx, cz), slot);
  if (!sp.exists) {
    return;
  }
  let d_xz = distance(sp.pos.xz, frame.camera_pos.xz);
  if (d_xz > region_r) {
    return;
  }

  // Bounding sphere: centred on the card centre height, widened by the
  // clump's horizontal offset so the yaw rotation needs no trig here.
  let centre = sp.pos + vec3f(0.0, info.center.y * sp.scale, 0.0);
  let rad = info.shape.x * sp.scale + 0.35;
  for (var i = 0; i < 6; i++) {
    let pl = info.planes[i];
    if (dot(pl.xyz, centre) + pl.w < -rad) {
      return;
    }
  }

  // Front-to-back bucket by true 3D distance.
  let dist = distance(centre, frame.camera_pos);
  let br = info.bucket_r;
  var bucket = 0u;
  if (dist > br.x) { bucket = 1u; }
  if (dist > br.y) { bucket = 2u; }
  if (dist > br.z) { bucket = 3u; }

  let caps = info.bucket_cap;
  let bases = info.bucket_base;
  let w = atomicAdd(&draw_args[bucket * 4u + 1u], 1u);
  if (w >= u32(caps[bucket])) {
    atomicSub(&draw_args[bucket * 4u + 1u], 1u); // full — the count stays clamped
    return;
  }

  let yaw_q = u32(sp.yaw * (1024.0 / 6.2831853)) & 1023u;
  let scale_q = u32(clamp(sp.scale * 0.25, 0.0, 1.0) * 4095.0) & 4095u;
  let phase_q = u32(sp.phase * (1024.0 / 6.2831853)) & 1023u;
  out_insts[u32(bases[bucket]) + w] = PlantInst(sp.pos, yaw_q | (scale_q << 10u) | (phase_q << 22u));
}
