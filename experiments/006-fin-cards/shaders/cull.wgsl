#include "src/wgsl/scatter.wgsl"
#include "./fin_shared.wgsl"

// Per-frame plant compaction for the fin cards.
//
// One thread per scatter candidate slot of the camera-bounded region (cells x
// 128 slots) for ONE stand entry: evaluate the shared placement hash ONCE,
// drop slots that hold no plant, drop plants outside the stand region, drop
// plants whose fade envelope is already zero, and append the survivors to a
// compact instance list + indirect draw args.
//
// Without this pass the vertex shader had to re-run scatter_candidate (hash +
// terrain heightmap tap) for all 30 vertices of every candidate, existing or
// not — ~86 placement evaluations per plant actually drawn.

struct DrawArgs {
  vertex_count: u32,
  instance_count: atomic<u32>,
  first_vertex: u32,
  first_instance: u32,
}

@group(1) @binding(0) var<uniform> fin: FinCfg;
@group(1) @binding(1) var<storage, read_write> plants: array<FinPlant>;
@group(1) @binding(2) var<storage, read_write> draw_args: DrawArgs;

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
  let side = u32(fin.side);
  let cell_lin = gid.x / SCATTER_MAX_PER_CELL;
  if (cell_lin >= side * side) { return; }
  let slot = gid.x % SCATTER_MAX_PER_CELL;
  let cxi = i32(fin.origin_cell_x) + i32(cell_lin % side);
  let czi = i32(fin.origin_cell_z) + i32(cell_lin / side);
  // The stand's region is the only place its plants exist (Scatter.region()).
  if (cxi < i32(fin.cell_min) || cxi > i32(fin.cell_max) ||
      czi < i32(fin.cell_min) || czi > i32(fin.cell_max)) { return; }

  let entry_index = u32(fin.entry_index);
  let sp = scatter_candidate(u32(fin.seed), entry_index, vec2i(cxi, czi), slot);
  if (!sp.exists) { return; }

  let center = fin_center(sp.pos, sp.scale, sp.yaw, fin);
  let to_cam = frame.camera_pos - center;
  let dcam = length(to_cam);
  if (fin_fade(dcam, fin.radius * sp.scale, fin.max_dist) < 0.004) { return; }

  let idx = atomicAdd(&draw_args.instance_count, 1u);
  if (idx >= arrayLength(&plants)) {
    atomicSub(&draw_args.instance_count, 1u);
    return;
  }
  plants[idx] = FinPlant(sp.pos.x, sp.pos.y, sp.pos.z, sp.yaw, sp.scale, sp.phase);
}
