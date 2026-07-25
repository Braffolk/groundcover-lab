#include "src/wgsl/scatter.wgsl"
#include "./fin_shared.wgsl"

// Per-frame plant compaction for the fin cards.
//
// One thread per scatter candidate slot of the camera-bounded region (cells x
// `fin.slots` slots) for ONE stand entry: evaluate the shared placement hash
// ONCE, drop slots that hold no plant, drop plants outside the stand region,
// drop plants whose fade envelope is already zero, and append the survivors to
// a compact instance list + indirect draw args.
//
// `fin.slots` is per ENTRY and comes from standEntrySlots(): carpet_div^2 for a
// carpet (484 for the bog moss), SCATTER_MAX_PER_CELL for ordinary scatter.
// Hardcoding the scatter budget here rendered ~26% of the moss mat — the first
// 5.8 of its 22 grid rows per cell — which looked exactly like a placement bug
// (corduroy stripes of tiles across the bog).
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
// Carpet only: tiles inside MOSS_NEAR_M, which are the only ones that draw more
// than their closure card. Splitting them out is what keeps the vertex cost
// sane: at the default region ~300k moss tiles survive but only ~8k of them are
// near, so the far field pays 6 vertices instead of 6 * MOSS_LEVELS. The near
// draw sets first_vertex = 6, so the SAME vertex shader emits cards 1..n-1.
@group(1) @binding(3) var<storage, read_write> near_plants: array<FinPlant>;
@group(1) @binding(4) var<storage, read_write> near_args: DrawArgs;

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
  let side = u32(fin.side);
  let slots = u32(fin.slots);
  let cell_lin = gid.x / slots;
  if (cell_lin >= side * side) { return; }
  let slot = gid.x % slots;
  let cxi = i32(fin.origin_cell_x) + i32(cell_lin % side);
  let czi = i32(fin.origin_cell_z) + i32(cell_lin / side);
  // The stand's region is the only place its plants exist (Scatter.region()).
  if (cxi < i32(fin.cell_min) || cxi > i32(fin.cell_max) ||
      czi < i32(fin.cell_min) || czi > i32(fin.cell_max)) { return; }
  // Whole-cell region reject before the hash: both fades are exactly zero past
  // max_dist, so a cell whose nearest point is beyond it holds nothing visible.
  // (Half a cell diagonal is 2.829 m.) ~27% of the dispatched square is corner
  // that can never contribute, and a carpet cell is 484 threads of it.
  let cell_c = (vec2f(f32(cxi), f32(czi)) + 0.5) * SCATTER_CELL_SIZE;
  if (length(frame.camera_pos.xz - cell_c) - 2.8285 > fin.max_dist) { return; }

  let entry_index = u32(fin.entry_index);
  let sp = scatter_candidate(u32(fin.seed), entry_index, vec2i(cxi, czi), slot);
  if (!sp.exists) { return; }

  var d_xz = 0.0;
  if (fin.carpet > 0.5) {
    // Carpet: region edge only, from the tile centre in XZ (moss.wgsl matches).
    d_xz = length(frame.camera_pos.xz - sp.pos.xz);
    if (carpet_fade(d_xz, fin.max_dist) < 0.004) { return; }
  } else {
    let center = fin_center(sp.pos, sp.scale, sp.yaw, fin);
    if (fin_fade(length(frame.camera_pos - center), fin.radius * sp.scale, fin.max_dist) < 0.004) { return; }
  }

  let record = FinPlant(sp.pos.x, sp.pos.y, sp.pos.z, sp.yaw, sp.scale, sp.phase);
  let idx = atomicAdd(&draw_args.instance_count, 1u);
  if (idx >= arrayLength(&plants)) {
    atomicSub(&draw_args.instance_count, 1u);
    return;
  }
  plants[idx] = record;

  if (fin.carpet > 0.5 && d_xz < MOSS_NEAR_M) {
    let n = atomicAdd(&near_args.instance_count, 1u);
    if (n >= arrayLength(&near_plants)) {
      atomicSub(&near_args.instance_count, 1u);
      return;
    }
    near_plants[n] = record;
  }
}
