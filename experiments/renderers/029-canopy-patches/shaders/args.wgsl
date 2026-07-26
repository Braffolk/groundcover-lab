// Indirect-args fan-out. The cull pass keeps exactly two counters per stand
// entry (patch-stack survivors, composite-card survivors); one 64-thread
// dispatch at the end of the same compute pass expands them into the 6 draw
// slots per entry:
//
//   slot 0..1  depth slabs (front, back)     -> near count, firstVertex q*6
//   slot 2..3  crown patches (upper, lower)  -> near count
//   slot 4     composite side card           -> far count
//   slot 5     composite crown card          -> far count
//
// firstVertex carries the quad index into the vertex shader, so all 6 draws
// share one pipeline and one 6-vertex quad.
// QUADS and FIRST_FAR must match main.ts's QUADS / NEAR_QUADS and the quad
// indices in patches.wgsl.

struct DrawArgs {
  vertex_count: u32,
  instance_count: u32,
  first_vertex: u32,
  first_instance: u32,
}

@group(1) @binding(0) var<storage, read> counters: array<u32>;
@group(1) @binding(1) var<storage, read_write> args: array<DrawArgs>;

const QUADS: u32 = 6u;
const FIRST_FAR: u32 = 4u;

@compute @workgroup_size(64)
fn cs_args(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&args)) {
    return;
  }
  let entry = i / QUADS;
  let q = i % QUADS;
  let lod = select(0u, 1u, q >= FIRST_FAR);
  args[i] = DrawArgs(6u, counters[entry * 2u + lod], q * 6u, 0u);
}
