#include "src/wgsl/gcmesh.wgsl"

// BAKE STAGE 1 — splat the stand's tile into a wrapping occupancy volume.
//
// The tile holds the REAL scatter plants of one stand entry inside an LxL
// window (their exact positions, yaws and scales), placed on a flat local
// ground. Source-mesh vertices are ~0.5mm apart, so point-splatting them into
// a 5mm grid fills the canopy solidly; a coarse 8^3 "any" mask is written in
// the same pass and gives the ray march its empty-space skip.
//
// Attribute normals go into a separate coarser (1.5cm) volume as an atomic
// running sum: the ray answer averages ~1.35cm of canopy per texel anyway, so
// per-voxel averages lose nothing and cost 1/16 of the memory.

struct VoxInfo {
  bmin: vec4f,    // xyz = mesh bounds min, w = base y to subtract
  bspan: vec4f,   // xyz = (bmax - bmin) / 65535
  pivot: vec4f,   // xy = mesh horizontal pivot, z = tile period, w = canopy height
  dims: vec4u,    // occupancy VX, VY, VZ, wordsX
  adims: vec4u,   // attribute AX, AY, AZ, vertexCount
  cdims: vec4u,   // coarse CX, CY, CZ, wordsCX
  chunk: vec4u,   // x = vertex chunk offset, y = chunk count, z = attr stride, w = 0
}

@group(0) @binding(0) var<uniform> vinfo: VoxInfo;
@group(0) @binding(1) var<storage, read> verts: array<vec4u>;
@group(0) @binding(2) var<storage, read> insts: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> occ: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> coarse: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> acc_n: array<atomic<i32>>;

fn wrap01(x: f32) -> f32 {
  return x - floor(x);
}

@compute @workgroup_size(64)
fn splat(@builtin(global_invocation_id) gid: vec3u) {
  let local = gid.x;
  if (local >= vinfo.chunk.y) { return; }
  let vi = vinfo.chunk.x + local;
  if (vi >= vinfo.adims.w) { return; }

  let inst = insts[gid.y];
  let rec = verts[local];   // the binding is a chunk sub-range of the mesh

  let q = vec3f(f32(rec.x & 0xffffu), f32(rec.x >> 16u), f32(rec.y & 0xffffu));
  let p0 = vinfo.bmin.xyz + q * vinfo.bspan.xyz;
  var p = vec3f(p0.x - vinfo.pivot.x, p0.y - vinfo.bmin.w, p0.z - vinfo.pivot.y) * inst.z;

  let c = cos(inst.w);
  let s = sin(inst.w);
  let rx = p.x * c - p.z * s;
  let rz = p.x * s + p.z * c;

  let period = vinfo.pivot.z;
  let height = vinfo.pivot.w;
  let fx = wrap01((rx + inst.x) / period);
  let fz = wrap01((rz + inst.y) / period);
  let fy = p.y / height;
  if (fy < 0.0 || fy >= 1.0) { return; }

  // --- occupancy bit (5mm) ------------------------------------------------
  let vx = min(u32(fx * f32(vinfo.dims.x)), vinfo.dims.x - 1u);
  let vy = min(u32(fy * f32(vinfo.dims.y)), vinfo.dims.y - 1u);
  let vz = min(u32(fz * f32(vinfo.dims.z)), vinfo.dims.z - 1u);
  let word = (vy * vinfo.dims.z + vz) * vinfo.dims.w + (vx >> 5u);
  atomicOr(&occ[word], 1u << (vx & 31u));

  // --- coarse "any" mask (4cm) --------------------------------------------
  let cx = min(u32(fx * f32(vinfo.cdims.x)), vinfo.cdims.x - 1u);
  let cy = min(u32(fy * f32(vinfo.cdims.y)), vinfo.cdims.y - 1u);
  let cz = min(u32(fz * f32(vinfo.cdims.z)), vinfo.cdims.z - 1u);
  let cword = (cy * vinfo.cdims.z + cz) * vinfo.cdims.w + (cx >> 5u);
  atomicOr(&coarse[cword], 1u << (cx & 31u));

  // --- normal accumulation (1.5cm), every Nth vertex ----------------------
  if ((vi % vinfo.chunk.z) != 0u) { return; }
  let nm = gcmesh_normal_decode_u16(rec.w & 0xffffu, rec.w >> 16u);
  let nrot = vec3f(nm.x * c - nm.z * s, nm.y, nm.x * s + nm.z * c);

  let ax = min(u32(fx * f32(vinfo.adims.x)), vinfo.adims.x - 1u);
  let ay = min(u32(fy * f32(vinfo.adims.y)), vinfo.adims.y - 1u);
  let az = min(u32(fz * f32(vinfo.adims.z)), vinfo.adims.z - 1u);
  let ai = ((ay * vinfo.adims.z + az) * vinfo.adims.x + ax) * 3u;
  atomicAdd(&acc_n[ai], i32(nrot.x * 64.0));
  atomicAdd(&acc_n[ai + 1u], i32(nrot.y * 64.0));
  atomicAdd(&acc_n[ai + 2u], i32(nrot.z * 64.0));
}
