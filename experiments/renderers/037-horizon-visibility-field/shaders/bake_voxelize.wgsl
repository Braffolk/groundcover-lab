// BAKE STAGE 1 — surface-density voxelization of a periodic canopy tile.
//
// One thread per (triangle x tile instance). Each triangle is splatted into the
// canopy volume as PROJECTED SURFACE AREA PER AXIS (not occupancy): every voxel
// accumulates area*|n.x|, area*|n.y|, area*|n.z| plus the total area, the
// area-weighted mean normal and mean authored colour.
//
// Per-axis projected area is the key quantity: a grass canopy is strongly
// ANISOTROPIC — near-vertical blades block a horizontal ray hard and a vertical
// ray hardly at all (you see between them, straight down to the soil). A single
// scalar density with a mean normal cannot express that (and worse, the mean
// normal cancels between the two faces of a blade); three numbers can, and they
// give the extinction for any direction as dot(abs(dir), A).
//
// The tile is periodic: xz voxel coordinates wrap, so plants that cross the
// tile edge come back in on the other side and the baked table tiles seamlessly.

#include "src/wgsl/gcmesh.wgsl"

struct VoxParams {
  dims: vec3u,          // (nx, ny, nz) — nx == nz == tile voxels, ny == height voxels
  tri_count: u32,
  mesh_min: vec3f,
  inst_count: u32,
  mesh_extent: vec3f,
  voxel: f32,           // normalized units per voxel
  center: vec2f,        // mesh xz centre (m)
  tile_norm: f32,       // tile size in normalized units
  mesh_scale_base: f32, // height_scale / mesh_height  (x instance scale)
  inv_hmax: f32,        // 1 / (height_scale * scale_max)
  area_scale: f32,      // AREA_FIXED / voxel^2
  base_y: f32,          // mesh bounds_min.y (subtracted so the plant base is y=0)
  _pad0: f32,
}

struct Inst {
  ox: f32,
  oz: f32,
  yaw: f32,
  scale: f32,
}

// Repacked on the CPU to 12B/vertex and 12B/triangle: the raw 16B records put
// the biggest source mesh (poa, 6.5M verts) over maxStorageBufferBindingSize.
@group(0) @binding(0) var<storage, read> verts: array<u32>;       // 3 u32 per vertex
@group(0) @binding(1) var<storage, read> tris: array<u32>;        // 3 u32 per triangle
@group(0) @binding(2) var<storage, read> insts: array<Inst>;
@group(0) @binding(3) var<storage, read_write> acc_u: array<atomic<u32>>;  // 4 per voxel
@group(0) @binding(4) var<storage, read_write> acc_i: array<atomic<i32>>;  // 4 per voxel
@group(0) @binding(5) var<uniform> vox: VoxParams;
@group(0) @binding(6) var<storage, read_write> acc_p: array<atomic<u32>>;  // 4 per voxel

struct Vtx {
  pos: vec3f,
  color: vec3f,
  normal: vec3f,
}

fn load_vertex(i: u32) -> Vtx {
  let w0 = verts[i * 3u];
  let w1 = verts[i * 3u + 1u];
  let w2 = verts[i * 3u + 2u];
  let q = vec3f(f32(w0 & 0xffffu), f32(w0 >> 16u), f32(w1 & 0xffffu)) / 65535.0;
  let c = vec3f(f32((w1 >> 16u) & 0xffu), f32((w1 >> 24u) & 0xffu), f32(w2 & 0xffu)) / 255.0;
  var out: Vtx;
  out.pos = vox.mesh_min + q * vox.mesh_extent;
  out.color = c;
  // The repack kept the SOURCE octahedral pair, only requantized to 8 bits.
  out.normal = gcmesh_normal_decode(f32((w2 >> 8u) & 0xffu) / 255.0, f32((w2 >> 16u) & 0xffu) / 255.0);
  return out;
}

/// Mesh frame -> normalized canopy frame (canopy height 1, tile `tile_norm` wide).
fn to_local(p: vec3f, inst: Inst) -> vec3f {
  let s = vox.mesh_scale_base * inst.scale;
  let cx = (p.x - vox.center.x) * s;
  let cz = (p.z - vox.center.y) * s;
  let cy = (p.y - vox.base_y) * s;
  let cs = cos(inst.yaw);
  let sn = sin(inst.yaw);
  let rx = cx * cs - cz * sn;
  let rz = cx * sn + cz * cs;
  // Instance offsets are already in normalized tile units.
  return vec3f(rx, cy, rz) * vox.inv_hmax + vec3f(inst.ox, 0.0, inst.oz);
}

fn splat(p: vec3f, color: vec3f, normal: vec3f, aq: u32) {
  let inv = 1.0 / vox.voxel;
  let ny = i32(floor(p.y * inv));
  if (ny < 0 || ny >= i32(vox.dims.y)) {
    return;
  }
  let nx = i32(vox.dims.x);
  let nz = i32(vox.dims.z);
  // Periodic tile: wrap xz.
  var ix = i32(floor(p.x * inv)) % nx;
  var iz = i32(floor(p.z * inv)) % nz;
  if (ix < 0) { ix += nx; }
  if (iz < 0) { iz += nz; }
  let base = (u32(ny) * vox.dims.z + u32(iz)) * vox.dims.x + u32(ix);
  let fq = f32(aq);
  atomicAdd(&acc_u[base * 4u], aq);
  atomicAdd(&acc_u[base * 4u + 1u], u32(color.r * fq));
  atomicAdd(&acc_u[base * 4u + 2u], u32(color.g * fq));
  atomicAdd(&acc_u[base * 4u + 3u], u32(color.b * fq));
  atomicAdd(&acc_i[base * 4u], i32(normal.x * fq));
  atomicAdd(&acc_i[base * 4u + 1u], i32(normal.y * fq));
  atomicAdd(&acc_i[base * 4u + 2u], i32(normal.z * fq));
  atomicAdd(&acc_p[base * 4u], u32(abs(normal.x) * fq));
  atomicAdd(&acc_p[base * 4u + 1u], u32(abs(normal.y) * fq));
  atomicAdd(&acc_p[base * 4u + 2u], u32(abs(normal.z) * fq));
}

fn radical2(i: u32) -> f32 {
  var bits = i;
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return f32(bits) * 2.3283064365386963e-10;
}

@compute @workgroup_size(64)
fn cs_voxelize(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let flat = (gid.y * (nwg.x * 64u)) + gid.x;
  let total = vox.tri_count * vox.inst_count;
  if (flat >= total) {
    return;
  }
  let tri = flat % vox.tri_count;
  let inst = insts[flat / vox.tri_count];

  let a = load_vertex(tris[tri * 3u]);
  let b = load_vertex(tris[tri * 3u + 1u]);
  let c = load_vertex(tris[tri * 3u + 2u]);
  let pa = to_local(a.pos, inst);
  let pb = to_local(b.pos, inst);
  let pc = to_local(c.pos, inst);
  let area = 0.5 * length(cross(pb - pa, pc - pa));
  if (!(area > 0.0)) {
    return;
  }
  // The mesh normals are authored; the local frame only rotates about y, so
  // rotating them keeps them unit-length.
  let cs = cos(inst.yaw);
  let sn = sin(inst.yaw);
  let rot = mat3x3f(vec3f(cs, 0.0, sn), vec3f(0.0, 1.0, 0.0), vec3f(-sn, 0.0, cs));

  let total_aq = area * vox.area_scale;
  let k = clamp(u32(ceil(area / (vox.voxel * vox.voxel))), 1u, 24u);
  let aq = max(1u, u32(total_aq / f32(k)));
  for (var s = 0u; s < k; s = s + 1u) {
    var u = (f32(s) + 0.5) / f32(k);
    var v = radical2(s + 1u);
    if (u + v > 1.0) {
      u = 1.0 - u;
      v = 1.0 - v;
    }
    let w = 1.0 - u - v;
    let p = pa * w + pb * u + pc * v;
    let col = a.color * w + b.color * u + c.color * v;
    let nrm = normalize(rot * (a.normal * w + b.normal * u + c.normal * v));
    splat(p, col, nrm, aq);
  }
}
