# Calamagrostis canescens source mesh binary

This is the exact indexed source mesh embedded in the accepted GCRP/v4 asset. It contains **2,049,985 vertices** and **2,171,134 triangles**, including the original authored RGB colours, per-vertex octahedral normals, index order, and triangle winding. Vertex and triangle records are copied bit-for-bit; the export adds only this standalone header. Units are metres.

## Binary layout (little-endian)

Header: 192 bytes. Magic at byte 0 is `GCMESH1\0`; version at byte 8 is `1`. It embeds the source and mesh-payload SHA-256 hashes; `manifest.json` also records the final-file SHA-256, counts, bounds, offsets, and sizes.

| byte | type | header field |
|---:|---|---|
| 0 | char[8] | `GCMESH1\0` |
| 8 | u32 | version = 1 |
| 12 | u32 | header bytes = 192 |
| 16 | u32 | vertex record bytes = 16 |
| 20 | u32 | triangle record bytes = 16 |
| 24 | u32 | vertex count = 2049985 |
| 28 | u32 | triangle count = 2171134 |
| 32 | u32 | profile id = 2 |
| 36 | u32 | flags = 7 (RGB, oct normals, indexed) |
| 40 | f32[3] | bounds min = -0.07772284001111984, 0.001295100199058652, -0.08573390543460846 |
| 52 | f32[3] | bounds max = 0.6704205274581909, 1.151517629623413, 0.6076181530952454 |
| 64 | f32[2] | tile origin XZ = 0, 0 |
| 72 | f32[2] | tile size XZ = 0.5199999809265137, 0.5199999809265137 |
| 80 | f32 | profile top H = 1.1765177249908447 |
| 84 | u32 | UNORM denominator = 65535 |
| 88 | u64 | vertex offset = 192 |
| 96 | u64 | triangle offset = 32799952 |
| 104 | u64 | total file bytes = 67538096 |
| 112 | u64 | source GCRP bytes = 119462112 |
| 120 | u8[32] | source GCRP SHA-256 |
| 152 | u8[32] | vertex+triangle payload SHA-256 |
| 184 | u8[8] | reserved zero |

Vertex record (16 bytes, beginning at byte 192):

| byte | type | meaning |
|---:|---|---|
| 0 | u16 | quantized X |
| 2 | u16 | quantized Y |
| 4 | u16 | quantized Z |
| 6 | u16 | authored R (UNORM) |
| 8 | u16 | authored G (UNORM) |
| 10 | u16 | authored B (UNORM) |
| 12 | u16 | octahedral normal U (UNORM) |
| 14 | u16 | octahedral normal V (UNORM) |

Decode position componentwise as `boundsMin + q/65535*(boundsMax-boundsMin)`. Decode colour as `rgb/65535`; the exporter performs no colour-space conversion.

Triangle record (16 bytes, beginning at byte 32799952): three u32 vertex indices at bytes 0, 4, 8 and one reserved zero u32 at byte 12.

The geometry is one periodic 0.5199999809265137 m x 0.5199999809265137 m community tile. Some vertices extend beyond the nominal tile bounds because foliage crosses a periodic edge; repeat copies by integer tile-size offsets.

## Minimal NumPy loader

~~~python
import struct
from pathlib import Path
import numpy as np

path = Path(__file__).with_name("calamagrostis-canescens-mesh-v1.bin")
with path.open("rb") as f:
    header = f.read(192)
vertex_count, triangle_count = struct.unpack_from("<II", header, 24)
bounds_min = np.asarray(struct.unpack_from("<3f", header, 40), dtype=np.float32)
bounds_max = np.asarray(struct.unpack_from("<3f", header, 52), dtype=np.float32)
vertex_offset, triangle_offset = struct.unpack_from("<QQ", header, 88)
vertex_dtype = np.dtype([("position", "<u2", 3), ("rgb", "<u2", 3), ("oct_normal", "<u2", 2)])
v = np.memmap(path, dtype=vertex_dtype, mode="r", offset=vertex_offset, shape=(vertex_count,))
tri4 = np.memmap(path, dtype="<u4", mode="r", offset=triangle_offset, shape=(triangle_count, 4))
positions = bounds_min + v["position"].astype(np.float32) / 65535.0 * (bounds_max - bounds_min)
colors = v["rgb"].astype(np.float32) / 65535.0
triangles = tri4[:, :3]
~~~
