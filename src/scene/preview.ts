import type { VramScope } from '../gpu/resources.ts'

/**
 * The preview geometry a MATERIAL is judged on — harness-owned, exactly like
 * a stand is for a renderer.
 *
 * WHY THE HARNESS OWNS IT: two materials authored on two slightly different
 * spheres turn every A/B into a comparison of geometry. So an experiment never
 * builds a preview mesh; it draws `ctx.preview.mesh` with `PREVIEW_VERTEX_LAYOUT`
 * and nothing else. Which object is shown is a runtime choice (`obj=` in the
 * URL / the toolbar picker), and both sides of an A/B always see the same one.
 *
 * ---------------------------------------------------------------------------
 * UV CONVENTION (the whole contract, in one place)
 * ---------------------------------------------------------------------------
 *
 * **UV IS IN METRES OF SURFACE.** One unit of u or of v is one metre measured
 * along the surface, on every object, in both axes. Not [0,1]-normalised.
 *
 * That choice is the whole point. A [0,1] chart is a per-object parametrisation
 * and forces every material to un-stretch it with a per-axis scale factor —
 * which is how a material ends up "fitted to the polygons" instead of laid out
 * at a physical size, and how a cylinder's caps end up at a different texel
 * density from its side. With metres, a material with a real-world period (the
 * Sphagnum tile is 0.18 m) divides by that period ONCE, isotropically, and is
 * at life size and square on every object with no per-object knowledge at all.
 *
 * The rest of the contract:
 *
 * - `u` increases along the tangent `T`; `v` increases along the bitangent
 *   `B = cross(N, T) * tangent.w`. `v` runs DOWNWARD in the object's frame
 *   (sphere: v=0 at the north pole; cylinder: v=0 at the top; plane: v runs
 *   along +z), which matches WebGPU texture space where row 0 is v=0.
 * - `tangent.w` is +-1 handedness. It is computed here from the analytic
 *   dp/dv of each surface, never hand-typed, so it cannot be inconsistent
 *   with the geometry.
 * - `uvBounds` is the metre rectangle the mesh actually covers. It is what a
 *   displacing material (parallax / silhouette POM) tests its offset uv
 *   against: stepping outside it means the ray left the patch, as opposed to
 *   merely leaving one tile.
 * - `uvPeriod` is the metre period of an axis that CLOSES on itself (the
 *   sphere's longitude, the cylinder's circumference), or 0 for an axis that
 *   ends at a mesh boundary. A tiling material must fit a whole number of tiles
 *   into a closing period or the seam shows two different parts of the tile
 *   side by side, so it snaps its period to the nearest divisor — an error of
 *   ~3% on these objects, applied to BOTH axes so the tiles stay square.
 *
 * NOTHING HERE HAS A PARAMETRISATION SINGULARITY. The sphere is deliberately a
 * tan-warped cube-sphere rather than the obvious lat/long one: a lat/long
 * sphere's u compresses by sin(colatitude), so a tiling material collapses into
 * a pinwheel of slivers at both poles — and the default three-quarter camera
 * looks slightly down, putting one of them straight in frame. The cube-sphere
 * trades that for BOUNDED distortion (5.7% scale, up to 30 degrees of shear, at
 * eight corner points) plus a tile-phase break at twelve chart edges. See
 * `buildSphere`.
 *
 * Every object is centred on the world origin and about 1 m across, so one set
 * of camera bookmarks frames all four.
 */

export type PreviewObjectId = 'sphere' | 'plane' | 'cube-edge' | 'cylinder'

export const PREVIEW_OBJECTS: readonly PreviewObjectId[] = ['sphere', 'plane', 'cube-edge', 'cylinder']

export const DEFAULT_PREVIEW_OBJECT: PreviewObjectId = 'sphere'

export function isPreviewObjectId(value: string | null | undefined): value is PreviewObjectId {
  return value !== null && value !== undefined && (PREVIEW_OBJECTS as readonly string[]).includes(value)
}

/** position(3) normal(3) tangent(3)+handedness(1) uv(2) = 12 floats. */
export const PREVIEW_VERTEX_FLOATS = 12
export const PREVIEW_VERTEX_STRIDE = PREVIEW_VERTEX_FLOATS * 4

/**
 * Hand this straight to `createRenderPipeline`. Shader locations:
 *   0 position (vec3f, world/object space — the objects are not transformed)
 *   1 normal   (vec3f, unit)
 *   2 tangent  (vec4f, xyz unit along +u, w = handedness for B = cross(N,T)*w)
 *   3 uv       (vec2f)
 */
export const PREVIEW_VERTEX_LAYOUT: GPUVertexBufferLayout = {
  arrayStride: PREVIEW_VERTEX_STRIDE,
  attributes: [
    { shaderLocation: 0, offset: 0, format: 'float32x3' },
    { shaderLocation: 1, offset: 12, format: 'float32x3' },
    { shaderLocation: 2, offset: 24, format: 'float32x4' },
    { shaderLocation: 3, offset: 40, format: 'float32x2' },
  ],
}

export interface PreviewMesh {
  id: PreviewObjectId
  title: string
  /** One-line description of what this object is good for judging. */
  purpose: string
  vertices: GPUBuffer
  indices: GPUBuffer
  indexCount: number
  vertexCount: number
  /** World-space AABB. */
  bounds: { min: readonly [number, number, number]; max: readonly [number, number, number] }
  /** The uv rectangle this mesh covers, IN METRES — see the module header. */
  uvBounds: { min: readonly [number, number]; max: readonly [number, number] }
  /**
   * Metre period of an axis that CLOSES on itself, or 0 for one that ends at a
   * mesh boundary. A tiling material fits a whole number of tiles into a
   * closing period (otherwise the seam butts two different parts of the tile
   * together); a ray-marching one uses it to decide whether stepping past the
   * end of the chart continues onto the object or leaves it.
   */
  uvPeriod: readonly [number, number]
}

/**
 * The material-specific slice of ExperimentContext — what MaterialStage merges
 * in, the way StandContextExt is what StandStage merges in. It lives here
 * rather than beside the stage so registry.ts can name it without importing
 * anything that pulls in a pipeline.
 */
export interface MaterialContextExt {
  /** Harness-owned preview geometry — an experiment never authors its own. */
  preview: PreviewScene
}

/** What an experiment sees as `ctx.preview`. */
export interface PreviewScene {
  /**
   * The object being previewed RIGHT NOW. Read it every frame — the toolbar
   * picker swaps it live, without rebuilding the experiment.
   */
  readonly mesh: PreviewMesh
  readonly objects: readonly PreviewMesh[]
  /** Build your vertex stage against this; it is the same for every object. */
  readonly vertexLayout: GPUVertexBufferLayout
}

// ---------------------------------------------------------------------------
// CPU mesh construction
// ---------------------------------------------------------------------------

interface RawMesh {
  // Explicitly ArrayBuffer-backed: `queue.writeBuffer` refuses the default
  // `ArrayBufferLike` parameterisation, which might be a SharedArrayBuffer.
  vertices: Float32Array<ArrayBuffer>
  indices: Uint32Array<ArrayBuffer>
  uvPeriod: [number, number]
}

interface VertexSink {
  push(
    p: readonly [number, number, number],
    n: readonly [number, number, number],
    t: readonly [number, number, number],
    /** Analytic dp/dv direction — the handedness is derived from it. */
    bv: readonly [number, number, number],
    uv: readonly [number, number],
  ): number
}

function makeSink(out: number[]): VertexSink {
  return {
    push(p, n, t, bv, uv) {
      const nl = Math.hypot(n[0], n[1], n[2]) || 1
      const nx = n[0] / nl
      const ny = n[1] / nl
      const nz = n[2] / nl
      // Gram-Schmidt the tangent against the normal so (T, N) is orthonormal
      // even where the parametrisation is skewed.
      const d = t[0] * nx + t[1] * ny + t[2] * nz
      let tx = t[0] - nx * d
      let ty = t[1] - ny * d
      let tz = t[2] - nz * d
      const tl = Math.hypot(tx, ty, tz) || 1
      tx /= tl
      ty /= tl
      tz /= tl
      // Handedness from the analytic +v direction: cross(N, T) either points
      // along dp/dv or against it, and that sign IS tangent.w.
      const cx = ny * tz - nz * ty
      const cy = nz * tx - nx * tz
      const cz = nx * ty - ny * tx
      const w = cx * bv[0] + cy * bv[1] + cz * bv[2] >= 0 ? 1 : -1
      const index = out.length / PREVIEW_VERTEX_FLOATS
      out.push(p[0], p[1], p[2], nx, ny, nz, tx, ty, tz, w, uv[0], uv[1])
      return index
    },
  }
}

/** A grid of (segU+1) x (segV+1) vertices, triangulated CCW seen from +N. */
function gridIndices(indices: number[], base: number, segU: number, segV: number): void {
  const row = segU + 1
  for (let j = 0; j < segV; j++) {
    for (let i = 0; i < segU; i++) {
      const a = base + j * row + i
      const b = a + 1
      const c = a + row
      const d = c + 1
      // v increases "downward", so this winding is CCW when viewed from +N.
      indices.push(a, c, b, b, c, d)
    }
  }
}

const SPHERE_RADIUS = 0.5
const CYL_RADIUS = 0.35
const CYL_HEIGHT = 1.0
const PLANE_SIZE = 1.0
const CUBE_SIZE = 1.0

interface FaceChart {
  n: [number, number, number]
  tu: [number, number, number]
  tv: [number, number, number]
}

/**
 * The six face charts shared by the cube and the cube-sphere: face normal, +u
 * direction, +v direction. `v` points down the four side faces and along +z on
 * the caps, matching the module's convention, and `cross(tv, tu) == n` on every
 * one of them, which is what makes `gridIndices`' winding front-facing.
 */
const CUBE_FACES: readonly FaceChart[] = [
  { n: [0, 0, 1], tu: [1, 0, 0], tv: [0, -1, 0] },
  { n: [0, 0, -1], tu: [-1, 0, 0], tv: [0, -1, 0] },
  { n: [1, 0, 0], tu: [0, 0, -1], tv: [0, -1, 0] },
  { n: [-1, 0, 0], tu: [0, 0, 1], tv: [0, -1, 0] },
  { n: [0, 1, 0], tu: [1, 0, 0], tv: [0, 0, 1] },
  { n: [0, -1, 0], tu: [1, 0, 0], tv: [0, 0, -1] },
]

/** Rotation about +Y, applied to positions and to chart axes alike. */
function rotationY(angle: number): (v: readonly [number, number, number]) => [number, number, number] {
  const c = Math.cos(angle)
  const s = Math.sin(angle)
  return (v) => [v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c]
}

/**
 * Ø1 m sphere as a tan-warped CUBE-SPHERE: six square charts pushed out onto
 * the sphere, no poles anywhere.
 *
 * The obvious lat/long UV sphere was tried first and is unusable for a material
 * preview. Its u compresses by sin(colatitude), so a tiling material collapses
 * into a pinwheel of ever-thinner slivers at both poles — and the default
 * three-quarter camera looks slightly DOWN, putting the north pole right in
 * frame. That is a singularity in the parametrisation, not a distortion you can
 * trade off: no scale factor fixes it.
 *
 * The cube-sphere replaces it with distortion that is BOUNDED: 5.7% scale
 * compression at the eight corners, and up to 30 degrees of shear there
 * (Gauss says a sphere has no distortion-free square chart, so the choice is
 * only which distortion). The `tan(s*pi/4)` warp is what makes it cheap — it
 * equalises texel area across a face, and it makes arc length EXACTLY linear in
 * the parameter along each face's central cross, which is what lets uv be
 * honest metres: one face spans r*pi/2 = 0.785 m and four of them close the
 * equator at 2*pi*r.
 *
 * Cost: the twelve chart edges break the tile's PHASE (not its scale, and not
 * its density). On a dense irregular material that reads as a different clump
 * arrangement rather than a line. `uvPeriod` is 0/0 because none of the six
 * charts closes on itself, so a tiling material keeps its exact real-world
 * period here instead of snapping — this object is at exact life size.
 *
 * The chart is rotated 45 degrees about Y so a face CENTRE, not a seam, faces
 * the three-quarter bookmark's azimuth.
 */
function buildSphere(seg = 40): RawMesh {
  const verts: number[] = []
  const indices: number[] = []
  const sink = makeSink(verts)
  const rot = rotationY(Math.PI / 4)
  const warp = (a: number): number => Math.tan((a * Math.PI) / 4)
  // Metres of arc per unit of the [-1,1] face parameter, exact on the centre cross.
  const arcPerUnit = (SPHERE_RADIUS * Math.PI) / 4
  for (const face of CUBE_FACES) {
    const base = verts.length / PREVIEW_VERTEX_FLOATS
    for (let j = 0; j <= seg; j++) {
      const t = (j / seg) * 2 - 1
      const tw = warp(t)
      for (let i = 0; i <= seg; i++) {
        const s = (i / seg) * 2 - 1
        const sw = warp(s)
        const qx = face.n[0] + face.tu[0] * sw + face.tv[0] * tw
        const qy = face.n[1] + face.tu[1] * sw + face.tv[1] * tw
        const qz = face.n[2] + face.tu[2] * sw + face.tv[2] * tw
        const len = Math.hypot(qx, qy, qz)
        const nrm: [number, number, number] = [qx / len, qy / len, qz / len]
        const p: [number, number, number] = [
          nrm[0] * SPHERE_RADIUS,
          nrm[1] * SPHERE_RADIUS,
          nrm[2] * SPHERE_RADIUS,
        ]
        // dp/ds and dp/dt are the face axes projected onto the tangent plane,
        // which is exactly what the sink's Gram-Schmidt does to them.
        sink.push(rot(p), rot(nrm), rot(face.tu), rot(face.tv), [s * arcPerUnit, t * arcPerUnit])
      }
    }
    gridIndices(indices, base, seg, seg)
  }
  return { vertices: new Float32Array(verts), indices: new Uint32Array(indices), uvPeriod: [0, 0] }
}

/**
 * 1 m x 1 m ground-parallel quad at y = 0, facing +Y. Deliberately horizontal
 * rather than camera-facing: groundcover materials live on the ground, and it
 * is the only object where the `grazing` bookmark means what it means for a
 * real field.
 */
function buildPlane(seg = 64): RawMesh {
  const verts: number[] = []
  const indices: number[] = []
  const sink = makeSink(verts)
  const n: [number, number, number] = [0, 1, 0]
  const t: [number, number, number] = [1, 0, 0]
  const bv: [number, number, number] = [0, 0, 1]
  for (let j = 0; j <= seg; j++) {
    const v = j / seg
    for (let i = 0; i <= seg; i++) {
      const u = i / seg
      const x = (u - 0.5) * PLANE_SIZE
      const z = (v - 0.5) * PLANE_SIZE
      sink.push([x, 0, z], n, t, bv, [u * PLANE_SIZE, v * PLANE_SIZE])
    }
  }
  gridIndices(indices, 0, seg, seg)
  return { vertices: new Float32Array(verts), indices: new Uint32Array(indices), uvPeriod: [0, 0] }
}

/**
 * 1 m cube rotated 45 degrees about Y, so a vertical EDGE faces the default
 * camera azimuth. That edge is the point of this object: a silhouette-POM
 * material only shows what it does where the surface turns away from the
 * viewer, and a hard 90-degree corner is the least forgiving version of that.
 * Each of the six faces carries its own 1 m x 1 m uv chart, so every face is at
 * the same texel density and none of them is stretched.
 */
function buildCubeEdge(seg = 32): RawMesh {
  const verts: number[] = []
  const indices: number[] = []
  const sink = makeSink(verts)
  const h = CUBE_SIZE / 2
  const rot = rotationY(Math.PI / 4)
  for (const face of CUBE_FACES) {
    const base = verts.length / PREVIEW_VERTEX_FLOATS
    for (let j = 0; j <= seg; j++) {
      const v = j / seg
      for (let i = 0; i <= seg; i++) {
        const u = i / seg
        const su = (u - 0.5) * CUBE_SIZE
        const sv = (v - 0.5) * CUBE_SIZE
        const p: [number, number, number] = [
          face.n[0] * h + face.tu[0] * su + face.tv[0] * sv,
          face.n[1] * h + face.tu[1] * su + face.tv[1] * sv,
          face.n[2] * h + face.tu[2] * su + face.tv[2] * sv,
        ]
        sink.push(rot(p), rot(face.n), rot(face.tu), rot(face.tv), [u * CUBE_SIZE, v * CUBE_SIZE])
      }
    }
    gridIndices(indices, base, seg, seg)
  }
  // Six independent charts — no edge of one is the other edge of it.
  return { vertices: new Float32Array(verts), indices: new Uint32Array(indices), uvPeriod: [0, 0] }
}

/**
 * Capped cylinder, r = 0.35 m, h = 1 m, axis +Y. uv is metres everywhere: the
 * side is (circumferential arc, distance down from the top) and each cap is a
 * planar (x, z) projection in the same metres. Because both charts are in the
 * same unit, the caps land at exactly the side's texel density with no per-part
 * scale factor — the [0,1]-per-part alternative is precisely what stretches a
 * cap by the ratio of circumference to height (2.2x here).
 */
function buildCylinder(segU = 96, segV = 32, capRings = 12): RawMesh {
  const verts: number[] = []
  const indices: number[] = []
  const sink = makeSink(verts)
  const top = CYL_HEIGHT / 2
  const circumference = 2 * Math.PI * CYL_RADIUS

  // --- side ---------------------------------------------------------------
  const sideBase = 0
  for (let j = 0; j <= segV; j++) {
    const v = j / segV
    const y = top - v * CYL_HEIGHT
    for (let i = 0; i <= segU; i++) {
      const phi = (i / segU) * Math.PI * 2
      const sp = Math.sin(phi)
      const cp = Math.cos(phi)
      sink.push(
        [CYL_RADIUS * sp, y, CYL_RADIUS * cp],
        [sp, 0, cp],
        [cp, 0, -sp],
        [0, -1, 0],
        [phi * CYL_RADIUS, v * CYL_HEIGHT],
      )
    }
  }
  gridIndices(indices, sideBase, segU, segV)

  // --- caps ---------------------------------------------------------------
  // Planar (x, z) in METRES — the same unit the side uses, so no rescaling.
  const capUv = (x: number, z: number, flip: number): [number, number] => [x, z * flip]
  for (const sign of [1, -1] as const) {
    const base = verts.length / PREVIEW_VERTEX_FLOATS
    const n: [number, number, number] = [0, sign, 0]
    const tu: [number, number, number] = [1, 0, 0]
    const tv: [number, number, number] = [0, 0, sign]
    for (let j = 0; j <= capRings; j++) {
      const r = (j / capRings) * CYL_RADIUS
      for (let i = 0; i <= segU; i++) {
        const phi = (i / segU) * Math.PI * 2
        const x = r * Math.sin(phi)
        const z = r * Math.cos(phi)
        sink.push([x, sign * top, z], n, tu, tv, capUv(x, z, sign))
      }
    }
    // The cap grid runs (angle, radius) rather than (u, v), so it needs its own
    // winding: from +Y, increasing-angle x increasing-radius is CW, hence the
    // swap on the top cap and the plain order on the bottom one.
    const row = segU + 1
    for (let j = 0; j < capRings; j++) {
      for (let i = 0; i < segU; i++) {
        const a = base + j * row + i
        const b = a + 1
        const c = a + row
        const d = c + 1
        if (sign > 0) indices.push(a, c, b, b, c, d)
        else indices.push(a, b, c, b, d, c)
      }
    }
  }
  // The side closes around the axis; height does not.
  return { vertices: new Float32Array(verts), indices: new Uint32Array(indices), uvPeriod: [circumference, 0] }
}

const BUILDERS: Record<PreviewObjectId, { title: string; purpose: string; build: () => RawMesh }> = {
  sphere: {
    title: 'sphere (Ø1 m, cube-sphere)',
    purpose: 'every surface orientation at once, with no pole — the default read on normals and lighting',
    build: () => buildSphere(),
  },
  plane: {
    title: 'plane (1×1 m, ground-parallel)',
    purpose: 'flat ground truth; the only object where `grazing` means what it means in a field',
    build: () => buildPlane(),
  },
  'cube-edge': {
    title: 'cube, edge-on (1 m)',
    purpose: 'hard 90° corners — where a silhouette-POM material either works or does not',
    build: () => buildCubeEdge(),
  },
  cylinder: {
    title: 'cylinder (Ø0.7 × 1 m)',
    purpose: 'one curved axis and two flat caps, at a single uv density',
    build: () => buildCylinder(),
  },
}

// ---------------------------------------------------------------------------
// GPU service
// ---------------------------------------------------------------------------

/**
 * All four objects, uploaded once and owned by the stage. Total is well under
 * a megabyte, so there is no reason to build them lazily and every reason for
 * the picker to switch instantly.
 */
export class PreviewGeometry implements PreviewScene {
  readonly objects: readonly PreviewMesh[]
  readonly vertexLayout = PREVIEW_VERTEX_LAYOUT
  /** Which object `mesh` returns. Set by the toolbar picker; read per frame. */
  active: PreviewObjectId

  constructor(scope: VramScope, queue: GPUQueue, active: PreviewObjectId = DEFAULT_PREVIEW_OBJECT) {
    this.active = active
    this.objects = PREVIEW_OBJECTS.map((id) => {
      const spec = BUILDERS[id]
      const raw = spec.build()
      const vertices = scope.createBuffer(
        {
          label: `preview/${id}/vertices`,
          size: raw.vertices.byteLength,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        },
        { tag: 'preview-geometry' },
      )
      queue.writeBuffer(vertices, 0, raw.vertices)
      const indices = scope.createBuffer(
        {
          label: `preview/${id}/indices`,
          size: raw.indices.byteLength,
          usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        },
        { tag: 'preview-geometry' },
      )
      queue.writeBuffer(indices, 0, raw.indices)

      const min: [number, number, number] = [Infinity, Infinity, Infinity]
      const max: [number, number, number] = [-Infinity, -Infinity, -Infinity]
      const uvMin: [number, number] = [Infinity, Infinity]
      const uvMax: [number, number] = [-Infinity, -Infinity]
      for (let i = 0; i < raw.vertices.length; i += PREVIEW_VERTEX_FLOATS) {
        for (let c = 0; c < 3; c++) {
          const value = raw.vertices[i + c]!
          if (value < min[c]!) min[c] = value
          if (value > max[c]!) max[c] = value
        }
        for (let c = 0; c < 2; c++) {
          const value = raw.vertices[i + 10 + c]!
          if (value < uvMin[c]!) uvMin[c] = value
          if (value > uvMax[c]!) uvMax[c] = value
        }
      }
      return {
        id,
        title: spec.title,
        purpose: spec.purpose,
        vertices,
        indices,
        indexCount: raw.indices.length,
        vertexCount: raw.vertices.length / PREVIEW_VERTEX_FLOATS,
        bounds: { min, max },
        uvBounds: { min: uvMin, max: uvMax },
        uvPeriod: raw.uvPeriod,
      }
    })
  }

  get mesh(): PreviewMesh {
    return this.objects.find((m) => m.id === this.active) ?? this.objects[0]!
  }
}
