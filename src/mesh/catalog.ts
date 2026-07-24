import { RAW_MESHES_AVAILABLE } from '../util/env.ts'
import { assetUrl } from '../util/paths.ts'
import { parseGcMesh, type GcMesh } from './gcmesh.ts'

/**
 * Source-mesh manifests come in two shapes:
 *  - flat (calamagrostis): { binary: "file.bin", vertexCount, tileSize, outputBytes, ... }
 *  - nested (elymus, poa): { binary: { file, bytes }, geometry: { vertexCount,
 *    tile?: { sizeX, sizeZ } | absent for finite specimens, ... } }
 * Normalize both here; nothing else in the app looks at raw manifests.
 */
interface FlatManifest {
  binary: string
  vertexCount: number
  triangleCount: number
  tileSize: number[]
  outputBytes: number
}

interface NestedManifest {
  binary: { file: string; bytes: number }
  geometry: {
    vertexCount: number
    triangleCount: number
    tile?: { sizeX: number; sizeZ: number }
  }
}

type RawManifest = FlatManifest | NestedManifest

export interface MeshInfo {
  id: string
  url: string
  vertexCount: number
  triangleCount: number
  /** Periodic community tile size (m), or null for finite single specimens. */
  tileSize: [number, number] | null
  bytes: number
  /**
   * The .bin is actually fetchable. False on static deployments built without
   * GC_DEPLOY_MESHES=1 — the manifests ship, the ~357MB of binaries do not.
   */
  available: boolean
}

/** Shown wherever a raw mesh is asked for but was not deployed. */
export const MESH_NOT_DEPLOYED = 'source mesh not included in this deployment'

function normalize(id: string, m: RawManifest): MeshInfo {
  if ('geometry' in m) {
    const g = m.geometry
    return {
      id,
      url: assetUrl(`/mesh/raw/${id}/${m.binary.file}`),
      vertexCount: g.vertexCount,
      triangleCount: g.triangleCount,
      tileSize: g.tile ? [g.tile.sizeX, g.tile.sizeZ] : null,
      bytes: m.binary.bytes,
      available: RAW_MESHES_AVAILABLE,
    }
  }
  const tile: [number, number] = [m.tileSize[0] ?? 0, m.tileSize[1] ?? 0]
  return {
    id,
    url: assetUrl(`/mesh/raw/${id}/${m.binary}`),
    vertexCount: m.vertexCount,
    triangleCount: m.triangleCount,
    tileSize: tile[0] > 0 && tile[1] > 0 ? tile : null,
    bytes: m.outputBytes,
    available: RAW_MESHES_AVAILABLE,
  }
}

const manifests = import.meta.glob<RawManifest>('/mesh/raw/*/manifest.json', {
  eager: true,
  import: 'default',
})

/**
 * Known source meshes under mesh/raw/<id>/ — discovered from their
 * manifest.json files; binaries are fetched and parsed lazily, then cached.
 */
export class MeshCatalog {
  private cache = new Map<string, Promise<GcMesh>>()

  list(): MeshInfo[] {
    return Object.entries(manifests)
      .map(([path, m]) => normalize(path.split('/')[3]!, m))
      .sort((a, b) => a.id.localeCompare(b.id))
  }

  info(id: string): MeshInfo {
    const info = this.list().find((m) => m.id === id)
    if (!info) throw new Error(`unknown mesh "${id}" — known: ${this.list().map((m) => m.id).join(', ') || '(none)'}`)
    return info
  }

  load(id: string): Promise<GcMesh> {
    let p = this.cache.get(id)
    if (!p) {
      const info = this.info(id)
      const { url } = info
      p = (async () => {
        if (!info.available) throw new Error(`${MESH_NOT_DEPLOYED} (mesh/raw/${id}/ holds only the manifest here)`)
        const r = await fetch(url)
        // A missing file under an SPA fallback answers 200 with index.html —
        // never hand that to the GCMESH1 parser.
        if (!r.ok || (r.headers.get('content-type') ?? '').includes('text/html')) {
          throw new Error(`${MESH_NOT_DEPLOYED} (fetch ${url}: ${r.status})`)
        }
        return parseGcMesh(await r.arrayBuffer())
      })()
      this.cache.set(id, p)
    }
    return p
  }
}

export const meshCatalog = new MeshCatalog()
