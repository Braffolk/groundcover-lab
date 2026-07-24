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
}

function normalize(id: string, m: RawManifest): MeshInfo {
  if ('geometry' in m) {
    const g = m.geometry
    return {
      id,
      url: `/mesh/raw/${id}/${m.binary.file}`,
      vertexCount: g.vertexCount,
      triangleCount: g.triangleCount,
      tileSize: g.tile ? [g.tile.sizeX, g.tile.sizeZ] : null,
      bytes: m.binary.bytes,
    }
  }
  const tile: [number, number] = [m.tileSize[0] ?? 0, m.tileSize[1] ?? 0]
  return {
    id,
    url: `/mesh/raw/${id}/${m.binary}`,
    vertexCount: m.vertexCount,
    triangleCount: m.triangleCount,
    tileSize: tile[0] > 0 && tile[1] > 0 ? tile : null,
    bytes: m.outputBytes,
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
      const { url } = this.info(id)
      p = fetch(url)
        .then((r) => {
          if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`)
          return r.arrayBuffer()
        })
        .then(parseGcMesh)
      this.cache.set(id, p)
    }
    return p
  }
}

export const meshCatalog = new MeshCatalog()
