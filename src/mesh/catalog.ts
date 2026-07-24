import { parseGcMesh, type GcMesh } from './gcmesh.ts'

interface RawManifest {
  binary: string
  vertexCount: number
  triangleCount: number
  boundsMin: number[]
  boundsMax: number[]
  tileSize: number[]
  outputBytes: number
}

export interface MeshInfo {
  id: string
  url: string
  vertexCount: number
  triangleCount: number
  tileSize: [number, number]
  bytes: number
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
      .map(([path, m]) => {
        const id = path.split('/')[3]!
        return {
          id,
          url: `/mesh/raw/${id}/${m.binary}`,
          vertexCount: m.vertexCount,
          triangleCount: m.triangleCount,
          tileSize: [m.tileSize[0] ?? 0, m.tileSize[1] ?? 0] as [number, number],
          bytes: m.outputBytes,
        }
      })
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
