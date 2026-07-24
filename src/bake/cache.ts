/**
 * OPFS-backed bake cache. Failures (no OPFS, quota, private mode) degrade to
 * cache misses — baking just reruns.
 */

async function bakeDir(): Promise<FileSystemDirectoryHandle | null> {
  try {
    const root = await navigator.storage.getDirectory()
    return await root.getDirectoryHandle('bake-cache', { create: true })
  } catch {
    return null
  }
}

export async function bakeCacheGet(key: string): Promise<ArrayBuffer | null> {
  const dir = await bakeDir()
  if (!dir) return null
  try {
    const handle = await dir.getFileHandle(`${key}.bin`)
    return await (await handle.getFile()).arrayBuffer()
  } catch {
    return null
  }
}

export async function bakeCachePut(key: string, data: ArrayBuffer): Promise<void> {
  const dir = await bakeDir()
  if (!dir) return
  try {
    const handle = await dir.getFileHandle(`${key}.bin`, { create: true })
    const writable = await handle.createWritable()
    await writable.write(data)
    await writable.close()
  } catch (err) {
    console.warn('[bake] cache write failed:', err)
  }
}

export async function bakeCacheClear(): Promise<void> {
  try {
    const root = await navigator.storage.getDirectory()
    await root.removeEntry('bake-cache', { recursive: true })
  } catch {
    /* nothing cached */
  }
}
