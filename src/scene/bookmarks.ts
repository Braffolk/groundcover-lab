import type { CameraPose } from './camera.ts'
import type { Terrain } from './terrain.ts'

/**
 * Standard cameras every experiment is judged at (keys 1-4 in the runner).
 * Defined relative to terrain height so they always sit correctly in the
 * scene; benches and goldens use these names for cross-experiment
 * comparability.
 */
export function standardBookmarks(terrain: Terrain): Record<string, CameraPose> {
  const h = (x: number, z: number): number => terrain.height(x, z)
  return {
    // Skimming the canopy top toward the origin — the classic impostor
    // killer. (Calamagrostis stands ~1.2m; yaw = atan2(-x, z) faces origin.)
    grazing: { x: 8, y: h(8, 8) + 1.35, z: 8, yaw: -Math.PI / 4, pitch: -0.06, fov: 60 },
    topdown: { x: 0, y: 42, z: 0.01, yaw: 0, pitch: -1.55, fov: 60 },
    // Camera inside foliage — checks the fade-out rule.
    'inside-plant': { x: 0.2, y: h(0.2, 0.2) + 0.55, z: 0.2, yaw: -Math.PI / 4, pitch: 0.12, fov: 70 },
    // Standing height looking to the horizon — LOD / plant-count scaling check.
    'far-horizon': { x: 0, y: h(0, 60) + 1.7, z: 60, yaw: 0, pitch: -0.06, fov: 60 },
    // 1m straight down — tile-level detail and seams for ground carpets. Named
    // rather than hand-written because URL `cam=x,y,z,...` poses are ABSOLUTE
    // while terrain here sits metres below zero, so "y=2" is not 2m up.
    'carpet-close': { x: 0, y: h(0, 0) + 1, z: 0.01, yaw: 0, pitch: -1.55, fov: 60 },
  }
}

const USER_KEY = 'gc-user-bookmarks'

export function loadUserBookmarks(): Record<string, CameraPose> {
  try {
    return JSON.parse(localStorage.getItem(USER_KEY) ?? '{}') as Record<string, CameraPose>
  } catch {
    return {}
  }
}

export function saveUserBookmark(name: string, pose: CameraPose): void {
  const all = loadUserBookmarks()
  all[name] = pose
  localStorage.setItem(USER_KEY, JSON.stringify(all))
}
