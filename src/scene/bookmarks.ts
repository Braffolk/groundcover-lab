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
    // Near-horizontal through the grass — the classic impostor killer.
    grazing: { x: 8, y: h(8, 8) + 0.45, z: 8, yaw: -2.35, pitch: -0.04, fov: 60 },
    topdown: { x: 0, y: 42, z: 0.01, yaw: 0, pitch: -1.55, fov: 60 },
    // Camera inside foliage — checks the fade-out rule.
    'inside-plant': { x: 0.2, y: h(0.2, 0.2) + 0.55, z: 0.2, yaw: 0.6, pitch: 0.12, fov: 70 },
    // Standing height looking to the horizon — LOD / plant-count scaling check.
    'far-horizon': { x: 0, y: h(0, 60) + 1.7, z: 60, yaw: Math.PI, pitch: -0.06, fov: 60 },
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
