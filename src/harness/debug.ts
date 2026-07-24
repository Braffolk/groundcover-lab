/**
 * The one global debug-view selector. Lives in the frame uniform so every
 * renderer answers to the same control (and A/B compares like-for-like).
 * Order must match the DEBUG_* constants in src/wgsl/debug.wgsl.
 */
export const DEBUG_VIEW_MODES = ['off', 'albedo', 'normals', 'lighting', 'coverage', 'depth'] as const

export type DebugViewMode = (typeof DEBUG_VIEW_MODES)[number]

export function debugModeIndex(mode: DebugViewMode): number {
  return DEBUG_VIEW_MODES.indexOf(mode)
}

export function parseDebugMode(raw: string | null): DebugViewMode {
  return raw !== null && (DEBUG_VIEW_MODES as readonly string[]).includes(raw) ? (raw as DebugViewMode) : 'off'
}
