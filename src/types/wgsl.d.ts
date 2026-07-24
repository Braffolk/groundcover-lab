declare module '*.wgsl' {
  export interface WgslSource {
    /** Project-root-relative path — stable identity across hot reloads. */
    id: string
    /** Shader source with all `#include` directives inlined. */
    code: string
  }
  const src: WgslSource
  export default src
}

/** Installed by the ShaderRegistry; called by the WGSL Vite plugin's HMR hook. */
// eslint-disable-next-line no-var
declare var __wgslHot: ((id: string, code: string | undefined) => void) | undefined
