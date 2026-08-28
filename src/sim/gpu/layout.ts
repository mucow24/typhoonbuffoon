/**
 * Buffer layout shared between the TS orchestration and the WGSL kernels.
 *
 * Storage buffers are section-indexed mega-buffers: one binding carries many
 * logical SoA arrays as consecutive sections of length `cap`, because the
 * baseline WebGPU limit is EIGHT storage buffers per stage and the sim has
 * dozens of arrays. Section ids below are interpolated into the WGSL prelude,
 * so TS and shaders can never disagree about where posY lives.
 */

/** Fixed-point scale for atomically-accumulated corrections: 2^-20 m (~1 um)
 *  quantum, orders of magnitude under every correction cap in the solver. */
export const FP_SCALE = 1 << 20

/** Particle f32 sections (buffer `pf`). */
export const PF = {
  posX: 0,
  posY: 1,
  prevX: 2,
  prevY: 3,
  velX: 4,
  velY: 5,
  accX: 6,
  accY: 7,
  invMass: 8,
  radius: 9,
  volume: 10,
  /** Velocity snapshot for Jacobi passes (hull viscosity, XSPH). */
  velTX: 11,
  velTY: 12,
  density: 13,
  lambda: 14,
  /** Summed fluid velocity at solids (census). */
  wetVX: 15,
  wetVY: 16,
  /** Pending fluid position correction (projection Jacobi apply). */
  dpX: 17,
  dpY: 18,
  /** Per-particle contact accumulator (own-thread writes only). */
  conX: 19,
  conY: 20,
  COUNT: 21,
} as const

/** Particle u32 sections (buffer `pu`). */
export const PU = {
  kind: 0,
  /** cluster index + 1; 0 = none. */
  cluster1: 1,
  alive: 2,
  grounded: 3,
  /** Fluid-neighbour census count at solids. */
  wet: 4,
  conHits: 5,
  COUNT: 6,
} as const

/** Atomic i32 fixed-point sections (buffer `fx`): cross-thread scatters only. */
export const FX = {
  memberX: 0,
  memberY: 1,
  solidX: 2,
  solidY: 3,
  COUNT: 4,
} as const

/** Atomic u32 sections (buffer `fc`). */
export const FC = {
  memberHits: 0,
  COUNT: 1,
} as const

/** Distance-constraint f32 sections (buffer `df`). */
export const DF = { rest: 0, compliance: 1, zeta: 2, lambda: 3, strain: 4, COUNT: 5 } as const
/** Distance-constraint u32 sections (buffer `du`). */
export const DU = { a: 0, b: 1, mat: 2, noCol1: 3, alive: 4, COUNT: 5 } as const
/** Bend-constraint f32 sections (buffer `bf`). */
export const BF = { restAngle: 0, compliance: 1, zeta: 2, lambda: 3, angle: 4, COUNT: 5 } as const
/** Bend-constraint u32 sections (buffer `bu`). */
export const BU = { a: 0, b: 1, c: 2, alive: 3, COUNT: 4 } as const

/** Cluster header floats per cluster (buffer `clF`, before the rest arrays). */
export const CL_HEADER_F = 8 // stiffness, maxCorrection, cx, cy, angle, pad*3

/**
 * Uniform fields, in buffer order. The WGSL struct is generated from this
 * list and the TS writer walks it, so the two cannot drift. All fields are
 * 4 bytes; the struct is padded to a 16-byte multiple.
 */
export const UNIFORM_FIELDS: readonly (readonly [string, 'u32' | 'f32'])[] = [
  ['n', 'u32'],
  ['cap', 'u32'],
  ['tableSize', 'u32'],
  /** Dense grid dimensions: tableSize = gridW * gridH, row-major keys. */
  ['gridW', 'u32'],
  ['gridH', 'u32'],
  ['gridX0', 'f32'],
  ['gridY0', 'f32'],
  ['distHW', 'u32'],
  ['bendHW', 'u32'],
  ['clusterCount', 'u32'],
  ['fluidIters', 'u32'],
  ['waveOn', 'u32'],
  ['terrCount', 'u32'],
  ['distCap', 'u32'],
  ['bendCap', 'u32'],
  ['padU', 'u32'],
  ['dt', 'f32'],
  ['h', 'f32'],
  ['gravity', 'f32'],
  ['cellSize', 'f32'],
  ['invCell', 'f32'],
  ['spacing', 'f32'],
  ['hK', 'f32'],
  ['h2', 'f32'],
  ['rq', 'f32'],
  ['rq2', 'f32'],
  /** Per-substep growth of the gather reach, (rq - h) / substeps: the
   *  support margin only needs to cover drift accrued SINCE the frame-start
   *  binning, so early substeps gather narrow and the last reaches rq. */
  ['reachSlope', 'f32'],
  ['poly6', 'f32'],
  ['spiky', 'f32'],
  ['pmass', 'f32'],
  ['restDensity', 'f32'],
  ['invRho0', 'f32'],
  ['selfRho', 'f32'],
  ['sCorrK', 'f32'],
  ['invSCorrDenom', 'f32'],
  ['eps', 'f32'],
  ['maxC', 'f32'],
  ['maxCorrSub', 'f32'],
  ['hullPressure', 'f32'],
  ['xsphVisc', 'f32'],
  ['hullC0', 'f32'],
  ['hullC1', 'f32'],
  ['hullCMax', 'f32'],
  ['maxSpeed', 'f32'],
  ['keepNode', 'f32'],
  ['keepFluid', 'f32'],
  ['fricSolid', 'f32'],
  ['fricFluid', 'f32'],
  ['restitution', 'f32'],
  ['contactRelax', 'f32'],
  ['normalDamping', 'f32'],
  ['maxContactCorr', 'f32'],
  ['maxTerrainPush', 'f32'],
  ['boundsX0', 'f32'],
  ['boundsX1', 'f32'],
  ['terrX0', 'f32'],
  ['terrInvDx', 'f32'],
  ['terrDx', 'f32'],
  ['waveX0', 'f32'],
  ['wavePush', 'f32'],
  ['waveBlend', 'f32'],
  ['fpScale', 'f32'],
  ['invFp', 'f32'],
  ['waterDensity', 'f32'],
  /** Union AABB of every member capsule (+reach), for the member-contact
   *  early-out: water far from all structure exits in four compares. */
  ['memAabbX0', 'f32'],
  ['memAabbY0', 'f32'],
  ['memAabbX1', 'f32'],
  ['memAabbY1', 'f32'],
  ['viscEverySub', 'u32'],
  ['padV', 'u32'],
] as const

export const UNIFORM_BYTES = Math.ceil((UNIFORM_FIELDS.length * 4) / 16) * 16

/** WGSL struct text for the uniforms, generated from the field list. */
export function uniformStructWgsl(): string {
  const lines = UNIFORM_FIELDS.map(([name, ty]) => `  ${name}: ${ty},`)
  return `struct UStruct {\n${lines.join('\n')}\n}`
}

/** Write uniform values into `out` (length >= UNIFORM_BYTES). */
export function writeUniforms(out: ArrayBuffer, values: Record<string, number>): void {
  const f32 = new Float32Array(out)
  const u32 = new Uint32Array(out)
  UNIFORM_FIELDS.forEach(([name, ty], i) => {
    const v = values[name]
    if (v === undefined) {
      if (name.startsWith('pad')) return
      throw new Error(`uniform field ${name} missing`)
    }
    if (ty === 'f32') f32[i] = v
    else u32[i] = v >>> 0
  })
}

/** Read-back staging layout: byte offsets of each copied region. */
export interface StagingLayout {
  posX: number
  posY: number
  velX: number
  velY: number
  density: number
  wetVX: number
  wetVY: number
  wet: number
  dStrain: number
  bAngle: number
  clusterPose: number
  totalBytes: number
}

export function stagingLayout(
  cap: number,
  distCap: number,
  bendCap: number,
  clusterCount: number,
): StagingLayout {
  let off = 0
  const take = (words: number): number => {
    const at = off
    off += words * 4
    return at
  }
  return {
    posX: take(cap),
    posY: take(cap),
    velX: take(cap),
    velY: take(cap),
    density: take(cap),
    wetVX: take(cap),
    wetVY: take(cap),
    wet: take(cap),
    dStrain: take(distCap),
    bAngle: take(bendCap),
    /** Full cluster header block; cx/cy/angle live at fixed header slots. */
    clusterPose: take(clusterCount * CL_HEADER_F),
    totalBytes: off,
  }
}
