import type { MaterialId } from '../sim/materials'

export const LEVEL_VERSION = 1
export const SOLUTION_VERSION = 1

export interface WorldObjectDoc {
  id: string
  x: number
  y: number
  width: number
  height: number
  /** kg/m^3. Below 1000 floats. */
  density: number
  label?: string
}

export interface AnchorDoc {
  id: string
  x: number
  y: number
  /**
   * null welds the anchor to static terrain. Otherwise it binds to a world
   * object, and a structure attached to it genuinely holds that object up -
   * build nothing and the object falls.
   */
  attachedTo: string | null
}

export interface LevelDoc {
  version: number
  name: string
  widthM: number
  budget: number
  materials: MaterialId[]
  objects: WorldObjectDoc[]
  anchors: AnchorDoc[]
}

export interface NodeDoc {
  id: string
  x: number
  y: number
}

export interface MemberDoc {
  id: string
  a: string
  b: string
  material: MaterialId
  /** Omitted means derive from the material and length. */
  segments?: number
}

/** The player's build. Serialised separately from the level. */
export interface Solution {
  version: number
  nodes: NodeDoc[]
  members: MemberDoc[]
}

export const emptySolution = (): Solution => ({
  version: SOLUTION_VERSION,
  nodes: [],
  members: [],
})

export function defaultLevel(widthM = 120): LevelDoc {
  return {
    version: LEVEL_VERSION,
    name: 'sandbox',
    widthM,
    budget: 50000,
    materials: ['wood', 'steel'],
    objects: [],
    anchors: [],
  }
}

/** Deep copy. Solutions are small, so snapshotting whole is simpler than a command log. */
export const cloneSolution = (s: Solution): Solution => ({
  version: s.version,
  nodes: s.nodes.map((n) => ({ ...n })),
  members: s.members.map((m) => ({ ...m })),
})

export const cloneLevel = (l: LevelDoc): LevelDoc => ({
  ...l,
  materials: [...l.materials],
  objects: l.objects.map((o) => ({ ...o })),
  anchors: l.anchors.map((a) => ({ ...a })),
})

/**
 * Versioned load. Migration hooks live here from day one - a saved level that
 * cannot be opened after a schema change is the kind of thing that quietly
 * costs a week later.
 */
export function migrateLevel(raw: unknown): LevelDoc {
  const doc = raw as Partial<LevelDoc>
  if (!doc || typeof doc !== 'object') return defaultLevel()
  const version = doc.version ?? 0
  if (version > LEVEL_VERSION) {
    throw new Error(`level version ${version} is newer than this build (${LEVEL_VERSION})`)
  }
  return {
    version: LEVEL_VERSION,
    name: doc.name ?? 'sandbox',
    widthM: doc.widthM ?? 120,
    budget: doc.budget ?? 50000,
    materials: doc.materials ?? ['wood', 'steel'],
    objects: doc.objects ?? [],
    anchors: doc.anchors ?? [],
  }
}

export function migrateSolution(raw: unknown): Solution {
  const sol = raw as Partial<Solution>
  if (!sol || typeof sol !== 'object') return emptySolution()
  const version = sol.version ?? 0
  if (version > SOLUTION_VERSION) {
    throw new Error(`solution version ${version} is newer than this build`)
  }
  return {
    version: SOLUTION_VERSION,
    nodes: sol.nodes ?? [],
    members: sol.members ?? [],
  }
}

let idCounter = 0
export const nextId = (prefix: string): string => `${prefix}${(++idCounter).toString(36)}`
