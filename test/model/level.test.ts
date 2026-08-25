import { describe, it, expect } from 'vitest'
import {
  allIds,
  claimIds,
  cloneLevel,
  cloneSolution,
  defaultLevel,
  emptySolution,
  migrateLevel,
  migrateSolution,
  nextId,
  LEVEL_VERSION,
} from '../../src/model/level'

/**
 * The document model. Small, but it is the authority everything projects from,
 * and the audit found a real corruption path in it: the id counter restarts at
 * zero each page load, so ids minted after loading a save collided with ids in
 * the save. Duplicate ids silently corrupt every map keyed on them.
 */

describe('id allocation', () => {
  it('never mints an id that a loaded document already contains', () => {
    // Simulate a fresh page load that opens a saved level: the module counter
    // is wherever previous tests left it, and the save contains ids from a
    // longer editing session - claim them all first.
    const doc = defaultLevel()
    const solution = emptySolution()
    doc.anchors.push({ id: 'a2z', x: 0, y: 0, attachedTo: null }) // base36 "2z" = 107
    doc.objects.push({ id: 'o1c', x: 0, y: 0, width: 1, height: 1, density: 100 })
    solution.members.push({ id: 'm2f', a: 'a1', b: 'a2', material: 'wood' })
    solution.nodes.push({ id: 'n9', x: 0, y: 0 })

    claimIds(allIds(doc, solution))

    const taken = new Set(allIds(doc, solution))
    for (let i = 0; i < 200; i++) {
      const id = nextId('a')
      expect(taken.has(id)).toBe(false)
      taken.add(id)
    }
  })

  it('ids stay unique across prefixes and calls', () => {
    const seen = new Set<string>()
    for (const prefix of ['a', 'm', 'n', 'o']) {
      for (let i = 0; i < 50; i++) {
        const id = nextId(prefix)
        expect(seen.has(id)).toBe(false)
        seen.add(id)
      }
    }
  })
})

describe('migration', () => {
  it('fills defaults for a missing or partial document', () => {
    expect(migrateLevel(undefined).widthM).toBe(120)
    expect(migrateLevel({ widthM: 60 }).widthM).toBe(60)
    expect(migrateLevel({}).materials).toEqual(['wood', 'steel'])
    expect(migrateSolution(undefined).members).toEqual([])
  })

  it('refuses documents from a NEWER build instead of silently mangling them', () => {
    expect(() => migrateLevel({ version: LEVEL_VERSION + 1 })).toThrow(/newer/)
    expect(() => migrateSolution({ version: 999 })).toThrow(/newer/)
  })
})

describe('clones', () => {
  it('are deep for every mutable collection', () => {
    const doc = defaultLevel()
    doc.anchors.push({ id: 'a1', x: 1, y: 2, attachedTo: null })
    doc.objects.push({ id: 'o1', x: 0, y: 0, width: 2, height: 2, density: 500 })
    const copy = cloneLevel(doc)
    copy.anchors[0]!.x = 99
    copy.objects[0]!.density = 1
    copy.materials.pop()
    expect(doc.anchors[0]!.x).toBe(1)
    expect(doc.objects[0]!.density).toBe(500)
    expect(doc.materials.length).toBe(2)

    const sol = emptySolution()
    sol.members.push({ id: 'm1', a: 'a', b: 'b', material: 'wood' })
    const solCopy = cloneSolution(sol)
    solCopy.members[0]!.material = 'steel'
    expect(sol.members[0]!.material).toBe('wood')
  })
})
