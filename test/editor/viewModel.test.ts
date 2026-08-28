import { describe, it, expect } from 'vitest'
import {
  nodePosition,
  pickAnchor,
  pickMember,
  pickNode,
  pickObject,
} from '../../src/editor/viewModel'
import type { StructureVM } from '../../src/runtime/protocol'

/**
 * Editor picking now runs on the main thread against the snapshot's structure
 * view-model. These are the same hit rules Session used against the live sim,
 * asserted against hand-placed geometry.
 */

const vm: StructureVM = {
  nodes: [
    { id: 'a1', x: 0, y: 0 },
    { id: 'n1', x: 2, y: 0 },
    { id: 'n2', x: 10, y: 10 },
  ],
  anchors: [{ id: 'a1', x: 0, y: 0, attachedTo: null }],
  members: [
    // A sagging two-segment member from (0,0) to (4,0) through (2,-1).
    { id: 'm1', points: [0, 0, 2, -1, 4, 0] },
    { id: 'm2', points: [10, 10, 10, 14] },
  ],
  objects: [
    { id: 'o1', x: 0, y: 5, width: 4, height: 2 },
    // Overlapping o1: later entries draw on top, so they pick first.
    { id: 'o2', x: 1, y: 5, width: 4, height: 2 },
  ],
}

describe('view-model picking', () => {
  it('picks the nearest node inside the radius, and nothing outside it', () => {
    expect(pickNode(vm, 0.4, 0.1, 1)).toBe('a1')
    expect(pickNode(vm, 1.4, 0.1, 1)).toBe('n1')
    expect(pickNode(vm, 5, 5, 1)).toBeNull()
  })

  it('picks a member from a point near a middle segment, not just endpoints', () => {
    // (1, -0.6) is ~0.07 from the first sagging segment, far from every node.
    expect(pickMember(vm, 1, -0.6, 0.3)).toBe('m1')
    expect(pickMember(vm, 10.2, 12, 0.3)).toBe('m2')
    expect(pickMember(vm, 6, -3, 0.3)).toBeNull()
  })

  it('clamps segment distance at the endpoints rather than extending the line', () => {
    // (5, 0) is 1.0 beyond m1's last endpoint along its axis; an unclamped
    // point-line distance would read 0 and pick it.
    expect(pickMember(vm, 5, 0, 0.5)).toBeNull()
  })

  it('picks anchors only from the anchor list', () => {
    expect(pickAnchor(vm, 0.2, 0, 1)).toBe('a1')
    expect(pickAnchor(vm, 2, 0, 1)).toBeNull() // n1 is a node, not an anchor
  })

  it('picks the topmost (last-listed) object and honours the pad', () => {
    expect(pickObject(vm, 1, 5)).toBe('o2')
    // Only o1 covers x=-1.5.
    expect(pickObject(vm, -1.5, 5)).toBe('o1')
    // o2's right edge is 3; the 0.4 pad reaches 3.4 and no further.
    expect(pickObject(vm, 3.3, 5)).toBe('o2')
    expect(pickObject(vm, 3.5, 5)).toBeNull()
  })

  it('resolves node positions by id', () => {
    expect(nodePosition(vm, 'n2')).toEqual({ x: 10, y: 10 })
    expect(nodePosition(vm, 'missing')).toBeNull()
  })
})
