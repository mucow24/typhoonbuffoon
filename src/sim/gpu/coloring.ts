/**
 * Greedy graph colouring for the GPU joints solve.
 *
 * The CPU reference solves constraints sequentially with in-place position
 * writes (Gauss-Seidel). On the GPU, constraints within one colour run in
 * parallel - which is EXACT Gauss-Seidel only if no two of them touch the
 * same particle. Colours are dispatched sequentially, preserving relaxation
 * between overlapping constraints; only the visiting order differs from the
 * CPU (colour-major instead of index-major), which the parity bands cover.
 *
 * Structure counts are 10^2-10^3, so this runs comfortably every frame after
 * any topology change; greedy is within one colour of optimal on chains,
 * which is what beams are.
 */

/**
 * `endpoints[i]` lists the particle indices constraint i touches; an empty
 * list means the slot is dead. Returns constraint indices grouped by colour.
 */
export function colorConstraints(endpoints: readonly (readonly number[])[]): number[][] {
  // particle index -> colours already used by constraints touching it.
  const usedByParticle = new Map<number, Set<number>>()
  const groups: number[][] = []

  for (let i = 0; i < endpoints.length; i++) {
    const pts = endpoints[i]!
    if (pts.length === 0) continue

    let colour = 0
    for (;;) {
      let clash = false
      for (const p of pts) {
        if (usedByParticle.get(p)?.has(colour)) {
          clash = true
          break
        }
      }
      if (!clash) break
      colour++
    }

    for (const p of pts) {
      let set = usedByParticle.get(p)
      if (!set) {
        set = new Set()
        usedByParticle.set(p, set)
      }
      set.add(colour)
    }
    ;(groups[colour] ??= []).push(i)
  }

  // Greedy never skips a colour, but be defensive about holes anyway.
  return groups.filter((g) => g !== undefined && g.length > 0)
}
