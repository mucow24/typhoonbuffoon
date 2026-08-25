/**
 * The material table. Wood and steel differ only in these numbers - there is no
 * separate solver path for either.
 *
 * On the two stiffness knobs:
 *
 * Real timber fails around 0.4% strain and steel yields near 0.2%, both far too
 * small to see. Rather than fake the visual, the axial stiffness is quoted
 * indirectly: `axialStrengthN` is the force at failure and `breakStrain` is the
 * strain reached at that force, so effective EA = strength / breakStrain. That
 * makes "how strong" and "how stretchy" independent, meaningful knobs, and the
 * stretch you see is genuine physical state.
 *
 * Bending is quoted as flexural rigidity EI, which stays realistic. Softening
 * axially to make squash visible does NOT make the structure jelly, because the
 * axial mode is damped near critical - jelly is undamped softness, not softness.
 */
export interface Material {
  id: MaterialId
  name: string
  /** kg/m^3. Drives float vs sink against water's 1000 - not a flag. */
  density: number
  /** Square section side, metres. Gives area, and the drawn thickness. */
  section: number
  costPerMetre: number
  /** Flexural rigidity E*I, N*m^2. Large = resists bending. */
  flexuralRigidity: number
  /** Axial force at failure, N. */
  axialStrengthN: number
  /** Strain reached at the failure load. This is the visible stretch. */
  breakStrain: number
  /** Strain past which deformation is permanent. Infinity means never. */
  yieldStrain: number
  /** How fast rest length migrates once yielding. 0 disables. */
  plasticRate: number
  /**
   * Bend angle per joint, radians, at which the member snaps in BENDING.
   * Without this a member can only fail by elongation, and a mast hit
   * broadside by wind or wave rotates without stretching - it could bend
   * double and never break.
   */
  breakAngle: number
  /** Bend angle past which the joint takes a permanent set. Infinity = never. */
  yieldAngle: number
  zetaAxial: number
  zetaBend: number
  /** Fraction of breakStrain above which damage starts accumulating. */
  damageOnset: number
  /** Damage accumulated per second at full load. */
  damageRate: number
  colour: number
  /** Drag coefficient against wind. ~1.2 for a squarish section. */
  dragCoefficient: number
  /** Default segment count for a member of this material, per metre of length. */
  segmentsPerMetre: number
}

export type MaterialId = 'wood' | 'steel'

export const MATERIALS: Record<MaterialId, Material> = {
  wood: {
    id: 'wood',
    name: 'wood',
    density: 500, // floats
    section: 0.3,
    costPerMetre: 10,
    flexuralRigidity: 6.75e6,
    // 420 kN is ~4.7 MPa over the 0.09 m^2 section - still far below real
    // timber's ~40 MPa, but the old 60 kN meant one strut could barely carry
    // one light house and read as tissue paper. EA = strength/breakStrain
    // rises with it, so members stop drooping like rope under modest loads.
    axialStrengthN: 420e3,
    breakStrain: 0.03, // stretches visibly near failure, then snaps with no warning
    yieldStrain: Infinity, // near-linear-elastic to fracture; never takes a set
    plasticRate: 0,
    breakAngle: 0.28, // bows legibly, then snaps
    yieldAngle: Infinity,
    zetaAxial: 0.9,
    zetaBend: 0.9,
    // Damage starts later and accumulates slower: a structure at 50% load is
    // holding, not dissolving. The old onset 0.6 / rate 0.35 meant anything
    // working near half its strength rotted away in seconds.
    damageOnset: 0.7,
    damageRate: 0.12,
    // Dark brown. The old tan was within a few percent of the sand palette,
    // so wooden members vanished into the beach behind them.
    colour: 0x7a4a26,
    dragCoefficient: 1.3,
    segmentsPerMetre: 0.5,
  },
  steel: {
    id: 'steel',
    name: 'steel',
    density: 7850, // sinks
    section: 0.14,
    costPerMetre: 34,
    flexuralRigidity: 1.2e7,
    axialStrengthN: 1.5e6,
    breakStrain: 0.015,
    yieldStrain: 0.008, // yields, then stays bent - it warns you before it goes
    plasticRate: 0.6,
    breakAngle: 0.6,
    yieldAngle: 0.12, // takes a visible permanent set long before it fails
    zetaAxial: 0.95,
    zetaBend: 0.95,
    damageOnset: 0.65,
    damageRate: 0.1,
    colour: 0x9aa7b4,
    dragCoefficient: 1.1,
    segmentsPerMetre: 0.35,
  },
}

export const MATERIAL_IDS: MaterialId[] = ['wood', 'steel']

/** Index into MATERIAL_IDS, for the Uint8 arrays in the constraint tables. */
export const materialIndex = (id: MaterialId): number => MATERIAL_IDS.indexOf(id)
export const materialAt = (index: number): Material =>
  MATERIALS[MATERIAL_IDS[index] ?? 'wood']!

/** Cross-sectional area, m^2. */
export const areaOf = (m: Material): number => m.section * m.section

/** kg per metre of member. */
export const massPerMetre = (m: Material): number => m.density * areaOf(m)

/** Effective axial stiffness E*A, derived so that failure lands at breakStrain. */
export const axialEA = (m: Material): number => m.axialStrengthN / m.breakStrain

/** XPBD axial compliance for a segment of the given length. */
export const axialCompliance = (m: Material, segmentLength: number): number =>
  segmentLength / axialEA(m)

/** XPBD bend compliance for a joint between segments of the given length. */
export const bendCompliance = (m: Material, segmentLength: number): number =>
  segmentLength / m.flexuralRigidity

/** Volume of a length of member, m^3. Used for buoyancy - always at REST size. */
export const restVolume = (m: Material, length: number): number => areaOf(m) * length

/** Sensible segment count for a member, given its length. */
export function segmentsFor(m: Material, length: number): number {
  return Math.max(1, Math.min(12, Math.round(length * m.segmentsPerMetre)))
}
