import { BF, BU, CA, CF, CL_HEADER_F, DF, DU, FC, FORCE_FP, FRC, FX, PF, PU, uniformStructWgsl } from './layout'

/**
 * WGSL kernels for the GPU solver - transcriptions of the CPU reference
 * (sim/fluid.ts, sim/solver.ts, sim/constraints/*, sim/clusters.ts), pass for
 * pass and clamp for clamp. Where the CPU is sequential and the GPU is
 * parallel, the reformulation is stated at the kernel:
 *
 *  - the fluid projection becomes gather-per-particle (kernels evaluated from
 *    both sides instead of cached per pair - identical maths, different float
 *    order);
 *  - member capsule contacts become gather-per-particle over members, which
 *    reproduces the CPU's stamp-dedupe (one contact per member per particle,
 *    closest point on the full segment) by construction;
 *  - joints run graph-coloured: within a colour no constraints share a
 *    particle, so parallel application is exact Gauss-Seidel; colours are
 *    dispatched sequentially, preserving pass order (clusters -> bend ->
 *    distance);
 *  - every cross-thread scatter accumulates in i32 fixed point (2^-20 m).
 *
 * Divergences from CPU accepted and covered by the parity bands: float
 * summation order, neighbour-set truncation order at MAX_NEIGHBOURS, bucket
 * entry order, colour-major constraint visiting order, per-substep (instead
 * of frame-frozen) neighbour pairs for hull viscosity.
 */

const KIND_NODE = 0
const KIND_FLUID = 1
const KIND_OBJECT = 2

export const MAX_NEIGHBOURS = 64

/** Shared prelude: uniforms, buffers by family, helpers. Buffer bindings are
 *  declared per kernel family below; this carries only the pure functions. */
const COMMON = /* wgsl */ `
${uniformStructWgsl()}

const KIND_NODE: u32 = ${KIND_NODE}u;
const KIND_FLUID: u32 = ${KIND_FLUID}u;
const KIND_OBJECT: u32 = ${KIND_OBJECT}u;
const MAX_NEIGHBOURS: u32 = ${MAX_NEIGHBOURS}u;

// Section index helpers - single source of truth interpolated from layout.ts.
fn pf_i(sec: u32, i: u32) -> u32 { return sec * U.cap + i; }
fn pu_i(sec: u32, i: u32) -> u32 { return sec * U.cap + i; }
fn df_i(sec: u32, i: u32) -> u32 { return sec * U.distCap + i; }
fn du_i(sec: u32, i: u32) -> u32 { return sec * U.distCap + i; }
fn bf_i(sec: u32, i: u32) -> u32 { return sec * U.bendCap + i; }
fn bu_i(sec: u32, i: u32) -> u32 { return sec * U.bendCap + i; }

const PF_POSX = ${PF.posX}u;   const PF_POSY = ${PF.posY}u;
const PF_PREVX = ${PF.prevX}u; const PF_PREVY = ${PF.prevY}u;
const PF_VELX = ${PF.velX}u;   const PF_VELY = ${PF.velY}u;
const PF_ACCX = ${PF.accX}u;   const PF_ACCY = ${PF.accY}u;
const PF_INVMASS = ${PF.invMass}u; const PF_RADIUS = ${PF.radius}u;
const PF_VOLUME = ${PF.volume}u;
const PF_VELTX = ${PF.velTX}u; const PF_VELTY = ${PF.velTY}u;
const PF_DENSITY = ${PF.density}u; const PF_LAMBDA = ${PF.lambda}u;
const PF_WETVX = ${PF.wetVX}u; const PF_WETVY = ${PF.wetVY}u;
const PF_DPX = ${PF.dpX}u;     const PF_DPY = ${PF.dpY}u;
const PF_CONX = ${PF.conX}u;   const PF_CONY = ${PF.conY}u;

const PU_KIND = ${PU.kind}u;   const PU_CLUSTER1 = ${PU.cluster1}u;
const PU_ALIVE = ${PU.alive}u; const PU_GROUNDED = ${PU.grounded}u;
const PU_WET = ${PU.wet}u;     const PU_CONHITS = ${PU.conHits}u;

const FX_MEMBERX = ${FX.memberX}u; const FX_MEMBERY = ${FX.memberY}u;
const FX_SOLIDX = ${FX.solidX}u;   const FX_SOLIDY = ${FX.solidY}u;
const FC_MEMBERHITS = ${FC.memberHits}u;

const DF_REST = ${DF.rest}u; const DF_COMPLIANCE = ${DF.compliance}u;
const DF_ZETA = ${DF.zeta}u; const DF_LAMBDA = ${DF.lambda}u;
const DF_STRAIN = ${DF.strain}u;
const DU_A = ${DU.a}u; const DU_B = ${DU.b}u; const DU_MAT = ${DU.mat}u;
const DU_NOCOL1 = ${DU.noCol1}u; const DU_ALIVE = ${DU.alive}u;
const DU_UNBREAK = ${DU.unbreakable}u;
const AFP: f32 = ${FORCE_FP}.0;

const BF_RESTANGLE = ${BF.restAngle}u; const BF_COMPLIANCE = ${BF.compliance}u;
const BF_ZETA = ${BF.zeta}u; const BF_LAMBDA = ${BF.lambda}u;
const BF_ANGLE = ${BF.angle}u;
const BU_A = ${BU.a}u; const BU_B = ${BU.b}u; const BU_C = ${BU.c}u;
const BU_ALIVE = ${BU.alive}u;

const CL_HEADER_F = ${CL_HEADER_F}u;
const PI = 3.14159265358979;

// DENSE row-major grid over the bounded field (the CPU hash table is hashed;
// dense keys make sorted entries SPATIALLY ordered, so a cell ring is a few
// contiguous runs and gather workgroups stay cache-coherent). Coordinates
// clamp to the edge rows/columns: the field-edge and terrain contacts keep
// everything inside anyway, and a stray spray particle merely lands in an
// edge bucket.
fn cell_x(x: f32) -> u32 {
  let c = i32(floor((x - U.gridX0) * U.invCell));
  return u32(clamp(c, 0, i32(U.gridW) - 1));
}
fn cell_y(y: f32) -> u32 {
  let c = i32(floor((y - U.gridY0) * U.invCell));
  return u32(clamp(c, 0, i32(U.gridH) - 1));
}
fn cell_key(cx: u32, cy: u32) -> u32 { return cy * U.gridW + cx; }

// Shortest signed difference between two angles (bending.ts angleDelta).
fn angle_delta(a: f32, b: f32) -> f32 {
  var d = (a - b) % (2.0 * PI);
  if (d > PI) { d -= 2.0 * PI; }
  else if (d < -PI) { d += 2.0 * PI; }
  return d;
}
`

/**
 * Grid-frame helpers - only for kernels that BIND gridS (WGSL requires every
 * module-scope reference to resolve, bound or not).
 *
 * One word of frame-progress state lives after the grid's scan workspace:
 * the substep counter (predict bumps it, gridClear zeroes it; cross-dispatch
 * storage visibility makes it safely readable without atomics). The gather
 * reach for a substep is the kernel radius plus however much support margin
 * the frame-start binning has EARNED - neighbour drift accrues over
 * substeps, so the margin does too, reaching the CPU's full rq by the last
 * substep (fluid.ts:104-111 documents why the margin exists at all).
 */
const GRID_FNS = /* wgsl */ `
fn substep_idx() -> u32 {
  return U.tableSize + 1u + U.cap + (U.tableSize + 255u) / 256u;
}
fn gather_reach() -> f32 {
  return U.hK + U.reachSlope * f32(gridS[substep_idx()]);
}
`

/** Terrain helpers - need the terrain heights binding in scope. */
const TERRAIN_FNS = /* wgsl */ `
fn height_at(x: f32) -> f32 {
  let n = U.terrCount;
  if (n == 0u) { return 0.0; }
  let t = (x - U.terrX0) * U.terrInvDx;
  if (t <= 0.0) { return terr[0]; }
  let last = f32(n - 1u);
  if (t >= last) { return terr[n - 1u]; }
  let i = u32(floor(t));
  let frac = t - floor(t);
  return terr[i] * (1.0 - frac) + terr[i + 1u] * frac;
}

fn normal_at(x: f32) -> vec2f {
  let h = U.terrDx;
  let dy = height_at(x + h) - height_at(x - h);
  let len = sqrt(dy * dy + 4.0 * h * h);
  return vec2f(-dy / len, (2.0 * h) / len);
}
`

/**
 * Neighbour-gather loop over the dense grid. `body` sees the candidate (slot
 * `k` when `sortedSlots`, else particle index `j`) and must `continue` on
 * its own filters; the \`neigh\` counter it increments caps candidates.
 *
 * Bounds are WORLD-SPACE: cells intersecting [pos - reach, pos + reach].
 * The reach carries the CPU's support margin (rq = h + 0.75*spacing): the
 * grid bins FRAME-START positions while gathers run per substep on CURRENT
 * ones, so without the margin a fast-closing neighbour still binned two
 * cells away is missed mid-frame - the exact failure fluid.ts:104-111
 * documents - and the miss is one-sided, which injects momentum. Reach-based
 * bounds give the exact margin at ~1.5x candidates instead of a full extra
 * ring's ~2.8x.
 */
const gatherLoop = (body: string, reach = 'U.rq', sortedSlots = false): string => /* wgsl */ `
  let reachW = ${reach};
  let cx0 = cell_x(xi - reachW);
  let cx1 = cell_x(xi + reachW);
  let cy0 = cell_y(yi - reachW);
  let cy1 = cell_y(yi + reachW);
  var neigh = 0u;
  for (var cy = cy0; cy <= cy1; cy++) {
    if (neigh >= MAX_NEIGHBOURS) { break; }
    // A row of the reach window is ONE contiguous run in the sorted entries.
    let rowBase = cell_key(cx0, cy);
    let end = gridS[cell_key(cx1, cy) + 1u];
    for (var k = gridS[rowBase]; k < end; k++) {
      if (neigh >= MAX_NEIGHBOURS) { break; }
      ${sortedSlots ? '' : 'let j = gridS[U.tableSize + 1u + k];'}
      ${body}
    }
  }
`

// ---------------------------------------------------------------- kernels

/** Bindings: uni, pf, pu, gridA. */
export const SRC_GRID_COUNT = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> gridA: array<atomic<u32>>;
${COMMON}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.n) { return; }
  if (pu[pu_i(PU_ALIVE, i)] != 1u) { return; }
  let kind = pu[pu_i(PU_KIND, i)];
  if (kind == KIND_NODE) { return; }
  let b = cell_key(cell_x(pf[pf_i(PF_POSX, i)]), cell_y(pf[pf_i(PF_POSY, i)]));
  atomicAdd(&gridA[b], 1u);
}
`

/** Exclusive scan over 65536 counts: 256 workgroups x 256. Bindings: uni, gridS, gridA. */
export const SRC_GRID_SCAN1 = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> gridS: array<u32>;
@group(0) @binding(2) var<storage, read_write> gridA: array<atomic<u32>>;
${COMMON}
var<workgroup> tmp: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
  let b = gid.x;
  let v = select(0u, atomicLoad(&gridA[b]), b < U.tableSize);
  tmp[lid.x] = v;
  workgroupBarrier();
  // Hillis-Steele inclusive scan in shared memory.
  var offset = 1u;
  while (offset < 256u) {
    var add = 0u;
    if (lid.x >= offset) { add = tmp[lid.x - offset]; }
    workgroupBarrier();
    tmp[lid.x] += add;
    workgroupBarrier();
    offset = offset << 1u;
  }
  // Exclusive value = inclusive of previous lane.
  var excl = 0u;
  if (lid.x > 0u) { excl = tmp[lid.x - 1u]; }
  gridS[b] = excl;
  if (lid.x == 255u) {
    // Block total into the scan workspace after starts+entries.
    gridS[U.tableSize + 1u + U.cap + wid.x] = tmp[255];
  }
}
`

/**
 * Scan the per-block sums (block count = ceil(tableSize/256), a few hundred
 * to a few thousand for a dense grid) - serial on one thread, which is
 * microseconds at this size and free of block-count limits. Writes the grand
 * total to starts[tableSize]. Bindings: uni, gridS.
 */
export const SRC_GRID_SCAN2 = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> gridS: array<u32>;
${COMMON}
@compute @workgroup_size(1)
fn main() {
  let base = U.tableSize + 1u + U.cap;
  let blocks = (U.tableSize + 255u) / 256u;
  var sum = 0u;
  for (var b = 0u; b < blocks; b++) {
    let v = gridS[base + b];
    gridS[base + b] = sum;
    sum += v;
  }
  gridS[U.tableSize] = sum;
}
`

/** Add block offsets; seed the scatter cursors. Bindings: uni, gridS, gridA. */
export const SRC_GRID_SCAN3 = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> gridS: array<u32>;
@group(0) @binding(2) var<storage, read_write> gridA: array<atomic<u32>>;
${COMMON}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let b = gid.x;
  if (b >= U.tableSize) { return; }
  let base = U.tableSize + 1u + U.cap;
  let start = gridS[b] + gridS[base + b / 256u];
  gridS[b] = start;
  atomicStore(&gridA[U.tableSize + b], start);
}
`

/** Bindings: uni, pf, pu, gridS, gridA. */
export const SRC_GRID_SCATTER = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> gridS: array<u32>;
@group(0) @binding(4) var<storage, read_write> gridA: array<atomic<u32>>;
${COMMON}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.n) { return; }
  if (pu[pu_i(PU_ALIVE, i)] != 1u) { return; }
  if (pu[pu_i(PU_KIND, i)] == KIND_NODE) { return; }
  let b = cell_key(cell_x(pf[pf_i(PF_POSX, i)]), cell_y(pf[pf_i(PF_POSY, i)]));
  let at = atomicAdd(&gridA[U.tableSize + b], 1u);
  gridS[U.tableSize + 1u + at] = i;
}
`

/**
 * Wetness census at solids, once per frame from frame-start state - the
 * candidate radius is the PAIR radius rq, exactly like the CPU pair build.
 * Bindings: uni, pf, pu, gridS.
 */
export const SRC_CENSUS = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> gridS: array<u32>;
${COMMON}
${GRID_FNS}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.n) { return; }
  var wet = 0u;
  var svx = 0.0;
  var svy = 0.0;
  let aliveI = pu[pu_i(PU_ALIVE, i)] == 1u;
  let kind = pu[pu_i(PU_KIND, i)];
  if (aliveI && kind != KIND_FLUID && kind != KIND_NODE) {
    let xi = pf[pf_i(PF_POSX, i)];
    let yi = pf[pf_i(PF_POSY, i)];
    ${gatherLoop(/* wgsl */ `
      if (j == i) { continue; }
      if (pu[pu_i(PU_KIND, j)] != KIND_FLUID) { continue; }
      let dx = xi - pf[pf_i(PF_POSX, j)];
      let dy = yi - pf[pf_i(PF_POSY, j)];
      if (dx * dx + dy * dy >= U.rq2) { continue; }
      neigh++;
      wet++;
      svx += pf[pf_i(PF_VELX, j)];
      svy += pf[pf_i(PF_VELY, j)];
    `)}
  }
  pu[pu_i(PU_WET, i)] = wet;
  pf[pf_i(PF_WETVX, i)] = svx;
  pf[pf_i(PF_WETVY, i)] = svy;
}
`

/** Frame-level wave forcing (solver.ts applyWaveDrive). Bindings: uni, pf, pu. */
export const SRC_WAVE = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
${COMMON}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.n || U.waveOn == 0u) { return; }
  if (pu[pu_i(PU_ALIVE, i)] != 1u) { return; }
  if (pu[pu_i(PU_KIND, i)] != KIND_FLUID) { return; }
  if (pf[pf_i(PF_POSX, i)] < U.waveX0) { return; }
  let v = pf[pf_i(PF_VELX, i)];
  pf[pf_i(PF_VELX, i)] = v + (U.wavePush - v) * U.waveBlend;
}
`

/** solver.ts predict, fused with the per-substep constraint lambda reset
 *  (disjoint index spaces, one dispatch). Bindings: uni, pf, pu, df, bf. */
export const SRC_PREDICT = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> df: array<f32>;
@group(0) @binding(4) var<storage, read_write> bf: array<f32>;
@group(0) @binding(5) var<storage, read_write> gridS: array<u32>;
${COMMON}
${GRID_FNS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  // Frame progress for the drift-scaled gather reach.
  if (i == 0u) { gridS[substep_idx()] = gridS[substep_idx()] + 1u; }
  if (i < U.distHW) { df[df_i(DF_LAMBDA, i)] = 0.0; }
  if (i < U.bendHW) { bf[bf_i(BF_LAMBDA, i)] = 0.0; }
  if (i >= U.n) { return; }
  if (pu[pu_i(PU_ALIVE, i)] != 1u) { return; }
  let x = pf[pf_i(PF_POSX, i)];
  let y = pf[pf_i(PF_POSY, i)];
  pf[pf_i(PF_PREVX, i)] = x;
  pf[pf_i(PF_PREVY, i)] = y;
  if (pf[pf_i(PF_INVMASS, i)] == 0.0) { return; }
  let vx = pf[pf_i(PF_VELX, i)] + pf[pf_i(PF_ACCX, i)] * U.h;
  let vy = pf[pf_i(PF_VELY, i)] + (U.gravity + pf[pf_i(PF_ACCY, i)]) * U.h;
  pf[pf_i(PF_VELX, i)] = vx;
  pf[pf_i(PF_VELY, i)] = vy;
  pf[pf_i(PF_POSX, i)] = x + vx * U.h;
  pf[pf_i(PF_POSY, i)] = y + vy * U.h;
}
`

/**
 * Shape matching, one workgroup per cluster (clusters.ts solve). Reductions
 * in shared memory; particle counts can exceed the workgroup, so phases
 * stride. Same atan2 best-fit rotation, same clamped correction.
 * Bindings: uni, pf, pu, clF, clU.
 * clU: per cluster {start, count} pairs at [0, 2*clusterCount), then indices.
 * clF: per cluster CL_HEADER_F floats (stiffness, maxCorrection, cx, cy,
 * angle, ...), then concatenated restX | restY | mass arrays per cluster
 * (their base = header area end + start*3 within each array block laid
 * consecutively per cluster: restX at base, restY at base+count, mass at
 * base+2*count where base = headerEnd + 3*start).
 */
export const SRC_CLUSTER = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> clF: array<f32>;
@group(0) @binding(4) var<storage, read_write> clU: array<u32>;
${COMMON}
var<workgroup> redA: array<f32, 256>;
var<workgroup> redB: array<f32, 256>;
var<workgroup> redC: array<f32, 256>;

fn cluster_reduce3(lid: u32) {
  workgroupBarrier();
  var half = 128u;
  while (half > 0u) {
    if (lid < half) {
      redA[lid] += redA[lid + half];
      redB[lid] += redB[lid + half];
      redC[lid] += redC[lid + half];
    }
    workgroupBarrier();
    half = half >> 1u;
  }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid3: vec3u) {
  let c = wid.x;
  let lid = lid3.x;
  let start = clU[c * 2u];
  let count = clU[c * 2u + 1u];
  let idxBase = U.clusterCount * 2u;
  let headerEnd = U.clusterCount * CL_HEADER_F;
  let base = headerEnd + 3u * start;

  // COM (mass-weighted current positions of ALIVE particles; the CPU sums
  // over the cached mass of every recorded particle - dead ones keep their
  // cached mass but have no position authority; matching it exactly, we sum
  // whatever the store holds, as the CPU does).
  var sx = 0.0; var sy = 0.0; var sm = 0.0;
  for (var k = lid; k < count; k += 256u) {
    let i = clU[idxBase + start + k];
    let m = clF[base + 2u * count + k];
    sx += pf[pf_i(PF_POSX, i)] * m;
    sy += pf[pf_i(PF_POSY, i)] * m;
    sm += m;
  }
  redA[lid] = sx; redB[lid] = sy; redC[lid] = sm;
  cluster_reduce3(lid);
  // workgroupUniformLoad: reads of workgroup memory are not provably uniform
  // to the compiler, and a return gated on one would be a barrier-divergence
  // validation error further down.
  let total = workgroupUniformLoad(&redC[0]);
  let comSumX = workgroupUniformLoad(&redA[0]);
  let comSumY = workgroupUniformLoad(&redB[0]);
  if (total <= 0.0) { return; }
  let comX = comSumX / total;
  let comY = comSumY / total;

  // Best-fit rotation: A = sum m (p'.q), B = sum m (p'y qx - p'x qy).
  var sA = 0.0; var sB = 0.0;
  for (var k = lid; k < count; k += 256u) {
    let i = clU[idxBase + start + k];
    let m = clF[base + 2u * count + k];
    let px = pf[pf_i(PF_POSX, i)] - comX;
    let py = pf[pf_i(PF_POSY, i)] - comY;
    let qx = clF[base + k];
    let qy = clF[base + count + k];
    sA += m * (px * qx + py * qy);
    sB += m * (py * qx - px * qy);
  }
  redA[lid] = sA; redB[lid] = sB; redC[lid] = 0.0;
  cluster_reduce3(lid);
  let sumA = workgroupUniformLoad(&redA[0]);
  let sumB = workgroupUniformLoad(&redB[0]);
  let theta = atan2(sumB, sumA);
  if (lid == 0u) {
    clF[c * CL_HEADER_F + 2u] = comX;
    clF[c * CL_HEADER_F + 3u] = comY;
    clF[c * CL_HEADER_F + 4u] = theta;
  }
  let cosT = cos(theta);
  let sinT = sin(theta);
  let stiffness = clF[c * CL_HEADER_F];
  let maxCorrFactor = clF[c * CL_HEADER_F + 1u];

  for (var k = lid; k < count; k += 256u) {
    let i = clU[idxBase + start + k];
    if (pu[pu_i(PU_ALIVE, i)] != 1u) { continue; }
    if (pf[pf_i(PF_INVMASS, i)] == 0.0) { continue; }
    let qx = clF[base + k];
    let qy = clF[base + count + k];
    let gx = comX + cosT * qx - sinT * qy;
    let gy = comY + sinT * qx + cosT * qy;
    var dx = (gx - pf[pf_i(PF_POSX, i)]) * stiffness;
    var dy = (gy - pf[pf_i(PF_POSY, i)]) * stiffness;
    let maxCorr = maxCorrFactor * pf[pf_i(PF_RADIUS, i)];
    let mag = sqrt(dx * dx + dy * dy);
    if (mag > maxCorr && mag > 1e-12) {
      let s = maxCorr / mag;
      dx *= s;
      dy *= s;
    }
    pf[pf_i(PF_POSX, i)] += dx;
    pf[pf_i(PF_POSY, i)] += dy;
  }
}
`

/** One colour-range uniform per dispatch (group 1). */
const COLOR_GROUP = /* wgsl */ `
struct ColorRange { start: u32, count: u32, padA: u32, padB: u32 }
@group(1) @binding(0) var<uniform> CR: ColorRange;
`

/**
 * XPBD distance solve (constraints/distance.ts), colour-parallel.
 * Bindings g0: uni, pf, pu, df, du, colorIdx; g1: colour range.
 */
export const SRC_DIST_SOLVE = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> df: array<f32>;
@group(0) @binding(4) var<storage, read_write> du: array<u32>;
@group(0) @binding(5) var<storage, read_write> colorIdx: array<u32>;
${COLOR_GROUP}
${COMMON}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= CR.count) { return; }
  let i = colorIdx[CR.start + gid.x];
  let ia = du[du_i(DU_A, i)];
  let ib = du[du_i(DU_B, i)];
  let w1 = pf[pf_i(PF_INVMASS, ia)];
  let w2 = pf[pf_i(PF_INVMASS, ib)];
  let wSum = w1 + w2;
  if (wSum == 0.0) { return; }

  let dx = pf[pf_i(PF_POSX, ia)] - pf[pf_i(PF_POSX, ib)];
  let dy = pf[pf_i(PF_POSY, ia)] - pf[pf_i(PF_POSY, ib)];
  let len = sqrt(dx * dx + dy * dy);
  if (len < 1e-9) { return; }

  let rest = df[df_i(DF_REST, i)];
  let C = len - rest;
  let nx = dx / len;
  let ny = dy / len;

  let invH2 = 1.0 / (U.h * U.h);
  let alphaTilde = df[df_i(DF_COMPLIANCE, i)] * invH2;
  let lambda = df[df_i(DF_LAMBDA, i)];
  let dLambda = (-C - alphaTilde * lambda) / (wSum + alphaTilde);
  df[df_i(DF_LAMBDA, i)] = lambda + dLambda;

  pf[pf_i(PF_POSX, ia)] += w1 * dLambda * nx;
  pf[pf_i(PF_POSY, ia)] += w1 * dLambda * ny;
  pf[pf_i(PF_POSX, ib)] -= w2 * dLambda * nx;
  pf[pf_i(PF_POSY, ib)] -= w2 * dLambda * ny;

  df[df_i(DF_STRAIN, i)] = select(0.0, C / rest, rest > 1e-9);
}
`

/** Distance velocity damping (distance.ts dampVelocities), colour-parallel. */
export const SRC_DIST_DAMP = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> df: array<f32>;
@group(0) @binding(4) var<storage, read_write> du: array<u32>;
@group(0) @binding(5) var<storage, read_write> colorIdx: array<u32>;
${COLOR_GROUP}
${COMMON}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= CR.count) { return; }
  let i = colorIdx[CR.start + gid.x];
  let zeta = df[df_i(DF_ZETA, i)];
  if (zeta <= 0.0) { return; }
  let ia = du[du_i(DU_A, i)];
  let ib = du[du_i(DU_B, i)];
  let w1 = pf[pf_i(PF_INVMASS, ia)];
  let w2 = pf[pf_i(PF_INVMASS, ib)];
  let wSum = w1 + w2;
  if (wSum == 0.0) { return; }

  let dx = pf[pf_i(PF_POSX, ia)] - pf[pf_i(PF_POSX, ib)];
  let dy = pf[pf_i(PF_POSY, ia)] - pf[pf_i(PF_POSY, ib)];
  let len = sqrt(dx * dx + dy * dy);
  if (len < 1e-9) { return; }
  let nx = dx / len;
  let ny = dy / len;

  let compliance = df[df_i(DF_COMPLIANCE, i)];
  let k = select(1e12, 1.0 / compliance, compliance > 1e-12);
  let omega = sqrt(k * wSum);
  let factor = 1.0 - exp(-2.0 * zeta * omega * U.h);

  let vrel = (pf[pf_i(PF_VELX, ia)] - pf[pf_i(PF_VELX, ib)]) * nx
           + (pf[pf_i(PF_VELY, ia)] - pf[pf_i(PF_VELY, ib)]) * ny;
  let dv = vrel * factor;
  let s1 = w1 / wSum;
  let s2 = w2 / wSum;
  pf[pf_i(PF_VELX, ia)] -= s1 * dv * nx;
  pf[pf_i(PF_VELY, ia)] -= s1 * dv * ny;
  pf[pf_i(PF_VELX, ib)] += s2 * dv * nx;
  pf[pf_i(PF_VELY, ib)] += s2 * dv * ny;
}
`

/** XPBD bend solve (constraints/bending.ts), colour-parallel. */
export const SRC_BEND_SOLVE = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> bf: array<f32>;
@group(0) @binding(4) var<storage, read_write> bu: array<u32>;
@group(0) @binding(5) var<storage, read_write> colorIdx: array<u32>;
${COLOR_GROUP}
${COMMON}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= CR.count) { return; }
  let i = colorIdx[CR.start + gid.x];
  let ia = bu[bu_i(BU_A, i)];
  let ib = bu[bu_i(BU_B, i)];
  let ic = bu[bu_i(BU_C, i)];
  let w0 = pf[pf_i(PF_INVMASS, ia)];
  let w1 = pf[pf_i(PF_INVMASS, ib)];
  let w2 = pf[pf_i(PF_INVMASS, ic)];
  if (w0 + w1 + w2 == 0.0) { return; }

  let abx = pf[pf_i(PF_POSX, ib)] - pf[pf_i(PF_POSX, ia)];
  let aby = pf[pf_i(PF_POSY, ib)] - pf[pf_i(PF_POSY, ia)];
  let bcx = pf[pf_i(PF_POSX, ic)] - pf[pf_i(PF_POSX, ib)];
  let bcy = pf[pf_i(PF_POSY, ic)] - pf[pf_i(PF_POSY, ib)];
  let lab2 = abx * abx + aby * aby;
  let lbc2 = bcx * bcx + bcy * bcy;
  if (lab2 < 1e-12 || lbc2 < 1e-12) { return; }

  let theta = atan2(abx * bcy - aby * bcx, abx * bcx + aby * bcy);
  let C = angle_delta(theta, bf[bf_i(BF_RESTANGLE, i)]);
  bf[bf_i(BF_ANGLE, i)] = C;

  let n1x = -aby / lab2;
  let n1y = abx / lab2;
  let n2x = -bcy / lbc2;
  let n2y = bcx / lbc2;
  let gbx = -(n1x + n2x);
  let gby = -(n1y + n2y);

  let denom = w0 * (n1x * n1x + n1y * n1y)
            + w1 * (gbx * gbx + gby * gby)
            + w2 * (n2x * n2x + n2y * n2y);
  if (denom < 1e-12) { return; }

  let invH2 = 1.0 / (U.h * U.h);
  let alphaTilde = bf[bf_i(BF_COMPLIANCE, i)] * invH2;
  let lambda = bf[bf_i(BF_LAMBDA, i)];
  let dLambda = (-C - alphaTilde * lambda) / (denom + alphaTilde);
  bf[bf_i(BF_LAMBDA, i)] = lambda + dLambda;

  pf[pf_i(PF_POSX, ia)] += w0 * dLambda * n1x;
  pf[pf_i(PF_POSY, ia)] += w0 * dLambda * n1y;
  pf[pf_i(PF_POSX, ib)] += w1 * dLambda * gbx;
  pf[pf_i(PF_POSY, ib)] += w1 * dLambda * gby;
  pf[pf_i(PF_POSX, ic)] += w2 * dLambda * n2x;
  pf[pf_i(PF_POSY, ic)] += w2 * dLambda * n2y;
}
`

/** Bend velocity damping (bending.ts dampVelocities), colour-parallel. */
export const SRC_BEND_DAMP = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> bf: array<f32>;
@group(0) @binding(4) var<storage, read_write> bu: array<u32>;
@group(0) @binding(5) var<storage, read_write> colorIdx: array<u32>;
${COLOR_GROUP}
${COMMON}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= CR.count) { return; }
  let i = colorIdx[CR.start + gid.x];
  let zeta = bf[bf_i(BF_ZETA, i)];
  if (zeta <= 0.0) { return; }
  let ia = bu[bu_i(BU_A, i)];
  let ib = bu[bu_i(BU_B, i)];
  let ic = bu[bu_i(BU_C, i)];
  let w0 = pf[pf_i(PF_INVMASS, ia)];
  let w1 = pf[pf_i(PF_INVMASS, ib)];
  let w2 = pf[pf_i(PF_INVMASS, ic)];
  if (w0 + w1 + w2 == 0.0) { return; }

  let abx = pf[pf_i(PF_POSX, ib)] - pf[pf_i(PF_POSX, ia)];
  let aby = pf[pf_i(PF_POSY, ib)] - pf[pf_i(PF_POSY, ia)];
  let bcx = pf[pf_i(PF_POSX, ic)] - pf[pf_i(PF_POSX, ib)];
  let bcy = pf[pf_i(PF_POSY, ic)] - pf[pf_i(PF_POSY, ib)];
  let lab2 = abx * abx + aby * aby;
  let lbc2 = bcx * bcx + bcy * bcy;
  if (lab2 < 1e-12 || lbc2 < 1e-12) { return; }

  let n1x = -aby / lab2;
  let n1y = abx / lab2;
  let n2x = -bcy / lbc2;
  let n2y = bcx / lbc2;
  let gbx = -(n1x + n2x);
  let gby = -(n1y + n2y);

  let denom = w0 * (n1x * n1x + n1y * n1y)
            + w1 * (gbx * gbx + gby * gby)
            + w2 * (n2x * n2x + n2y * n2y);
  if (denom < 1e-12) { return; }

  let thetaDot = n1x * pf[pf_i(PF_VELX, ia)] + n1y * pf[pf_i(PF_VELY, ia)]
               + gbx * pf[pf_i(PF_VELX, ib)] + gby * pf[pf_i(PF_VELY, ib)]
               + n2x * pf[pf_i(PF_VELX, ic)] + n2y * pf[pf_i(PF_VELY, ic)];

  let compliance = bf[bf_i(BF_COMPLIANCE, i)];
  let k = select(1e12, 1.0 / compliance, compliance > 1e-12);
  let omega = sqrt(k * denom);
  let factor = 1.0 - exp(-2.0 * zeta * omega * U.h);
  let s = (thetaDot * factor) / denom;

  pf[pf_i(PF_VELX, ia)] -= w0 * s * n1x;
  pf[pf_i(PF_VELY, ia)] -= w0 * s * n1y;
  pf[pf_i(PF_VELX, ib)] -= w1 * s * gbx;
  pf[pf_i(PF_VELY, ib)] -= w1 * s * gby;
  pf[pf_i(PF_VELX, ic)] -= w2 * s * n2x;
  pf[pf_i(PF_VELY, ic)] -= w2 * s * n2y;
}
`

/**
 * Pack the gather-hot per-particle state into one vec4 - position, SIGNED
 * kernel mass (fluid: +pmass; solid: -rho0*volume, the displaced-water mass
 * of Akinci et al., sign carrying the kind), lambda slot. The projection
 * gathers are memory-latency bound on iGPUs: one 16-byte read per neighbour
 * instead of three to five scattered scalars is the difference between over
 * and under the frame budget. Rebuilt per projection iteration, because the
 * joints and previous iteration moved positions. Bindings: uni, pf, pu, gath.
 */
export const SRC_PACK_GATHER = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> gridS: array<u32>;
@group(0) @binding(4) var<storage, read_write> gath: array<vec4f>;
${COMMON}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  // GRID-SORTED order: slot k holds entry k's particle, so a bucket's
  // particles are CONTIGUOUS vec4s and the gather's inner loop reads
  // sequential memory - the difference between latency-bound and
  // bandwidth-bound on an iGPU.
  let k = gid.x;
  if (k >= gridS[U.tableSize]) { return; }
  let i = gridS[U.tableSize + 1u + k];
  let kind = pu[pu_i(PU_KIND, i)];
  var km = U.pmass;
  if (kind != KIND_FLUID) { km = -(U.waterDensity * pf[pf_i(PF_VOLUME, i)]); }
  gath[k] = vec4f(pf[pf_i(PF_POSX, i)], pf[pf_i(PF_POSY, i)], km, 0.0);
}
`

/**
 * PBF density + lambda, gather-per-fluid-particle (fluid.ts project's
 * accumulate pass, both sides evaluated from this side) over the packed
 * vec4 neighbour data. Fluid and displaced-solid contributions share one
 * code path - the maths is identical, only the mass source differs, and the
 * mass sign carries the kind. Bindings: uni, pf, pu, gridS, gath.
 */
export const SRC_FLUID_DENSITY = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> gridS: array<u32>;
@group(0) @binding(4) var<storage, read_write> gath: array<vec4f>;
${COMMON}
${GRID_FNS}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let kSelf = gid.x;
  if (kSelf >= gridS[U.tableSize]) { return; }
  let own = gath[kSelf];
  if (own.z <= 0.0) { return; } // solids carry negative kernel mass
  let i = gridS[U.tableSize + 1u + kSelf];
  let xi = own.x;
  let yi = own.y;

  var density = U.selfRho;
  var gradSum = 0.0;
  var gIX = 0.0;
  var gIY = 0.0;

  ${gatherLoop(/* wgsl */ `
    if (k == kSelf) { continue; }
    let q = gath[k];
    let dx = xi - q.x;
    let dy = yi - q.y;
    let r2 = dx * dx + dy * dy;
    if (r2 >= U.h2 || r2 <= 1e-18) { continue; }
    neigh++;
    let mj = abs(q.z);
    let d = U.h2 - r2;
    density += U.poly6 * d * d * d * mj;
    let r = sqrt(r2);
    let s = (U.spiky * (U.hK - r) * (U.hK - r) * mj / r) * U.invRho0;
    let gx = s * dx;
    let gy = s * dy;
    gIX += gx;
    gIY += gy;
    gradSum += gx * gx + gy * gy;
  `, 'gather_reach()', true)}

  pf[pf_i(PF_DENSITY, i)] = density;
  // Resist compression only (fluid.ts lambda pass), with the same error clamp.
  var C = density * U.invRho0 - 1.0;
  var lambda = 0.0;
  if (C > 0.0) {
    C = min(C, U.maxC);
    let gs = gradSum + gIX * gIX + gIY * gIY;
    lambda = -C / (gs + U.eps);
  }
  pf[pf_i(PF_LAMBDA, i)] = lambda;
  gath[kSelf] = vec4f(xi, yi, own.z, lambda);
}
`

/**
 * PBF correction, gather-per-fluid (fluid.ts project's correct pass) over
 * the packed vec4 data (position, signed kernel mass, lambda). Own
 * displacement into dp (applied by fluid_apply - Jacobi, like the CPU);
 * solid reaction shares accumulate atomically in fixed point. Only the rare
 * solid neighbour costs an extra scattered read (its invMass).
 * Bindings: uni, pf, pu, gridS, fx, gath.
 */
export const SRC_FLUID_CORRECT = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> gridS: array<u32>;
@group(0) @binding(4) var<storage, read_write> fx: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> gath: array<vec4f>;
${COMMON}
${GRID_FNS}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let kSelf = gid.x;
  if (kSelf >= gridS[U.tableSize]) { return; }
  let own = gath[kSelf];
  if (own.z <= 0.0) { return; } // solids carry negative kernel mass
  let i = gridS[U.tableSize + 1u + kSelf];
  var dpx = 0.0;
  var dpy = 0.0;
  {
    let xi = own.x;
    let yi = own.y;
    let lambdaI = own.w;
    let wi = pf[pf_i(PF_INVMASS, i)];

    ${gatherLoop(/* wgsl */ `
      if (k == kSelf) { continue; }
      let q = gath[k];
      let dx = xi - q.x;
      let dy = yi - q.y;
      let r2 = dx * dx + dy * dy;
      if (r2 >= U.h2 || r2 <= 1e-18) { continue; }
      neigh++;
      let d = U.h2 - r2;
      let w6 = U.poly6 * d * d * d;
      let r = sqrt(r2);
      let wSpiky = U.spiky * (U.hK - r) * (U.hK - r);
      if (q.z > 0.0) {
        let s = (wSpiky * U.pmass / r) * U.invRho0;
        let gx = s * dx;
        let gy = s * dy;
        // Artificial pressure (surfaceTensionN = 4: a multiply chain).
        let ratio = w6 * U.invSCorrDenom;
        let r2p = ratio * ratio;
        let corr = -U.sCorrK * r2p * r2p;
        let f = lambdaI + q.w + corr;
        dpx += f * gx;
        dpy += f * gy;
      } else {
        let mj = -q.z;
        let s = (wSpiky * mj / r) * U.invRho0;
        let gx = s * dx;
        let gy = s * dy;
        let j = gridS[U.tableSize + 1u + k];
        let wj = pf[pf_i(PF_INVMASS, j)];
        let boost = select(1.0, U.hullPressure, wj > 0.0);
        let f = lambdaI * boost;
        let wSum = wi + wj;
        if (wSum > 0.0) {
          let shareI = wi / wSum;
          let shareJ = wj / wSum;
          dpx += f * gx * shareI;
          dpy += f * gy * shareI;
          if (shareJ > 0.0) {
            atomicAdd(&fx[${FX.solidX}u * U.cap + j], i32(-f * gx * shareJ * U.fpScale));
            atomicAdd(&fx[${FX.solidY}u * U.cap + j], i32(-f * gy * shareJ * U.fpScale));
          }
        }
      }
    `, 'gather_reach()', true)}
  }
  pf[pf_i(PF_DPX, i)] = dpx;
  pf[pf_i(PF_DPY, i)] = dpy;
}
`

/**
 * Apply the projection's position corrections, clamped: the fluid's own dp
 * (fluid.ts apply pass) and the solids' atomically-accumulated reaction
 * shares (fluid.ts solidTouched loop, drained and zeroed here). Disjoint
 * kinds, one dispatch. Bindings: uni, pf, pu, fx.
 */
export const SRC_FLUID_APPLY = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> fx: array<atomic<i32>>;
${COMMON}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.n) { return; }
  if (pu[pu_i(PU_KIND, i)] == KIND_FLUID) {
    if (pu[pu_i(PU_ALIVE, i)] != 1u) { return; }
    if (pf[pf_i(PF_INVMASS, i)] == 0.0) { return; }
    var dx = pf[pf_i(PF_DPX, i)];
    var dy = pf[pf_i(PF_DPY, i)];
    let mag = sqrt(dx * dx + dy * dy);
    if (mag > U.maxCorrSub && mag > 1e-12) {
      let k = U.maxCorrSub / mag;
      dx *= k;
      dy *= k;
    }
    pf[pf_i(PF_POSX, i)] += dx;
    pf[pf_i(PF_POSY, i)] += dy;
  } else {
    let sxRaw = atomicExchange(&fx[${FX.solidX}u * U.cap + i], 0);
    let syRaw = atomicExchange(&fx[${FX.solidY}u * U.cap + i], 0);
    if (sxRaw == 0 && syRaw == 0) { return; }
    var sx = f32(sxRaw) * U.invFp;
    var sy = f32(syRaw) * U.invFp;
    let mag = sqrt(sx * sx + sy * sy);
    if (mag > U.maxCorrSub) {
      let k = U.maxCorrSub / mag;
      sx *= k;
      sy *= k;
    }
    pf[pf_i(PF_POSX, i)] += sx;
    pf[pf_i(PF_POSY, i)] += sy;
  }
}
`

/**
 * Object-object contacts, gather-per-particle (solver.ts solveSolidContacts,
 * each side computing its own inverse-mass share - the maths the CPU applies
 * to both sides of each pair, evaluated from each side). Own-thread
 * accumulators; no atomics. Bindings: uni, pf, pu, gridS.
 */
export const SRC_CONTACTS_SOLID = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> gridS: array<u32>;
${COMMON}
${GRID_FNS}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.n) { return; }
  if (pu[pu_i(PU_ALIVE, i)] != 1u) { return; }
  if (pu[pu_i(PU_KIND, i)] != KIND_OBJECT) { return; }
  let wi = pf[pf_i(PF_INVMASS, i)];
  if (wi <= 0.0) { return; }
  let ci = pu[pu_i(PU_CLUSTER1, i)];
  let xi = pf[pf_i(PF_POSX, i)];
  let yi = pf[pf_i(PF_POSY, i)];
  let ri = pf[pf_i(PF_RADIUS, i)];

  var accX = 0.0;
  var accY = 0.0;
  var hits = 0u;

  ${gatherLoop(/* wgsl */ `
    if (j == i) { continue; }
    if (pu[pu_i(PU_ALIVE, j)] != 1u) { continue; }
    if (pu[pu_i(PU_KIND, j)] != KIND_OBJECT) { continue; }
    if (pu[pu_i(PU_CLUSTER1, j)] == ci) { continue; }
    let dx = xi - pf[pf_i(PF_POSX, j)];
    let dy = yi - pf[pf_i(PF_POSY, j)];
    let minDist = ri + pf[pf_i(PF_RADIUS, j)];
    let d2 = dx * dx + dy * dy;
    if (d2 >= minDist * minDist || d2 < 1e-12) { continue; }
    neigh++;
    let dist = sqrt(d2);
    let nx = dx / dist;
    let ny = dy / dist;
    let wj = pf[pf_i(PF_INVMASS, j)];
    let wSum = wi + wj;
    if (wSum <= 0.0) { continue; }
    let pen = min(minDist - dist, U.maxContactCorr);
    let scale = (pen * U.contactRelax) / wSum;
    accX += nx * scale * wi;
    accY += ny * scale * wi;
    hits++;
  `, "gather_reach()")}

  pf[pf_i(PF_CONX, i)] += accX;
  pf[pf_i(PF_CONY, i)] += accY;
  pu[pu_i(PU_CONHITS, i)] += hits;
}
`

/**
 * Member capsule contacts, gather-per-particle over members (solver.ts
 * solveMemberContacts). Per (member, particle) exactly one contact from the
 * closest point on the full segment - what the CPU's sample-point walk with
 * the stamp dedupe computes. Endpoint reactions scatter in fixed point.
 * Bindings: uni, pf, pu, df, du, fx, fc, matProps.
 */
export const SRC_CONTACTS_MEMBER = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> df: array<f32>;
@group(0) @binding(4) var<storage, read_write> du: array<u32>;
@group(0) @binding(5) var<storage, read_write> fx: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> fc: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> matProps: array<f32>;
${COMMON}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let j = gid.x;
  if (j >= U.n || U.distHW == 0u) { return; }
  if (pu[pu_i(PU_ALIVE, j)] != 1u) { return; }
  let kindJ = pu[pu_i(PU_KIND, j)];
  if (kindJ == KIND_NODE) { return; }
  let wj = pf[pf_i(PF_INVMASS, j)];
  if (wj == 0.0) { return; }
  let xj = pf[pf_i(PF_POSX, j)];
  let yj = pf[pf_i(PF_POSY, j)];
  let rj = pf[pf_i(PF_RADIUS, j)];
  // Whole-structure early-out: most water is nowhere near any member.
  if (xj < U.memAabbX0 || xj > U.memAabbX1 || yj < U.memAabbY0 || yj > U.memAabbY1) { return; }
  let cl1 = pu[pu_i(PU_CLUSTER1, j)];

  var accX = 0.0;
  var accY = 0.0;
  var hits = 0u;

  for (var m = 0u; m < U.distHW; m++) {
    if (du[du_i(DU_ALIVE, m)] != 1u) { continue; }
    let rest = df[df_i(DF_REST, m)];
    if (rest <= 1e-6) { continue; } // welds are points, not surfaces
    let ia = du[du_i(DU_A, m)];
    let ib = du[du_i(DU_B, m)];
    if (j == ia || j == ib) { continue; }
    let noCol1 = du[du_i(DU_NOCOL1, m)];
    if (noCol1 != 0u && cl1 == noCol1) { continue; }

    let ax = pf[pf_i(PF_POSX, ia)];
    let ay = pf[pf_i(PF_POSY, ia)];
    let bx = pf[pf_i(PF_POSX, ib)];
    let by = pf[pf_i(PF_POSY, ib)];
    let mat = du[du_i(DU_MAT, m)];
    let radius = max(matProps[mat * 2u] * 0.5, U.spacing * 0.75);
    let reach = radius + rj;
    // Per-member AABB reject.
    if (xj < min(ax, bx) - reach || xj > max(ax, bx) + reach) { continue; }
    if (yj < min(ay, by) - reach || yj > max(ay, by) + reach) { continue; }

    let ex = bx - ax;
    let ey = by - ay;
    let len2 = ex * ex + ey * ey;
    if (len2 < 1e-12) { continue; }
    let segLen = sqrt(len2);

    let px = xj - ax;
    let py = yj - ay;
    var u = (px * ex + py * ey) / len2;
    u = clamp(u, 0.0, 1.0);
    let cx = ax + ex * u;
    let cy = ay + ey * u;
    var nx = xj - cx;
    var ny = yj - cy;
    let dist = sqrt(nx * nx + ny * ny);
    if (dist >= reach) { continue; }

    if (dist < 1e-6) {
      nx = -ey / segLen;
      ny = ex / segLen;
    } else {
      nx /= dist;
      ny /= dist;
    }

    let wa = pf[pf_i(PF_INVMASS, ia)];
    let wb = pf[pf_i(PF_INVMASS, ib)];
    let u1 = 1.0 - u;
    let wSeg = wa * u1 * u1 + wb * u * u;
    let wSum = wj + wSeg;
    if (wSum <= 0.0) { continue; }
    let pen = min(reach - dist, U.maxContactCorr);
    let push = pen * U.contactRelax;
    let scale = push / wSum;

    accX += nx * scale * wj;
    accY += ny * scale * wj;
    hits++;

    if (wa > 0.0) {
      atomicAdd(&fx[${FX.memberX}u * U.cap + ia], i32(-nx * scale * wa * u1 * U.fpScale));
      atomicAdd(&fx[${FX.memberY}u * U.cap + ia], i32(-ny * scale * wa * u1 * U.fpScale));
      atomicAdd(&fc[${FC.memberHits}u * U.cap + ia], 1u);
    }
    if (wb > 0.0) {
      atomicAdd(&fx[${FX.memberX}u * U.cap + ib], i32(-nx * scale * wb * u * U.fpScale));
      atomicAdd(&fx[${FX.memberY}u * U.cap + ib], i32(-ny * scale * wb * u * U.fpScale));
      atomicAdd(&fc[${FC.memberHits}u * U.cap + ib], 1u);
    }
  }

  pf[pf_i(PF_CONX, j)] += accX;
  pf[pf_i(PF_CONY, j)] += accY;
  pu[pu_i(PU_CONHITS, j)] += hits;
}
`

/**
 * The per-particle substep tail, fused into one dispatch: contact resolution
 * (solver.ts resolveContacts: averaged, both position halves, damped inbound
 * normal), member-endpoint reactions (summed, position only - endpoints are
 * nodes, contact targets are hashed kinds, disjoint populations), terrain and
 * field-edge contacts (solveContacts), and the velocity derivation
 * (updateVelocities). Every pass reads and writes only its own particle, so
 * fusing them is free of hazards and saves three full-buffer barriers per
 * substep. Bindings: uni, pf, pu, fx, fc, terr.
 */
export const SRC_INTEGRATE = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> fx: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> fc: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> terr: array<f32>;
${COMMON}
${TERRAIN_FNS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.n) { return; }
  let alive = pu[pu_i(PU_ALIVE, i)] == 1u;
  let wi = pf[pf_i(PF_INVMASS, i)];

  let hits = pu[pu_i(PU_CONHITS, i)];
  if (hits > 0u && alive && wi != 0.0) {
    var dx = pf[pf_i(PF_CONX, i)] / f32(hits);
    var dy = pf[pf_i(PF_CONY, i)] / f32(hits);
    let mag = sqrt(dx * dx + dy * dy);
    if (mag >= 1e-12) {
      if (mag > U.maxContactCorr) {
        let k = U.maxContactCorr / mag;
        dx *= k;
        dy *= k;
      }
      // BOTH halves together (see the CPU comment): the push itself carries
      // no velocity, then the inbound normal part of the existing motion is
      // damped rather than annihilated.
      pf[pf_i(PF_POSX, i)] += dx;
      pf[pf_i(PF_POSY, i)] += dy;
      pf[pf_i(PF_PREVX, i)] += dx;
      pf[pf_i(PF_PREVY, i)] += dy;

      let inv = 1.0 / sqrt(dx * dx + dy * dy);
      let nx = dx * inv;
      let ny = dy * inv;
      let rx = pf[pf_i(PF_POSX, i)] - pf[pf_i(PF_PREVX, i)];
      let ry = pf[pf_i(PF_POSY, i)] - pf[pf_i(PF_PREVY, i)];
      let vn = rx * nx + ry * ny;
      if (vn < 0.0) {
        let damp = U.normalDamping;
        pf[pf_i(PF_PREVX, i)] = pf[pf_i(PF_POSX, i)] - (rx - vn * nx * damp);
        pf[pf_i(PF_PREVY, i)] = pf[pf_i(PF_POSY, i)] - (ry - vn * ny * damp);
      }
    }
  }
  pf[pf_i(PF_CONX, i)] = 0.0;
  pf[pf_i(PF_CONY, i)] = 0.0;
  pu[pu_i(PU_CONHITS, i)] = 0u;

  let mHits = atomicExchange(&fc[${FC.memberHits}u * U.cap + i], 0u);
  let mxRaw = atomicExchange(&fx[${FX.memberX}u * U.cap + i], 0);
  let myRaw = atomicExchange(&fx[${FX.memberY}u * U.cap + i], 0);
  if (mHits > 0u && alive && wi != 0.0) {
    var sx = f32(mxRaw) * U.invFp;
    var sy = f32(myRaw) * U.invFp;
    let mmag = sqrt(sx * sx + sy * sy);
    if (mmag >= 1e-12) {
      if (mmag > U.maxContactCorr) {
        let k = U.maxContactCorr / mmag;
        sx *= k;
        sy *= k;
      }
      pf[pf_i(PF_POSX, i)] += sx;
      pf[pf_i(PF_POSY, i)] += sy;
    }
  }

  // ---- terrain + field-edge contacts (solver.ts solveContacts) ----
  pu[pu_i(PU_GROUNDED, i)] = 0u;
  if (!alive || wi == 0.0) { return; }
  let r = pf[pf_i(PF_RADIUS, i)];

  var x = pf[pf_i(PF_POSX, i)];
  if (x < U.boundsX0 + r || x > U.boundsX1 - r) {
    x = select(U.boundsX1 - r, U.boundsX0 + r, x < U.boundsX0 + r);
    pf[pf_i(PF_POSX, i)] = x;
    pf[pf_i(PF_PREVX, i)] = x;
  }

  let kind = pu[pu_i(PU_KIND, i)];
  if (U.terrCount > 0u) {
    let floorY = height_at(x) + r;
    var y = pf[pf_i(PF_POSY, i)];
    if (y < floorY) {
      let pushCap = select(U.maxTerrainPush, U.maxTerrainPush * 0.1, kind == KIND_NODE);
      let nrm = normal_at(x);
      let gap = min(floorY - y, pushCap);
      let d = gap * nrm.y;
      x += nrm.x * d;
      y += nrm.y * d;
      pf[pf_i(PF_POSX, i)] = x;
      pf[pf_i(PF_POSY, i)] = y;
      pf[pf_i(PF_PREVX, i)] += nrm.x * d;
      pf[pf_i(PF_PREVY, i)] += nrm.y * d;

      let rx = x - pf[pf_i(PF_PREVX, i)];
      let ry = y - pf[pf_i(PF_PREVY, i)];
      let vn = rx * nrm.x + ry * nrm.y;
      if (vn < 0.0) {
        pf[pf_i(PF_PREVX, i)] = x - (rx - vn * nrm.x);
        pf[pf_i(PF_PREVY, i)] = y - (ry - vn * nrm.y);
      }
      pu[pu_i(PU_GROUNDED, i)] = 1u;
    }
  }

  // ---- velocity derivation + damping + friction (updateVelocities) ----
  let invH = 1.0 / U.h;
  var vx = (pf[pf_i(PF_POSX, i)] - pf[pf_i(PF_PREVX, i)]) * invH;
  var vy = (pf[pf_i(PF_POSY, i)] - pf[pf_i(PF_PREVY, i)]) * invH;

  let sp2 = vx * vx + vy * vy;
  if (sp2 > U.maxSpeed * U.maxSpeed) {
    let k = U.maxSpeed / sqrt(sp2);
    vx *= k;
    vy *= k;
  }

  if (kind == KIND_NODE) {
    vx *= U.keepNode;
    vy *= U.keepNode;
  } else if (kind == KIND_FLUID && U.keepFluid < 1.0) {
    vx *= U.keepFluid;
    vy *= U.keepFluid;
  }

  if (pu[pu_i(PU_GROUNDED, i)] == 1u) {
    let friction = select(U.fricSolid, U.fricFluid, kind == KIND_FLUID);
    vx *= friction;
    if (vy > 0.0) { vy *= U.restitution; }
  }

  pf[pf_i(PF_VELX, i)] = vx;
  pf[pf_i(PF_VELY, i)] = vy;
}
`

/** Copy vel into the Jacobi snapshot sections. Bindings: uni, pf, pu. */
export const SRC_SNAP_VEL = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
${COMMON}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.n) { return; }
  pf[pf_i(PF_VELTX, i)] = pf[pf_i(PF_VELX, i)];
  pf[pf_i(PF_VELTY, i)] = pf[pf_i(PF_VELY, i)];
}
`

/**
 * Hull viscosity, fluid side (fluid.ts applyHullViscosity): gather DYNAMIC
 * solid neighbours, apply own share from the velocity snapshot. The CPU
 * applies pairs sequentially over the frame-frozen pair list; this is the
 * Jacobi per-substep-pairs form - a stated divergence. Bindings: uni, pf, pu, gridS.
 */
export const SRC_HULL_FLUID = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> gridS: array<u32>;
${COMMON}
${GRID_FNS}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.n) { return; }
  if (U.hullC0 <= 0.0 && U.hullC1 <= 0.0) { return; }
  if (pu[pu_i(PU_ALIVE, i)] != 1u) { return; }
  if (pu[pu_i(PU_KIND, i)] != KIND_FLUID) { return; }
  let wi = pf[pf_i(PF_INVMASS, i)];
  let xi = pf[pf_i(PF_POSX, i)];
  let yi = pf[pf_i(PF_POSY, i)];
  let vix = pf[pf_i(PF_VELTX, i)];
  let viy = pf[pf_i(PF_VELTY, i)];
  var dvxAcc = 0.0;
  var dvyAcc = 0.0;

  ${gatherLoop(/* wgsl */ `
    if (j == i) { continue; }
    let kindJ = pu[pu_i(PU_KIND, j)];
    if (kindJ == KIND_FLUID) { continue; }
    let wj = pf[pf_i(PF_INVMASS, j)];
    if (wj <= 0.0) { continue; } // static bed: not a hull
    let dx = xi - pf[pf_i(PF_POSX, j)];
    let dy = yi - pf[pf_i(PF_POSY, j)];
    if (dx * dx + dy * dy >= U.rq2) { continue; }
    neigh++;
    let wSum = wi + wj;
    if (wSum <= 0.0) { continue; }
    let dvx = pf[pf_i(PF_VELTX, j)] - vix;
    let dvy = pf[pf_i(PF_VELTY, j)] - viy;
    var c = U.hullC0 + U.hullC1 * sqrt(dvx * dvx + dvy * dvy);
    c = min(c, U.hullCMax);
    c /= max(1.0, f32(pu[pu_i(PU_WET, j)]));
    let shareI = wi / wSum;
    dvxAcc += dvx * c * shareI;
    dvyAcc += dvy * c * shareI;
  `, "(gather_reach() + (U.rq - U.hK))")}

  pf[pf_i(PF_VELX, i)] += dvxAcc;
  pf[pf_i(PF_VELY, i)] += dvyAcc;
}
`

/** Hull viscosity, solid side: gather fluid neighbours, apply own share. */
export const SRC_HULL_SOLID = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> gridS: array<u32>;
${COMMON}
${GRID_FNS}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let j = gid.x;
  if (j >= U.n) { return; }
  if (U.hullC0 <= 0.0 && U.hullC1 <= 0.0) { return; }
  if (pu[pu_i(PU_ALIVE, j)] != 1u) { return; }
  let kindJ = pu[pu_i(PU_KIND, j)];
  if (kindJ == KIND_FLUID || kindJ == KIND_NODE) { return; }
  let wj = pf[pf_i(PF_INVMASS, j)];
  if (wj <= 0.0) { return; }
  let xi = pf[pf_i(PF_POSX, j)];
  let yi = pf[pf_i(PF_POSY, j)];
  let vjx = pf[pf_i(PF_VELTX, j)];
  let vjy = pf[pf_i(PF_VELTY, j)];
  let wetJ = max(1.0, f32(pu[pu_i(PU_WET, j)]));
  var dvxAcc = 0.0;
  var dvyAcc = 0.0;

  ${gatherLoop(/* wgsl */ `
    if (j2 == j) { continue; }
    if (pu[pu_i(PU_KIND, j2)] != KIND_FLUID) { continue; }
    let wi = pf[pf_i(PF_INVMASS, j2)];
    let dx = xi - pf[pf_i(PF_POSX, j2)];
    let dy = yi - pf[pf_i(PF_POSY, j2)];
    if (dx * dx + dy * dy >= U.rq2) { continue; }
    neigh++;
    let wSum = wi + wj;
    if (wSum <= 0.0) { continue; }
    let dvx = vjx - pf[pf_i(PF_VELTX, j2)];
    let dvy = vjy - pf[pf_i(PF_VELTY, j2)];
    var c = U.hullC0 + U.hullC1 * sqrt(dvx * dvx + dvy * dvy);
    c = min(c, U.hullCMax);
    c /= wetJ;
    let shareJ = wj / wSum;
    dvxAcc -= dvx * c * shareJ;
    dvyAcc -= dvy * c * shareJ;
  `, "(gather_reach() + (U.rq - U.hK))").replace(/\blet j = /g, 'let j2 = ')}

  pf[pf_i(PF_VELX, j)] += dvxAcc;
  pf[pf_i(PF_VELY, j)] += dvyAcc;
}
`

/**
 * Pack (position, velocity) into the sorted gather buffer for XSPH - the
 * velocity here IS the Jacobi snapshot: taken before any thread writes.
 * Bindings: uni, pf, pu, gridS, gath.
 */
export const SRC_PACK_XSPH = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> gridS: array<u32>;
@group(0) @binding(4) var<storage, read_write> gath: array<vec4f>;
${COMMON}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= gridS[U.tableSize]) { return; }
  let i = gridS[U.tableSize + 1u + k];
  gath[k] = vec4f(
    pf[pf_i(PF_POSX, i)],
    pf[pf_i(PF_POSY, i)],
    pf[pf_i(PF_VELX, i)],
    pf[pf_i(PF_VELY, i)],
  );
}
`

/**
 * XSPH viscosity (fluid.ts applyViscosity), sorted-gather form over the
 * packed (pos, vel) snapshot. Every hashed neighbour contributes, solid or
 * fluid - the CPU's ff and fs passes in one loop; only fluid is nudged.
 * Bindings: uni, pf, pu, gridS, gath.
 */
export const SRC_XSPH = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> gridS: array<u32>;
@group(0) @binding(4) var<storage, read_write> gath: array<vec4f>;
${COMMON}
${GRID_FNS}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let kSelf = gid.x;
  if (kSelf >= gridS[U.tableSize]) { return; }
  if (U.xsphVisc <= 0.0) { return; }
  let i = gridS[U.tableSize + 1u + kSelf];
  if (pu[pu_i(PU_KIND, i)] != KIND_FLUID) { return; }
  if (pf[pf_i(PF_INVMASS, i)] == 0.0) { return; }
  let own = gath[kSelf];
  let xi = own.x;
  let yi = own.y;
  let vix = own.z;
  let viy = own.w;
  var numX = 0.0;
  var numY = 0.0;
  var wsum = 0.0;

  ${gatherLoop(/* wgsl */ `
    if (k == kSelf) { continue; }
    let q = gath[k];
    let dx = xi - q.x;
    let dy = yi - q.y;
    let r2 = dx * dx + dy * dy;
    if (r2 >= U.h2) { continue; }
    neigh++;
    let d = U.h2 - r2;
    let w = U.poly6 * d * d * d;
    numX += (q.z - vix) * w;
    numY += (q.w - viy) * w;
    wsum += w;
  `, 'gather_reach()', true)}

  if (wsum <= 1e-12) { return; }
  pf[pf_i(PF_VELX, i)] = vix + (numX / wsum) * U.xsphVisc;
  pf[pf_i(PF_VELY, i)] = viy + (numY / wsum) * U.xsphVisc;
}
`

/** Zero the grid bucket counts (per frame). Bindings: uni, gridA. */
export const SRC_GRID_CLEAR = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> gridA: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> gridS: array<u32>;
${COMMON}
${GRID_FNS}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x == 0u) { gridS[substep_idx()] = 0u; }
  if (gid.x >= U.tableSize) { return; }
  atomicStore(&gridA[gid.x], 0u);
}
`

/**
 * Column-field helpers - for kernels that bind colF. Mirrors sim/water.ts
 * surfaceAt / velocityAt / submergedFraction exactly; -1e30 is the "dry"
 * sentinel standing in for the CPU's -Infinity.
 */
const COL_FNS = /* wgsl */ `
fn col_of(x: f32) -> i32 { return i32(floor((x - U.colX0) * U.colInvRes)); }
fn surface_at(x: f32) -> f32 {
  let c = col_of(x);
  if (c < 0 || c >= i32(U.colN)) { return -1.0e30; }
  return colF[${CF.surface}u * U.colN + u32(c)];
}
fn col_vel(x: f32) -> vec2f {
  let c = col_of(x);
  if (c < 0 || c >= i32(U.colN)) { return vec2f(0.0, 0.0); }
  return vec2f(colF[${CF.velX}u * U.colN + u32(c)], colF[${CF.velY}u * U.colN + u32(c)]);
}
fn submerged_frac(x: f32, y: f32, r: f32) -> f32 {
  let s = surface_at(x);
  if (s < -1.0e29) { return 0.0; }
  let depth = s - (y - r);
  if (depth <= 0.0) { return 0.0; }
  let span = 2.0 * r;
  return select(depth / span, 1.0, depth >= span);
}
`

/** Reset the column accumulators and field. Bindings: uni, colA, colF. */
export const SRC_COL_CLEAR = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> colA: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> colF: array<f32>;
${COMMON}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let c = gid.x;
  if (c >= U.colN) { return; }
  atomicStore(&colA[${CA.count}u * U.colN + c], 0);
  atomicStore(&colA[${CA.floorFP}u * U.colN + c], 2147483647);
  atomicStore(&colA[${CA.velXFP}u * U.colN + c], 0);
  atomicStore(&colA[${CA.velYFP}u * U.colN + c], 0);
  colF[${CF.surface}u * U.colN + c] = -1.0e30;
  colF[${CF.velX}u * U.colN + c] = 0.0;
  colF[${CF.velY}u * U.colN + c] = 0.0;
}
`

/** Scatter fluid particles into their columns (water.ts build loop):
 *  count, velocity sum (FORCE_FP), column floor (position-grade FP min).
 *  Bindings: uni, pf, pu, colA. */
export const SRC_COL_ACCUM = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> colA: array<atomic<i32>>;
${COMMON}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.n) { return; }
  if (pu[pu_i(PU_ALIVE, i)] != 1u) { return; }
  if (pu[pu_i(PU_KIND, i)] != KIND_FLUID) { return; }
  let c = i32(floor((pf[pf_i(PF_POSX, i)] - U.colX0) * U.colInvRes));
  if (c < 0 || c >= i32(U.colN)) { return; }
  let cu = u32(c);
  atomicAdd(&colA[${CA.count}u * U.colN + cu], 1);
  atomicAdd(&colA[${CA.velXFP}u * U.colN + cu], i32(pf[pf_i(PF_VELX, i)] * AFP));
  atomicAdd(&colA[${CA.velYFP}u * U.colN + cu], i32(pf[pf_i(PF_VELY, i)] * AFP));
  atomicMin(&colA[${CA.floorFP}u * U.colN + cu], i32(pf[pf_i(PF_POSY, i)] * U.fpScale));
}
`

/** Surface from column VOLUME, not the topmost particle (water.ts:64-79 -
 *  spray must not define the waterline), plus the mean column velocity.
 *  Bindings: uni, colA, colF. */
export const SRC_COL_SURFACE = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> colA: array<i32>;
@group(0) @binding(2) var<storage, read_write> colF: array<f32>;
${COMMON}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let c = gid.x;
  if (c >= U.colN) { return; }
  let k = colA[${CA.count}u * U.colN + c];
  if (k < 3) { return; } // stays at the dry sentinel
  let depth = f32(k) * (U.spacing * U.spacing) * U.colInvRes;
  let floorY = f32(colA[${CA.floorFP}u * U.colN + c]) * U.invFp;
  colF[${CF.surface}u * U.colN + c] = floorY + depth;
  let inv = 1.0 / (f32(k) * AFP);
  colF[${CF.velX}u * U.colN + c] = f32(colA[${CA.velXFP}u * U.colN + c]) * inv;
  colF[${CF.velY}u * U.colN + c] = f32(colA[${CA.velYFP}u * U.colN + c]) * inv;
}
`

/** One smoothing pass so a choppy surface does not make lift chatter -
 *  SEQUENTIAL and in place, matching the CPU's ascending in-place loop
 *  (water.ts:81-98) where a column's smooth reads the already-smoothed
 *  left neighbour. ~120 columns; one thread, like gridScan2.
 *  Bindings: uni, colF. */
export const SRC_COL_SMOOTH = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> colF: array<f32>;
${COMMON}
@compute @workgroup_size(1)
fn main() {
  if (U.colN < 3u) { return; }
  for (var c = 1u; c < U.colN - 1u; c++) {
    let b = colF[${CF.surface}u * U.colN + c];
    if (b < -1.0e29) { continue; }
    var sum = b;
    var k = 1.0;
    let a = colF[${CF.surface}u * U.colN + c - 1u];
    if (a > -1.0e29) { sum += a; k += 1.0; }
    let d = colF[${CF.surface}u * U.colN + c + 1u];
    if (d > -1.0e29) { sum += d; k += 1.0; }
    colF[${CF.surface}u * U.colN + c] = sum / k;
  }
}
`

/**
 * Member coupling forces, per distance constraint: hydrostatic wall load
 * (world.ts applyHydrostaticLoad - horizontal only, the vertical pressure
 * difference IS buoyancy) and quadratic water drag (applyWaterDrag - the
 * impulse cap keeps light nodes from overshooting the shared velocity).
 * Guards mirror the CPU exactly: joinery (unbreakable, zero-rest) carries
 * no wall area and catches no drag. Endpoint accelerations scatter into
 * frc in FORCE_FP fixed point; forcesApply resolves them.
 * Bindings: uni, pf, pu, df, du, frc, matProps, colF.
 */
export const SRC_FORCES_MEMBER = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> df: array<f32>;
@group(0) @binding(4) var<storage, read_write> du: array<u32>;
@group(0) @binding(5) var<storage, read_write> frc: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> matProps: array<f32>;
@group(0) @binding(7) var<storage, read_write> colF: array<f32>;
${COMMON}
${COL_FNS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let m = gid.x;
  if (m >= U.distHW) { return; }
  if (du[du_i(DU_ALIVE, m)] != 1u) { return; }
  if (du[du_i(DU_UNBREAK, m)] == 1u) { return; }
  let rest = df[df_i(DF_REST, m)];
  if (rest <= 1e-6) { return; }
  let ia = du[du_i(DU_A, m)];
  let ib = du[du_i(DU_B, m)];
  let wa = pf[pf_i(PF_INVMASS, ia)];
  let wb = pf[pf_i(PF_INVMASS, ib)];
  if (wa == 0.0 && wb == 0.0) { return; }

  let ax = pf[pf_i(PF_POSX, ia)];
  let ay = pf[pf_i(PF_POSY, ia)];
  let bx = pf[pf_i(PF_POSX, ib)];
  let by = pf[pf_i(PF_POSY, ib)];
  let midX = (ax + bx) * 0.5;
  let midY = (ay + by) * 0.5;

  // Per-endpoint force shares (before the endpoint's own invMass).
  var fxA = 0.0;
  var fyA = 0.0;
  var fxB = 0.0;
  var fyB = 0.0;

  // Hydrostatic wall load - horizontal only.
  let vertical = abs(by - ay);
  if (vertical >= 0.05) {
    let rhoG = U.waterDensity * (-U.gravity);
    let sL = surface_at(midX - U.hydroOff);
    let sR = surface_at(midX + U.hydroOff);
    let pL = select(0.0, max(0.0, rhoG * (sL - midY)), sL > -1.0e29);
    let pR = select(0.0, max(0.0, rhoG * (sR - midY)), sR > -1.0e29);
    let net = pL - pR;
    if (net != 0.0) {
      let fx = net * vertical;
      fxA += fx * 0.5;
      fxB += fx * 0.5;
    }
  }

  // Quadratic drag against the local column flow.
  let mat = du[du_i(DU_MAT, m)];
  let section = matProps[mat * 2u];
  let submerged = submerged_frac(midX, midY, section * 0.5);
  if (submerged > 0.01) {
    let cv = col_vel(midX);
    let relX = cv.x - (pf[pf_i(PF_VELX, ia)] + pf[pf_i(PF_VELX, ib)]) * 0.5;
    let relY = cv.y - (pf[pf_i(PF_VELY, ia)] + pf[pf_i(PF_VELY, ib)]) * 0.5;
    let relSpeed = sqrt(relX * relX + relY * relY);
    let ex = bx - ax;
    let ey = by - ay;
    let len = sqrt(ex * ex + ey * ey);
    if (relSpeed >= 1e-6 && len >= 1e-9) {
      let dirX = relX / relSpeed;
      let dirY = relY / relSpeed;
      let cross = abs((ex * dirY - ey * dirX) / len);
      let frontal = rest * cross + section;
      var force = 0.5 * U.waterDensity * matProps[mat * 2u + 1u] * frontal * relSpeed * relSpeed * submerged;
      // Drag may slow relative motion, never reverse it (world.ts:467-472).
      let massAB = select(0.0, 1.0 / wa, wa > 0.0) + select(0.0, 1.0 / wb, wb > 0.0);
      let maxForce = massAB * relSpeed / U.dt;
      force = min(force, maxForce);
      fxA += force * dirX * 0.5;
      fyA += force * dirY * 0.5;
      fxB += force * dirX * 0.5;
      fyB += force * dirY * 0.5;
    }
  }

  if (wa > 0.0 && (fxA != 0.0 || fyA != 0.0)) {
    atomicAdd(&frc[${FRC.accX}u * U.cap + ia], i32(fxA * wa * AFP));
    atomicAdd(&frc[${FRC.accY}u * U.cap + ia], i32(fyA * wa * AFP));
  }
  if (wb > 0.0 && (fxB != 0.0 || fyB != 0.0)) {
    atomicAdd(&frc[${FRC.accX}u * U.cap + ib], i32(fxB * wb * AFP));
    atomicAdd(&frc[${FRC.accY}u * U.cap + ib], i32(fyB * wb * AFP));
  }
}
`

/**
 * Resolve the member force scatters and add per-particle buoyancy
 * (world.ts applyBuoyancy): analytic lift from REST volume, wetness-gated
 * and accel-capped for object particles. Runs AFTER census (PU_WET) and
 * the column pass, BEFORE the substeps consume accX/accY. Also clears frc
 * behind itself for the next frame.
 * Bindings: uni, pf, pu, frc, colF.
 */
export const SRC_FORCES_APPLY = /* wgsl */ `
@group(0) @binding(0) var<uniform> U: UStruct;
@group(0) @binding(1) var<storage, read_write> pf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pu: array<u32>;
@group(0) @binding(3) var<storage, read_write> frc: array<i32>;
@group(0) @binding(4) var<storage, read_write> colF: array<f32>;
${COMMON}
${COL_FNS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.n) { return; }
  let rawX = frc[${FRC.accX}u * U.cap + i];
  let rawY = frc[${FRC.accY}u * U.cap + i];
  frc[${FRC.accX}u * U.cap + i] = 0;
  frc[${FRC.accY}u * U.cap + i] = 0;
  if (pu[pu_i(PU_ALIVE, i)] != 1u) { return; }
  let w = pf[pf_i(PF_INVMASS, i)];
  if (w == 0.0) { return; }

  var addX = f32(rawX) / AFP;
  var addY = f32(rawY) / AFP;

  let kind = pu[pu_i(PU_KIND, i)];
  let vol = pf[pf_i(PF_VOLUME, i)];
  if (kind != KIND_FLUID && vol > 0.0) {
    var frac = submerged_frac(pf[pf_i(PF_POSX, i)], pf[pf_i(PF_POSY, i)], pf[pf_i(PF_RADIUS, i)]);
    if (frac > 0.0) {
      let g = -U.gravity;
      if (kind == KIND_OBJECT) {
        // Wetness-gated: below the waterline says "maybe"; actual fluid
        // contact says "in water" (world.ts:315-323). Freshly from THIS
        // frame's census - the whole point of computing forces on-device.
        frac *= min(1.0, f32(pu[pu_i(PU_WET, i)]) / 8.0);
        if (frac > 0.0) {
          addY += min(g * (U.waterDensity * vol * w) * frac, U.maxObjBuoy);
        }
      } else {
        addY += g * (U.waterDensity * vol * w) * frac;
      }
    }
  }

  if (addX != 0.0 || addY != 0.0) {
    pf[pf_i(PF_ACCX, i)] += addX;
    pf[pf_i(PF_ACCY, i)] += addY;
  }
}
`

/** Names -> sources, for pipeline creation. */
export const KERNELS = {
  colClear: SRC_COL_CLEAR,
  colAccum: SRC_COL_ACCUM,
  colSurface: SRC_COL_SURFACE,
  colSmooth: SRC_COL_SMOOTH,
  forcesMember: SRC_FORCES_MEMBER,
  forcesApply: SRC_FORCES_APPLY,
  gridClear: SRC_GRID_CLEAR,
  packGather: SRC_PACK_GATHER,
  gridCount: SRC_GRID_COUNT,
  gridScan1: SRC_GRID_SCAN1,
  gridScan2: SRC_GRID_SCAN2,
  gridScan3: SRC_GRID_SCAN3,
  gridScatter: SRC_GRID_SCATTER,
  census: SRC_CENSUS,
  wave: SRC_WAVE,
  predict: SRC_PREDICT,
  cluster: SRC_CLUSTER,
  distSolve: SRC_DIST_SOLVE,
  distDamp: SRC_DIST_DAMP,
  bendSolve: SRC_BEND_SOLVE,
  bendDamp: SRC_BEND_DAMP,
  fluidDensity: SRC_FLUID_DENSITY,
  fluidCorrect: SRC_FLUID_CORRECT,
  fluidApply: SRC_FLUID_APPLY,
  contactsSolid: SRC_CONTACTS_SOLID,
  contactsMember: SRC_CONTACTS_MEMBER,
  integrate: SRC_INTEGRATE,
  snapVel: SRC_SNAP_VEL,
  packXsph: SRC_PACK_XSPH,
  hullFluid: SRC_HULL_FLUID,
  hullSolid: SRC_HULL_SOLID,
  xsph: SRC_XSPH,
} as const

export type KernelName = keyof typeof KERNELS
