/*
 * 多视图几何：本质矩阵、位姿恢复、三角化、PnP、LM 位姿/点优化
 * 对应 COLMAP 中 SfM 的两视图几何与增量式定位阶段。
 */

import {
  EPS, identity, m3mul, m3t, m3mulv, m3det, rodrigues, cross, dot,
  norm, smallestEigenvector, gramMatrix, svd3, enforceEssential, solveLinear,
  sampleIndices, mulberry32, huberWeight,
} from './mathx.js';

/** 归一化相机坐标（x 以焦距归一） */
export function toNormalized(cam, x, y) {
  return [(x - cam.cx) / cam.f, (y - cam.cy) / cam.f];
}

export function projectPose(cam, X) {
  const R = cam.R, t = cam.t;
  const xc = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0];
  const yc = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1];
  const zc = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
  if (zc <= 1e-9) return { u: NaN, v: NaN, z: zc, ok: false };
  return {
    u: cam.f * (xc / zc) + cam.cx,
    v: cam.f * (yc / zc) + cam.cy,
    z: zc,
    ok: true,
  };
}

/** Sampson 距离平方（归一化坐标） */
function sampsonError2(E, x1, y1, x2, y2) {
  const ex = E[0] * x1 + E[1] * y1 + E[2];
  const ey = E[3] * x1 + E[4] * y1 + E[5];
  const ez = E[6] * x1 + E[7] * y1 + E[8];
  const r = x2 * ex + y2 * ey + ez;
  const e2x = E[0] * x2 + E[3] * y2 + E[6];
  const e2y = E[1] * x2 + E[4] * y2 + E[7];
  const denom = ex * ex + ey * ey + e2x * e2x + e2y * e2y;
  return (r * r) / (denom + 1e-30);
}

function fitEssential(nx1, ny1, nx2, ny2, idx) {
  const n = idx.length;
  const A = new Float64Array(n * 9);
  for (let r = 0; r < n; r++) {
    const i = idx[r];
    const x1 = nx1[i], y1 = ny1[i], x2 = nx2[i], y2 = ny2[i];
    const o = r * 9;
    A[o] = x1 * x2;
    A[o + 1] = y1 * x2;
    A[o + 2] = x2;
    A[o + 3] = x1 * y2;
    A[o + 4] = y1 * y2;
    A[o + 5] = y2;
    A[o + 6] = x1;
    A[o + 7] = y1;
    A[o + 8] = 1;
  }
  const AtA = gramMatrix(A, n, 9);
  const { vector } = smallestEigenvector(AtA, 9);
  const E = new Float64Array(9);
  for (let i = 0; i < 9; i++) E[i] = vector[i];
  return enforceEssential(E);
}

/**
 * 本质矩阵 RANSAC（归一化 8 点法）
 * @param {object} camA @param {object} camB
 * @param {Float32Array} ax 图像 A 的特征 x ...
 * @param {Array<{a:number,b:number}>} matches
 */
export function estimateEssentialRansac(camA, camB, fa, fb, matches, opts = {}) {
  const thresholdPx = opts.threshold ?? 3.0;
  const fAvg = (camA.f + camB.f) * 0.5;
  const threshN = thresholdPx / fAvg;
  const thresh2 = threshN * threshN;
  const maxIters = opts.maxIters ?? 1500;
  const conf = opts.confidence ?? 0.9995;
  const rand = mulberry32(opts.seed ?? 0x51ed);

  const n = matches.length;
  if (n < 8) return null;
  const sampleSize = Math.max(8, Math.min(opts.sampleSize ?? 8, n));

  const nx1 = new Float64Array(n), ny1 = new Float64Array(n);
  const nx2 = new Float64Array(n), ny2 = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const m = matches[i];
    nx1[i] = (fa.x[m.a] - camA.cx) / camA.f;
    ny1[i] = (fa.y[m.a] - camA.cy) / camA.f;
    nx2[i] = (fb.x[m.b] - camB.cx) / camB.f;
    ny2[i] = (fb.y[m.b] - camB.cy) / camB.f;
  }

  let bestE = null;
  let bestInliers = null;
  let bestCount = 0;
  let iters = maxIters;
  let iter = 0;
  const sample = new Int32Array(sampleSize);
  const camW = camA.width || camA.cx * 2 || 1;
  const camH = camA.height || camA.cy * 2 || 1;

  while (iter < iters && iter < maxIters) {
    iter++;
    // 分层随机采样：把图像划分为网格，尽量避免全部取到同一平面区域
    const used = new Set();
    let k = 0;
    const cols = 4, rows = 3;
    while (k < sampleSize) {
      const cell = (rand() * cols * rows) | 0;
      const cx0 = cell % cols, cy0 = (cell / cols) | 0;
      let pick = -1;
      for (let attempt = 0; attempt < 12; attempt++) {
        const v = (rand() * n) | 0;
        if (used.has(v)) continue;
        const gx = Math.min(cols - 1, Math.floor((nx1[v] * fAvg + camA.cx) / camW * cols));
        const gy = Math.min(rows - 1, Math.floor((ny1[v] * fAvg + camA.cy) / camH * rows));
        if (gx === cx0 && gy === cy0) { pick = v; break; }
        if (pick < 0) pick = v;
      }
      if (pick < 0) pick = (rand() * n) | 0;
      if (used.has(pick)) continue;
      used.add(pick);
      sample[k++] = pick;
    }

    let E;
    try {
      E = fitEssential(nx1, ny1, nx2, ny2, sample);
    } catch {
      continue;
    }
    if (!E || !Number.isFinite(E[0])) continue;

    let count = 0;
    const inl = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      if (sampsonError2(E, nx1[i], ny1[i], nx2[i], ny2[i]) <= thresh2) {
        inl[i] = 1;
        count++;
      }
    }

    if (count > bestCount) {
      bestCount = count;
      bestE = E;
      bestInliers = inl;
      const ratio = count / n;
      if (ratio > 0.01) {
        const pGood = Math.pow(ratio, sampleSize);
        if (pGood > 1e-12) {
          const need = Math.ceil(Math.log(1 - conf) / Math.log(1 - pGood));
          if (need < iters) iters = Math.min(maxIters, Math.max(need, 64));
        }
      }
    }
  }

  if (!bestE || bestCount < 8) return null;

  // 用全部内点做一次最小二乘精化
  const idx = [];
  for (let i = 0; i < n; i++) if (bestInliers[i]) idx.push(i);
  let E = bestE;
  if (idx.length >= 8) {
    try {
      E = fitEssential(nx1, ny1, nx2, ny2, idx);
      const inl2 = new Uint8Array(n);
      let c2 = 0;
      for (let i = 0; i < n; i++) {
        if (sampsonError2(E, nx1[i], ny1[i], nx2[i], ny2[i]) <= thresh2 * 1.5) {
          inl2[i] = 1; c2++;
        }
      }
      if (c2 >= bestCount) {
        bestInliers = inl2;
        bestCount = c2;
      } else {
        E = bestE;
      }
    } catch {
      E = bestE;
    }
  }

  const inlierIdx = [];
  for (let i = 0; i < n; i++) if (bestInliers[i]) inlierIdx.push(i);

  return { E, inlierIdx, inlierCount: inlierIdx.length, iterations: iter };
}

/* ---------------- 位姿恢复 ---------------- */

function triangulateNormalized(x1, y1, x2, y2, R, t) {
  // P1 = [I|0], P2 = [R|t]
  const rows = new Float64Array(4 * 4); // 4 行 4 列
  // 视图 1
  let o = 0;
  rows[o] = x1; rows[o + 1] = 0; rows[o + 2] = 0; rows[o + 3] = 0;
  o = 4;
  rows[o] = 0; rows[o + 1] = y1; rows[o + 2] = 0; rows[o + 3] = 0;
  // 实际上 P1 = [I|0]：u*(P1_3 X) - P1_1 X = u*Z - X = 0
  // 行向量应为 [ -1, 0, u, 0 ] 和 [ 0, -1, v, 0 ]
  rows[0] = -1; rows[1] = 0; rows[2] = x1; rows[3] = 0;
  rows[4] = 0; rows[5] = -1; rows[6] = y1; rows[7] = 0;

  // 视图 2
  const P2 = [
    R[0], R[1], R[2], t[0],
    R[3], R[4], R[5], t[1],
    R[6], R[7], R[8], t[2],
  ];
  // u * P3 - P1, v * P3 - P2
  for (let r = 0; r < 2; r++) {
    const uu = r === 0 ? x2 : y2;
    const pr = r === 0 ? 0 : 1;
    const base = (2 + r) * 4;
    for (let c = 0; c < 4; c++) {
      rows[base + c] = uu * P2[8 + c] - P2[pr * 4 + c];
    }
  }

  const AtA = gramMatrix(rows, 4, 4);
  const { vector } = smallestEigenvector(AtA, 4);
  const w = vector[3];
  if (Math.abs(w) < 1e-12) return null;
  return [vector[0] / w, vector[1] / w, vector[2] / w];
}

/**
 * 从 E 恢复相对位姿（4 组候选 + 手性检验）
 */
export function recoverPoseFromEssential(E, camA, camB, fa, fb, matches, inlierIdx) {
  const { U, V } = svd3(E);
  // 强制 U、V 行列式为 +1（第三列奇异值为 0，可任意取反）
  if (m3det(U) < 0) {
    U[2] = -U[2]; U[5] = -U[5]; U[8] = -U[8];
  }
  if (m3det(V) < 0) {
    V[2] = -V[2]; V[5] = -V[5]; V[8] = -V[8];
  }
  const W = new Float64Array([0, -1, 0, 1, 0, 0, 0, 0, 1]);
  const R1 = m3mul(m3mul(U, W), m3t(V));
  const R2 = m3mul(m3mul(U, m3t(W)), m3t(V));
  const t0 = new Float64Array([U[2], U[5], U[8]]);

  const candidates = [
    { R: R1, t: t0 },
    { R: R1, t: [-t0[0], -t0[1], -t0[2]] },
    { R: R2, t: t0 },
    { R: R2, t: [-t0[0], -t0[1], -t0[2]] },
  ];

  let best = null;
  const sampleN = Math.min(inlierIdx.length, 160);
  const step = Math.max(1, Math.floor(inlierIdx.length / sampleN));

  for (const cand of candidates) {
    let good = 0;
    for (let s = 0; s < inlierIdx.length; s += step) {
      const mi = inlierIdx[s];
      const m = matches[mi];
      const x1 = (fa.x[m.a] - camA.cx) / camA.f;
      const y1 = (fa.y[m.a] - camA.cy) / camA.f;
      const x2 = (fb.x[m.b] - camB.cx) / camB.f;
      const y2 = (fb.y[m.b] - camB.cy) / camB.f;
      const X = triangulateNormalized(x1, y1, x2, y2, cand.R, cand.t);
      if (!X) continue;
      const z1 = X[2];
      const z2 = cand.R[6] * X[0] + cand.R[7] * X[1] + cand.R[8] * X[2] + cand.t[2];
      if (z1 > 1e-6 && z2 > 1e-6) good++;
    }
    if (!best || good > best.good) best = { R: cand.R, t: cand.t, good };
  }

  if (!best) return null;
  return { R: best.R, t: best.t, goodCount: best.good };
}

/* ---------------- 三角化 ---------------- */

/**
 * 多视图三角化
 * @param {Array<{cam:object,x:number,y:number}>} obs
 * @param {object} opts { maxError: 6 }
 */
export function triangulateObservations(obs, opts = {}) {
  const n = obs.length;
  if (n < 2) return null;
  const A = new Float64Array(2 * n * 4);
  const rays = [];
  let row = 0;

  for (const o of obs) {
    const cam = o.cam;
    const R = cam.R, t = cam.t;
    const u = (o.x - cam.cx) / cam.f;
    const v = (o.y - cam.cy) / cam.f;
    const P = [
      R[0], R[1], R[2], t[0],
      R[3], R[4], R[5], t[1],
      R[6], R[7], R[8], t[2],
    ];
    for (let r = 0; r < 2; r++) {
      const uu = r === 0 ? u : v;
      const pr = r;
      for (let c = 0; c < 4; c++) {
        A[(row + r) * 4 + c] = uu * P[8 + c] - P[pr * 4 + c];
      }
    }
    row += 2;

    // 世界坐标系下的视线方向
    const dCam = [u, v, 1];
    const Rt = m3t(R);
    rays.push(m3mulv(Rt, dCam));
  }

  const AtA = gramMatrix(A, 2 * n, 4);
  const { vector } = smallestEigenvector(AtA, 4);
  const w = vector[3];
  if (!Number.isFinite(w) || Math.abs(w) < 1e-12) return null;
  const X = new Float64Array([vector[0] / w, vector[1] / w, vector[2] / w]);
  if (!Number.isFinite(X[0]) || !Number.isFinite(X[1]) || !Number.isFinite(X[2])) return null;

  let maxErr = 0;
  for (const o of obs) {
    const p = projectPose(o.cam, X);
    if (!p.ok) return null;
    const err = Math.hypot(p.u - o.x, p.v - o.y);
    if (err > maxErr) maxErr = err;
  }
  const maxError = opts.maxError ?? 6;
  if (maxErr > maxError) return null;

  // 最大视线夹角（度）
  let maxAngle = 0;
  for (let i = 0; i < rays.length; i++) {
    const a = rays[i];
    const na = norm(a);
    for (let j = i + 1; j < rays.length; j++) {
      const b = rays[j];
      const nb = norm(b);
      const c = Math.max(-1, Math.min(1, dot(a, b) / (na * nb)));
      const ang = (Math.acos(c) * 180) / Math.PI;
      if (ang > maxAngle) maxAngle = ang;
    }
  }

  return { X, error: maxErr, angle: maxAngle };
}

/* ---------------- PnP ---------------- */

export function pnpDlt(corr, cam) {
  const n = corr.length;
  if (n < 6) return null;

  // 三维点归一化
  let cx = 0, cy = 0, cz = 0;
  for (const c of corr) { cx += c.X[0]; cy += c.X[1]; cz += c.X[2]; }
  cx /= n; cy /= n; cz /= n;
  let md = 0;
  for (const c of corr) md += Math.hypot(c.X[0] - cx, c.X[1] - cy, c.X[2] - cz);
  md /= n;
  if (!(md > 1e-12)) return null;
  const s = Math.sqrt(3) / md;

  const A = new Float64Array(2 * n * 12);
  for (let i = 0; i < n; i++) {
    const c = corr[i];
    const X = s * (c.X[0] - cx);
    const Y = s * (c.X[1] - cy);
    const Z = s * (c.X[2] - cz);
    const u = (c.x - cam.cx) / cam.f;
    const v = (c.y - cam.cy) / cam.f;
    const r0 = (2 * i) * 12;
    A[r0] = X; A[r0 + 1] = Y; A[r0 + 2] = Z; A[r0 + 3] = 1;
    A[r0 + 8] = -u * X; A[r0 + 9] = -u * Y; A[r0 + 10] = -u * Z; A[r0 + 11] = -u;
    const r1 = r0 + 12;
    A[r1 + 4] = X; A[r1 + 5] = Y; A[r1 + 6] = Z; A[r1 + 7] = 1;
    A[r1 + 8] = -v * X; A[r1 + 9] = -v * Y; A[r1 + 10] = -v * Z; A[r1 + 11] = -v;
  }

  const AtA = gramMatrix(A, 2 * n, 12);
  const { vector } = smallestEigenvector(AtA, 12);
  const P = vector;

  const M = new Float64Array([
    P[0], P[1], P[2],
    P[4], P[5], P[6],
    P[8], P[9], P[10],
  ]);
  const { U, S, V } = svd3(M);
  let R = m3mul(U, m3t(V));
  if (m3det(R) < 0) {
    // 修正为旋转矩阵
    for (let i = 0; i < 3; i++) U[i * 3 + 2] = -U[i * 3 + 2];
    R = m3mul(U, m3t(V));
  }
  const sc = (S[0] + S[1] + S[2]) / 3;
  if (!(sc > 1e-12)) return null;
  const tNorm = new Float64Array([P[3] / sc, P[7] / sc, P[11] / sc]);

  // 反归一化
  const Rc = m3mulv(R, [cx, cy, cz]);
  const t = new Float64Array([
    tNorm[0] / s - Rc[0],
    tNorm[1] / s - Rc[1],
    tNorm[2] / s - Rc[2],
  ]);

  return { R, t };
}

function reprojectionResidual(cam, X, x, y) {
  const p = projectPose(cam, X);
  if (!p.ok) return [1e6, 1e6];
  return [p.u - x, p.v - y];
}

/**
 * PnP RANSAC（6 点 DLT）
 * @param {object} cam 相机（使用 f/cx/cy，R/t 会被写入）
 * @param {Array<{X:Float64Array,x:number,y:number}>} corr
 */
export function pnpRansac(cam, corr, opts = {}) {
  const threshold = opts.threshold ?? 5;
  const maxIters = opts.maxIters ?? 800;
  const n = corr.length;
  if (n < 6) return null;

  const rand = mulberry32(opts.seed ?? 0x9e37);
  const thresh2 = threshold * threshold;
  let best = null;
  let bestCount = 0;
  let iters = maxIters;
  let iter = 0;

  const sample = new Int32Array(6);
  while (iter < iters && iter < maxIters) {
    iter++;
    const used = new Set();
    let k = 0;
    while (k < 6) {
      const v = (rand() * n) | 0;
      if (used.has(v)) continue;
      used.add(v);
      sample[k++] = v;
    }
    const sub = [];
    for (let i = 0; i < 6; i++) sub.push(corr[sample[i]]);
    let pose;
    try {
      pose = pnpDlt(sub, cam);
    } catch {
      continue;
    }
    if (!pose || !Number.isFinite(pose.R[0]) || !Number.isFinite(pose.t[0])) continue;

    const testCam = { ...cam, R: pose.R, t: pose.t };
    let count = 0;
    for (let i = 0; i < n; i++) {
      const c = corr[i];
      const p = projectPose(testCam, c.X);
      if (!p.ok) continue;
      const du = p.u - c.x, dv = p.v - c.y;
      if (du * du + dv * dv <= thresh2) count++;
    }
    if (count > bestCount) {
      bestCount = count;
      best = pose;
      const ratio = count / n;
      if (ratio > 0.01) {
        const pGood = Math.pow(ratio, 6);
        if (pGood > 1e-12) {
          const need = Math.ceil(Math.log(0.0005) / Math.log(1 - pGood));
          if (need < iters) iters = Math.min(maxIters, Math.max(need, 48));
        }
      }
    }
  }

  if (!best || bestCount < 6) return null;

  const inliers = [];
  const testCam = { ...cam, R: best.R, t: best.t };
  for (let i = 0; i < n; i++) {
    const c = corr[i];
    const p = projectPose(testCam, c.X);
    if (p.ok) {
      const du = p.u - c.x, dv = p.v - c.y;
      if (du * du + dv * dv <= thresh2 * 1.7) inliers.push(c);
    }
  }

  let pose = best;
  if (inliers.length >= 6) {
    const refined = pnpDlt(inliers, cam);
    if (refined) pose = refined;
  }

  const resultCam = { ...cam, R: pose.R, t: pose.t };
  refineCameraPose(resultCam, inliers.length >= 6 ? inliers : corr, 6);

  // 统计最终误差
  let sum = 0, count = 0;
  for (const c of inliers) {
    const p = projectPose(resultCam, c.X);
    if (!p.ok) continue;
    sum += Math.hypot(p.u - c.x, p.v - c.y);
    count++;
  }

  return {
    R: resultCam.R,
    t: resultCam.t,
    inliers,
    inlierCount: inliers.length,
    meanError: count ? sum / count : Infinity,
  };
}

/* ---------------- LM 优化 ---------------- */

/**
 * 相机位姿 LM 优化（固定三维点）
 * @param {object} cam 输入输出
 * @param {Array<{X:Float64Array,x:number,y:number}>} corr
 */
export function refineCameraPose(cam, corr, iterations = 5) {
  if (!corr.length) return 0;
  let R = Float64Array.from(cam.R);
  let t = Float64Array.from(cam.t);

  const project = (RR, tt, X) => {
    const xc = RR[0] * X[0] + RR[1] * X[1] + RR[2] * X[2] + tt[0];
    const yc = RR[3] * X[0] + RR[4] * X[1] + RR[5] * X[2] + tt[1];
    const zc = RR[6] * X[0] + RR[7] * X[1] + RR[8] * X[2] + tt[2];
    if (zc <= 1e-9) return null;
    return [cam.f * (xc / zc) + cam.cx, cam.f * (yc / zc) + cam.cy, xc, yc, zc];
  };

  const costOf = (RR, tt) => {
    let c = 0, cnt = 0;
    for (const o of corr) {
      const p = project(RR, tt, o.X);
      if (!p) continue;
      const e = Math.hypot(p[0] - o.x, p[1] - o.y);
      c += e * e;
      cnt++;
    }
    return cnt ? c / cnt : Infinity;
  };

  let cost = costOf(R, t);
  let lambda = 1e-3;

  for (let it = 0; it < iterations; it++) {
    const JtJ = new Float64Array(36);
    const Jtr = new Float64Array(6);
    let used = 0;

    for (const o of corr) {
      const p = project(R, t, o.X);
      if (!p) continue;
      const [u, v, xc, yc, zc] = p;
      const ru = u - o.x;
      const rv = v - o.y;
      const w = huberWeight(Math.hypot(ru, rv), 3);
      const sw = Math.sqrt(w);

      // R X
      const rx = R[0] * o.X[0] + R[1] * o.X[1] + R[2] * o.X[2];
      const ry = R[3] * o.X[0] + R[4] * o.X[1] + R[5] * o.X[2];
      const rz = R[6] * o.X[0] + R[7] * o.X[1] + R[8] * o.X[2];

      // d(u,v)/d(xc,yc,zc)
      const fz = cam.f / zc;
      const g = [fz, 0, -cam.f * xc / (zc * zc)];
      const h = [0, fz, -cam.f * yc / (zc * zc)];

      // d(xc)/dw = -skew(RX) -> du/dw = -(g x RX)
      const gx = -(g[1] * rz - g[2] * ry);
      const gy = -(g[2] * rx - g[0] * rz);
      const gz = -(g[0] * ry - g[1] * rx);
      const hx = -(h[1] * rz - h[2] * ry);
      const hy = -(h[2] * rx - h[0] * rz);
      const hz = -(h[0] * ry - h[1] * rx);

      const Ju = [gx, gy, gz, g[0], g[1], g[2]];
      const Jv = [hx, hy, hz, h[0], h[1], h[2]];

      for (let a = 0; a < 6; a++) {
        const jua = Ju[a] * sw, jva = Jv[a] * sw;
        Jtr[a] += jua * ru * sw + jva * rv * sw;
        for (let b = a; b < 6; b++) {
          JtJ[a * 6 + b] += jua * Ju[b] * sw + jva * Jv[b] * sw;
        }
      }
      used++;
    }

    if (!used) break;

    // 对称化
    for (let a = 0; a < 6; a++) {
      for (let b = 0; b < a; b++) JtJ[a * 6 + b] = JtJ[b * 6 + a];
    }
    for (let a = 0; a < 6; a++) {
      JtJ[a * 6 + a] += lambda * (JtJ[a * 6 + a] + 1e-9);
      Jtr[a] = -Jtr[a];
    }

    const delta = solveLinear(JtJ, Jtr, 6);
    if (!delta) { lambda *= 4; continue; }

    const dR = rodrigues([delta[0], delta[1], delta[2]]);
    const Rn = m3mul(dR, R);
    const tn = new Float64Array([t[0] + delta[3], t[1] + delta[4], t[2] + delta[5]]);
    const newCost = costOf(Rn, tn);

    if (newCost < cost) {
      R = Rn; t = tn;
      cost = newCost;
      lambda = Math.max(1e-9, lambda * 0.4);
    } else {
      lambda *= 5;
    }
  }

  cam.R = R;
  cam.t = t;
  return cost;
}

/** 三维点 LM 优化（固定相机） */
export function refinePoint(obs, X0, iterations = 4) {
  let X = Float64Array.from(X0);
  if (obs.length < 2) return X;
  let lambda = 1e-3;

  const residualOf = (P) => {
    let c = 0, n = 0;
    for (const o of obs) {
      const p = projectPose(o.cam, P);
      if (!p.ok) return Infinity;
      const du = p.u - o.x, dv = p.v - o.y;
      c += du * du + dv * dv;
      n++;
    }
    return n ? c / n : Infinity;
  };

  let cost = residualOf(X);

  for (let it = 0; it < iterations; it++) {
    const JtJ = new Float64Array(9);
    const Jtr = new Float64Array(3);
    let used = 0;

    for (const o of obs) {
      const cam = o.cam;
      const p = projectPose(cam, X);
      if (!p.ok) continue;
      const xc = cam.R[0] * X[0] + cam.R[1] * X[1] + cam.R[2] * X[2] + cam.t[0];
      const yc = cam.R[3] * X[0] + cam.R[4] * X[1] + cam.R[5] * X[2] + cam.t[1];
      const zc = cam.R[6] * X[0] + cam.R[7] * X[1] + cam.R[8] * X[2] + cam.t[2];
      if (zc <= 1e-9) continue;

      const du = p.u - o.x, dv = p.v - o.y;
      const w = huberWeight(Math.hypot(du, dv), 3);
      const sw = Math.sqrt(w);

      const gx = cam.f / zc, gy = 0, gz = -cam.f * xc / (zc * zc);
      const hx = 0, hy = cam.f / zc, hz = -cam.f * yc / (zc * zc);

      // J = (R^T g)^T
      const Rt = m3t(cam.R);
      const Ju = [Rt[0] * gx + Rt[3] * gy + Rt[6] * gz,
                  Rt[1] * gx + Rt[4] * gy + Rt[7] * gz,
                  Rt[2] * gx + Rt[5] * gy + Rt[8] * gz];
      const Jv = [Rt[0] * hx + Rt[3] * hy + Rt[6] * hz,
                  Rt[1] * hx + Rt[4] * hy + Rt[7] * hz,
                  Rt[2] * hx + Rt[5] * hy + Rt[8] * hz];

      for (let a = 0; a < 3; a++) {
        Jtr[a] += Ju[a] * du * w + Jv[a] * dv * w;
        for (let b = a; b < 3; b++) {
          JtJ[a * 3 + b] += Ju[a] * Ju[b] * w + Jv[a] * Jv[b] * w;
        }
      }
      used++;
    }
    if (!used) break;

    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < a; b++) JtJ[a * 3 + b] = JtJ[b * 3 + a];
      JtJ[a * 3 + a] += lambda * (JtJ[a * 3 + a] + 1e-9);
      Jtr[a] = -Jtr[a];
    }

    const delta = solveLinear(JtJ, Jtr, 3);
    if (!delta) { lambda *= 4; continue; }
    const Xn = new Float64Array([X[0] + delta[0], X[1] + delta[1], X[2] + delta[2]]);
    const newCost = residualOf(Xn);
    if (newCost < cost) {
      X = Xn; cost = newCost;
      lambda = Math.max(1e-9, lambda * 0.4);
    } else {
      lambda *= 5;
    }
  }
  return X;
}

/** 观测重投影误差（像素） */
export function observationError(cam, X, x, y) {
  const p = projectPose(cam, X);
  if (!p.ok) return Infinity;
  return Math.hypot(p.u - x, p.v - y);
}