/*
 * 多视图稠密重建（MVS）
 *
 * 与 COLMAP 的 PatchMatch Stereo / Fusion / Meshing 阶段对应，这里采用
 * 更适合纯 JS、可离线运行的简化方案：
 *
 *   1. 对每张参考视图做平面扫描（Plane-Sweep），用 Census + 灰度差
 *      在多个邻域视图上做光度一致性匹配，得到每像素深度
 *   2. 左右一致性（前向-后向）检查剔除遮挡/误匹配
 *   3. 把所有深度图融合进 TSDF 体素场
 *   4. 用 Marching Tetrahedra 提取三角网格（顶点颜色来自图像）
 *
 * 不含任何 DOM / WebGL 依赖，可在 Node 中进行数值测试。
 */

import { rgbaToGray, resizeRGBA, censusTransform, sampleRGBNearest } from './image.js';
import { projectPose } from './geometry.js';
import { popcount32 } from './orb.js';
import { percentile } from './mathx.js';

const QUALITY = {
  fast: { targetSize: 224, depths: 56, maxRef: 10, neighbors: 4, gridSize: 112, minCost: 0.4, dilate: 1 },
  standard: { targetSize: 240, depths: 64, maxRef: 16, neighbors: 4, gridSize: 128, minCost: 0.4, dilate: 3 },
  high: { targetSize: 300, depths: 80, maxRef: 24, neighbors: 5, gridSize: 160, minCost: 0.38, dilate: 3 },
};

function cloneCamForWork(cam, scale) {
  return {
    ...cam,
    f: cam.f * scale,
    fx: cam.f * scale,
    fy: cam.f * scale,
    cx: cam.cx * scale,
    cy: cam.cy * scale,
    w: Math.max(8, Math.round(cam.width * scale)),
    h: Math.max(8, Math.round(cam.height * scale)),
    R: cam.R,
    t: cam.t,
  };
}

function prepareWorkImage(img, scale) {
  const w = Math.max(8, Math.round(img.width * scale));
  const h = Math.max(8, Math.round(img.height * scale));
  const rgba = resizeRGBA(img.rgba, img.width, img.height, w, h);
  const gray = rgbaToGray(rgba, w, h);
  const census = censusTransform(gray, w, h, 2);
  return { w, h, rgba, gray, census };
}

function camCenter(cam) {
  const R = cam.R, t = cam.t;
  return [
    -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
    -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
    -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
  ];
}

function viewDirection(cam) {
  // 相机 +Z 轴在世界中的方向
  return [cam.R[6], cam.R[7], cam.R[8]];
}

function depthRangeFor(cam, points) {
  const visible = [];
  const positive = [];
  for (const pt of points) {
    const p = projectPose(cam, pt.X);
    if (!p.ok) continue;
    positive.push(p.z);
    if (p.u >= 0 && p.v >= 0 && p.u < cam.width && p.v < cam.height) {
      visible.push(p.z);
    }
  }
  let zs = visible.length >= 10 ? visible : positive;
  if (zs.length < 4) return null;
  let lo = percentile(zs, 0.02);
  let hi = percentile(zs, 0.98);
  if (!(hi > lo)) {
    const m = (lo + hi) / 2;
    lo = m * 0.75;
    hi = m * 1.25;
  }
  return { lo: Math.max(1e-6, lo * 0.72), hi: hi * 1.35 };
}

function buildShareMatrix(cameras, points) {
  const m = cameras.length;
  const share = Array.from({ length: m }, () => new Uint32Array(m));
  const indexOf = new Map();
  cameras.forEach((c, i) => indexOf.set(c, i));

  for (const pt of points) {
    const ids = [];
    for (const o of pt.obs) {
      const idx = indexOf.get(o.cam);
      if (idx !== undefined) ids.push(idx);
    }
    for (let a = 0; a < ids.length; a++) {
      for (let b = a + 1; b < ids.length; b++) {
        share[ids[a]][ids[b]]++;
        share[ids[b]][ids[a]]++;
      }
    }
  }
  return share;
}

function selectNeighbors(refIdx, cameras, share, K) {
  const ref = cameras[refIdx];
  const refCenter = camCenter(ref);
  const refDir = viewDirection(ref);
  const candidates = [];

  for (let j = 0; j < cameras.length; j++) {
    if (j === refIdx) continue;
    const c = cameras[j];
    const dir = viewDirection(c);
    const cosA = Math.max(-1, Math.min(1,
      refDir[0] * dir[0] + refDir[1] * dir[1] + refDir[2] * dir[2]));
    const angle = (Math.acos(cosA) * 180) / Math.PI;
    const ctr = camCenter(c);
    const dist = Math.hypot(ctr[0] - refCenter[0], ctr[1] - refCenter[1], ctr[2] - refCenter[2]);
    candidates.push({ idx: j, share: share[refIdx][j], angle, dist });
  }

  const good = candidates
    .filter((c) => c.angle < 72)
    .sort((a, b) => {
      if (b.share !== a.share) return b.share - a.share;
      return a.dist - b.dist;
    });

  if (good.length >= 1) return good.slice(0, K).map((c) => c.idx);

  // 没有共视点时退化为最近的相机
  return candidates.sort((a, b) => a.dist - b.dist).slice(0, K).map((c) => c.idx);
}

/* ---------------- 平面扫描深度估计 ---------------- */

async function estimateDepthMap(refIdx, neighborIdx, cameras, works, points, cfg, emit, yieldFn, check) {
  const refCam = cameras[refIdx];
  const refW = works[refIdx];
  const workCam = cloneCamForWork(refCam, refW.w / refCam.width);
  const { w, h, gray, census } = refW;

  const range = depthRangeFor(refCam, points);
  if (!range) return null;

  const D = cfg.depths;
  const invLo = 1 / range.hi;
  const invHi = 1 / range.lo;
  const invStep = (invHi - invLo) / (D - 1);

  // 邻域相机投影参数： X_j = z * (A_j * d_world) + b_j
  const refR = refCam.R, refT = refCam.t;
  const refRt = [
    refR[0], refR[3], refR[6],
    refR[1], refR[4], refR[7],
    refR[2], refR[5], refR[8],
  ];
  const neighbors = neighborIdx.map((j) => {
    const cj = cameras[j];
    const wj = works[j];
    const wcam = cloneCamForWork(cj, wj.w / cj.width);
    const Rj = cj.R;
    const A = new Float64Array(9);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        A[r * 3 + c] =
          Rj[r * 3] * refRt[0 * 3 + c] +
          Rj[r * 3 + 1] * refRt[1 * 3 + c] +
          Rj[r * 3 + 2] * refRt[2 * 3 + c];
      }
    }
    const At = [
      A[0] * refT[0] + A[1] * refT[1] + A[2] * refT[2],
      A[3] * refT[0] + A[4] * refT[1] + A[5] * refT[2],
      A[6] * refT[0] + A[7] * refT[1] + A[8] * refT[2],
    ];
    const b = [cj.t[0] - At[0], cj.t[1] - At[1], cj.t[2] - At[2]];
    return { A, b, wcam, wj, gray: wj.gray, census: wj.census, w: wj.w, h: wj.h };
  });

  const depth = new Float32Array(w * h);
  const costOut = new Float32Array(w * h).fill(1);
  const confOut = new Float32Array(w * h);
  const bestNeighbor = new Int16Array(w * h).fill(-1);

  const margin = 2;
  const invDepths = new Float64Array(D);
  for (let d = 0; d < D; d++) invDepths[d] = 1 / (invLo + d * invStep);

  const neighborCost = (adList, bList, z, x, y) => {
    let min1 = Infinity, min2 = Infinity;
    const total = neighbors.length;
    for (let k = 0; k < total; k++) {
      const n = neighbors[k];
      const ad = adList[k];
      const b = bList[k];
      const xc = z * ad[0] + b[0];
      const yc = z * ad[1] + b[1];
      const zc = z * ad[2] + b[2];
      if (zc <= 1e-9) continue;
      const u = n.wcam.fx * (xc / zc) + n.wcam.cx;
      const v = n.wcam.fy * (yc / zc) + n.wcam.cy;
      const ui = Math.round(u), vi = Math.round(v);
      let c;
      if (ui < margin || vi < margin || ui >= n.w - margin || vi >= n.h - margin) {
        c = 1;
      } else {
        const idxN = vi * n.w + ui;
        const idxR = y * w + x;
        const censusCost = popcount32(census[idxR] ^ n.census[idxN]) / 24;
        const grayCost = Math.abs(gray[idxR] - n.gray[idxN]) / 255;
        c = 0.8 * censusCost + 0.2 * grayCost;
      }
      if (c < min1) { min2 = min1; min1 = c; }
      else if (c < min2) { min2 = c; }
    }
    if (!Number.isFinite(min1)) return 1;
    if (Number.isFinite(min2)) return (min1 + min2) * 0.5;
    return min1;
  };

  for (let y = margin; y < h - margin; y++) {
    for (let x = margin; x < w - margin; x++) {
      const idx = y * w + x;
      if (!census[idx]) continue; // 边界/无纹理位置

      // 参考相机坐标系中的射线方向；X_j = z * (A * ray) + b
      const ray = [
        (x - workCam.cx) / workCam.fx,
        (y - workCam.cy) / workCam.fy,
        1,
      ];
      const adList = [];
      const bList = [];
      for (const n of neighbors) {
        adList.push([
          n.A[0] * ray[0] + n.A[1] * ray[1] + n.A[2] * ray[2],
          n.A[3] * ray[0] + n.A[4] * ray[1] + n.A[5] * ray[2],
          n.A[6] * ray[0] + n.A[7] * ray[1] + n.A[8] * ray[2],
        ]);
        bList.push(n.b);
      }

      let bestCost = Infinity;
      let secondCost = Infinity;
      let bestD = -1;

      for (let d = 0; d < D; d++) {
        const z = invDepths[d];
        const c = neighborCost(adList, bList, z, x, y);
        if (c < bestCost) {
          secondCost = bestCost;
          bestCost = c;
          bestD = d;
        } else if (c < secondCost) {
          secondCost = c;
        }
      }

      if (cfg.debug && refIdx === 0 && x === 50 && y === 60) {
        console.log('[sweep] census', census[idx], 'bestCost', bestCost.toFixed(3), 'bestD', bestD, 'z', invDepths[bestD]?.toFixed(2));
        for (let k = 0; k < neighbors.length; k++) {
          const n = neighbors[k];
          const ad = adList[k], b = n.b;
          const z = invDepths[bestD];
          const xc = z * ad[0] + b[0], yc = z * ad[1] + b[1], zc = z * ad[2] + b[2];
          let line = `  nb${k} zc=${zc.toFixed(2)}`;
          if (zc > 1e-9) {
            const uu = n.wcam.fx * (xc / zc) + n.wcam.cx;
            const vv = n.wcam.fy * (yc / zc) + n.wcam.cy;
            line += ` uv=(${uu.toFixed(1)},${vv.toFixed(1)}) dims=${n.w}x${n.h}`;
          }
          console.log(line);
        }
      }
      if (bestD < 0 || bestCost > cfg.minCost) continue;

      // 抛物线亚像素精化（在逆深度参数上）
      let offset = 0;
      if (bestD > 0 && bestD < D - 1) {
        const cp = neighborCost(adList, bList, invDepths[bestD - 1], x, y);
        const cn = neighborCost(adList, bList, invDepths[bestD + 1], x, y);
        const denom = cp - 2 * bestCost + cn;
        if (Math.abs(denom) > 1e-9) {
          offset = 0.5 * (cp - cn) / denom;
          offset = Math.max(-0.75, Math.min(0.75, offset));
        }
      }

      const invD = invLo + (bestD + offset) * invStep;
      depth[idx] = 1 / Math.max(1e-9, invD);
      costOut[idx] = bestCost;
      confOut[idx] = Number.isFinite(secondCost) ? Math.max(0, secondCost - bestCost) : 0;

      // 最优邻域视图
      let bestK = 0, bestNc = Infinity;
      for (let k = 0; k < neighbors.length; k++) {
        const n = neighbors[k];
        const ad = adList[k];
        const b = n.b;
        const z = depth[idx];
        const xc = z * ad[0] + b[0];
        const yc = z * ad[1] + b[1];
        const zc = z * ad[2] + b[2];
        if (zc <= 1e-9) continue;
        const u = n.wcam.fx * (xc / zc) + n.wcam.cx;
        const v = n.wcam.fy * (yc / zc) + n.wcam.cy;
        const ui = Math.round(u), vi = Math.round(v);
        if (ui < margin || vi < margin || ui >= n.w - margin || vi >= n.h - margin) continue;
        const idxN = vi * n.w + ui;
        const c = 0.8 * (popcount32(census[idx] ^ n.census[idxN]) / 24) +
          0.2 * (Math.abs(gray[idx] - n.gray[idxN]) / 255);
        if (c < bestNc) { bestNc = c; bestK = k; }
      }
      bestNeighbor[idx] = neighbors.length ? bestK : -1;
    }
  }

  if (cfg.debug && refIdx === 0) {
    const hist = [0,0,0,0,0];
    let minC = Infinity, sumC = 0, nC = 0;
    for (let i = 0; i < costOut.length; i++) {
      const c = costOut[i];
      if (c < 0.4) hist[0]++;
      else if (c < 0.6) hist[1]++;
      else if (c < 0.8) hist[2]++;
      else if (c < 1) hist[3]++;
      else hist[4]++;
      if (c < 0.999) { minC = Math.min(minC, c); sumC += c; nC++; }
    }
    console.log('[cost] ref0 hist<.4,.6,.8,1,>=1', hist.join(','), 'sampleN', nC, 'mean', (sumC/Math.max(1,nC)).toFixed(3));
    // 检查若干像素的邻域投影
    for (const [px,py] of [[80,60],[120,90],[60,100],[100,50]]) {
      const idx=py*w+px; const z=depth[idx];
      console.log('[px]',px,py,'depth',z.toFixed(2),'cost',costOut[idx].toFixed(3));
    }
  }
  return {
    refIdx,
    cam: workCam,
    neighborIdx,
    w, h,
    depth,
    cost: costOut,
    confidence: confOut,
    bestNeighbor,
    zRange: range,
  };
}

/* ---------------- 深度后处理 ---------------- */

function medianFilterDepth(map) {
  const { w, h, depth } = map;
  const src = Float32Array.from(depth);
  const valid = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      if (!(src[idx] > 0)) continue;
      const vals = [];
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const v = src[(y + dy) * w + (x + dx)];
          if (v > 0) vals.push(v);
        }
      }
      if (vals.length < 4) continue;
      vals.sort((a, b) => a - b);
      const m = vals[vals.length >> 1];
      // 与中值相差过大则视为孤立点
      if (Math.abs(m - src[idx]) / src[idx] < 0.08) valid[idx] = m;
    }
  }
  map.depth = valid;
}

function forwardBackwardCheck(map, maps, debug = false) {
  const { w, h, depth, bestNeighbor, neighborIdx } = map;
  const out = new Float32Array(w * h);
  const margin = 2;

  // 预取邻域深度图
  const nbMaps = neighborIdx.map((j) => maps[j]).filter(Boolean);
  if (!nbMaps.length) return;

  // 邻域视图索引 -> map
  const byRef = new Map();
  for (const m of nbMaps) byRef.set(m.refIdx, m);

  let nTotal = 0, nInside = 0, nHave = 0, nPass = 0;
  const diffs = [];
  const perNb = new Array(neighborIdx.length).fill(0);
  const perNbInside = new Array(neighborIdx.length).fill(0);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const z = depth[idx];
      if (!(z > 0)) continue;
      nTotal++;
      if (bestNeighbor[idx] >= 0) perNb[bestNeighbor[idx]]++;
      const k = bestNeighbor[idx];
      if (k < 0) continue;
      const neighborRef = neighborIdx[k];
      const nm = byRef.get(neighborRef);
      if (!nm) continue;

      // 反投影到世界，再投影到邻域
      const cam = map.cam;
      const ray = [
        (x - cam.cx) / cam.fx,
        (y - cam.cy) / cam.fy,
        1,
      ];
      const Xc = [ray[0] * z, ray[1] * z, z];
      const R = cam.R;
      const Xw = [
        R[0] * Xc[0] + R[3] * Xc[1] + R[6] * Xc[2] - (R[0] * cam.t[0] + R[3] * cam.t[1] + R[6] * cam.t[2]),
        R[1] * Xc[0] + R[4] * Xc[1] + R[7] * Xc[2] - (R[1] * cam.t[0] + R[4] * cam.t[1] + R[7] * cam.t[2]),
        R[2] * Xc[0] + R[5] * Xc[1] + R[8] * Xc[2] - (R[2] * cam.t[0] + R[5] * cam.t[1] + R[8] * cam.t[2]),
      ];
      const ncam = nm.cam;
      const Rj = ncam.R, tj = ncam.t;
      const xj = Rj[0] * Xw[0] + Rj[1] * Xw[1] + Rj[2] * Xw[2] + tj[0];
      const yj = Rj[3] * Xw[0] + Rj[4] * Xw[1] + Rj[5] * Xw[2] + tj[1];
      const zj = Rj[6] * Xw[0] + Rj[7] * Xw[1] + Rj[8] * Xw[2] + tj[2];
      if (zj <= 1e-9) continue;
      const u = ncam.fx * (xj / zj) + ncam.cx;
      const v = ncam.fy * (yj / zj) + ncam.cy;
      const ui = Math.round(u), vi = Math.round(v);
      if (ui < margin || vi < margin || ui >= nm.w - margin || vi >= nm.h - margin) continue;
      nInside++;
      perNbInside[k]++;
      const d2 = nm.depth[vi * nm.w + ui];
      if (!(d2 > 0)) continue;
      nHave++;
      const rel = Math.abs(d2 - zj) / zj;
      diffs.push(rel);
      if (rel > 0.06) continue;
      nPass++;
      out[idx] = z;
    }
  }
  if (debug) {
    diffs.sort((a, b) => a - b);
    console.log('[fb-stats] ref', map.refIdx, 'valid', nTotal, 'inside', nInside,
      'haveDepth', nHave, 'pass', nPass, 'reldiff p50',
      (diffs[diffs.length >> 1] || 0).toFixed(3),
      'bestNb', perNb.join('/'), 'insideNb', perNbInside.join('/'));
  }
  map.depth = out;
}

/* ---------------- TSDF 融合 ---------------- */

function robustBounds(points) {
  const xs = [], ys = [], zs = [];
  for (const pt of points) {
    xs.push(pt.X[0]); ys.push(pt.X[1]); zs.push(pt.X[2]);
  }
  if (!xs.length) return null;
  const lo = [percentile(xs, 0.01), percentile(ys, 0.01), percentile(zs, 0.01)];
  const hi = [percentile(xs, 0.99), percentile(ys, 0.99), percentile(zs, 0.99)];
  const ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  const maxExt = Math.max(...ext) || 1;
  const minExt = maxExt * 0.08;
  for (let a = 0; a < 3; a++) {
    if (ext[a] < minExt) {
      const c = (lo[a] + hi[a]) * 0.5;
      lo[a] = c - minExt / 2;
      hi[a] = c + minExt / 2;
    }
  }
  // 外扩 12%
  for (let a = 0; a < 3; a++) {
    const pad = (hi[a] - lo[a]) * 0.12;
    lo[a] -= pad; hi[a] += pad;
  }
  return { lo, hi };
}

async function fuseTSDF(maps, cameras, works, points, cfg, emit, yieldFn, check) {
  const bounds = robustBounds(points);
  if (!bounds) throw new Error('没有可用于稠密重建的稀疏点');

  const ext = [
    bounds.hi[0] - bounds.lo[0],
    bounds.hi[1] - bounds.lo[1],
    bounds.hi[2] - bounds.lo[2],
  ];
  const maxExt = Math.max(...ext);
  let voxel = maxExt / Math.max(24, cfg.gridSize - 1);
  let dims = ext.map((e) => Math.min(320, Math.max(4, Math.ceil(e / voxel) + 1)));
  // 控制总体素数量
  while (dims[0] * dims[1] * dims[2] > 4_200_000) {
    voxel *= 1.08;
    dims = ext.map((e) => Math.min(320, Math.max(4, Math.ceil(e / voxel) + 1)));
  }
  const [nx, ny, nz] = dims;
  const total = nx * ny * nz;
  const tsdf = new Float32Array(total).fill(1);
  const weight = new Float32Array(total);
  const cr = new Float32Array(total);
  const cg = new Float32Array(total);
  const cb = new Float32Array(total);
  const frontVotes = new Uint8Array(total);
  const behindVotes = new Uint8Array(total);
  const trunc = voxel * 3.5;

  const origin = [bounds.lo[0] - voxel * 0.5, bounds.lo[1] - voxel * 0.5, bounds.lo[2] - voxel * 0.5];

  const nxny = nx * ny;
  const world = [0, 0, 0];

  for (let mi = 0; mi < maps.length; mi++) {
    check();
    const map = maps[mi];
    if (!map) continue;
    const cam = map.cam;
    const { w, h, depth } = map;
    const work = works[map.refIdx];
    const rgba = work.rgba;

    // 计算该深度图在世界空间中的包围盒，限制体素遍历范围
    const corners = [];
    for (const z of [map.zRange.lo, map.zRange.hi]) {
      for (const [px, py] of [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]]) {
        const xc = ((px - cam.cx) / cam.fx) * z;
        const yc = ((py - cam.cy) / cam.fy) * z;
        const zc = z;
        const R = cam.R;
        const Xw = [
          R[0] * xc + R[3] * yc + R[6] * zc - (R[0] * cam.t[0] + R[3] * cam.t[1] + R[6] * cam.t[2]),
          R[1] * xc + R[4] * yc + R[7] * zc - (R[1] * cam.t[0] + R[4] * cam.t[1] + R[7] * cam.t[2]),
          R[2] * xc + R[5] * yc + R[8] * zc - (R[2] * cam.t[0] + R[5] * cam.t[1] + R[8] * cam.t[2]),
        ];
        corners.push(Xw);
      }
    }
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (const c of corners) {
      minX = Math.min(minX, c[0]); maxX = Math.max(maxX, c[0]);
      minY = Math.min(minY, c[1]); maxY = Math.max(maxY, c[1]);
      minZ = Math.min(minZ, c[2]); maxZ = Math.max(maxZ, c[2]);
    }
    const i0 = Math.max(0, Math.floor((minX - origin[0]) / voxel));
    const i1 = Math.min(nx - 1, Math.ceil((maxX - origin[0]) / voxel));
    const j0 = Math.max(0, Math.floor((minY - origin[1]) / voxel));
    const j1 = Math.min(ny - 1, Math.ceil((maxY - origin[1]) / voxel));
    const k0 = Math.max(0, Math.floor((minZ - origin[2]) / voxel));
    const k1 = Math.min(nz - 1, Math.ceil((maxZ - origin[2]) / voxel));

    const R = cam.R, t = cam.t;
    for (let k = k0; k <= k1; k++) {
      world[2] = origin[2] + k * voxel;
      const idxK = k * nxny;
      for (let j = j0; j <= j1; j++) {
        world[1] = origin[1] + j * voxel;
        const idxJ = idxK + j * nx;
        for (let i = i0; i <= i1; i++) {
          world[0] = origin[0] + i * voxel;
          const xc = R[0] * world[0] + R[1] * world[1] + R[2] * world[2] + t[0];
          const yc = R[3] * world[0] + R[4] * world[1] + R[5] * world[2] + t[1];
          const zc = R[6] * world[0] + R[7] * world[1] + R[8] * world[2] + t[2];
          if (zc <= 1e-9) continue;
          const u = cam.fx * (xc / zc) + cam.cx;
          const v = cam.fy * (yc / zc) + cam.cy;
          if (u < 0 || v < 0 || u >= w - 1 || v >= h - 1) continue;
          const d = depth[(v | 0) * w + (u | 0)];
          if (!(d > 0)) continue;

          const idx = idxJ + i;
          const sdf = zc - d;
          if (sdf < -trunc) {
            if (behindVotes[idx] < 255) behindVotes[idx]++;
          } else if (sdf > trunc) {
            if (frontVotes[idx] < 255) frontVotes[idx]++;
          } else {
            const wgt = 1;
            const val = Math.max(-1, Math.min(1, sdf / trunc));
            const oldW = weight[idx];
            const newW = oldW + wgt;
            tsdf[idx] = (tsdf[idx] * oldW + val * wgt) / newW;
            weight[idx] = newW;
            const col = sampleRGBNearest(rgba, w, h, u, v);
            if (col) {
              cr[idx] += col[0] * wgt;
              cg[idx] += col[1] * wgt;
              cb[idx] += col[2] * wgt;
            }
          }
        }
      }
    }
    emit('fusion', `融合深度图 ${mi + 1}/${maps.length}`, 0.62 + 0.2 * ((mi + 1) / maps.length));
    await yieldFn();
  }

  if (cfg.debug) {
    let wCount = 0;
    for (let i = 0; i < total; i++) if (weight[i] > 0) wCount++;
    console.log('[tsdf] dims', dims, 'voxel', voxel.toFixed(4), 'origin', origin.map(v=>v.toFixed(2)), 'weightedVoxels', wCount);
    // 用第一张深度图的几个有效像素验证体素反投影
    const dm = maps[0];
    if (dm) {
      const cam = dm.cam;
      let shown = 0;
      for (let y = 4; y < dm.h - 4 && shown < 5; y += 9) {
        for (let x = 4; x < dm.w - 4 && shown < 5; x += 9) {
          const z = dm.depth[y * dm.w + x];
          if (!(z > 0)) continue;
          const xc = (x - cam.cx) / cam.fx * z;
          const yc = (y - cam.cy) / cam.fy * z;
          const X = [
            cam.R[0]*xc + cam.R[3]*yc + cam.R[6]*z - (cam.R[0]*cam.t[0]+cam.R[3]*cam.t[1]+cam.R[6]*cam.t[2]),
            cam.R[1]*xc + cam.R[4]*yc + cam.R[7]*z - (cam.R[1]*cam.t[0]+cam.R[4]*cam.t[1]+cam.R[7]*cam.t[2]),
            cam.R[2]*xc + cam.R[5]*yc + cam.R[8]*z - (cam.R[2]*cam.t[0]+cam.R[5]*cam.t[1]+cam.R[8]*cam.t[2]),
          ];
          const vi = Math.round((X[0]-origin[0])/voxel), vj = Math.round((X[1]-origin[1])/voxel), vk = Math.round((X[2]-origin[2])/voxel);
          if (vi<0||vj<0||vk<0||vi>=nx||vj>=ny||vk>=nz) { console.log('[tsdf-check] out', X.map(v=>v.toFixed(2))); shown++; continue; }
          const idx = vk*nxny + vj*nx + vi;
          console.log('[tsdf-check] X', X.map(v=>v.toFixed(2)), 'idx', vi,vj,vk, 'tsdf', tsdf[idx].toFixed(2), 'w', weight[idx].toFixed(1), 'front', frontVotes[idx], 'behind', behindVotes[idx]);
          shown++;
        }
      }
    }
  }
  // 未落在窄带内但被多数视图判定为“表面之后”的体素标记为内部
  const red = new Float32Array(total);
  const green = new Float32Array(total);
  const blue = new Float32Array(total);
  for (let i = 0; i < total; i++) {
    if (weight[i] > 0) {
      red[i] = cr[i] / weight[i];
      green[i] = cg[i] / weight[i];
      blue[i] = cb[i] / weight[i];
    } else {
      tsdf[i] = behindVotes[i] > frontVotes[i] ? -1 : 1;
      red[i] = 210; green[i] = 214; blue[i] = 224;
    }
  }

  // 已知体素掩码 + 形态学膨胀（沿 6 邻域外推 TSDF/颜色），
  // 让 Marching Tetrahedra 能跨越深度图的细小空洞，避免表面碎成孤岛。
  const known = new Uint8Array(total);
  for (let i = 0; i < total; i++) if (weight[i] > 0) known[i] = 1;
  const strideZ = nxny, strideY = nx, strideX = 1;
  for (let pass = 0; pass < (cfg.dilate ?? 3); pass++) {
    const addIdx = [];
    const addT = [];
    const addR = [];
    const addG = [];
    const addB = [];
    for (let k = 1; k < nz - 1; k++) {
      for (let j = 1; j < ny - 1; j++) {
        const rowBase = k * strideZ + j * strideY;
        for (let i = 1; i < nx - 1; i++) {
          const idx = rowBase + i;
          if (known[idx]) continue;
          let cnt = 0, st = 0, sr = 0, sg = 0, sb = 0;
          const nbrs = [idx + strideX, idx - strideX, idx + strideY,
            idx - strideY, idx + strideZ, idx - strideZ];
          for (const q of nbrs) {
            if (!known[q]) continue;
            cnt++; st += tsdf[q]; sr += red[q]; sg += green[q]; sb += blue[q];
          }
          if (cnt < 2) continue;
          addIdx.push(idx);
          addT.push(st / cnt);
          addR.push(sr / cnt);
          addG.push(sg / cnt);
          addB.push(sb / cnt);
        }
      }
    }
    if (!addIdx.length) break;
    for (let a = 0; a < addIdx.length; a++) {
      const idx = addIdx[a];
      tsdf[idx] = addT[a];
      red[idx] = addR[a];
      green[idx] = addG[a];
      blue[idx] = addB[a];
      weight[idx] = 1e-6;
      known[idx] = 1;
    }
    if (cfg.debug) console.log('[tsdf] dilation pass', pass + 1, 'added', addIdx.length);
  }

  return { tsdf, weight, color: [red, green, blue], dims, origin, voxel, bounds };
}

/* ---------------- Marching Tetrahedra ---------------- */

const CUBE_CORNERS = [
  [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
];

// 6 个四面体（共享 0-6 主对角线），保证相邻立方体剖分一致
const TETS = [
  [0, 1, 2, 6],
  [0, 2, 3, 6],
  [0, 3, 7, 6],
  [0, 7, 4, 6],
  [0, 4, 5, 6],
  [0, 5, 1, 6],
];

const TET_EDGES = [
  [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
];

function marchingTetrahedra(volume, emit, yieldFn, check) {
  const { tsdf, weight, color, dims, origin, voxel } = volume;
  const [nx, ny, nz] = dims;
  const nxny = nx * ny;
  const totalVox = nx * ny * nz;

  const positions = [];
  const colors = [];
  const indices = [];
  const vertexMap = new Map();

  const cornerWorld = (i, j, k) => [
    origin[0] + i * voxel,
    origin[1] + j * voxel,
    origin[2] + k * voxel,
  ];

  const addVertex = (g0, g1, v0, v1, c0, c1) => {
    const key = g0 < g1 ? g0 * totalVox + g1 : g1 * totalVox + g0;
    const existing = vertexMap.get(key);
    if (existing !== undefined) return existing;
    const t = v0 / (v0 - v1);

    const i0 = g0 % nx, j0 = ((g0 / nx) | 0) % ny, k0 = (g0 / nxny) | 0;
    const i1 = g1 % nx, j1 = ((g1 / nx) | 0) % ny, k1 = (g1 / nxny) | 0;
    const p0 = cornerWorld(i0, j0, k0);
    const p1 = cornerWorld(i1, j1, k1);
    const idx = positions.length / 3;
    positions.push(
      p0[0] + (p1[0] - p0[0]) * t,
      p0[1] + (p1[1] - p0[1]) * t,
      p0[2] + (p1[2] - p0[2]) * t,
    );
    const o0 = g0, o1 = g1;
    colors.push(
      (color[0][o0] + (color[0][o1] - color[0][o0]) * t) / 255,
      (color[1][o0] + (color[1][o1] - color[1][o0]) * t) / 255,
      (color[2][o0] + (color[2][o1] - color[2][o0]) * t) / 255,
    );
    vertexMap.set(key, idx);
    return idx;
  };

  const emitTri = (a, b, c, insideCentroid, outsideCentroid) => {
    if (a === b || b === c || a === c) return;
    const ax = positions[a * 3], ay = positions[a * 3 + 1], az = positions[a * 3 + 2];
    const bx = positions[b * 3], by = positions[b * 3 + 1], bz = positions[b * 3 + 2];
    const cx = positions[c * 3], cy = positions[c * 3 + 1], cz = positions[c * 3 + 2];
    const ux = bx - ax, uy = by - ay, uz = bz - az;
    const vx = cx - ax, vy = cy - ay, vz = cz - az;
    const nx0 = uy * vz - uz * vy;
    const ny0 = uz * vx - ux * vz;
    const nz0 = ux * vy - uy * vx;
    const dirx = outsideCentroid[0] - insideCentroid[0];
    const diry = outsideCentroid[1] - insideCentroid[1];
    const dirz = outsideCentroid[2] - insideCentroid[2];
    if (nx0 * dirx + ny0 * diry + nz0 * dirz < 0) {
      indices.push(a, c, b);
    } else {
      indices.push(a, b, c);
    }
  };

  const vals = new Float64Array(8);
  const gids = new Int32Array(8);

  for (let k = 0; k < nz - 1; k++) {
    if ((k & 15) === 0) {
      check();
      emit('mesh', `Marching Tetrahedra ${Math.round((k / (nz - 1)) * 100)}%`, 0.82 + 0.15 * (k / Math.max(1, nz - 1)));
    }
    for (let j = 0; j < ny - 1; j++) {
      const baseJ = k * nxny + j * nx;
      for (let i = 0; i < nx - 1; i++) {
        const base = baseJ + i;
        let anyPos = false, anyNeg = false;
        let allKnown = true;
        for (let c = 0; c < 8; c++) {
          const [dx, dy, dz] = CUBE_CORNERS[c];
          const g = base + dx + dy * nx + dz * nxny;
          gids[c] = g;
          if (!(weight[g] > 0)) { allKnown = false; break; }
          const v = tsdf[g];
          vals[c] = v;
          if (v < 0) anyNeg = true; else anyPos = true;
        }
        // 只对 8 个角都被深度图观测到的立方体做等值面提取，
        // 避免在未观测区域生成大片虚假表面。
        if (!allKnown || !anyNeg || !anyPos) continue;

        for (let ti = 0; ti < TETS.length; ti++) {
          const tet = TETS[ti];
          const inside = [];
          const outside = [];
          for (let c = 0; c < 4; c++) {
            if (vals[tet[c]] < 0) inside.push(c);
            else outside.push(c);
          }
          if (!inside.length || !outside.length) continue;

          const inCentroid = [0, 0, 0];
          for (const c of inside) {
            const g = gids[tet[c]];
            inCentroid[0] += origin[0] + (g % nx) * voxel;
            inCentroid[1] += origin[1] + (((g / nx) | 0) % ny) * voxel;
            inCentroid[2] += origin[2] + ((g / nxny) | 0) * voxel;
          }
          inCentroid[0] /= inside.length; inCentroid[1] /= inside.length; inCentroid[2] /= inside.length;

          const outCentroid = [0, 0, 0];
          for (const c of outside) {
            const g = gids[tet[c]];
            outCentroid[0] += origin[0] + (g % nx) * voxel;
            outCentroid[1] += origin[1] + (((g / nx) | 0) % ny) * voxel;
            outCentroid[2] += origin[2] + ((g / nxny) | 0) * voxel;
          }
          outCentroid[0] /= outside.length; outCentroid[1] /= outside.length; outCentroid[2] /= outside.length;

          const edgeVertex = (a, b) => {
            const ga = gids[tet[a]], gb = gids[tet[b]];
            return addVertex(ga, gb, vals[tet[a]], vals[tet[b]],
              [color[0][ga], color[1][ga], color[2][ga]],
              [color[0][gb], color[1][gb], color[2][gb]]);
          };

          if (inside.length === 1) {
            const iv = inside[0];
            const v0 = edgeVertex(iv, outside[0]);
            const v1 = edgeVertex(iv, outside[1]);
            const v2 = edgeVertex(iv, outside[2]);
            emitTri(v0, v1, v2, inCentroid, outCentroid);
          } else if (inside.length === 3) {
            const ov = outside[0];
            const v0 = edgeVertex(ov, inside[0]);
            const v1 = edgeVertex(ov, inside[1]);
            const v2 = edgeVertex(ov, inside[2]);
            emitTri(v0, v1, v2, inCentroid, outCentroid);
          } else {
            const i0 = inside[0], i1 = inside[1];
            const o0 = outside[0], o1 = outside[1];
            const a = edgeVertex(i0, o0);
            const b = edgeVertex(i0, o1);
            const c = edgeVertex(i1, o1);
            const d = edgeVertex(i1, o0);
            emitTri(a, b, c, inCentroid, outCentroid);
            emitTri(a, c, d, inCentroid, outCentroid);
          }
        }
      }
    }
  }

  return {
    positions: new Float32Array(positions),
    colors: new Float32Array(colors),
    indices: new Uint32Array(indices),
  };
}

function smoothMesh(positions, indices, iterations = 1, lambda = 0.35) {
  const vcount = positions.length / 3;
  const sums = new Float32Array(vcount * 3);
  const counts = new Uint32Array(vcount);
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i], b = indices[i + 1], c = indices[i + 2];
    for (const [u, v] of [[a, b], [b, c], [c, a]]) {
      sums[u * 3] += positions[v * 3];
      sums[u * 3 + 1] += positions[v * 3 + 1];
      sums[u * 3 + 2] += positions[v * 3 + 2];
      counts[u]++;
    }
  }
  for (let it = 0; it < iterations; it++) {
    const next = Float32Array.from(positions);
    for (let v = 0; v < vcount; v++) {
      if (!counts[v]) continue;
      const inv = 1 / counts[v];
      next[v * 3] = positions[v * 3] * (1 - lambda) + sums[v * 3] * inv * lambda;
      next[v * 3 + 1] = positions[v * 3 + 1] * (1 - lambda) + sums[v * 3 + 1] * inv * lambda;
      next[v * 3 + 2] = positions[v * 3 + 2] * (1 - lambda) + sums[v * 3 + 2] * inv * lambda;
    }
    positions = next;
  }
  return positions;
}

function computeNormals(positions, indices) {
  const normals = new Float32Array(positions.length);
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i] * 3, b = indices[i + 1] * 3, c = indices[i + 2] * 3;
    const ux = positions[b] - positions[a];
    const uy = positions[b + 1] - positions[a + 1];
    const uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a];
    const vy = positions[c + 1] - positions[a + 1];
    const vz = positions[c + 2] - positions[a + 2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    for (const d of [a, b, c]) {
      normals[d] += nx;
      normals[d + 1] += ny;
      normals[d + 2] += nz;
    }
  }
  for (let i = 0; i < normals.length; i += 3) {
    const n = Math.hypot(normals[i], normals[i + 1], normals[i + 2]) || 1;
    normals[i] /= n;
    normals[i + 1] /= n;
    normals[i + 2] /= n;
  }
  return normals;
}

/* ---------------- 主入口 ---------------- */

/**
 * @param {{cameras:Array,points:Array,images:Array}} sfm
 * @param {object} opts
 * @param {{onProgress?:Function,yieldFn?:Function,shouldCancel?:Function}} hooks
 */
export async function reconstructDense(sfm, opts = {}, hooks = {}) {
  const cfg = { ...QUALITY.standard, ...(QUALITY[opts.quality] || {}), ...opts };
  const progress = hooks.onProgress || (() => {});
  const yieldFn = hooks.yieldFn || (async () => {});
  const cancelled = hooks.shouldCancel || (() => false);
  const emit = (stage, message, fraction, extra) => progress({ stage, message, fraction, ...extra });
  const check = () => { if (cancelled()) throw new Error('已取消'); };

  const { cameras, points, images } = sfm;
  if (!cameras || cameras.length < 2) throw new Error('稠密重建至少需要 2 个已定位相机');
  if (!points || points.length < 10) throw new Error('稀疏点不足，无法进行稠密重建');

  // 选择参考视图：数量超过上限时均匀抽样
  let refIndices = [];
  if (cameras.length <= cfg.maxRef) {
    refIndices = cameras.map((_, i) => i);
  } else {
    for (let k = 0; k < cfg.maxRef; k++) {
      refIndices.push(Math.round((k * (cameras.length - 1)) / Math.max(1, cfg.maxRef - 1)));
    }
    refIndices = [...new Set(refIndices)];
  }

  // 准备降采样工作图像
  emit('dense', '准备多尺度图像…', 0.02);
  const works = cameras.map((cam) => {
    const scale = Math.min(1, cfg.targetSize / Math.max(cam.width, cam.height));
    return prepareWorkImage(images[cam.index], scale);
  });
  await yieldFn();

  const share = buildShareMatrix(cameras, points);
  const maps = new Array(cameras.length).fill(null);

  // 深度图估计
  for (let ri = 0; ri < refIndices.length; ri++) {
    check();
    const refIdx = refIndices[ri];
    const neighborIdx = selectNeighbors(refIdx, cameras, share, cfg.neighbors);
    emit('dense', `平面扫描深度图 ${ri + 1}/${refIndices.length}`, 0.03 + 0.5 * (ri / refIndices.length));
    const map = await estimateDepthMap(
      refIdx, neighborIdx, cameras, works, points, cfg,
      emit, yieldFn, check
    );
    if (map) {
      medianFilterDepth(map);
      if (cfg.debug) {
        let nv = 0;
        for (let q = 0; q < map.depth.length; q++) if (map.depth[q] > 0) nv++;
        console.log('[depth] ref', refIdx, 'neighbors', neighborIdx.join(','),
          'range', map.zRange.lo.toFixed(2), map.zRange.hi.toFixed(2),
          'valid', nv, '/', map.w * map.h);
      }
      maps[refIdx] = map;
    }
    await yieldFn();
  }

  // 前向-后向一致性
  emit('dense', '左右一致性检查…', 0.55);
  for (const map of maps) {
    if (map) forwardBackwardCheck(map, maps, cfg.debug);
  }
  await yieldFn();

  let validDepthPixels = 0;
  for (const m of maps) if (m) {
    let nv = 0;
    for (let i = 0; i < m.depth.length; i++) if (m.depth[i] > 0) nv++;
    if (cfg.debug) console.log('[fb] ref', m.refIdx, 'valid', nv, '/', m.w * m.h);
    validDepthPixels += nv;
  }
  if (validDepthPixels < 500) {
    throw new Error('稠密深度估计失败（有效像素过少）。请增加照片数量与重叠度，或改用更高质量的稠密设置。');
  }

  // TSDF 融合
  check();
  const validMaps = maps.filter(Boolean);
  emit('fusion', '构建 TSDF 体素场…', 0.6);
  const volume = await fuseTSDF(validMaps, cameras, works, points, cfg, emit, yieldFn, check);

  // 网格提取
  check();
  emit('mesh', '提取三角网格…', 0.82);
  const mesh = marchingTetrahedra(volume, emit, yieldFn, check);
  await yieldFn();

  if (mesh.indices.length < 30) {
    throw new Error('网格提取结果为空。请增加照片数量，或降低稠密重建的体素精度后重试。');
  }

  // 轻微平滑
  let positions = mesh.positions;
  if (cfg.smoothMesh !== false) {
    positions = smoothMesh(Float32Array.from(positions), mesh.indices, 1, 0.28);
  }
  const normals = computeNormals(positions, mesh.indices);

  const stats = {
    depthMaps: validMaps.length,
    validDepthPixels,
    grid: volume.dims,
    voxelSize: volume.voxel,
    vertices: positions.length / 3,
    triangles: mesh.indices.length / 3,
  };

  emit('mesh', `稠密重建完成 · ${stats.vertices.toLocaleString()} 顶点 / ${stats.triangles.toLocaleString()} 三角面`, 1, { stats });

  return {
    mesh: {
      positions,
      colors: mesh.colors,
      normals,
      indices: mesh.indices,
    },
    depthMaps: validMaps,
    volumeBounds: volume.bounds,
    stats,
  };
}