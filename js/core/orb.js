/*
 * ORB 特征提取与匹配（纯 JS 实现，无依赖）
 *
 * 流程与 ORB / COLMAP 的前端特征阶段一致：
 *   FAST-9 角点 -> Harris 评分与 NMS -> 金字塔尺度分配
 *   -> 灰度质心方向 -> 旋转 BRIEF 256 位描述子 -> Hamming 距离 + 比值检验
 */

import { mulberry32 } from './mathx.js';

/* ---------------- 位运算 ---------------- */

// 16 位 popcount 查找表，加速 Hamming 距离
const POP16 = new Uint8Array(65536);
for (let i = 0; i < 65536; i++) {
  let v = i, c = 0;
  while (v) { v &= v - 1; c++; }
  POP16[i] = c;
}

export function popcount32(x) {
  return POP16[x & 0xffff] + POP16[(x >>> 16) & 0xffff];
}

/* ---------------- 基础图像操作 ---------------- */

/** 5x5 二项式模糊 [1,4,6,4,1]/16，边界复制 */
export function blur5(src, w, h) {
  const tmp = new Uint8Array(w * h);
  const out = new Uint8Array(w * h);

  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const x0 = x < 2 ? 0 : x - 2;
      const x1 = x < 1 ? 0 : x - 1;
      const x3 = x + 1 >= w ? w - 1 : x + 1;
      const x4 = x + 2 >= w ? w - 1 : x + 2;
      tmp[row + x] = (src[row + x0] + 4 * src[row + x1] + 6 * src[row + x] +
        4 * src[row + x3] + src[row + x4] + 8) >> 4;
    }
  }
  for (let y = 0; y < h; y++) {
    const y0 = (y < 2 ? 0 : y - 2) * w;
    const y1 = (y < 1 ? 0 : y - 1) * w;
    const y3 = (y + 1 >= h ? h - 1 : y + 1) * w;
    const y4 = (y + 2 >= h ? h - 1 : y + 2) * w;
    const row = y * w;
    for (let x = 0; x < w; x++) {
      out[row + x] = (tmp[y0 + x] + 4 * tmp[y1 + x] + 6 * tmp[row + x] +
        4 * tmp[y3 + x] + tmp[y4 + x] + 8) >> 4;
    }
  }
  return out;
}

/** 双线性灰度采样 */
export function bilinearGray(gray, w, h, x, y) {
  const xc = x < 0 ? 0 : x > w - 1 ? w - 1 : x;
  const yc = y < 0 ? 0 : y > h - 1 ? h - 1 : y;
  const x0 = xc | 0;
  const y0 = yc | 0;
  const x1 = x0 + 1 >= w ? w - 1 : x0 + 1;
  const y1 = y0 + 1 >= h ? h - 1 : y0 + 1;
  const fx = xc - x0, fy = yc - y0;
  const a = gray[y0 * w + x0], b = gray[y0 * w + x1];
  const c = gray[y1 * w + x0], d = gray[y1 * w + x1];
  return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

/** 双线性缩小（配合模糊使用） */
export function downsample(src, w, h, nw, nh) {
  const out = new Uint8Array(nw * nh);
  const sx = w / nw;
  const sy = h / nh;
  for (let y = 0; y < nh; y++) {
    const yy = Math.min(h - 1, (y + 0.5) * sy - 0.5);
    for (let x = 0; x < nw; x++) {
      const xx = Math.min(w - 1, (x + 0.5) * sx - 0.5);
      out[y * nw + x] = bilinearGray(src, w, h, xx, yy) | 0;
    }
  }
  return out;
}

/* ---------------- ORB 常量 ---------------- */

const CIRCLE = [
  [0, -3], [1, -3], [2, -2], [3, -1], [3, 0], [3, 1], [2, 2], [1, 3],
  [0, 3], [-1, 3], [-2, 2], [-3, 1], [-3, 0], [-3, -1], [-2, -2], [-1, -3],
];

const PATCH_RADIUS = 13;
const ORIENT_RADIUS = 15;
const MARGIN = 20;

/* 固定的旋转 BRIEF 采样模式（高斯分布，半径 13） */
const BRIEF_PATTERN = (() => {
  const rand = mulberry32(0x9e3779b9);
  const gauss = () => {
    let u = 0, v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const pat = new Float32Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    let x1, y1, x2, y2;
    do {
      x1 = gauss() * 5.2; y1 = gauss() * 5.2;
    } while (x1 * x1 + y1 * y1 > PATCH_RADIUS * PATCH_RADIUS);
    do {
      x2 = gauss() * 5.2; y2 = gauss() * 5.2;
    } while (x2 * x2 + y2 * y2 > PATCH_RADIUS * PATCH_RADIUS);
    pat[i * 4] = x1; pat[i * 4 + 1] = y1; pat[i * 4 + 2] = x2; pat[i * 4 + 3] = y2;
  }
  return pat;
})();

/* ---------------- Harris 响应 ---------------- */

function harrisResponse(gray, w, h) {
  const resp = new Float32Array(w * h);
  const ix = new Float32Array(w * h);
  const iy = new Float32Array(w * h);

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const p = y * w + x;
      ix[p] = (gray[p + 1] - gray[p - 1]) * 0.5;
      iy[p] = (gray[p + w] - gray[p - w]) * 0.5;
    }
  }

  // 用积分图计算 5x5 窗口和
  const W = w + 1;
  const sxx = new Float64Array(W * (h + 1));
  const syy = new Float64Array(W * (h + 1));
  const sxy = new Float64Array(W * (h + 1));
  for (let y = 0; y < h; y++) {
    let rxx = 0, ryy = 0, rxy = 0;
    for (let x = 0; x < w; x++) {
      const p = y * w + x;
      rxx += ix[p] * ix[p];
      ryy += iy[p] * iy[p];
      rxy += ix[p] * iy[p];
      const q = (y + 1) * W + (x + 1);
      sxx[q] = sxx[q - W] + rxx;
      syy[q] = syy[q - W] + ryy;
      sxy[q] = sxy[q - W] + rxy;
    }
  }

  const R = 2;
  for (let y = R; y < h - R; y++) {
    for (let x = R; x < w - R; x++) {
      const x0 = x - R, y0 = y - R, x1 = x + R + 1, y1 = y + R + 1;
      const a = sxx[y1 * W + x1] - sxx[y0 * W + x1] - sxx[y1 * W + x0] + sxx[y0 * W + x0];
      const b = syy[y1 * W + x1] - syy[y0 * W + x1] - syy[y1 * W + x0] + syy[y0 * W + x0];
      const c = sxy[y1 * W + x1] - sxy[y0 * W + x1] - sxy[y1 * W + x0] + sxy[y0 * W + x0];
      const det = a * b - c * c;
      const trace = a + b;
      resp[y * w + x] = det - 0.04 * trace * trace;
    }
  }
  return resp;
}

/* ---------------- FAST-9 ---------------- */

function fastCandidates(gray, w, h, threshold) {
  const cand = [];
  for (let y = 3; y < h - 3; y++) {
    for (let x = 3; x < w - 3; x++) {
      const p = y * w + x;
      const c = gray[p];
      // 高速预测试：圆上 4 个正交位置的亮/暗计数
      const p0 = gray[p - 3 * w], p4 = gray[p + 3 * w];
      const p8 = gray[p - 3], p12 = gray[p + 3];
      const hi = c + threshold, lo = c - threshold;
      let bright = 0, dark = 0;
      if (p0 > hi) bright++; else if (p0 < lo) dark++;
      if (p4 > hi) bright++; else if (p4 < lo) dark++;
      if (p8 > hi) bright++; else if (p8 < lo) dark++;
      if (p12 > hi) bright++; else if (p12 < lo) dark++;
      if (bright < 3 && dark < 3) continue;

      let bRun = 0, dRun = 0, maxB = 0, maxD = 0;
      for (let k = 0; k < 32; k++) {
        const [dx, dy] = CIRCLE[k & 15];
        const v = gray[(y + dy) * w + (x + dx)];
        if (v > hi) { bRun++; if (bRun > maxB) maxB = bRun; dRun = 0; }
        else if (v < lo) { dRun++; if (dRun > maxD) maxD = dRun; bRun = 0; }
        else { bRun = 0; dRun = 0; }
        if (maxB >= 9 || maxD >= 9) break;
      }
      if (maxB >= 9 || maxD >= 9) cand.push({ x, y });
    }
  }
  return cand;
}

/* ---------------- 方向 ---------------- */

function orientation(gray, w, h, x, y) {
  let m10 = 0, m01 = 0;
  for (let dy = -ORIENT_RADIUS; dy <= ORIENT_RADIUS; dy++) {
    const yy = y + dy;
    for (let dx = -ORIENT_RADIUS; dx <= ORIENT_RADIUS; dx++) {
      if (dx * dx + dy * dy > ORIENT_RADIUS * ORIENT_RADIUS) continue;
      const xx = x + dx;
      const v = gray[yy * w + xx];
      m10 += dx * v;
      m01 += dy * v;
    }
  }
  return Math.atan2(m01, m10);
}

/* ---------------- 描述子 ---------------- */

function describe(gray, w, h, x, y, angle, out, outOff) {
  const c = Math.cos(angle), s = Math.sin(angle);
  for (let i = 0; i < 256; i++) {
    const o = i * 4;
    const x1 = x + c * BRIEF_PATTERN[o] - s * BRIEF_PATTERN[o + 1];
    const y1 = y + s * BRIEF_PATTERN[o] + c * BRIEF_PATTERN[o + 1];
    const x2 = x + c * BRIEF_PATTERN[o + 2] - s * BRIEF_PATTERN[o + 3];
    const y2 = y + s * BRIEF_PATTERN[o + 2] + c * BRIEF_PATTERN[o + 3];
    const v1 = bilinearGray(gray, w, h, x1, y1);
    const v2 = bilinearGray(gray, w, h, x2, y2);
    if (v1 < v2) {
      const word = i >>> 5;
      out[outOff + word] |= (1 << (i & 31));
    }
  }
}

/* ---------------- 主入口 ---------------- */

/**
 * @param {Uint8Array} rgba RGBA 像素
 * @param {number} w
 * @param {number} h
 * @param {object} opts
 * @returns {{x:Float32Array,y:Float32Array,angle:Float32Array,level:Uint8Array,
 *            score:Float32Array,desc:Uint32Array,count:number,width:number,height:number}}
 */
export function extractOrb(rgba, w, h, opts = {}) {
  const nfeatures = opts.nfeatures || 1200;
  const levels = opts.levels || 4;
  const scaleFactor = opts.scaleFactor || 1.25;
  const threshold = opts.fastThreshold || 10;
  const nmsRadius = 1;

  const gray0 = opts.gray || (() => {
    const g = new Uint8Array(w * h);
    for (let i = 0, p = 0; i < g.length; i++, p += 4) {
      g[i] = (rgba[p] * 77 + rgba[p + 1] * 150 + rgba[p + 2] * 29) >> 8;
    }
    return g;
  })();

  // 构建金字塔
  const pyr = [];
  let curW = w, curH = h, curScale = 1;
  let curGray = gray0;
  for (let l = 0; l < levels; l++) {
    if (l > 0) {
      const nw = Math.max(32, Math.round(w / Math.pow(scaleFactor, l)));
      const nh = Math.max(32, Math.round(h / Math.pow(scaleFactor, l)));
      if (nw < 48 || nh < 48) break;
      curGray = downsample(blur5(curGray, curW, curH), curW, curH, nw, nh);
      curW = nw; curH = nh; curScale = nw / w;
    }
    const blurred = blur5(curGray, curW, curH);
    pyr.push({ gray: blurred, w: curW, h: curH, scale: curScale, index: l });
  }

  // 每层的候选角点
  const perLevel = [];
  const debugLevels = [];
  let totalArea = 0;
  for (const lv of pyr) totalArea += lv.w * lv.h;

  for (const lv of pyr) {
    const { gray, w: lw, h: lh } = lv;
    const resp = harrisResponse(gray, lw, lh);
    const cands = fastCandidates(gray, lw, lh, threshold);
    const pts = [];
    let harrisPos = 0;
    for (const pt of cands) {
      if (pt.x < MARGIN || pt.y < MARGIN || pt.x >= lw - MARGIN || pt.y >= lh - MARGIN) continue;
      const r = resp[pt.y * lw + pt.x];
      if (r <= 0) continue;
      harrisPos++;
      let isMax = true;
      for (let dy = -nmsRadius; dy <= nmsRadius && isMax; dy++) {
        for (let dx = -nmsRadius; dx <= nmsRadius; dx++) {
          if (dx === 0 && dy === 0) continue;
          if (resp[(pt.y + dy) * lw + (pt.x + dx)] > r) { isMax = false; break; }
        }
      }
      if (isMax) pts.push({ x: pt.x, y: pt.y, score: r });
    }
    pts.sort((a, b) => b.score - a.score);
    debugLevels.push({ level: lv.index, fast: cands.length, harrisPos, afterNms: pts.length });

    // 网格化均匀分配
    const quota = Math.max(40, Math.round((nfeatures * lv.w * lv.h) / totalArea));
    const cell = 28;
    const cols = Math.ceil(lw / cell);
    const rows = Math.ceil(lh / cell);
    const cellCount = new Uint16Array(cols * rows);
    const cellCap = Math.max(2, Math.ceil((quota * 4) / (cols * rows)));
    const selected = [];
    for (const p of pts) {
      const ci = (p.y / cell) | 0;
      const cj = (p.x / cell) | 0;
      const ci2 = ci * cols + cj;
      if (cellCount[ci2] >= cellCap) continue;
      cellCount[ci2]++;
      selected.push(p);
      if (selected.length >= quota) break;
    }
    perLevel.push({ level: lv, pts: selected });
  }

  // 提取描述子
  const all = [];
  for (const { level, pts } of perLevel) {
    const { gray, w: lw, h: lh, scale } = level;
    for (const p of pts) {
      const ang = orientation(gray, lw, lh, p.x, p.y);
      all.push({ x: p.x / scale, y: p.y / scale, angle: ang, level: level.index, score: p.score, gray, w: lw, h: lh, lx: p.x, ly: p.y });
    }
  }

  const count = all.length;
  const xs = new Float32Array(count);
  const ys = new Float32Array(count);
  const angles = new Float32Array(count);
  const levelArr = new Uint8Array(count);
  const scores = new Float32Array(count);
  const desc = new Uint32Array(count * 8);

  for (let i = 0; i < count; i++) {
    const f = all[i];
    xs[i] = f.x;
    ys[i] = f.y;
    angles[i] = f.angle;
    levelArr[i] = f.level;
    scores[i] = f.score;
    describe(f.gray, f.w, f.h, f.lx, f.ly, f.angle, desc, i * 8);
  }

  return { x: xs, y: ys, angle: angles, level: levelArr, score: scores, desc, count, width: w, height: h, debugLevels };
}

/* ---------------- 匹配 ---------------- */

/**
 * Hamming 距离 + Lowe 比值检验 + 双向交叉验证
 * @returns {Array<{a:number,b:number,dist:number}>}
 */
export function matchDescriptors(fa, fb, opts = {}) {
  const ratio = opts.ratio ?? 0.8;
  const maxDist = opts.maxDist ?? 88;
  const crossCheck = opts.crossCheck ?? true;
  const na = fa.count;
  const nb = fb.count;
  if (!na || !nb) return [];

  const bestA = new Int32Array(na).fill(-1);
  const bestDistA = new Int32Array(na).fill(1e9);

  for (let i = 0; i < na; i++) {
    const offA = i * 8;
    let best = 1e9, second = 1e9, bestIdx = -1;
    for (let j = 0; j < nb; j++) {
      const offB = j * 8;
      let d = 0;
      d += POP16[(fa.desc[offA] ^ fb.desc[offB]) & 0xffff];
      d += POP16[((fa.desc[offA] ^ fb.desc[offB]) >>> 16) & 0xffff];
      d += POP16[(fa.desc[offA + 1] ^ fb.desc[offB + 1]) & 0xffff];
      d += POP16[((fa.desc[offA + 1] ^ fb.desc[offB + 1]) >>> 16) & 0xffff];
      d += POP16[(fa.desc[offA + 2] ^ fb.desc[offB + 2]) & 0xffff];
      d += POP16[((fa.desc[offA + 2] ^ fb.desc[offB + 2]) >>> 16) & 0xffff];
      d += POP16[(fa.desc[offA + 3] ^ fb.desc[offB + 3]) & 0xffff];
      d += POP16[((fa.desc[offA + 3] ^ fb.desc[offB + 3]) >>> 16) & 0xffff];
      d += POP16[(fa.desc[offA + 4] ^ fb.desc[offB + 4]) & 0xffff];
      d += POP16[((fa.desc[offA + 4] ^ fb.desc[offB + 4]) >>> 16) & 0xffff];
      d += POP16[(fa.desc[offA + 5] ^ fb.desc[offB + 5]) & 0xffff];
      d += POP16[((fa.desc[offA + 5] ^ fb.desc[offB + 5]) >>> 16) & 0xffff];
      d += POP16[(fa.desc[offA + 6] ^ fb.desc[offB + 6]) & 0xffff];
      d += POP16[((fa.desc[offA + 6] ^ fb.desc[offB + 6]) >>> 16) & 0xffff];
      d += POP16[(fa.desc[offA + 7] ^ fb.desc[offB + 7]) & 0xffff];
      d += POP16[((fa.desc[offA + 7] ^ fb.desc[offB + 7]) >>> 16) & 0xffff];

      if (d < best) {
        second = best;
        best = d;
        bestIdx = j;
      } else if (d < second) {
        second = d;
      }
    }
    if (bestIdx >= 0 && best <= maxDist && best < ratio * second) {
      bestA[i] = bestIdx;
      bestDistA[i] = best;
    }
  }

  if (!crossCheck) {
    const out = [];
    for (let i = 0; i < na; i++) {
      if (bestA[i] >= 0) out.push({ a: i, b: bestA[i], dist: bestDistA[i] });
    }
    return out;
  }

  const bestB = new Int32Array(nb).fill(-1);
  const bestDistB = new Int32Array(nb).fill(1e9);
  for (let j = 0; j < nb; j++) {
    const offB = j * 8;
    let best = 1e9, bestIdx = -1;
    for (let i = 0; i < na; i++) {
      const offA = i * 8;
      let d = 0;
      for (let k = 0; k < 8; k++) {
        const x = fa.desc[offA + k] ^ fb.desc[offB + k];
        d += POP16[x & 0xffff] + POP16[(x >>> 16) & 0xffff];
      }
      if (d < best) { best = d; bestIdx = i; }
    }
    if (bestIdx >= 0) { bestB[j] = bestIdx; bestDistB[j] = best; }
  }

  const out = [];
  for (let i = 0; i < na; i++) {
    const j = bestA[i];
    if (j >= 0 && bestB[j] === i && bestDistB[j] <= maxDist) {
      out.push({ a: i, b: j, dist: bestDistA[i] });
    }
  }
  return out;
}