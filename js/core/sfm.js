/*
 * 增量式 Structure-from-Motion（SfM）
 *
 * 参考 COLMAP 的 incremental mapping 流程：
 *   1. ORB 特征提取
 *   2. 邻域/环形图像匹配 + 本质矩阵 RANSAC 几何验证
 *   3. 特征轨迹（track）并查集
 *   4. 选择最佳初始像对，恢复相对位姿并三角化
 *   5. 逐张图像 PnP 定位 -> 三角化新点 -> 局部/全局优化
 *
 * 不依赖任何 DOM / 第三方库，可在浏览器和 Node 中运行。
 */

import { extractOrb, matchDescriptors } from './orb.js';
import { rgbaToGray, focalFromFov } from './image.js';
import {
  estimateEssentialRansac, recoverPoseFromEssential, triangulateObservations,
  pnpRansac, refineCameraPose, refinePoint, projectPose,
} from './geometry.js';
import {
  identity, zeros, cameraCenter, median, m3orthonormalize,
  m3mul, m3mulv, m3t, solveLinear, averageRotations, dot, normalize,
} from './mathx.js';

const DEFAULTS = {
  nfeatures: 1200,
  levels: 4,
  matchWindow: 3,
  ring: true,
  exhaustive: false,
  minMatches: 22,
  minInliers: 16,
  minInlierRatio: 0.5,
  fovFallback: 55,
  reprojThreshold: 6,
  minTriangulationAngle: 1.2,
  pnpMinInliers: 8,
  pnpThreshold: 5.5,
  refIters: 2,
  seed: 0x1234abcd,
};

function createCameras(images, cfg) {
  return images.map((img, index) => {
    let f = Number(img.focal);
    if (!(f > 1) || !Number.isFinite(f)) {
      f = focalFromFov(img.width, img.fov || cfg.fovFallback);
    }
    return {
      index,
      name: img.name || `image_${index}`,
      width: img.width,
      height: img.height,
      f,
      cx: img.width / 2,
      cy: img.height / 2,
      R: identity(3),
      t: zeros(3),
      center: zeros(3),
      registered: false,
    };
  });
}

/* ---------------- 并查集 ---------------- */

function makeUnionFind(n) {
  const parent = new Int32Array(n).fill(-1);
  const rank = new Uint8Array(n);
  function find(x) {
    let r = x;
    while (parent[r] >= 0) r = parent[r];
    while (parent[x] >= 0) {
      const next = parent[x];
      parent[x] = r;
      x = next;
    }
    return r;
  }
  function union(a, b) {
    let ra = find(a), rb = find(b);
    if (ra === rb) return;
    if (rank[ra] < rank[rb]) { const t = ra; ra = rb; rb = t; }
    parent[rb] = ra;
    if (rank[ra] === rank[rb]) rank[ra]++;
  }
  return { find, union };
}

/**
 * @param {Array<{name,width,height,rgba,focal?,fov?}>} images
 * @param {object} opts
 * @param {{onProgress?:Function,yieldFn?:Function,shouldCancel?:Function}} hooks
 */
export async function reconstruct(images, opts = {}, hooks = {}) {
  const cfg = { ...DEFAULTS, ...opts };
  const progress = hooks.onProgress || (() => {});
  const yieldFn = hooks.yieldFn || (async () => {});
  const cancelled = hooks.shouldCancel || (() => false);
  const n = images.length;
  if (n < 2) throw new Error('多视角重建至少需要 2 张图片');

  const emit = (stage, message, fraction, extra) => {
    progress({ stage, message, fraction, ...extra });
  };
  const check = () => {
    if (cancelled()) throw new Error('已取消');
  };

  const cams = createCameras(images, cfg);
  const gray = images.map((img) => img.gray || rgbaToGray(img.rgba, img.width, img.height));

  /* ---------- 1. 特征提取 ---------- */
  emit('features', '提取 ORB 特征…', 0);
  const feats = [];
  for (let i = 0; i < n; i++) {
    check();
    emit('features', `提取特征 ${i + 1}/${n}`, 0.03 + 0.22 * (i / n), { image: i });
    feats.push(extractOrb(images[i].rgba, images[i].width, images[i].height, {
      gray: gray[i],
      nfeatures: cfg.nfeatures,
      levels: cfg.levels,
    }));
    await yieldFn();
  }

  /* ---------- 2. 特征匹配 + 两视图几何验证 ---------- */
  const pairs = [];
  if (cfg.exhaustive && n <= 32) {
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) pairs.push([i, j]);
  } else {
    for (let i = 0; i < n; i++) {
      for (let d = 1; d <= cfg.matchWindow; d++) {
        const j = i + d;
        if (j < n) pairs.push([i, j]);
      }
    }
    if (cfg.ring && n > 2) {
      for (let d = 1; d <= Math.min(cfg.matchWindow, n - 1); d++) {
        pairs.push([n - d, 0]);
        if (d > 1) pairs.push([n - d, 1]);
      }
    }
  }

  // 去重
  const seen = new Set();
  const uniquePairs = [];
  for (const [i, j] of pairs) {
    if (i === j) continue;
    const key = i < j ? `${i}_${j}` : `${j}_${i}`;
    if (seen.has(key)) continue;
    seen.add(key);
    uniquePairs.push(i < j ? [i, j] : [j, i]);
  }

  emit('matching', `匹配 ${uniquePairs.length} 个图像对…`, 0.25);
  const verifiedPairs = [];
  let pairDone = 0;
  for (const [i, j] of uniquePairs) {
    check();
    const matches = matchDescriptors(feats[i], feats[j], {
      ratio: 0.8,
      maxDist: 88,
      crossCheck: true,
    });
    if (matches.length >= cfg.minMatches) {
      const baseSeed = cfg.seed + i * 7919 + j * 104729;
      const eopts = {
        threshold: 4,
        sampleSize: 12,
        maxIters: 1400,
      };
      const g1 = estimateEssentialRansac(cams[i], cams[j], feats[i], feats[j], matches, {
        ...eopts, seed: baseSeed,
      });
      const g2 = estimateEssentialRansac(cams[i], cams[j], feats[i], feats[j], matches, {
        ...eopts, seed: baseSeed + 1,
      });
      const geo = !g1 ? g2 : !g2 ? g1 : (g1.inlierCount >= g2.inlierCount ? g1 : g2);
      if (geo && geo.inlierCount >= cfg.minInliers && geo.inlierCount >= matches.length * cfg.minInlierRatio) {
        const inlierMatches = geo.inlierIdx.map((idx) => matches[idx]);
        let relPose = null;
        try {
          const rel = recoverPoseFromEssential(
            geo.E, cams[i], cams[j], feats[i], feats[j], matches, geo.inlierIdx
          );
          if (rel) relPose = { R: rel.R, t: rel.t, goodCount: rel.goodCount };
        } catch {
          /* 忽略，仅作为后续定位的候选初始化 */
        }
        verifiedPairs.push({
          i, j,
          matches: inlierMatches,
          inlierIdx: geo.inlierIdx,
          rawCount: matches.length,
          inlierCount: geo.inlierCount,
          E: geo.E,
          relPose,
        });
      }
    }
    pairDone++;
    emit('matching',
      `匹配图像对 ${pairDone}/${uniquePairs.length} · 已确认 ${verifiedPairs.length} 对`,
      0.25 + 0.25 * (pairDone / uniquePairs.length));
    await yieldFn();
  }

  if (!verifiedPairs.length) {
    throw new Error(
      '没有找到可靠的图像对应关系。请确认多张照片拍摄的是同一物体、相邻照片有 60% 以上重叠，且光线充足。'
    );
  }

  verifiedPairs.sort((a, b) => b.inlierCount - a.inlierCount);

  /* ---------- 3. 构建特征轨迹 ---------- */
  emit('matching', '构建特征轨迹…', 0.5);
  let totalFeatures = 0;
  const imgOffsets = new Array(n);
  for (let i = 0; i < n; i++) {
    imgOffsets[i] = totalFeatures;
    totalFeatures += feats[i].count;
  }
  if (totalFeatures > 4_000_000) {
    throw new Error('特征点过多，请减少图片数量后重试');
  }

  const uf = makeUnionFind(totalFeatures);
  for (const pair of verifiedPairs) {
    const oi = imgOffsets[pair.i], oj = imgOffsets[pair.j];
    for (const m of pair.matches) {
      uf.union(oi + m.a, oj + m.b);
    }
  }

  const rootToTrack = new Map();
  const tracks = [];

  for (let img = 0; img < n; img++) {
    const off = imgOffsets[img];
    for (let f = 0; f < feats[img].count; f++) {
      const root = uf.find(off + f);
      let tr = rootToTrack.get(root);
      if (tr === undefined) {
        tr = tracks.length;
        rootToTrack.set(root, tr);
        tracks.push({ obs: [], pointId: -1, valid: true });
      }
      const track = tracks[tr];
      // 允许同一张图有多个候选观测：伪匹配会把不同特征并进同一条轨迹，
      // 后续三角化/定位时用几何一致性挑选正确的那一个。
      track.obs.push({ img, feat: f });
    }
  }

  // 只保留观测覆盖 >=2 张不同图像的轨迹
  const validTracks = [];
  for (const track of tracks) {
    const imgs = new Set(track.obs.map((o) => o.img));
    if (imgs.size >= 2) {
      track.id = validTracks.length;
      track.images = imgs;
      validTracks.push(track);
    } else {
      track.valid = false;
    }
  }
  tracks.length = 0;
  tracks.push(...validTracks);

  // 每张图包含哪些轨迹（每条轨迹每张图只登记一次）
  const imageTracks = Array.from({ length: n }, () => []);
  for (let t = 0; t < tracks.length; t++) {
    const tr = tracks[t];
    for (const img of tr.images) imageTracks[img].push(t);
  }

  // 同一张图内的候选观测按图像分组
  const groupObs = (track, registeredOnly = true) => {
    const map = new Map();
    for (const o of track.obs) {
      if (registeredOnly && !cams[o.img].registered) continue;
      let arr = map.get(o.img);
      if (!arr) { arr = []; map.set(o.img, arr); }
      arr.push(o);
    }
    return [...map.entries()].map(([img, arr]) => ({ img, arr }));
  };

  const makeObs = (o) => {
    const feat = feats[o.img];
    return {
      cam: cams[o.img],
      img: o.img,
      feat: o.feat,
      x: feat.x[o.feat],
      y: feat.y[o.feat],
    };
  };

  /**
   * 每张图挑一个观测；若给定 X，则选重投影误差最小的那个。
   */
  const selectObs = (track, registeredOnly = true, X = null) => {
    const groups = groupObs(track, registeredOnly);
    const out = [];
    for (const { img, arr } of groups) {
      let best = arr[0];
      if (X) {
        let bestErr = Infinity;
        for (const o of arr) {
          const p = projectPose(cams[img], X);
          if (!p.ok) continue;
          const e = Math.hypot(p.u - feats[img].x[o.feat], p.v - feats[img].y[o.feat]);
          if (e < bestErr) { bestErr = e; best = o; }
        }
      }
      out.push(makeObs(best));
    }
    return out;
  };

  const trackObs = (track, registeredOnly = true, X = null) =>
    selectObs(track, registeredOnly, X);

  /**
   * 对一条轨迹做一致性三角化：先尝试各种观测组合，
   * 再用选出的三维点重新挑每张图最吻合的观测并精化。
   */
  const tryTriangulateTrack = (track) => {
    const groups = groupObs(track, true);
    if (groups.length < 2) return null;

    let best = null;
    const choice = new Array(groups.length).fill(0);

    const evaluate = () => {
      const obs = groups.map((g, gi) => makeObs(g.arr[choice[gi]]));
      const tri = triangulateObservations(obs, { maxError: cfg.reprojThreshold });
      if (!tri) return;
      let maxErr = 0;
      for (const o of obs) {
        const p = projectPose(o.cam, tri.X);
        if (!p.ok || p.z <= 0) return;
        maxErr = Math.max(maxErr, Math.hypot(p.u - o.x, p.v - o.y));
      }
      if (!best || maxErr < best.maxErr) best = { tri, maxErr };
    };

    evaluate();
    // 每个有多候选的图，逐一替换后尝试
    let tries = 0;
    for (let gi = 0; gi < groups.length && tries < 20; gi++) {
      if (groups[gi].arr.length < 2) continue;
      const saved = choice[gi];
      for (let ai = 1; ai < groups[gi].arr.length && tries < 20; ai++) {
        choice[gi] = ai;
        evaluate();
        tries++;
      }
      choice[gi] = saved;
    }

    if (!best) return null;
    if (best.tri.angle < cfg.minTriangulationAngle) return null;

    // 用三角化结果重新挑观测并精化
    const obs = selectObs(track, true, best.tri.X);
    const tri2 = triangulateObservations(obs, { maxError: cfg.reprojThreshold });
    if (tri2 && tri2.angle >= cfg.minTriangulationAngle) {
      for (const o of obs) {
        const p = projectPose(o.cam, tri2.X);
        if (!p.ok || p.z <= 0) return null;
      }
      return { tri: tri2, obs };
    }
    return null;
  };

  const sampleColor = (img, featIdx) => {
    const im = images[img];
    const x = Math.max(0, Math.min(im.width - 1, Math.round(feats[img].x[featIdx])));
    const y = Math.max(0, Math.min(im.height - 1, Math.round(feats[img].y[featIdx])));
    const p = (y * im.width + x) * 4;
    return [im.rgba[p], im.rgba[p + 1], im.rgba[p + 2]];
  };

  /* ---------- 4. 初始像对 ---------- */
  emit('mapping', '选择初始像对…', 0.5);
  let init = null;
  let initTrackPoints = 0;

  for (let pi = 0; pi < Math.min(verifiedPairs.length, 14); pi++) {
    const pair = verifiedPairs[pi];
    const camA = cams[pair.i];
    const camB = cams[pair.j];

    // 重置临时相机
    camA.R = identity(3); camA.t = zeros(3);
    camB.R = identity(3); camB.t = zeros(3);
    const pose = recoverPoseFromEssential(
      pair.E, camA, camB, feats[pair.i], feats[pair.j], pair.matches,
      pair.inlierIdx.map((_, k) => k)
    );
    if (!pose) continue;

    // 临时注册
    camA.registered = true; camB.registered = true;
    camB.R = pose.R; camB.t = pose.t;

    let good = 0;
    for (let t = 0; t < tracks.length; t++) {
      const tr = tracks[t];
      let hasA = false, hasB = false;
      for (const o of tr.obs) {
        if (o.img === pair.i) hasA = true;
        else if (o.img === pair.j) hasB = true;
      }
      if (!hasA || !hasB) continue;
      const res = tryTriangulateTrack(tr);
      const tri = res && res.tri;
      if (tri && tri.X[2] > 0) {
        good++;
      }
    }

    camA.registered = false; camB.registered = false;

    const score = good;
    if (good >= 28 && (!init || score > initTrackPoints)) {
      init = { pair, pose };
      initTrackPoints = good;
    }
    if (good >= 120) break;
  }

  if (!init) {
    throw new Error(
      '无法从图像中恢复相机位姿（初始像对三角化失败）。请让相邻照片有更大重叠、避免纯平面 / 纯白物体，并尽量保持相机焦距一致。'
    );
  }

  const { pair: initPair, pose: initPose } = init;
  const camA = cams[initPair.i];
  const camB = cams[initPair.j];
  camA.R = identity(3); camA.t = zeros(3);
  camB.R = Float64Array.from(initPose.R);
  camB.t = Float64Array.from(initPose.t);
  camA.registered = true;
  camB.registered = true;
  camA.center = cameraCenter(camA.R, camA.t);
  camB.center = cameraCenter(camB.R, camB.t);

  const registered = new Set([initPair.i, initPair.j]);
  const refCameraIndex = initPair.i;

  const updateCamCenter = (cam) => {
    cam.center = cameraCenter(cam.R, cam.t);
  };

  // 初始三角化
  const points = []; // { X, color, trackId, valid }
  const triangulateTrack = (tr) => {
    const res = tryTriangulateTrack(tr);
    if (!res) return;
    const { tri, obs } = res;
    let pid = tr.pointId;
    if (pid >= 0 && points[pid]) {
      points[pid].X = tri.X;
      points[pid].valid = true;
    } else {
      pid = points.length;
      points.push({
        X: tri.X,
        color: sampleColor(obs[0].img, obs[0].feat),
        trackId: tr.id,
        valid: true,
      });
      tr.pointId = pid;
    }
  };

  for (const tr of tracks) triangulateTrack(tr);
  await yieldFn();

  /* ---------- 5. 增量式定位 ---------- */
  const getCorrespondences = (img) => {
    const corr = [];
    for (const ti of imageTracks[img]) {
      const tr = tracks[ti];
      if (tr.pointId < 0) continue;
      const pt = points[tr.pointId];
      if (!pt || !pt.valid) continue;
      // 同一轨迹可能在同一张图里有多个候选特征，全部交给 RANSAC 选择
      for (const o of tr.obs) {
        if (o.img !== img) continue;
        corr.push({
          X: pt.X,
          x: feats[img].x[o.feat],
          y: feats[img].y[o.feat],
        });
      }
    }
    return corr;
  };

  // 图像 -> 已验证像对
  const pairsByImage = Array.from({ length: n }, () => []);
  for (const pair of verifiedPairs) {
    pairsByImage[pair.i].push(pair);
    pairsByImage[pair.j].push(pair);
  }

  const invertRel = (rel) => {
    const R = m3t(rel.R);
    const t = m3mulv(R, [-rel.t[0], -rel.t[1], -rel.t[2]]);
    return { R, t };
  };

  // 与未注册图像相连、质量最高的已验证像对（用于校验候选位姿）
  const strongestPairFor = (img) => {
    let best = null;
    for (const pair of pairsByImage[img]) {
      const reg = pair.i === img ? pair.j : pair.i;
      if (!registered.has(reg) || !pair.relPose) continue;
      if (!best || pair.inlierCount > best.pair.inlierCount) {
        best = { pair, reg, rel: pair.i === img ? invertRel(pair.relPose) : pair.relPose };
      }
    }
    return best;
  };

  const rotationAngleBetween = (Ra, Rb) => {
    const rel = m3mul(Ra, m3t(Rb));
    const tr = rel[0] + rel[4] + rel[8];
    return (Math.acos(Math.max(-1, Math.min(1, (tr - 1) / 2))) * 180) / Math.PI;
  };

  // 候选位姿必须与最强两视图像对的旋转/基线方向一致（防止平面镜面歧义）
  const candidateConsistent = (img, cand) => {
    const strong = strongestPairFor(img);
    if (!strong) return true;
    const Rreg = cams[strong.reg].R;
    const Rexp = m3mul(strong.rel.R, Rreg);
    const rotErr = rotationAngleBetween(cand.R, Rexp);

    // 相机中心应位于该像对给出的射线上（正方向）
    const C = cameraCenter(cand.R, cand.t);
    const Creg = cams[strong.reg].center;
    const cImgReg = m3mulv(m3t(strong.rel.R), [-strong.rel.t[0], -strong.rel.t[1], -strong.rel.t[2]]);
    const d = normalize(m3mulv(m3t(Rreg), cImgReg));
    const v = normalize([C[0] - Creg[0], C[1] - Creg[1], C[2] - Creg[2]]);
    const cos = dot(v, d);
    if (cfg.debug) console.log('[check] img', img, cand.kind, 'strongReg', strong.reg, 'rotErr', rotErr.toFixed(1), 'cos', cos.toFixed(2));
    return rotErr <= 12 && cos > 0.35;
  };

  // 当前已注册相机的典型基线长度（用于搜索尺度）
  const baselineScale = () => {
    const cs = [...registered].map((i) => cams[i].center);
    const vals = [];
    for (let a = 0; a < cs.length; a++) {
      for (let b = a + 1; b < cs.length; b++) {
        vals.push(Math.hypot(
          cs[a][0] - cs[b][0], cs[a][1] - cs[b][1], cs[a][2] - cs[b][2]
        ));
      }
    }
    return vals.length ? (median(vals) || 1) : 1;
  };

  // 相邻相机的基线长度（最近邻距离的中位数），用于尺度先验
  const nearestNeighborBaseline = () => {
    const cs = [...registered].map((i) => cams[i].center);
    if (cs.length < 2) return 1;
    const nns = [];
    for (let a = 0; a < cs.length; a++) {
      let dmin = Infinity;
      for (let b = 0; b < cs.length; b++) {
        if (a === b) continue;
        const d = Math.hypot(
          cs[a][0] - cs[b][0], cs[a][1] - cs[b][1], cs[a][2] - cs[b][2]
        );
        if (d < dmin) dmin = d;
      }
      if (Number.isFinite(dmin)) nns.push(dmin);
    }
    return median(nns) || 1;
  };

  const poseStats = (img, R, t, corr, threshold = 6) => {
    const c = cams[img];
    let count = 0;
    let sum = 0;
    const inliers = [];
    for (const q of corr) {
      const p = projectPose({ f: c.f, cx: c.cx, cy: c.cy, R, t }, q.X);
      if (!p.ok) continue;
      const e = Math.hypot(p.u - q.x, p.v - q.y);
      if (e <= threshold) { count++; sum += e; inliers.push(q); }
    }
    return { count, mean: count ? sum / count : Infinity, inliers };
  };

  let mapRound = 0;

  const registerCamera = async (img, R, t, inlierObs = []) => {
    const cam = cams[img];
    cam.R = R;
    cam.t = t;
    if (inlierObs.length >= 4) refineCameraPose(cam, inlierObs, 5);
    cam.R = m3orthonormalize(cam.R);
    updateCamCenter(cam);
    cam.registered = true;
    registered.add(img);

    // 用新相机重新三角化相关轨迹（包括已有轨迹，加入新观测后会更准确）
    for (const ti of imageTracks[img]) {
      triangulateTrack(tracks[ti]);
    }

    mapRound++;
    emit('mapping',
      `相机定位 ${registered.size}/${n} · 稀疏点 ${points.filter((p) => p.valid).length}`,
      0.5 + 0.35 * (registered.size / n));
    if (mapRound % 2 === 0) await yieldFn();
  };

  /**
   * 兜底定位：当新相机几乎没有可用的 2D-3D 点时，用已验证像对的
   * 相对位姿 + 相邻基线先验直接给出位姿，再用该像对的二维匹配做手性校验。
   * 这一步保证顺序拍摄的环形数据集不会因为中间某张图重叠不足而中断。
   */
  const fallbackRegister = () => {
    const baseline = Math.max(1e-3, nearestNeighborBaseline());
    let best = null;

    for (let img = 0; img < n; img++) {
      if (registered.has(img)) continue;
      for (const pair of pairsByImage[img]) {
        const reg = pair.i === img ? pair.j : pair.i;
        if (!registered.has(reg) || !pair.relPose || pair.relPose.goodCount < 12) continue;
        let rel = pair.relPose;
        if (pair.i === img) rel = invertRel(rel);

        const Rk = m3mul(rel.R, cams[reg].R);
        const base = m3mulv(rel.R, cams[reg].t);

        const mkCam = (which, tNew) => {
          const c = cams[which];
          const useNew = which === img;
          return {
            ...c,
            R: useNew ? Rk : c.R,
            t: useNew ? tNew : c.t,
          };
        };

        const step = Math.max(1, Math.floor(pair.matches.length / 70));
        for (const factor of [0.75, 0.9, 1.0, 1.1, 1.25, 1.45]) {
          const lam = baseline * factor;
          const t = [
            base[0] + lam * rel.t[0],
            base[1] + lam * rel.t[1],
            base[2] + lam * rel.t[2],
          ];
          const camI = mkCam(pair.i, t);
          const camJ = mkCam(pair.j, t);
          let good = 0;
          let total = 0;
          for (let mi = 0; mi < pair.matches.length; mi += step) {
            const mm = pair.matches[mi];
            total++;
            const tri = triangulateObservations([
              { cam: camI, x: feats[pair.i].x[mm.a], y: feats[pair.i].y[mm.a] },
              { cam: camJ, x: feats[pair.j].x[mm.b], y: feats[pair.j].y[mm.b] },
            ], { maxError: 8 });
            if (tri) good++;
          }
          if (!(total > 0)) continue;
          const cand = {
            kind: 'fallback', R: Rk, t,
            count: good, mean: 0,
            inliers: [],
          };
          if (good < Math.max(8, total * 0.35)) continue;
          if (!candidateConsistent(img, cand)) continue;
          if (!best || good > best.count ||
              (good === best.count && Math.abs(lam - baseline) < Math.abs(best.lam - baseline))) {
            best = { img, R: Rk, t, count: good, total, lam };
          }
        }
      }
    }
    return best;
  };

  while (registered.size < n) {
    check();
    const baseLen = Math.max(1e-3, baselineScale());
    let best = null;

    for (let img = 0; img < n; img++) {
      if (registered.has(img)) continue;
      const corr = getCorrespondences(img);
      if (corr.length < 6) continue;

      const candidates = [];

      // 候选 A：经典 PnP RANSAC
      const pnpCam = { ...cams[img] };
      const pnp = pnpRansac(pnpCam, corr, {
        threshold: cfg.pnpThreshold,
        maxIters: 900,
        seed: cfg.seed + img * 7717 + mapRound * 131,
      });
      if (pnp && Number.isFinite(pnp.meanError) &&
          pnp.inlierCount >= Math.max(8, corr.length * 0.25)) {
        const st = poseStats(img, pnp.R, pnp.t, corr, Math.max(4, cfg.pnpThreshold));
        candidates.push({
          kind: 'pnp',
          R: pnp.R, t: pnp.t,
          count: st.count, mean: st.mean,
          inliers: st.inliers.length >= 6 ? st.inliers : pnp.inliers,
        });
      }

      // 候选 B：与已注册相机的两视图相对位姿 + 基线尺度搜索
      // （仅在 2D-3D 点较少、PnP 容易退化时提供强约束）
      const rayCandidates = [];
      for (const pair of pairsByImage[img]) {
        const regIndex = pair.i === img ? pair.j : pair.i;
        if (!registered.has(regIndex)) continue;
        const relPose = pair.relPose;
        if (!relPose || relPose.goodCount < 12) continue;

        let rel = relPose;
        if (pair.i === img) rel = invertRel(relPose);

        const Rk = m3mul(rel.R, cams[regIndex].R);
        const base = m3mulv(rel.R, cams[regIndex].t);
        // 该像对给出的世界空间“相机中心射线”：C_img = C_reg + λ·d
        const cImgReg = m3mulv(m3t(rel.R), [-rel.t[0], -rel.t[1], -rel.t[2]]);
        const dWorld = normalize(m3mulv(m3t(cams[regIndex].R), cImgReg));
        rayCandidates.push({
          regIndex, Rk, d: dWorld,
          origin: cams[regIndex].center,
          weight: relPose.goodCount,
        });

        let bestLam = null;
        const evalAt = (lam) => {
          if (!Number.isFinite(lam) || lam <= 1e-6) return;
          const t = [
            base[0] + lam * rel.t[0],
            base[1] + lam * rel.t[1],
            base[2] + lam * rel.t[2],
          ];
          const st = poseStats(img, Rk, t, corr, 5);
          const score = st.count * 4 - st.mean;
          if (!bestLam || score > bestLam.score) {
            bestLam = { lam, score, R: Rk, t, count: st.count, mean: st.mean, inliers: st.inliers };
          }
        };

        // 由每条 2D-3D 对应解析求基线尺度 λ：
        //   x = f (a_x + λ b_x) / (a_z + λ b_z) + cx
        // 这些估计应聚集在真实尺度附近，用投票 + 中值得到可靠初值。
        const cImg = cams[img];
        const lamCands = [];
        for (const q of corr) {
          const a = [
            Rk[0] * q.X[0] + Rk[1] * q.X[1] + Rk[2] * q.X[2] + base[0],
            Rk[3] * q.X[0] + Rk[4] * q.X[1] + Rk[5] * q.X[2] + base[1],
            Rk[6] * q.X[0] + Rk[7] * q.X[1] + Rk[8] * q.X[2] + base[2],
          ];
          const rx = q.x - cImg.cx;
          const ry = q.y - cImg.cy;
          const denU = rx * rel.t[2] - cImg.f * rel.t[0];
          const denV = ry * rel.t[2] - cImg.f * rel.t[1];
          if (Math.abs(denU) > 1e-9) {
            const lam = (cImg.f * a[0] - rx * a[2]) / denU;
            if (Number.isFinite(lam) && Math.abs(lam) < 12 * baseLen) lamCands.push(lam);
          }
          if (Math.abs(denV) > 1e-9) {
            const lam = (cImg.f * a[1] - ry * a[2]) / denV;
            if (Number.isFinite(lam) && Math.abs(lam) < 12 * baseLen) lamCands.push(lam);
          }
        }

        if (lamCands.length >= 3) {
          lamCands.sort((x, y) => x - y);
          const tol = Math.max(1e-3, 0.05 * baseLen);
          let bestLo = 0, bestHi = 0, bestN = 0;
          for (let s = 0; s < lamCands.length; s++) {
            let hi = s;
            while (hi + 1 < lamCands.length && lamCands[hi + 1] - lamCands[s] <= tol) hi++;
            if (hi - s + 1 > bestN) { bestN = hi - s + 1; bestLo = s; bestHi = hi; }
          }
          const hint = lamCands[(bestLo + bestHi) >> 1];
          if (cfg.debug) console.log('[lam] img', img, 'reg', regIndex, 'baseLen', baseLen.toFixed(3), 'nLam', lamCands.length, 'cluster', bestN, 'hint', hint.toFixed(3));
          evalAt(hint);
          const d = Math.max(1e-3, 0.02 * baseLen);
          for (let s = -10; s <= 10; s++) evalAt(hint + (d * s) / 10);
        }

        // 粗扫描兜底
        for (let s = 0.02 * baseLen, n = 0; n <= 90; s += (3.4 * baseLen) / 90, n++) evalAt(s);
        if (bestLam) {
          const d = Math.max(1e-3, 0.01 * baseLen);
          for (let s = -8; s <= 8; s++) evalAt(bestLam.lam + (d * s) / 8);
        }

        if (cfg.debug && bestLam) console.log('[lam-best] img', img, 'reg', regIndex, 'lam', bestLam.lam.toFixed(3), 'count', bestLam.count, 'mean', bestLam.mean.toFixed(2));
        if (bestLam && bestLam.count >= Math.max(5, corr.length * 0.35) &&
            bestLam.mean < 5.5) {
          candidates.push({
            kind: 'rel',
            R: bestLam.R, t: bestLam.t,
            count: bestLam.count, mean: bestLam.mean,
            inliers: bestLam.inliers,
          });
        }
      }

      // 候选 C：多条“相机中心射线”交会。两视图相对位姿只给方向，
      // 多个已注册邻居的射线交会可直接确定新相机的世界坐标与基线尺度。
      if (rayCandidates.length >= 2) {
        const A = new Float64Array(9);
        const bv = new Float64Array(3);
        for (const rc of rayCandidates) {
          const d = rc.d;
          const P = new Float64Array([
            1 - d[0] * d[0], -d[0] * d[1], -d[0] * d[2],
            -d[1] * d[0], 1 - d[1] * d[1], -d[1] * d[2],
            -d[2] * d[0], -d[2] * d[1], 1 - d[2] * d[2],
          ]);
          for (let a = 0; a < 9; a++) A[a] += P[a];
          const po = [
            P[0] * rc.origin[0] + P[1] * rc.origin[1] + P[2] * rc.origin[2],
            P[3] * rc.origin[0] + P[4] * rc.origin[1] + P[5] * rc.origin[2],
            P[6] * rc.origin[0] + P[7] * rc.origin[1] + P[8] * rc.origin[2],
          ];
          bv[0] += po[0]; bv[1] += po[1]; bv[2] += po[2];
        }
        const C = solveLinear(A, bv, 3);
        if (C) {
          let positive = 0;
          for (const rc of rayCandidates) {
            const lam = (C[0] - rc.origin[0]) * rc.d[0] +
              (C[1] - rc.origin[1]) * rc.d[1] +
              (C[2] - rc.origin[2]) * rc.d[2];
            if (lam > 0) positive++;
          }
          if (positive >= Math.ceil(rayCandidates.length * 0.7)) {
            const Rk = averageRotations(rayCandidates.map((rc) => rc.Rk));
            const t = [
              -(Rk[0] * C[0] + Rk[1] * C[1] + Rk[2] * C[2]),
              -(Rk[3] * C[0] + Rk[4] * C[1] + Rk[5] * C[2]),
              -(Rk[6] * C[0] + Rk[7] * C[1] + Rk[8] * C[2]),
            ];
            const st = poseStats(img, Rk, t, corr, 6);
            if (st.count >= Math.max(6, corr.length * 0.25)) {
              candidates.push({
                kind: 'rays',
                R: Rk, t,
                count: st.count, mean: st.mean,
                inliers: st.inliers,
              });
            }
            if (cfg.debug) {
              console.log('[rays] img', img, 'n', rayCandidates.length,
                'C', C.map((v) => v.toFixed(3)), 'count', st.count,
                'mean', st.mean.toFixed(2));
            }
          }
        }
      }

      if (!candidates.length) continue;
      candidates.sort((a, b) => {
        if (b.count !== a.count) return b.count - a.count;
        return a.mean - b.mean;
      });
      const cand = candidates.find((c) => candidateConsistent(img, c));
      if (!cand) {
        if (cfg.debug) console.log('[reject] img', img, 'all candidates inconsistent with strongest pair');
        continue;
      }
      if (cfg.debug) {
        console.log('[cand] img', img, 'corr', corr.length, cand.kind,
          'inliers', cand.count, 'mean', cand.mean.toFixed(2));
      }
      if (!best || cand.count > best.cand.count ||
          (cand.count === best.cand.count && cand.mean < best.cand.mean)) {
        best = { img, cand };
      }
    }

    if (!best) {
      const fb = fallbackRegister();
      if (!fb) break;
      if (cfg.debug) {
        console.log('[fallback] img', fb.img, 'lam', fb.lam.toFixed(2),
          'cheirality', fb.count, '/', fb.total);
      }
      await registerCamera(fb.img, fb.R, fb.t, []);
      continue;
    }

    const testCam = { ...cams[best.img], R: best.cand.R, t: best.cand.t };
    refineCameraPose(testCam, best.cand.inliers, 5);
    const finalStats = poseStats(best.img, testCam.R, testCam.t, getCorrespondences(best.img), 5);
    if (finalStats.count < Math.max(5, best.cand.count * 0.6)) {
      // 位姿不可信，放弃本轮（避免把错误相机写入模型）
      break;
    }

    await registerCamera(best.img, testCam.R, testCam.t, []);
  }

  if (registered.size < 2) {
    throw new Error('相机注册数量不足，无法完成重建');
  }

  /* ---------- 6. 优化与清理 ---------- */
  const pointObs = (pt, registeredOnly = true) => {
    const tr = tracks[pt.trackId];
    return trackObs(tr, registeredOnly, pt.X);
  };

  const prunePoints = (maxError = 9) => {
    let removed = 0;
    for (const pt of points) {
      if (!pt.valid) continue;
      const obs = pointObs(pt);
      if (obs.length < 2) { pt.valid = false; removed++; continue; }
      let bad = false;
      for (const o of obs) {
        const p = projectPose(o.cam, pt.X);
        if (!p.ok || p.z <= 0 || Math.hypot(p.u - o.x, p.v - o.y) > maxError) {
          bad = true; break;
        }
      }
      if (bad) { pt.valid = false; removed++; }
    }
    return removed;
  };

  const refineAll = async (iterations = cfg.refIters) => {
    for (let it = 0; it < iterations; it++) {
      check();
      emit('mapping', `光束法平差（位姿+点）${it + 1}/${iterations}`, 0.87 + 0.1 * (it / iterations));

      // 相机位姿
      for (const img of registered) {
        if (img === refCameraIndex) continue;
        const cam = cams[img];
        const corr = [];
        for (const pt of points) {
          if (!pt.valid) continue;
          const o = pointObs(pt).find((q) => q.img === img);
          if (o) {
            corr.push({ X: pt.X, x: o.x, y: o.y });
          }
        }
        if (corr.length >= 4) {
          refineCameraPose(cam, corr, 3);
          updateCamCenter(cam);
          cam.R = m3orthonormalize(cam.R);
        }
      }

      // 三维点
      for (const pt of points) {
        if (!pt.valid) continue;
        const obs = pointObs(pt);
        if (obs.length < 2) continue;
        pt.X = refinePoint(obs, pt.X, 3);
      }

      prunePoints(9);
      await yieldFn();
      if ((it + 1) % 2 === 0) await yieldFn();
    }
  };

  /**
   * 剔除与整体模型不一致的相机：
   * 观测点太少（说明只与本段少量点自洽）或重投影误差明显偏大。
   */
  const pruneCameras = () => {
    const stats = [];
    for (const img of registered) {
      if (img === refCameraIndex) continue;
      const cam = cams[img];
      let n = 0;
      const errs = [];
      for (const pt of points) {
        if (!pt.valid) continue;
        const o = pointObs(pt).find((q) => q.img === img);
        if (!o) continue;
        const p = projectPose(cam, pt.X);
        if (!p.ok || p.z <= 0) continue;
        n++;
        errs.push(Math.hypot(p.u - o.x, p.v - o.y));
      }
      stats.push({ img, n, med: errs.length ? median(errs) : Infinity });
    }
    if (!stats.length) return [];
    const medObs = median(stats.map((s) => s.n));
    const medErr = median(stats.filter((s) => Number.isFinite(s.med)).map((s) => s.med));
    const minObs = Math.max(12, medObs * 0.3);
    const maxErr = Math.max(1.5, medErr * 2.5);
    const removed = [];
    for (const s of stats) {
      if (s.n < minObs || (Number.isFinite(s.med) && s.med > maxErr)) {
        cams[s.img].registered = false;
        registered.delete(s.img);
        removed.push(s.img);
      }
    }
    return removed;
  };

  await refineAll(cfg.refIters);
  prunePoints(12);
  const removedCams = pruneCameras();
  if (removedCams.length) {
    prunePoints(10);
    await refineAll(1);
    prunePoints(10);
    pruneCameras();
  }
  if (registered.size < 2) {
    throw new Error('相机一致性检查后有效相机不足 2 个。请增加照片重叠度或减少模糊/重复纹理照片。');
  }

  // 收集最终结果
  const validPoints = [];
  for (const pt of points) {
    if (!pt.valid) continue;
    const obs = pointObs(pt);
    if (obs.length < 2) continue;
    validPoints.push({
      X: pt.X,
      color: pt.color,
      trackId: pt.trackId,
      obs: obs.map((o) => ({
        cam: o.cam,
        img: o.img,
        feat: o.feat,
        x: o.x,
        y: o.y,
      })),
    });
  }

  const registeredCams = [];
  for (let i = 0; i < n; i++) if (cams[i].registered) registeredCams.push(cams[i]);

  // 稀疏点重投影误差
  let errSum = 0, errCount = 0;
  for (const pt of validPoints) {
    for (const o of pt.obs) {
      const p = projectPose(o.cam, pt.X);
      if (p.ok) { errSum += Math.hypot(p.u - o.x, p.v - o.y); errCount++; }
    }
  }

  const stats = {
    images: n,
    registeredCameras: registeredCams.length,
    trackedFeatures: feats.reduce((s, f) => s + f.count, 0),
    verifiedPairs: verifiedPairs.length,
    totalPairs: uniquePairs.length,
    initialPair: [initPair.i, initPair.j],
    tracks: tracks.length,
    sparsePoints: validPoints.length,
    meanReprojectionError: errCount ? errSum / errCount : 0,
  };

  emit('mapping', `SfM 完成 · ${stats.registeredCameras} 个相机 / ${stats.sparsePoints} 个稀疏点`, 1, { stats });

  return {
    cameras: registeredCams,
    allCameras: cams,
    points: validPoints,
    tracks,
    pairs: verifiedPairs,
    stats,
    images,
  };
}