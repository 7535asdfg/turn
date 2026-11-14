/*
 * SfM + MVS 数值测试（Node）
 *
 * 使用 tests/gen_synthetic.py 生成的 10 个已知位姿的合成视角，检查：
 *   1. SfM 注册的相机数量
 *   2. 相机中心 / 旋转（Umeyama 对齐后）与真值的误差
 *   3. 稀疏点到真实场景表面的距离
 *   4. MVS 稠密网格到真实表面的距离
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { reconstruct } from '../js/core/sfm.js';
import { reconstructDense } from '../js/core/mvs.js';
import {
  svd3, m3mul, m3t, m3det, m3mulv, cameraCenter, percentile, median,
} from '../js/core/mathx.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SYN = path.join(__dirname, 'synthetic');

const meta = JSON.parse(fs.readFileSync(path.join(SYN, 'meta.json'), 'utf8'));
const raw = fs.readFileSync(path.join(SYN, 'views.bin'));

const images = [];
for (let i = 0; i < meta.nViews; i++) {
  const start = i * meta.bytesPerView;
  const rgba = new Uint8Array(raw.buffer, raw.byteOffset + start, meta.bytesPerView);
  images.push({
    name: `view_${i}`,
    width: meta.width,
    height: meta.height,
    rgba,
    focal: meta.focal,
  });
}

/* ---------------- Umeyama 相似变换 ---------------- */

function umeyama(src, dst) {
  const n = src.length;
  const meanS = [0, 0, 0], meanD = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < 3; a++) { meanS[a] += src[i][a] / n; meanD[a] += dst[i][a] / n; }
  }
  const H = new Float64Array(9);
  let varS = 0;
  for (let i = 0; i < n; i++) {
    const s = [src[i][0] - meanS[0], src[i][1] - meanS[1], src[i][2] - meanS[2]];
    const d = [dst[i][0] - meanD[0], dst[i][1] - meanD[1], dst[i][2] - meanD[2]];
    varS += (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]) / n;
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) H[a * 3 + b] += s[a] * d[b] / n;
    }
  }
  const { U, S, V } = svd3(H);
  const Sfix = new Float64Array([1, 1, Math.sign(m3det(U) * m3det(V)) || 1]);
  const D = new Float64Array([Sfix[0] * S[0], Sfix[1] * S[1], Sfix[2] * S[2]]);
  // R = V * diag(Sfix) * U^T
  const Ufix = new Float64Array(U);
  if (Sfix[2] < 0) {
    for (let r = 0; r < 3; r++) Ufix[r * 3 + 2] = -Ufix[r * 3 + 2];
  }
  const R = m3mul(V, m3t(Ufix));
  const scale = (D[0] + D[1] + D[2]) / (varS || 1);
  const Rm = m3mulv(R, meanS);
  const t = [meanD[0] - scale * Rm[0], meanD[1] - scale * Rm[1], meanD[2] - scale * Rm[2]];
  return { R, scale, t };
}

function rotErrorDeg(Ra, Rb) {
  // Ra, Rb 均为世界->相机；比较同一世界下的旋转
  const rel = m3mul(Ra, m3t(Rb));
  const tr = rel[0] + rel[4] + rel[8];
  return (Math.acos(Math.max(-1, Math.min(1, (tr - 1) / 2))) * 180) / Math.PI;
}

function surfaceDistance(p, scene) {
  let best = Infinity;
  for (const sp of scene.spheres) {
    const d = Math.abs(Math.hypot(p[0] - sp.center[0], p[1] - sp.center[1], p[2] - sp.center[2]) - sp.r);
    if (d < best) best = d;
  }
  best = Math.min(best, Math.abs(p[1] - scene.groundY));
  best = Math.min(best, Math.abs(p[2] - scene.wallZ));
  return best;
}

function writePLY(file, mesh) {
  const { positions, colors, indices } = mesh;
  const nv = positions.length / 3;
  const nt = indices.length / 3;
  const lines = [
    'ply',
    'format ascii 1.0',
    `element vertex ${nv}`,
    'property float x',
    'property float y',
    'property float z',
    'property uchar red',
    'property uchar green',
    'property uchar blue',
    `element face ${nt}`,
    'property list uchar int vertex_indices',
    'end_header',
  ];
  for (let i = 0; i < nv; i++) {
    lines.push(`${positions[i * 3].toFixed(5)} ${positions[i * 3 + 1].toFixed(5)} ${positions[i * 3 + 2].toFixed(5)} ` +
      `${Math.round(colors[i * 3] * 255)} ${Math.round(colors[i * 3 + 1] * 255)} ${Math.round(colors[i * 3 + 2] * 255)}`);
  }
  for (let i = 0; i < indices.length; i += 3) {
    lines.push(`3 ${indices[i]} ${indices[i + 1]} ${indices[i + 2]}`);
  }
  fs.writeFileSync(file, lines.join('\n'));
}

async function main() {
  const t0 = performance.now();
  console.log('=== SfM ===');
  const sfm = await reconstruct(images, {
    nfeatures: 1200,
    matchWindow: 3,
    ring: true,
    refIters: 2,
  }, {
    onProgress: (e) => {
      if (e.fraction === undefined) return;
      process.stdout.write(`\r${e.stage.padEnd(9)} ${(e.fraction * 100).toFixed(0).padStart(3)}%  ${e.message.slice(0, 56).padEnd(57)}`);
    },
  });
  console.log('\n');

  console.log('SfM stats:', sfm.stats);
  console.log(`耗时 ${((performance.now() - t0) / 1000).toFixed(1)} s`);

  const byIndex = new Map(sfm.cameras.map((c) => [c.index, c]));
  const pairs = [];
  for (const [idx, cam] of byIndex) {
    const gt = meta.cameras[idx];
    pairs.push({ cam, gt });
  }

  const src = pairs.map((p) => Array.from(cameraCenter(p.cam.R, p.cam.t)));
  const dst = pairs.map((p) => p.gt.center);
  const align = umeyama(src, dst);
  const sceneScale = Math.max(
    ...meta.cameras.map((c) => Math.hypot(c.center[0], c.center[1], c.center[2]))
  );

  console.log(`\n对齐 scale=${align.scale.toFixed(4)}`);
  let sumErr = 0, maxErr = 0, sumRot = 0;
  for (const { cam, gt } of pairs) {
    const c = cameraCenter(cam.R, cam.t);
    const q = m3mulv(align.R, c);
    const aligned = [q[0] * align.scale + align.t[0], q[1] * align.scale + align.t[1], q[2] * align.scale + align.t[2]];
    const err = Math.hypot(aligned[0] - gt.center[0], aligned[1] - gt.center[1], aligned[2] - gt.center[2]);
    // 世界变换 X_gt = s R X_est + t  =>  R_cam_gt = R_cam_est * R^T
    const Rgt = m3mul(cam.R, m3t(align.R));
    const re = rotErrorDeg(Rgt, new Float64Array(gt.R.flat()));
    sumErr += err; maxErr = Math.max(maxErr, err); sumRot += re;
    console.log(`  cam ${String(cam.index).padStart(2)}  中心误差 ${err.toFixed(4)}  旋转误差 ${re.toFixed(2)}°`);
  }
  console.log(`相机中心误差: 平均 ${(sumErr / pairs.length).toFixed(4)} (${((sumErr / pairs.length / sceneScale) * 100).toFixed(2)}% 场景半径), 最大 ${maxErr.toFixed(4)}`);
  console.log(`相机旋转误差: 平均 ${(sumRot / pairs.length).toFixed(2)}°`);

  // 稀疏点表面误差
  const pdist = [];
  for (const pt of sfm.points) {
    const q = m3mulv(align.R, pt.X);
    const p = [q[0] * align.scale + align.t[0], q[1] * align.scale + align.t[1], q[2] * align.scale + align.t[2]];
    pdist.push(surfaceDistance(p, meta.scene));
  }
  pdist.sort((a, b) => a - b);
  console.log(`\n稀疏点 ${pdist.length} 个，到真实表面距离: 中位数 ${median(pdist).toFixed(4)}, p90 ${percentile(pdist, 0.9).toFixed(4)}, p99 ${percentile(pdist, 0.99).toFixed(4)}`);

  console.log('\n=== MVS ===');
  const t1 = performance.now();
  const dense = await reconstructDense(sfm, {
    quality: process.argv[2] || 'standard',
  }, {
    onProgress: (e) => {
      if (e.fraction === undefined) return;
      process.stdout.write(`\r${e.stage.padEnd(9)} ${(e.fraction * 100).toFixed(0).padStart(3)}%  ${e.message.slice(0, 56).padEnd(57)}`);
    },
  });
  console.log('\n');
  console.log('MVS stats:', dense.stats);
  console.log(`耗时 ${((performance.now() - t1) / 1000).toFixed(1)} s`);

  const { positions } = dense.mesh;
  const md = [];
  const nv = positions.length / 3;
  const step = Math.max(1, Math.floor(nv / 50000));
  for (let i = 0; i < nv; i += step) {
    const q = m3mulv(align.R, [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]]);
    const p = [q[0] * align.scale + align.t[0], q[1] * align.scale + align.t[1], q[2] * align.scale + align.t[2]];
    md.push(surfaceDistance(p, meta.scene));
  }
  md.sort((a, b) => a - b);
  console.log(`网格顶点 ${nv}，到真实表面距离: 中位数 ${median(md).toFixed(4)}, p90 ${percentile(md, 0.9).toFixed(4)}, p99 ${percentile(md, 0.99).toFixed(4)}`);

  writePLY(path.join(SYN, 'recon.ply'), dense.mesh);
  console.log(`已写出 ${path.join(SYN, 'recon.ply')}`);
}

main().catch((e) => {
  console.error('\n测试失败:', e);
  process.exit(1);
});