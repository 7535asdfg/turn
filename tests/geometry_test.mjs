/*
 * 几何核心单元测试：PnP + 多视图三角化
 * 运行：node tests/geometry_test.mjs
 */
import { pnpRansac, triangulateObservations, projectPose } from '../js/core/geometry.js';
import { identity, rodrigues, m3mul, m3t, m3mulv, mulberry32 } from '../js/core/mathx.js';

let failed = false;
const check = (cond, msg) => {
  if (!cond) { console.error('FAIL:', msg); failed = true; }
  else console.log('ok  :', msg);
};

const rand = mulberry32(42);
const R = rodrigues([0.2, -0.4, 0.15]);
const t = new Float64Array([0.3, -0.2, 4.5]);
const cam = { R, t, f: 480, cx: 240, cy: 180, index: 0 };
const corr = [];
const gt = [];
for (let i = 0; i < 80; i++) {
  const X = new Float64Array([(rand() - 0.5) * 2, (rand() - 0.5) * 2, (rand() - 0.5) * 2 + 1]);
  const p = projectPose(cam, X);
  if (!p.ok) continue;
  const x = p.u + (rand() - 0.5) * 1.5;
  const y = p.v + (rand() - 0.5) * 1.5;
  corr.push({ X, x, y });
  gt.push(X);
}
console.log('correspondences:', corr.length);

const init = { ...cam, R: identity(3), t: new Float64Array(3) };
const pnp = pnpRansac(init, corr, { threshold: 5, maxIters: 1000 });
check(!!pnp, 'PnP RANSAC 返回结果');
if (pnp) {
  const rel = m3mul(pnp.R, m3t(R));
  const tr = rel[0] + rel[4] + rel[8];
  const rotErr = (Math.acos(Math.max(-1, Math.min(1, (tr - 1) / 2))) * 180) / Math.PI;
  const tErr = Math.hypot(pnp.t[0] - t[0], pnp.t[1] - t[1], pnp.t[2] - t[2]);
  console.log(`      inliers ${pnp.inlierCount}/${corr.length}, rotErr ${rotErr.toFixed(3)}°, tErr ${tErr.toFixed(4)}`);
  check(pnp.inlierCount >= corr.length - 3, 'PnP 内点数量');
  check(rotErr < 0.5, 'PnP 旋转误差 < 0.5°');
  check(tErr < 0.05, 'PnP 平移误差 < 0.05');
}

// 两视图三角化
const R2 = rodrigues([-0.1, 0.35, 0.05]);
const C2 = [1.2, 0.1, 0.2];
const rc = m3mulv(R2, C2);
const cam2 = { R: R2, t: new Float64Array([-rc[0], -rc[1], -rc[2]]), f: 480, cx: 240, cy: 180, index: 1 };
let ok = 0;
const errs = [];
for (let i = 0; i < 30; i++) {
  const X = gt[i];
  const p1 = projectPose(cam, X);
  const p2 = projectPose(cam2, X);
  if (!p1.ok || !p2.ok) continue;
  const tri = triangulateObservations(
    [{ cam, x: p1.u, y: p1.v }, { cam: cam2, x: p2.u, y: p2.v }],
    { maxError: 5 }
  );
  if (!tri) continue;
  ok++;
  errs.push(Math.hypot(tri.X[0] - X[0], tri.X[1] - X[1], tri.X[2] - X[2]));
}
errs.sort((a, b) => a - b);
const med = errs[errs.length >> 1] || Infinity;
const max = errs[errs.length - 1] || Infinity;
console.log(`      triangulated ${ok}/30, median err ${med.toFixed(5)}, max ${max.toFixed(5)}`);
check(ok >= 29, '三角化成功数量');
check(med < 1e-4 && max < 1e-2, '三角化 3D 精度');

if (failed) {
  console.error('几何测试失败');
  process.exitCode = 1;
} else {
  console.log('几何测试通过');
}