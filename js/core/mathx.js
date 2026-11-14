/*
 * 轻量线性代数工具（无任何依赖，可在浏览器与 Node 中运行）
 *
 * 矩阵一律使用行主序 Float64Array：
 *   3x3 -> [m00,m01,m02, m10,m11,m12, m20,m21,m22]
 *   4x4 -> 按行展开
 */

export const EPS = 1e-12;

export function zeros(n) {
  return new Float64Array(n);
}

export function identity(n) {
  const m = new Float64Array(n * n);
  for (let i = 0; i < n; i++) m[i * n + i] = 1;
  return m;
}

export function diag3(a, b, c) {
  const m = new Float64Array(9);
  m[0] = a; m[4] = b; m[8] = c;
  return m;
}

/* ---------------- 3x3 / 4x4 ---------------- */

export function m3mul(a, b) {
  const o = new Float64Array(9);
  for (let r = 0; r < 3; r++) {
    const a0 = a[r * 3], a1 = a[r * 3 + 1], a2 = a[r * 3 + 2];
    o[r * 3] = a0 * b[0] + a1 * b[3] + a2 * b[6];
    o[r * 3 + 1] = a0 * b[1] + a1 * b[4] + a2 * b[7];
    o[r * 3 + 2] = a0 * b[2] + a1 * b[5] + a2 * b[8];
  }
  return o;
}

export function m3t(a) {
  return new Float64Array([
    a[0], a[3], a[6],
    a[1], a[4], a[7],
    a[2], a[5], a[8],
  ]);
}

export function m3mulv(a, v) {
  return new Float64Array([
    a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
    a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
    a[6] * v[0] + a[7] * v[1] + a[8] * v[2],
  ]);
}

export function m3det(a) {
  return (
    a[0] * (a[4] * a[8] - a[5] * a[7]) -
    a[1] * (a[3] * a[8] - a[5] * a[6]) +
    a[2] * (a[3] * a[7] - a[4] * a[6])
  );
}

export function m3inv(a) {
  const d = m3det(a);
  if (Math.abs(d) < 1e-18) return null;
  const id = 1 / d;
  const o = new Float64Array(9);
  o[0] = (a[4] * a[8] - a[5] * a[7]) * id;
  o[1] = (a[2] * a[7] - a[1] * a[8]) * id;
  o[2] = (a[1] * a[5] - a[2] * a[4]) * id;
  o[3] = (a[5] * a[6] - a[3] * a[8]) * id;
  o[4] = (a[0] * a[8] - a[2] * a[6]) * id;
  o[5] = (a[2] * a[3] - a[0] * a[5]) * id;
  o[6] = (a[3] * a[7] - a[4] * a[6]) * id;
  o[7] = (a[1] * a[6] - a[0] * a[7]) * id;
  o[8] = (a[0] * a[4] - a[1] * a[3]) * id;
  return o;
}

export function m3orthonormalize(a) {
  // Gram-Schmidt，防止数值漂移
  const x = new Float64Array([a[0], a[3], a[6]]);
  let n = Math.hypot(x[0], x[1], x[2]);
  if (n < EPS) return identity(3);
  x[0] /= n; x[1] /= n; x[2] /= n;

  const y = new Float64Array([a[1], a[4], a[7]]);
  const d = x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
  y[0] -= d * x[0]; y[1] -= d * x[1]; y[2] -= d * x[2];
  n = Math.hypot(y[0], y[1], y[2]);
  if (n < EPS) return identity(3);
  y[0] /= n; y[1] /= n; y[2] /= n;

  const z = cross(x, y);
  return new Float64Array([
    x[0], y[0], z[0],
    x[1], y[1], z[1],
    x[2], y[2], z[2],
  ]);
}

export function cross(a, b) {
  return new Float64Array([
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]);
}

export function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function norm(a) {
  return Math.hypot(a[0], a[1], a[2]);
}

export function normalize(a) {
  const n = norm(a) || 1;
  return new Float64Array([a[0] / n, a[1] / n, a[2] / n]);
}

export function sub(a, b) {
  return new Float64Array([a[0] - b[0], a[1] - b[1], a[2] - b[2]]);
}

export function add(a, b) {
  return new Float64Array([a[0] + b[0], a[1] + b[1], a[2] + b[2]]);
}

export function scale(a, s) {
  return new Float64Array([a[0] * s, a[1] * s, a[2] * s]);
}

export function crossMatrix(a) {
  return new Float64Array([
    0, -a[2], a[1],
    a[2], 0, -a[0],
    -a[1], a[0], 0,
  ]);
}

/* ---------------- 旋转 / 位姿 ---------------- */

/** 轴角（行向量 w，模长为角度）-> 旋转矩阵 */
export function rodrigues(w) {
  const th = Math.hypot(w[0], w[1], w[2]);
  if (th < 1e-14) {
    return new Float64Array([
      1, -w[2], w[1],
      w[2], 1, -w[0],
      -w[1], w[0], 1,
    ]);
  }
  const x = w[0] / th, y = w[1] / th, z = w[2] / th;
  const c = Math.cos(th), s = Math.sin(th), C = 1 - c;
  return new Float64Array([
    c + x * x * C, x * y * C - z * s, x * z * C + y * s,
    y * x * C + z * s, c + y * y * C, y * z * C - x * s,
    z * x * C - y * s, z * y * C + x * s, c + z * z * C,
  ]);
}

/** 把世界点 X 变换到相机坐标系：R X + t */
export function applyPose(R, t, X) {
  return new Float64Array([
    R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0],
    R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1],
    R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2],
  ]);
}

export function cameraCenter(R, t) {
  const Rt = m3t(R);
  return new Float64Array([
    -(Rt[0] * t[0] + Rt[1] * t[1] + Rt[2] * t[2]),
    -(Rt[3] * t[0] + Rt[4] * t[1] + Rt[5] * t[2]),
    -(Rt[6] * t[0] + Rt[7] * t[1] + Rt[8] * t[2]),
  ]);
}

export function rotationAngle(R) {
  const tr = R[0] + R[4] + R[8];
  return Math.acos(Math.max(-1, Math.min(1, (tr - 1) / 2)));
}

/* ---------------- 特征值 ---------------- */

/**
 * 对称矩阵 Jacobi 特征分解。
 * @param {ArrayLike<number>} Ain 行主序 n x n 对称矩阵
 * @returns {{values:Float64Array, vectors:Float64Array}} vectors 的第 i 列是第 i 个特征向量
 */
export function jacobiEigen(Ain, n, maxSweeps = 80) {
  const A = Float64Array.from(Ain);
  const V = identity(n);

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const v = A[p * n + q];
        off += v * v;
      }
    }
    if (off < 1e-26) break;

    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q];
        if (Math.abs(apq) < 1e-18) continue;
        const app = A[p * n + p];
        const aqq = A[q * n + q];
        const theta = (aqq - app) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;

        // A <- J^T A J
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p];
          const akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq;
          A[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k];
          const aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk;
          A[q * n + k] = s * apk + c * aqk;
        }
        // V <- V J
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p];
          const vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq;
          V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }

  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = A[i * n + i];
  return { values, vectors: V };
}

/** 返回绝对值最小的特征值对应的特征向量 */
export function smallestEigenvector(AtA, n) {
  const { values, vectors } = jacobiEigen(AtA, n);
  let best = 0;
  let bestAbs = Infinity;
  for (let i = 0; i < n; i++) {
    const a = Math.abs(values[i]);
    if (a < bestAbs) { bestAbs = a; best = i; }
  }
  const out = new Float64Array(n);
  for (let r = 0; r < n; r++) out[r] = vectors[r * n + best];
  return { vector: out, values };
}

/** 由 A (m x n) 构造 A^T A */
export function gramMatrix(A, m, n) {
  const G = new Float64Array(n * n);
  for (let r = 0; r < m; r++) {
    const base = r * n;
    for (let i = 0; i < n; i++) {
      const ai = A[base + i];
      if (ai === 0) continue;
      for (let j = i; j < n; j++) {
        G[i * n + j] += ai * A[base + j];
      }
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) G[i * n + j] = G[j * n + i];
  }
  return G;
}

/**
 * 3x3 奇异值分解 A = U diag(S) V^T
 * 适用于秩 >= 2 的矩阵（本项目中用于本质矩阵与 PnP 的极分解）。
 */
export function svd3(A) {
  const AtA = gramMatrix(A, 3, 3);
  const { values, vectors } = jacobiEigen(AtA, 3);

  const order = [0, 1, 2].sort((i, j) => values[j] - values[i]);
  const S = new Float64Array(3);
  const V = new Float64Array(9);
  for (let c = 0; c < 3; c++) {
    const src = order[c];
    S[c] = Math.sqrt(Math.max(0, values[src]));
    for (let r = 0; r < 3; r++) V[r * 3 + c] = vectors[r * 3 + src];
  }

  const U = new Float64Array(9);
  for (let c = 0; c < 2; c++) {
    const v = new Float64Array([V[c], V[3 + c], V[6 + c]]);
    const av = m3mulv(A, v);
    const s = S[c];
    if (s > 1e-12) {
      U[c] = av[0] / s; U[3 + c] = av[1] / s; U[6 + c] = av[2] / s;
    }
  }
  // 第三列由前两列叉乘得到，保证正交
  const u0 = new Float64Array([U[0], U[3], U[6]]);
  const u1 = new Float64Array([U[1], U[4], U[7]]);
  const u2 = cross(u0, u1);
  // 特征向量的符号是任意的；选择 V 第三列的符号使 A v2 = s2 u2
  const v2 = new Float64Array([V[2], V[5], V[8]]);
  const av2 = m3mulv(A, v2);
  if (dot(u2, av2) < 0) {
    V[2] = -V[2]; V[5] = -V[5]; V[8] = -V[8];
  }
  U[2] = u2[0]; U[5] = u2[1]; U[8] = u2[2];

  return { U, S, V };
}

/** 把 E 投影到本质矩阵流形（两个相等奇异值 + 一个 0） */
export function enforceEssential(E) {
  const { U, V } = svd3(E);
  // E = U diag(1,1,0) V^T
  const D = diag3(1, 1, 0);
  return m3mul(m3mul(U, D), m3t(V));
}

/** 从旋转矩阵提取四元数 [x,y,z,w] */
export function quatFromMat3(R) {
  const tr = R[0] + R[4] + R[8];
  let x, y, z, w;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    w = 0.25 * s;
    x = (R[7] - R[5]) / s;
    y = (R[2] - R[6]) / s;
    z = (R[3] - R[1]) / s;
  } else if (R[0] > R[4] && R[0] > R[8]) {
    const s = Math.sqrt(1 + R[0] - R[4] - R[8]) * 2;
    w = (R[7] - R[5]) / s;
    x = 0.25 * s;
    y = (R[1] + R[3]) / s;
    z = (R[2] + R[6]) / s;
  } else if (R[4] > R[8]) {
    const s = Math.sqrt(1 + R[4] - R[0] - R[8]) * 2;
    w = (R[2] - R[6]) / s;
    x = (R[1] + R[3]) / s;
    y = 0.25 * s;
    z = (R[5] + R[7]) / s;
  } else {
    const s = Math.sqrt(1 + R[8] - R[0] - R[4]) * 2;
    w = (R[3] - R[1]) / s;
    x = (R[2] + R[6]) / s;
    y = (R[5] + R[7]) / s;
    z = 0.25 * s;
  }
  const n = Math.hypot(x, y, z, w) || 1;
  return [x / n, y / n, z / n, w / n];
}

/** 四元数 [x,y,z,w] -> 旋转矩阵 */
export function mat3FromQuat(q) {
  let x = q[0], y = q[1], z = q[2], w = q[3];
  const n = Math.hypot(x, y, z, w) || 1;
  x /= n; y /= n; z /= n; w /= n;
  return new Float64Array([
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ]);
}

/** 旋转矩阵->四元数，然后求和归一化（用于旋转平均） */
export function averageRotations(rotations) {
  if (!rotations.length) return identity(3);
  const qs = rotations.map(quatFromMat3);
  const ref = qs[0];
  const acc = [0, 0, 0, 0];
  for (const q of qs) {
    const s = q[0] * ref[0] + q[1] * ref[1] + q[2] * ref[2] + q[3] * ref[3] < 0 ? -1 : 1;
    acc[0] += q[0] * s; acc[1] += q[1] * s; acc[2] += q[2] * s; acc[3] += q[3] * s;
  }
  return mat3FromQuat(acc);
}

/* ---------------- 线性方程 / LM ---------------- */

/** 高斯消元（带部分主元），A 为 n x n 行主序，返回 x */
export function solveLinear(Ain, b, n) {
  const A = Float64Array.from(Ain);
  const x = Float64Array.from(b);
  for (let col = 0; col < n; col++) {
    let piv = col;
    let maxv = Math.abs(A[col * n + col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(A[r * n + col]);
      if (v > maxv) { maxv = v; piv = r; }
    }
    if (maxv < 1e-20) return null;
    if (piv !== col) {
      for (let c = 0; c < n; c++) {
        const tmp = A[col * n + c];
        A[col * n + c] = A[piv * n + c];
        A[piv * n + c] = tmp;
      }
      const tmp = x[col]; x[col] = x[piv]; x[piv] = tmp;
    }
    const d = A[col * n + col];
    for (let r = col + 1; r < n; r++) {
      const f = A[r * n + col] / d;
      if (f === 0) continue;
      for (let c = col; c < n; c++) A[r * n + c] -= f * A[col * n + c];
      x[r] -= f * x[col];
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    let s = x[r];
    for (let c = r + 1; c < n; c++) s -= A[r * n + c] * x[c];
    const d = A[r * n + r];
    if (Math.abs(d) < 1e-20) return null;
    x[r] = s / d;
  }
  return x;
}

/**
 * 一维 Huber 权重（残差以像素计）
 */
export function huberWeight(r, delta = 3) {
  const a = Math.abs(r);
  return a <= delta ? 1 : delta / a;
}

/* ---------------- 随机数 ---------------- */

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 无放回抽取 k 个下标 */
export function sampleIndices(n, k, rand) {
  const out = new Int32Array(k);
  const used = new Set();
  let i = 0;
  while (i < k) {
    const v = (rand() * n) | 0;
    if (used.has(v)) continue;
    used.add(v);
    out[i++] = v;
  }
  return out;
}

export function median(values) {
  if (!values.length) return NaN;
  const a = Array.from(values).sort((x, y) => x - y);
  return a[(a.length - 1) >> 1];
}

export function percentile(values, q) {
  if (!values.length) return NaN;
  const a = Array.from(values).sort((x, y) => x - y);
  const idx = Math.min(a.length - 1, Math.max(0, Math.round(q * (a.length - 1))));
  return a[idx];
}