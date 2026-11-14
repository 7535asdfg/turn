/*
 * 图像与相机基础工具（无 DOM 依赖）
 * 图像统一表示为 { width, height, data: RGBA Uint8Array|Uint8ClampedArray }
 */

export function rgbaToGray(rgba, w, h) {
  const g = new Uint8Array(w * h);
  for (let i = 0, p = 0; i < g.length; i++, p += 4) {
    g[i] = (rgba[p] * 77 + rgba[p + 1] * 150 + rgba[p + 2] * 29) >> 8;
  }
  return g;
}

/** 面积平均缩放：先缩小再放大都适用 */
export function resizeRGBA(src, sw, sh, dw, dh) {
  if (sw === dw && sh === dh) return new Uint8Array(src);
  const out = new Uint8Array(dw * dh * 4);
  const xRatio = sw / dw;
  const yRatio = sh / dh;

  for (let y = 0; y < dh; y++) {
    const sy0 = y * yRatio;
    const sy1 = Math.min(sh, (y + 1) * yRatio);
    const iy0 = Math.floor(sy0), iy1 = Math.min(sh - 1, Math.ceil(sy1) - 1);
    for (let x = 0; x < dw; x++) {
      const sx0 = x * xRatio;
      const sx1 = Math.min(sw, (x + 1) * xRatio);
      const ix0 = Math.floor(sx0), ix1 = Math.min(sw - 1, Math.ceil(sx1) - 1);
      let r = 0, g = 0, b = 0, n = 0;
      // 当缩放比例不大时只采样 2x2，比例大时按整数步长采样
      const stepX = Math.max(1, Math.floor((ix1 - ix0) / 3));
      const stepY = Math.max(1, Math.floor((iy1 - iy0) / 3));
      for (let yy = iy0; yy <= iy1; yy += stepY) {
        let p = (yy * sw + ix0) * 4;
        for (let xx = ix0; xx <= ix1; xx += stepX, p += stepX * 4) {
          r += src[p]; g += src[p + 1]; b += src[p + 2]; n++;
        }
      }
      const o = (y * dw + x) * 4;
      out[o] = r / n; out[o + 1] = g / n; out[o + 2] = b / n; out[o + 3] = 255;
    }
  }
  return out;
}

/** 双线性采样 RGBA，返回 [r,g,b] */
export function sampleRGB(rgba, w, h, x, y) {
  const xc = Math.max(0, Math.min(w - 1, x));
  const yc = Math.max(0, Math.min(h - 1, y));
  const x0 = Math.floor(xc), y0 = Math.floor(yc);
  const x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1);
  const fx = xc - x0, fy = yc - y0;
  const p00 = (y0 * w + x0) * 4, p10 = (y0 * w + x1) * 4;
  const p01 = (y1 * w + x0) * 4, p11 = (y1 * w + x1) * 4;
  const r = (rgba[p00] * (1 - fx) + rgba[p10] * fx) * (1 - fy) +
            (rgba[p01] * (1 - fx) + rgba[p11] * fx) * fy;
  const g = (rgba[p00 + 1] * (1 - fx) + rgba[p10 + 1] * fx) * (1 - fy) +
            (rgba[p01 + 1] * (1 - fx) + rgba[p11 + 1] * fx) * fy;
  const b = (rgba[p00 + 2] * (1 - fx) + rgba[p10 + 2] * fx) * (1 - fy) +
            (rgba[p01 + 2] * (1 - fx) + rgba[p11 + 2] * fx) * fy;
  return [r, g, b];
}

/** 最近邻采样 */
export function sampleRGBNearest(rgba, w, h, x, y) {
  const xi = Math.round(x), yi = Math.round(y);
  if (xi < 0 || yi < 0 || xi >= w || yi >= h) return null;
  const p = (yi * w + xi) * 4;
  return [rgba[p], rgba[p + 1], rgba[p + 2]];
}

/**
 * Census 变换（5x5，24 位）
 * 对光照变化鲁棒，适合多视角稠密匹配。
 */
export function censusTransform(gray, w, h, radius = 2) {
  const out = new Uint32Array(w * h);
  const offsets = [];
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      if (dx === 0 && dy === 0) continue;
      offsets.push(dy * w + dx);
    }
  }
  for (let y = radius; y < h - radius; y++) {
    for (let x = radius; x < w - radius; x++) {
      const idx = y * w + x;
      const c = gray[idx];
      let bits = 0;
      for (let k = 0; k < 24; k++) {
        if (gray[idx + offsets[k]] < c) bits |= (1 << k);
      }
      out[idx] = bits >>> 0;
    }
  }
  return out;
}

/* ---------------- EXIF 焦距解析 ---------------- */

function readU16(view, off, little) {
  return little ? view.getUint16(off, true) : view.getUint16(off, false);
}
function readU32(view, off, little) {
  return little ? view.getUint32(off, true) : view.getUint32(off, false);
}

function readRational(view, off, little) {
  const num = readU32(view, off, little);
  const den = readU32(view, off + 4, little);
  return den ? num / den : 0;
}

/**
 * 从 JPEG ArrayBuffer 中读取焦距信息。
 * @param {ArrayBuffer} buffer
 * @param {number} width  当前图像宽度（解码后）
 * @param {number} height 当前图像高度
 * @returns {{focal:number|null, source:string}}
 */
export function focalFromJpegBuffer(buffer, width, height) {
  try {
    const view = new DataView(buffer);
    if (view.byteLength < 4 || view.getUint16(0, false) !== 0xffd8) {
      return { focal: null, source: 'none' };
    }

    let off = 2;
    let tiffOff = -1;
    while (off + 4 < view.byteLength) {
      if (view.getUint8(off) !== 0xff) { off++; continue; }
      const marker = view.getUint8(off + 1);
      if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
        off += 2;
        continue;
      }
      if (marker === 0xd9 || marker === 0xda) break;
      const len = view.getUint16(off + 2, false);
      if (marker === 0xe1 && len >= 8) {
        const start = off + 4;
        if (view.getUint32(start, false) === 0x45786966) { // "Exif"
          tiffOff = start + 6;
          break;
        }
      }
      off += 2 + len;
    }
    if (tiffOff < 0 || tiffOff + 8 > view.byteLength) return { focal: null, source: 'none' };

    const little = view.getUint16(tiffOff, false) === 0x4949;
    if (!little && view.getUint16(tiffOff, false) !== 0x4d4d) return { focal: null, source: 'none' };
    if (readU16(view, tiffOff + 2, little) !== 0x002a) return { focal: null, source: 'none' };

    const ifd0 = tiffOff + readU32(view, tiffOff + 4, little);
    if (ifd0 + 2 > view.byteLength) return { focal: null, source: 'none' };

    const tags = new Map();
    let exifIfd = -1;

    function readIFD(ifdOff) {
      if (ifdOff + 2 > view.byteLength) return;
      const count = readU16(view, ifdOff, little);
      for (let i = 0; i < count; i++) {
        const e = ifdOff + 2 + i * 12;
        if (e + 12 > view.byteLength) return;
        const tag = readU16(view, e, little);
        const type = readU16(view, e + 2, little);
        const n = readU32(view, e + 4, little);
        let valueOff = e + 8;
        const typeSize = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8 }[type] || 1;
        if (typeSize * n > 4) valueOff = tiffOff + readU32(view, e + 8, little);
        tags.set(tag, { type, n, valueOff });
        if (tag === 0x8769) exifIfd = tiffOff + readU32(view, e + 8, little);
      }
    }

    readIFD(ifd0);
    if (exifIfd > 0) readIFD(exifIfd);

    const f35 = tags.get(0xa405);
    if (f35) {
      const v = f35.type === 3 ? readU16(view, f35.valueOff, little) : readU32(view, f35.valueOff, little);
      if (v > 1) {
        const diag = Math.hypot(width, height);
        return { focal: (v * diag) / 43.266615, mi35: v, source: '35mm' };
      }
    }

    const focalTag = tags.get(0x920a);
    const xresTag = tags.get(0xa20e);
    const unitTag = tags.get(0xa210);
    if (focalTag && xresTag) {
      const focalMm = readRational(view, focalTag.valueOff, little);
      const pxPerUnit = readRational(view, xresTag.valueOff, little);
      const unit = unitTag ? readU16(view, unitTag.valueOff, little) : 2;
      const unitMm = unit === 3 ? 10 : unit === 2 ? 25.4 : 0;
      if (focalMm > 0 && pxPerUnit > 0 && unitMm > 0) {
        return { focal: (focalMm * pxPerUnit) / unitMm, focalMm, source: 'focal-plane' };
      }
    }
  } catch {
    /* ignore */
  }
  return { focal: null, source: 'none' };
}

/** 由视场角（水平方向）计算焦距（像素） */
export function focalFromFov(width, fovDeg) {
  return width / (2 * Math.tan((fovDeg * Math.PI) / 360));
}

export function makeCameraIntrinsics(f, width, height) {
  return { f, cx: width / 2, cy: height / 2, width, height };
}