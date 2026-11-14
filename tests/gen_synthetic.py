#!/usr/bin/env python3
"""
生成用于测试 SfM + MVS 的合成多视角数据（物体环绕拍摄）。

场景：3 个带唯一程序纹理的球体，深色渐变背景（无背景特征）
相机：围绕物体一圈的 24 个视角，已知内参和世界->相机外参

输出（tests/synthetic/）：
  view_XX.png   可视化用
  views.bin     所有视角的 RGBA 原始字节
  depths.bin    所有视角的相机 z 深度（Float32，0 表示背景）
  meta.json     相机参数、场景参数、偏移量
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

OUT = Path(__file__).resolve().parent / "synthetic"
OUT.mkdir(parents=True, exist_ok=True)

WIDTH, HEIGHT = 480, 360
FOCAL = 480.0
CX, CY = WIDTH / 2, HEIGHT / 2
N_VIEWS = 36

SPHERES = [
    {"center": (-0.72, 0.10, 0.05), "r": 0.86, "seed": 11, "tint": (1.00, 0.52, 0.30)},
    {"center": (0.82, -0.18, 0.32), "r": 0.60, "seed": 23, "tint": (0.40, 0.75, 1.00)},
    {"center": (0.48, 0.92, -0.42), "r": 0.42, "seed": 37, "tint": (0.62, 1.00, 0.55)},
]

LIGHT = np.array([0.45, 0.85, 0.55], dtype=np.float64)
LIGHT /= np.linalg.norm(LIGHT)


def look_at(center, target, up=(0.0, 1.0, 0.0)):
    c = np.asarray(center, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    fwd = tgt - c
    fwd /= np.linalg.norm(fwd)
    up = np.asarray(up, dtype=np.float64)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd], axis=0)
    t = -R @ c
    return R, t, c


def make_texture(seed, tint):
    """随机平滑色块 + 细噪声；色块尺寸适中，跨视角稳定。"""
    rng = np.random.default_rng(seed)
    small = rng.uniform(0.10, 0.98, (48, 48, 3))
    img = Image.fromarray((small * 255).astype(np.uint8)).resize((512, 512), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float64) / 255.0
    yy, xx = np.mgrid[0:512, 0:512]
    detail = (0.07 * np.sin(xx * 0.33 + 0.7) * np.sin(yy * 0.29 + 1.1)
              + 0.05 * np.sin(xx * 0.71 + 2.0) * np.sin(yy * 0.63 + 0.4))
    arr = np.clip(arr + detail[..., None], 0, 1)
    fine = rng.uniform(-0.03, 0.03, (512, 512, 1))
    arr = np.clip(arr + fine, 0, 1)
    tint = np.asarray(tint, dtype=np.float64)
    arr = arr * tint[None, None, :] + (1.0 - tint[None, None, :]) * 0.10
    return np.clip(arr, 0, 1)


TEXTURES = [make_texture(s["seed"], s["tint"]) for s in SPHERES]


def sample_tex(tex, u, v):
    n = tex.shape[0] - 1
    x = np.clip((np.mod(u, 1.0) * n).astype(np.int64), 0, n)
    y = np.clip((np.mod(v, 1.0) * n).astype(np.int64), 0, n)
    return tex[y, x]


def sphere_hit(origin, dirs, center, radius):
    oc = origin - np.asarray(center, dtype=np.float64)
    a = np.sum(dirs * dirs, axis=1)
    b = 2.0 * (dirs @ oc)
    c = oc @ oc - radius * radius
    disc = b * b - 4 * a * c
    valid = disc > 0
    t = np.full(dirs.shape[0], np.inf)
    sq = np.sqrt(np.maximum(disc, 0))
    t0 = (-b - sq) / (2 * a)
    t1 = (-b + sq) / (2 * a)
    tt = np.where(t0 > 1e-4, t0, t1)
    valid &= tt > 1e-4
    t[valid] = tt[valid]
    return t


def render_one(Rw, C, seed):
    rng = np.random.default_rng(seed)
    us, vs = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    u = us.ravel().astype(np.float64)
    v = vs.ravel().astype(np.float64)
    dcam = np.stack([(u - CX) / FOCAL, (v - CY) / FOCAL, np.ones_like(u)], axis=1)
    dworld = dcam @ Rw

    n = u.shape[0]
    best_t = np.full(n, np.inf)
    best_id = np.full(n, -1, dtype=np.int32)

    for si, sp in enumerate(SPHERES):
        t = sphere_hit(C, dworld, sp["center"], sp["r"])
        mask = t < best_t
        best_t[mask] = t[mask]
        best_id[mask] = si

    P = C[None, :] + best_t[:, None] * dworld
    normal = np.zeros_like(P)
    color = np.zeros_like(P)

    for si, sp in enumerate(SPHERES):
        m = best_id == si
        if not np.any(m):
            continue
        center = np.asarray(sp["center"])
        r = sp["r"]
        rel = P[m] - center
        normal[m] = rel / r
        theta = np.arctan2(rel[:, 2], rel[:, 0])
        phi = np.arcsin(np.clip(rel[:, 1] / r, -1, 1))
        color[m] = sample_tex(TEXTURES[si], theta / (2 * np.pi) + 0.5, phi / np.pi + 0.5)

    # 深色背景渐变（不产生稳定特征）
    m = best_id < 0
    if np.any(m):
        sky = np.clip(0.16 - 0.10 * dcam[m, 1], 0.015, 0.22)
        color[m] = np.stack([
            sky * 0.55 + 0.012,
            sky * 0.75 + 0.02,
            sky * 1.15 + 0.05,
        ], axis=1)

    diffuse = np.clip(normal @ LIGHT, 0, 1) * 0.42 + 0.72
    color = np.clip(color * diffuse[:, None], 0, 1)

    img = (color.reshape(HEIGHT, WIDTH, 3) * 255.0)
    img = np.clip(img + rng.normal(0, 1.6, img.shape), 0, 255).astype(np.uint8)
    pil = Image.fromarray(img, "RGB").filter(ImageFilter.GaussianBlur(0.7))
    depth = np.where(np.isfinite(best_t), best_t, 0.0).astype(np.float32).reshape(HEIGHT, WIDTH)
    return pil, depth


def main():
    cameras = []
    blobs = []
    depths = []

    for i in range(N_VIEWS):
        ang = 2 * np.pi * i / N_VIEWS
        radius = 3.55 + 0.22 * math.sin(i * 1.7)
        height = 0.92 + 0.18 * math.cos(i * 2.3)
        C = np.array([radius * math.cos(ang), height, radius * math.sin(ang)])
        target = np.array([0.0, 0.12, 0.0])
        R, t, C = look_at(C, target)
        pil, depth = render_one(R, C, 1000 + i)
        pil.save(OUT / f"view_{i:02d}.png")

        rgba = np.dstack([np.asarray(pil), np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)])
        blobs.append(rgba.astype(np.uint8).tobytes())
        depths.append(depth.tobytes())
        cameras.append({
            "index": i,
            "width": WIDTH,
            "height": HEIGHT,
            "focal": FOCAL,
            "R": R.tolist(),
            "t": t.tolist(),
            "center": C.tolist(),
        })

    with open(OUT / "views.bin", "wb") as fh:
        fh.write(b"".join(blobs))
    with open(OUT / "depths.bin", "wb") as fh:
        fh.write(b"".join(depths))

    meta = {
        "width": WIDTH,
        "height": HEIGHT,
        "focal": FOCAL,
        "nViews": N_VIEWS,
        "bytesPerView": WIDTH * HEIGHT * 4,
        "bytesPerDepth": WIDTH * HEIGHT * 4,
        "cameras": cameras,
        "scene": {
            "spheres": SPHERES,
            "groundY": -100.0,
            "wallZ": -100.0,
        },
    }
    with open(OUT / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"生成 {N_VIEWS} 个视角 -> {OUT}")
    print("  view_XX.png / views.bin / depths.bin / meta.json")


if __name__ == "__main__":
    main()