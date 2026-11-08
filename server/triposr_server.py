"""
Pic2Mesh · TripoSR 本地推理服务
=================================

把 TripoSR（单图 → 三平面 NeRF → 网格）包装成一个简单的 HTTP 接口，供 index.html 调用：

    POST /api/reconstruct   multipart/form-data, 字段名 image
        -> 返回 GLB 二进制（model/gltf-binary）

    GET  /api/health        -> {"status": "ok", "device": "cuda"}

启动：
    pip install -r requirements.txt
    pip install -e /path/to/TripoSR          # 官方仓库，提供 tsr 包
    python triposr_server.py                 # 默认 127.0.0.1:7861

说明：
    * 首次运行会自动从 HuggingFace 下载 stabilityai/TripoSR 权重（约 1.7 GB）。
      国内网络可以先执行：set HF_ENDPOINT=https://hf-mirror.com
    * GPU（CUDA）下 ~2-5 秒/张；纯 CPU 下可能需要数分钟。
"""

from __future__ import annotations

import io
import os

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image

# ----------------------------------------------------------------------------
# 应用 & 模型
# ----------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 131072 if DEVICE == "cuda" else 8192          # 显存/内存吃紧时调小
FOREGROUND_RATIO = 0.85                                     # 主体缩放比例（官方默认）
DEFAULT_RESOLUTION = 256                                    # marching cubes 分辨率

app = FastAPI(title="Pic2Mesh · TripoSR Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 本机使用；如需收紧请改成 http://localhost:8000
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

_model = None
_rembg_session = None


def get_model():
    """懒加载 TripoSR 模型。"""
    global _model
    if _model is None:
        from tsr.system import TSR

        print(f"[TripoSR] 载入模型到 {DEVICE} …")
        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.renderer.set_chunk_size(CHUNK_SIZE)
        model.to(DEVICE)
        model.eval()
        _model = model
        print("[TripoSR] 模型就绪")
    return _model


def remove_background(image: Image.Image) -> Image.Image:
    """使用 rembg 抠图；未安装则原样返回。"""
    global _rembg_session
    try:
        import rembg
    except ImportError:
        print("[warn] 未安装 rembg，跳过背景移除")
        return image

    if _rembg_session is None:
        _rembg_session = rembg.new_session("u2net")
    return rembg.remove(image, session=_rembg_session)


def preprocess(img_bytes: bytes, background_removal: bool) -> Image.Image:
    """按 TripoSR 官方 run.py 的流程处理输入图。"""
    image = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    if background_removal:
        image = remove_background(image)

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    from tsr.utils import resize_foreground

    image = resize_foreground(image, FOREGROUND_RATIO)

    # 透明背景合成到中性灰（0.5），与官方一致
    arr = np.array(image).astype(np.float32) / 255.0
    rgb = arr[:, :, :3] * arr[:, :, 3:4] + (1.0 - arr[:, :, 3:4]) * 0.5
    return Image.fromarray((rgb * 255.0).astype(np.uint8))


# ----------------------------------------------------------------------------
# 路由
# ----------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_loaded": _model is not None,
    }


@app.post("/api/reconstruct")
async def reconstruct(
    image: UploadFile = File(...),
    background_removal: bool = True,
    resolution: int = DEFAULT_RESOLUTION,
):
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传的图片为空")

    if not (64 <= resolution <= 512):
        raise HTTPException(status_code=400, detail="resolution 需在 64~512 之间")

    try:
        pil_image = preprocess(data, background_removal)
    except Exception as exc:  # 图片损坏 / 依赖缺失
        raise HTTPException(status_code=400, detail=f"图片处理失败：{exc}") from exc

    try:
        model = get_model()
        with torch.no_grad():
            scene_codes = model([pil_image], device=DEVICE)
        with torch.no_grad():
            meshes = model.extract_mesh(
                scene_codes, has_vertex_color=True, resolution=resolution
            )
        mesh = meshes[0]
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise HTTPException(
            status_code=507,
            detail="显存不足：请降低 resolution，或把 CHUNK_SIZE 调小后重试",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"推理失败：{exc}") from exc

    glb = mesh.export(file_type="glb")
    if isinstance(glb, str):  # 某些 trimesh 版本返回临时文件路径
        path = glb
        with open(path, "rb") as fh:
            glb = fh.read()
        try:
            os.unlink(path)
        except OSError:
            pass

    return Response(
        content=glb,
        media_type="model/gltf-binary",
        headers={"Content-Disposition": 'attachment; filename="model.glb"'},
    )


@app.get("/")
def index():
    return JSONResponse({"service": "Pic2Mesh TripoSR", "endpoints": ["/api/health", "/api/reconstruct"]})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7861)