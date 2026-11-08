# Pic2Mesh · 单张照片生成可旋转的 3D 效果图

一个纯前端实现的网页应用：**上传一张照片 → 算法自动估计深度 → 实时构建 3D 模型
→ 鼠标自由旋转 / 缩放 / 平移**。

全部推理都在浏览器本地完成，照片不会上传到任何服务器。

```
项目/
├── index.html              ← 整个网页应用（UI + 3D 渲染 + 算法），单文件无构建
├── start.bat               ← Windows 一键启动本地服务并打开网页
├── README.md
├── models/                 ← 内置的深度模型（74 MB，可选，缺失时自动回退在线下载）
│   └── onnx-community/depth-anything-v2-small/
│       ├── config.json
│       ├── preprocessor_config.json
│       └── onnx/model_fp16.onnx (50 MB) + model_quantized.onnx (27 MB)
└── server/                 ← 旧版 TripoSR 服务的残留文件，网页已不再使用，可自行删除
```

---

## 一、快速开始

**方式 1（推荐）**：双击 `start.bat`，会自动起一个本地 HTTP 服务并打开
<http://localhost:8000/index.html>。

**方式 2**：在项目目录执行

```bash
python -m http.server 8000
```

然后浏览器访问 <http://localhost:8000/index.html>。

**方式 3**：直接双击 `index.html` 也能用（CDN 资源都带 CORS 头），但 `file://`
下无法读取同目录的 `models/`（会退回到在线下载），且部分浏览器对 WebGPU 有限制，
遇到问题请改用方式 1 或 2。

> 深度模型已放在 `models/` 里，**默认完全离线推理**，不会上传照片，也无需等待下载。
> 若把 `models/` 删掉，网页会自动改为在线下载（默认走 huggingface 国内镜像
> hf-mirror.com，可在界面里切换）。
> Three.js / Transformers.js 两个库仍从 jsDelivr CDN 加载（约 1 MB）。

## 二、界面与操作

| 位置 | 功能 |
| --- | --- |
| 左上「选择文件」 | 点击选择本地照片；也支持**拖拽**到页面，或 `Ctrl + V` 直接粘贴截图 |
| 深度强度 | 3D 起伏的强弱，越大立体感越强，过大会拉出尖刺 |
| 网格精度 | 网格分辨率（60–360）。越高越细腻，也越吃显卡 |
| 平滑 | 对深度图做模糊，抑制噪点与台阶 |
| 反转深度 | 默认「越亮=越近」；勾选后变成「越亮=越远」，适合深度图明暗颠倒的模型 |
| 显示 | 自动旋转 / 线框 / 点云 / 原色（无光照） |
| 导出 GLB | 把当前重建结果导出为 `.glb`，可导入 Blender、Windows 3D 查看器、Three.js 等 |

**3D 视区操作**：

- 左键拖拽 → 旋转
- 滚轮 / 双指捏合 → 缩放
- 右键拖拽 / 双指拖动 → 平移
- 触屏：单指旋转、双指缩放平移
- 右上角实时显示顶点数、三角面数、FPS

## 三、算法说明

```
照片 ──► Depth Anything V2 Small (ONNX, transformers.js)
     ──► 单目相对深度图（每个像素一个深度值）
     ──► 归一化 + 可分离盒式模糊去噪
     ──► 按网格精度双线性采样，把深度当作 Z 位移，生成顶点/UV/索引
     ──► 原图作为纹理贴回 + 环境光/方向光 + 顶点法线 → Three.js 实时渲染
```

- 模型：`onnx-community/depth-anything-v2-small`（已打包在 `models/`），优先使用 **WebGPU + fp16** 推理，不支持时自动回退到 **WASM + q8**。
- 这是「2.5D 浮雕式重建」：正面视角与原图高度一致，旋转时能看到清晰的立体层次；
  但照片里被遮挡的部分（比如人的后脑勺）无法凭空生成，转到背面会看到镜像/拉伸。

## 四、常见问题

**1. 提示「存在安全风险 / 模型加载失败」**
先确认 `models/` 目录完整（约 74 MB）。若已删除，网页会改为在线下载——国内网络访问
huggingface.co 会超时，此时请确认勾选了「在线下载时优先用国内镜像」。
若用 `file://` 直接打开，浏览器禁止读取本地模型文件，请改用 `start.bat`。

**2. 生成很慢**
第一次生成要初始化推理引擎（编译 ONNX 模型）。支持 **WebGPU** 的 Chrome / Edge
通常 3–10 秒；不支持 WebGPU 的浏览器会回退到 WASM，首次可能要 1–3 分钟、
之后每张 10–30 秒。用 `start.bat` 通过 `localhost` 打开可以启用更快的路径。

若浏览器控制台出现 `createBuffer failed ... mappedAtCreation` 之类的 WebGPU 报错，
网页会自动改用 **WASM** 重新推理（状态栏会显示「回退 WASM 重新推理…」），
不需要手动处理，只是首次会多等一会儿。

**3. 3D 太「平」或者太「尖」**
调「深度强度」。太尖可以把「平滑」调到 2–3。

**4. 边缘有明显的拉伸/撕裂**
这是深度图的固有特性（前后景边界处深度跳变）。可以适当增加「平滑」。

**5. 导出 GLB 后贴图丢失**
贴图以 PNG 内嵌在 GLB 里，Blender 导入时请使用「文件 → 导入 → glTF 2.0」，不要用「导入 OBJ」。

## 五、技术栈

- [Three.js](https://threejs.org/) r169 —— WebGL 渲染、OrbitControls 轨道控制、GLTFExporter
- [Transformers.js](https://github.com/huggingface/transformers.js) v3 + ONNX Runtime Web —— 浏览器内运行 Depth Anything V2

没有任何构建步骤，改完 `index.html` 刷新即可。