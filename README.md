# FacePass - 本地人脸识别系统

FacePass 是一个课程大作业项目，目标是在本地完成图片上传、人脸检测、身份识别与结果可视化。当前版本完成了模型端、后端、前端三层分离的最小闭环，并加入边界错误处理、日志和重试机制。

当前模型层支持两种实现：

- `insightface`：默认运行实现，使用 InsightFace 完成人脸检测与特征提取。
- `fake`：测试替身实现，不下载权重，用于在 CI / 本地测试中验证后端与接口解耦。

## 结构

```text
src/
|-- common/              # 日志、重试、安全图片解码、通用异常
|-- face_model/          # 模型端：FaceModel 抽象、DetectedFace、InsightFace 实现
|-- backend/             # 后端：Gallery、Recognizer、FastAPI API、配置
`-- frontend/            # 前端：Gradio 界面，仅通过 HTTP 调后端
scripts/
|-- download_models.py   # HF Hub 下载占位脚本
`-- run_dev.py           # 可选的一键开发启动脚本
data/
|-- registered/          # p01..p20 注册照目录，本地放置，不进 git
`-- test/                # 测试图片，本地放置，不进 git
models/                 # gallery.pkl 和自托管模型文件，本地生成/下载，不进 git
tests/                  # 接口、边界和错误路径测试
```

## 依赖

本项目使用 `uv` 管理依赖。`onnxruntime` 当前固定为 `1.22.1`，因为较新的 `1.24.x` 在本项目的 CPython 3.10 Windows 环境没有可用 wheel。

```powershell
uv sync
```

后端默认配置在 `src/backend/config.py`，当前 `model_name="insightface"`。如果只想跑不依赖权重的后端集成测试，可以在测试里把配置切到 `fake`。

## 运行

首次运行 InsightFace `buffalo_l` 会自动下载约 300MB 模型到用户目录下的 InsightFace 缓存，不会进入本仓库。

```powershell
# 1. 下载项目自托管大文件。目前 HF 仓库还未创建，此脚本只会创建 models/ 并提示 TODO。
uv run python scripts/download_models.py

# 2. 准备注册集：将 p01..p20 的注册照放到 data/registered/p01/ 等目录。

# 3. 启动后端
uv run uvicorn src.backend.api:app --port 8000

# 4. 另开终端启动前端
uv run python src/frontend/app.py
```

健康检查：

```powershell
curl http://127.0.0.1:8000/health
```

## API

- `GET /health`：返回服务状态。
- `GET /identities`：返回身份库中每个身份的注册图数量。
- `POST /recognize`：multipart 上传图片，返回 `RecognitionResult` 列表。

`RecognitionResult` 字段保持为：

- `bbox`: `(x, y, w, h)`
- `identity_id`: 命中身份或 `unknown`
- `name`: 显示名，unknown 时为 `null`
- `similarity`: 相似度
- `is_unknown`: 是否为 unknown

## 错误处理

错误按三档处理：

- 致命错误：后端启动阶段模型加载失败、身份库不存在且无法构建、身份库为空，会记录清晰错误并以非零状态退出。
- 瞬时错误：前端请求后端、HF 下载等 IO/网络操作使用有限次数指数退避重试；耗尽后返回友好提示，不让前端进程退出。
- 可恢复输入错误：坏图片、空文件、非图片、单张注册图读取失败不重试；API 返回 400/413，建库时跳过坏图并继续。

## 大文件与 Hugging Face

仓库不提交任何超过 50MB 的权重、构建好的身份库或大数组文件。`.gitignore` 已排除：

- `models/`
- `data/`
- `*.onnx`
- `*.pkl`
- `*.npy`
- `.venv/`
- Python 缓存和日志

模型下载链接：`<待填 HF 链接>`

`scripts/download_models.py` 中的 `HF_REPO_ID` 目前是 `TODO_org/face-recognition-assignment-models`。后续创建 HF 仓库后，将自托管的大文件上传到该仓库，运行时下载到本地 `models/`。

## 测试

```powershell
uv run pytest tests -q
```

当前测试覆盖：

- Gallery 注册、保存、加载、最大相似度匹配。
- Gallery 空库匹配返回空结果，unknown 语义由 Recognizer 统一持有。
- Recognizer 阈值判定和 unknown 输出。
- API 健康检查、身份列表、坏图片 400、正常图片 200。
- `load_identities` 支持注入路径，不再绑定定义时默认配置。
- `safe_load_image` 坏图/空文件处理。
- `with_retry` 只重试瞬时异常，不重试 `ValueError`。
- 前端后端不可达时返回友好提示。
- `FakeFaceModel` 可在不下载权重的情况下完整跑通建库和 `/recognize` HTTP 流程。
- 后端不导入具体模型实现，前端不导入任何 `src.*` 内部模块。

## 待填项

- 创建 Hugging Face 仓库并替换 `HF_REPO_ID` 与模型下载链接。
- 收集并放置真实 `data/registered/p01` 到 `p20` 注册照。
- 基于注册集内部相似度分布确定最终 unknown 阈值，不能用测试集调参。
- 最终提交前按课程要求补充打包脚本和报告。

## 小组成员

| 姓名 | 学号 | 分工 |
| --- | --- | --- |
|  |  |  |
|  |  |  |
|  |  |  |
