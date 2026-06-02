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
|-- run_dev.py           # 可选的一键开发启动脚本
|-- eval_self.py         # 单脸裁剪口径评测脚本
|-- analyze_threshold.py # 注册集相似度分布与阈值分析
`-- eval_end2end.py      # 多脸端到端评测与出图
dataset/
|-- identities.csv       # 身份 ID 到显示名映射，已纳入版本控制
|-- registered/          # 真实注册集：p01..p20 注册照目录，已纳入版本控制
`-- test/                # 自采测试图与标注目录，测试图已纳入版本控制
data/
|-- registered/          # 占位目录，仅保留 .gitkeep
|-- test/                # 占位目录，仅保留 .gitkeep
`-- tmp_*                # 测试 / 脚本用的临时 fixture，不是正式数据集
models/                  # gallery.pkl 等本地产物，不进 git
reports/                 # 评测 JSON / PNG 输出，本地产物，不进 git
tests/                  # 接口、边界和错误路径测试
```

## 依赖

本项目使用 `uv` 管理依赖。`onnxruntime` 当前固定为 `1.22.1`，因为较新的 `1.24.x` 在本项目的 CPython 3.10 Windows 环境没有可用 wheel。

```powershell
uv sync
```

后端默认配置在 `src/backend/config.py`，当前 `model_name="insightface"`。如果只想跑不依赖权重的后端集成测试，可以在测试里把配置切到 `fake`。

## 运行

运行 `insightface` 模型前，需要你自己准备本地 `buffalo_l` 目录。项目不会再自动下载或托管模型文件。当前适配器按“`buffalo_l` 模型目录本身”加载，不是传 `models_cache` 根目录。

```powershell
# 1. 准备本地 buffalo_l 模型目录，例如：
#    F:\InsightFace\models_cache\models\buffalo_l
#
# 2. 准备注册集：将 p01..p20 的注册照放到 dataset/registered/p01/ 等目录。

# 3. 启动后端和前端
uv run python scripts/run_dev.py --model-path F:\InsightFace\models_cache\models\buffalo_l

# 或者只启动后端
$env:FACEPASS_MODEL_PATH = "F:\InsightFace\models_cache\models\buffalo_l"
uv run uvicorn src.backend.api:app --port 8000

# 另开终端启动前端
uv run python src/frontend/app.py
```

`run_dev.py` 会同时启动：

- FastAPI 后端：`http://127.0.0.1:8000`
- Gradio 前端：`http://127.0.0.1:7860`

浏览器应打开 `7860`。`8000` 只提供 API，所以直接访问 `GET /` 或请求 `/favicon.ico` 返回 `404` 是预期行为，不代表后端启动失败。

如果路径校验通过，显式传入的模型目录会被写入项目根的 `config.toml`。TOML 里建议使用正斜杠：

```toml
[model]
path = "F:/InsightFace/models_cache/models/buffalo_l"

[recognition]
threshold = 0.30
```

也可以参考并复制仓库里的 [`config.example.toml`](/F:/facepass/config.example.toml)：

```toml
[model]
# Fill this with the buffalo_l model directory itself, not the models cache root.
path = "models/buffalo_l"

[recognition]
# Placeholder default. Replace this with a threshold chosen from real evaluation reports.
threshold = 0.30
```

后续未显式传 `--model-path` 时，后端会按 `CLI > GUI > config.toml` 的优先级解析模型路径。

当前 `InsightFaceModel` 默认显式使用 `CPUExecutionProvider`，因此在没有 CUDA 版 `onnxruntime` 的机器上也能按 CPU 跑通。

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

## 数据与评测

自采测试集标注沿用 JSONL，一行一张图：

```json
{"image_path":"images/group_01.jpg","faces":[{"identity_id":"p01","bbox":[12,34,80,80]},{"identity_id":"unknown","bbox":[120,40,76,76]}]}
```

其中 `bbox` 固定为 `[x, y, w, h]`。

目录约定需要特别说明：

- `dataset/`：真实实验数据目录，当前已纳入版本控制；其中 `dataset/registered` 已准备完毕，`dataset/test` 当前主要包含测试图。
- `data/`：仓库内测试与脚本 fixture 目录，主要放临时合成样本和 `.gitkeep`，不是正式数据集根目录。

目前仓库内有三类相关入口：

- `scripts/eval_self.py`：复用标注框裁剪后的单脸口径，适合先看识别本身是否区分开。
- `scripts/eval_end2end.py`：复用真实 `Recognizer.recognize_image()` 整图链路，做“检测框 ↔ 标注框 IoU 贪心配对 + 端到端 top-1”评测。
- `scripts/analyze_threshold.py`：只看注册集内部相似度分布，用来给 unknown 阈值找候选值。

单脸评测：

```powershell
uv run python scripts/eval_self.py `
  --annotations-path dataset/test/annotations.jsonl `
  --test-root dataset/test `
  --registered-root dataset/registered `
  --model-name insightface `
  --threshold 0.30
```

阈值分析：

```powershell
uv run python scripts/analyze_threshold.py `
  --registered-root dataset/registered `
  --model-name insightface `
  --histogram-path reports/threshold_hist.png
```

多脸端到端评测：

```powershell
uv run python scripts/eval_end2end.py `
  --annotations-path dataset/test/annotations.jsonl `
  --test-root dataset/test `
  --registered-root dataset/registered `
  --model-name insightface `
  --threshold 0.30
```

`eval_end2end.py` 会输出：

- `reports/end2end_eval.json`
- `reports/end2end_confusion_matrix.png`
- `reports/end2end_detection.png`
- `reports/end2end_accuracy.png`

端到端评测的指标口径：

- 检测层：`detection_recall`、`detection_precision`、`false_positives`
- 识别层：严格 `strict_top1_accuracy` 与宽松 `matched_top1_accuracy`
- `unknown`：标注为 unknown 的检出准确率，以及系统判成 unknown 的精确率

如果本机还没有准备 `dataset/test` 标注或 `dataset/registered` 注册集，`eval_end2end.py` 会打印提示并直接退出，不会抛异常。

启动建库时的几个常见 warning 目前属于预期行为：

- 某张注册图检测到多张人脸时，系统会记录 warning，并使用面积最大的一张做人脸注册。
- 图片像素总数超过 `25_000_000` 时，`safe_load_image` 会拒绝加载并跳过该图，日志里会显示“图片尺寸过大”。

## 错误处理

错误按三档处理：

- 致命错误：后端启动阶段模型加载失败、身份库不存在且无法构建、身份库为空，会记录清晰错误并以非零状态退出。
- 瞬时错误：前端请求后端、HF 下载等 IO/网络操作使用有限次数指数退避重试；耗尽后返回友好提示，不让前端进程退出。
- 可恢复输入错误：坏图片、空文件、非图片、单张注册图读取失败不重试；API 返回 400/413，建库时跳过坏图并继续。

## 本地模型与大文件

仓库不提交任何超过 50MB 的权重、构建好的身份库或大数组文件。`.gitignore` 已排除：

- `models/`
- `data/`
- `*.onnx`
- `*.pkl`
- `*.npy`
- `.venv/`
- Python 缓存和日志

当前约定是：模型目录由使用者自行下载并放在本地任意位置，再通过 `--model-path`、`FACEPASS_MODEL_PATH` 或 `config.toml` 指定。项目本身不负责下载、不依赖 Hugging Face 托管，也不会把模型目录纳入 git。

`dataset/` 与上述模型产物不同：当前仓库已经提交了 `dataset/identities.csv`、`dataset/registered/` 和部分 `dataset/test/` 图片。

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
- 多脸端到端评测的 IoU 配对、严格/宽松 top-1、unknown 指标、漏检/误检日志与 bench 脚本产图。
- 后端不导入具体模型实现，前端不导入任何 `src.*` 内部模块。

## 待填项

- 补齐 `dataset/test/annotations.jsonl`，让自采评测脚本可以直接按默认参数运行。
- 基于注册集内部相似度分布确定最终 unknown 阈值，不能用测试集调参。
- 最终提交前按课程要求补充打包脚本和报告。

## 小组成员

| 姓名 | 学号 | 分工 |
| --- | --- | --- |
|  |  |  |
|  |  |  |
|  |  |  |
