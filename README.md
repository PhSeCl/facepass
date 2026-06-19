# FacePass — 本地人脸识别系统

FacePass 是一个在**本地运行**的人脸识别系统：上传图片 → 检测人脸 → 与身份库比对 → 在原图上标出人脸框、身份与相似度。检测与识别基于 [InsightFace](https://github.com/deepinsight/insightface) 的 `buffalo_l` 预训练模型，全程本地推理，不调用任何云端 API。

- **检索式识别**：注册照提取 embedding 建库，识别时做余弦最近邻 + 阈值判定，支持开集（库外人脸判为 `unknown`）。
- **三层架构**：模型端 / 后端（FastAPI）/ 前端（纯 HTML + JS），单进程即可运行。
- **开箱即用的 Web 界面**：识别、录入、底库、批量评估四个标签页。
- **CPU 默认可跑**，可选切换 GPU 加速。

---

## 快速开始

FacePass 需要 **Python 3.10+**，推荐用虚拟环境隔离依赖。

### 1. 创建环境并安装依赖

用标准 Python：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

或使用 [`uv`](https://docs.astral.sh/uv/)（推荐，更快）：

```powershell
uv sync
```

> 下文命令以 `uv run` 开头；如果用标准 Python 环境，去掉 `uv run`、直接用激活环境里的 `python` 即可（例如 `python scripts/run_dev.py`）。

### 2. 下载模型

FacePass 使用 InsightFace 的 `buffalo_l` 预训练模型（人脸检测 + 关键点对齐 + ArcFace 识别一体，约 326 MB，不随仓库分发）。用 insightface 自动下载（推荐）：

```powershell
uv run python -c "import insightface; insightface.app.FaceAnalysis(name='buffalo_l')"
```

模型会下载到 `~/.insightface/models/buffalo_l/`（Windows 为 `C:\Users\<用户名>\.insightface\models\buffalo_l\`）。

或手动下载 [`buffalo_l.zip`](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip) 解压，得到一个包含 5 个 `.onnx`（`det_10g`、`w600k_r50`、`1k3d68`、`2d106det`、`genderage`）的 `buffalo_l/` 目录。

### 3. 配置模型路径

在项目根新建 `config.toml`（可复制 [`config.example.toml`](config.example.toml)），把 `[model].path` 指向上一步那个**直接包含 `.onnx` 文件的 `buffalo_l` 目录**：

```toml
[model]
path = "C:/Users/<用户名>/.insightface/models/buffalo_l"   # TOML 建议用正斜杠

[recognition]
threshold = 0.30   # 余弦相似度阈值，低于此值判为 unknown
```

> 也可用命令行 `--model-path` 或环境变量 `FACEPASS_MODEL_PATH` 临时覆盖，优先级为 `CLI > 启动向导 > config.toml`。

### 4. 启动并打开浏览器

**双击仓库根的 `run.bat`** 即可一键启动（首次会有向导自动装依赖、选 CPU/GPU，无需手敲命令）。非 Windows 或需要命令行时，也可以用：

```powershell
uv run python scripts/run_dev.py
```

启动后访问 <http://127.0.0.1:8000>，在「人脸识别」页上传图片即可看到检测框与识别结果。注册集 `dataset/registered/`（p01–p20）已随仓库提供，启动即建库。

---

## 功能与界面

启动后访问 <http://127.0.0.1:8000>，Web 界面包含四个标签页：

| 标签页 | 功能 |
| --- | --- |
| 人脸识别 | 上传图片，检测所有人脸并标注身份、相似度，库外人脸标为 `unknown` |
| 录入人脸 | 一人多张照片批量注册为新身份，实时重建底库 |
| 身份库 | 查看当前已注册的全部身份及注册图数量 |
| 批量评估 | 上传 `test.zip` 或选本机数据集目录，跑端到端评测并内联展示混淆矩阵等图表 |

---

## 运行方式

### 双击 `run.bat`（Windows 一键）

双击仓库根的 `run.bat` 即可启动，无需手敲命令。它是一个瘦壳，会找到一个可用的 Python 去运行启动向导 `scripts/launcher.py`，由向导完成首次配置：检测 `uv` / `.venv` 环境、按需安装依赖、选择 CPU 或 GPU，并把选择写入 `config.toml` 的 `[runtime]` 表，之后直接据此启动。需要重新选择时运行 `run.bat --reconfigure`。

> 进程异常退出时窗口会停下显示错误码，便于排查。启动后浏览器打开 <http://127.0.0.1:8000>。

### 命令行启动

```powershell
# 启动后端（Web 前端由后端 GET / 直接返回，单进程）
uv run python scripts/run_dev.py

# 临时指定模型目录
uv run python scripts/run_dev.py --model-path D:/models/buffalo_l

# 健康检查
curl http://127.0.0.1:8000/health
```

### GPU 加速（可选）

`InsightFaceModel` 默认优先尝试 `CUDAExecutionProvider`，不可用时自动回退到 CPU。要启用 GPU，在当前虚拟环境里把 onnxruntime 换成 GPU 版（不要改动仓库默认依赖，以免影响其他环境与 CI 的可移植性）：

```powershell
uv pip uninstall onnxruntime
uv pip install "onnxruntime-gpu[cuda,cudnn]"
```

`run.bat` 向导选择 GPU 时会用独立的 `.venv-gpu`（约 2.8 GB，含 CUDA/cuDNN 运行库），与项目 `.venv` 隔离。用下面的脚本确认 GPU provider 是否真正生效：

```powershell
uv run python scripts/check_runtime.py --model-path D:/models/buffalo_l
```

输出中 `available_providers` 与 `session_providers` 都包含 `CUDAExecutionProvider` 即代表 GPU 推理可用。不再使用 GPU 时，删除 `.venv-gpu` 目录即可回收空间，再运行 `run.bat --reconfigure` 把设备切回 CPU。

---

## API 参考

所有接口返回 JSON，错误时为 `{"message": "..."}`，并带稳定的 Pydantic `response_model`。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/` | 返回 Web 前端页面 |
| `GET` | `/health` | 存活检查，返回 `{"status":"ok"}` |
| `GET` | `/identities` | 列出底库中所有已注册身份 |
| `POST` | `/recognize` | 上传图片，检测人脸并返回识别结果 |
| `POST` | `/register` | 上传单张图片，注册到指定身份 |
| `POST` | `/register/batch` | 批量上传图片，注册到同一身份 |
| `POST` | `/dataset-eval/inspect` | 检查上传的 `test.zip` 或本机数据集目录布局是否合法 |
| `POST` | `/dataset-eval/run` | 对数据集跑端到端评测，返回指标与内联图表 |

### `POST /recognize`

`multipart/form-data`，参数 `file`（图片，JPG/PNG/WEBP，≤10 MB）。返回图中每张人脸的识别结果：

```json
[
  {"bbox": [120, 80, 200, 240], "identity_id": "p01", "name": "成龙", "similarity": 0.873, "is_unknown": false},
  {"bbox": [400, 60, 180, 220], "identity_id": "unknown", "name": null, "similarity": 0.215, "is_unknown": true}
]
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `bbox` | `[int,int,int,int]` | 人脸框 `[x, y, width, height]`（绝对像素） |
| `identity_id` | `string` | 匹配身份 ID，未匹配为 `"unknown"` |
| `name` | `string \| null` | 显示名（来自 `identities.csv`），unknown 时为 `null` |
| `similarity` | `float` | 与底库最佳匹配的余弦相似度 |
| `is_unknown` | `bool` | 是否低于阈值被判为 unknown |

错误：`400`（图片无效 / 未检测到人脸）、`413`（超过 10 MB）。

### 注册接口

- `POST /register`：表单 `file` + `identity_id`（如 `p21`，仅限字母数字下划线连字符）+ 可选 `name`。注册后自动重建底库，无需重启。
- `POST /register/batch`：表单 `files`（多张）+ `identity_id` + 可选 `name`，返回 `{"identity_id", "name", "saved"}`。

错误：`400`（图片无效 / `identity_id` 非法 / 无一张成功）、`413`（图片过大）、`503`（识别器未初始化）。

### 数据集评测接口

- `inspect`：仅校验布局，返回 `{"has_registered": bool}`。
- `run`：表单 `gallery_choice` 选 `local`（仓库注册集）或数据集自带注册集建库；返回评测指标、混淆对、漏检/误检列表，以及内联（base64）的混淆矩阵、检测、准确率三张图。

前端「批量评估」页已接入这两个接口：支持上传 `test.zip`，或填入本机数据集目录绝对路径（超大数据集免上传）。目录模式接受两种布局：选中目录本身就是测试目录（直接含 `images/` 与标注文件），或选中数据集根目录（含 `test/images/` 与标注文件）。脚本化评测见[评测](#评测)。

---

## 项目结构

```text
src/
├── common/        # 日志、重试、安全图片解码、通用异常
├── face_model/    # 模型端：FaceModel 抽象、DetectedFace、InsightFace / Fake 实现
├── backend/       # 后端：Gallery、Recognizer、FastAPI API、配置、外部数据集导入
├── eval/          # 评测内核：CelebA / 单脸 / 端到端数据集、指标与出图，脚本和后端复用
└── frontend/static/index.html   # Web 前端（纯 HTML + JS），由后端 GET / 返回
scripts/           # 启动向导、评测与数据维护脚本（见下文）
dataset/           # 自采数据集：identities.csv、registered/（p01–p20）、test/
celeba_100_identities_3reg_3test/   # CelebA 100 类子集（各 3 注册 3 测试，裁剪脸）
tests/             # 接口、边界与错误路径测试
run.bat            # Windows 一键启动入口
config.example.toml
```

模型层有两种实现：`insightface`（默认，真实推理）与 `fake`（测试替身，不下载权重，用于 CI / 单元测试解耦后端与模型）。

`scripts/` 下的主要入口：

| 脚本 | 用途 |
| --- | --- |
| `run_dev.py` | 启动后端服务 |
| `launcher.py` | `run.bat` 调用的首次启动向导 |
| `check_runtime.py` | 诊断 onnxruntime provider（确认是否吃到 GPU） |
| `preannotate_test.py` | 为 `dataset/test/images` 生成预标注草稿 |
| `json2jsonl.py` | 由 `annotation.json` 生成规范的 `annotations.jsonl` |
| `summarize_test_annotations.py` | 统计 test 标注并校验图片与标注一一对应 |
| `eval_self.py` / `eval_end2end.py` / `eval_celeba.py` | 单脸 / 端到端 / CelebA 评测 |
| `analyze_threshold.py` | 注册集相似度分布与阈值分析 |
| `check_celeba_leakage.py` | 检查 CelebA register/test 是否有重复文件 |

---

## 数据与评测

### 数据集结构

```text
dataset/
├── identities.csv     # identity_id,name,domain 映射表，注册时自动 upsert
├── registered/        # 注册集：p01/ … p20/，每身份多张 pXX_rNN.jpg
└── test/
    ├── images/        # 测试图：单人照 pXX_tNN.jpg、合照 group_NN.jpg
    ├── annotation.json    # 人工维护的主标注（可读、分组式）
    └── annotations.jsonl  # 机器生成的规范标注（见下）
```

### 标注格式

人工维护的主文件是 **`annotation.json`**（按图片名分组，便于阅读与人工核对）：

```json
{
  "group_01.jpg": [
    {"bbox": [12, 34, 80, 80], "identity": "p01", "score": 0.71},
    {"bbox": [120, 40, 76, 76], "identity": "unknown", "score": 0.08}
  ],
  "p01_t01.jpg": [
    {"bbox": [220, 65, 140, 180], "identity": "p01", "score": 0.93}
  ]
}
```

- 顶层 key 为图片文件名；value 是该图所有人脸的列表。
- `bbox` 为 `[x, y, w, h]` 绝对像素整数；`identity` 取 `p01`–`p20` 或 `unknown`。
- `score` 是预标注阶段的相似度参考值，评测忽略，人工核对时保留。

规范的 **`annotations.jsonl`** 由 `scripts/json2jsonl.py` 从 `annotation.json` **自动生成**（字段重命名为 `image` / `image_type` / `faces[].identity_id`，并按图像真实尺寸把越界 bbox 钳制到 `[0, 0, W, H]`）。**不要手改 jsonl**——推送 `annotation.json` 到 `main` 后，GitHub Actions 会自动重新生成并提交 jsonl。本地手动同步：

```powershell
uv run python scripts/json2jsonl.py          # 生成
uv run python scripts/json2jsonl.py --check  # 仅校验是否同步（CI 用）
```

### 添加图片或身份

- **新增测试图**：把图片放进 `dataset/test/images/`（单人照 `pXX_tNN.jpg`、合照 `group_NN.jpg`），同一提交里更新 `annotation.json`。可先跑 `preannotate_test.py` 生成草稿再人工核对，提交前用 `summarize_test_annotations.py` 校验图片与标注一一对应。
- **新增身份**：用 Web「录入人脸」页批量上传，或调用 `/register/batch`：

```python
import requests
from pathlib import Path

files = [("files", (p.name, p.read_bytes(), "image/jpeg"))
         for p in Path("dataset/registered/p21").glob("*.jpg")]
requests.post("http://127.0.0.1:8000/register/batch",
              data={"identity_id": "p21", "name": "示例"}, files=files)
```

### 评测

所有评测脚本读取 `dataset/`，把指标 JSON 与图表写到 `reports/`（不进 git）。

```powershell
# 自采 20 类端到端评测（检测 + 识别 + 阈值判定整图链路）
uv run python scripts/eval_end2end.py `
  --annotations-path dataset/test/annotation.json `
  --test-root dataset/test --registered-root dataset/registered `
  --model-name insightface --threshold 0.30

# CelebA 100 类纯识别 top-1 评测（裁剪脸）
uv run python scripts/eval_celeba.py --data-dir celeba_100_identities_3reg_3test
```

- `eval_end2end.py` 输出 `reports/end2end_eval.json` 与混淆矩阵、检测、准确率三张图；指标含检测召回/精确率、严格/宽松 top-1、unknown 检出准确率与精确率。
- `eval_celeba.py` 输出 top-1 准确率、逐类准确率与成功/失败样本；可先用 `check_celeba_leakage.py` 确认 register/test 无重复图片。
- `eval_self.py` 走标注框裁剪后的单脸口径；`analyze_threshold.py` 分析注册集相似度分布以选取 unknown 阈值（不得用测试集调参）。

> 建库时若一张注册图含多张人脸，系统取面积最大的一张并记 warning；像素总数超过 2500 万的图会被跳过。这些均为预期行为。

### 当前 `dataset/test` 标注统计

<!-- TEST_ANNOTATION_COUNTS:START -->
当前 [dataset/test/annotation.json](dataset/test/annotation.json) 统计如下：

- 标注图片数：`66`
- 标注人脸总数：`165`
- 其中 `unknown`：`97`
- 图片目录：`dataset/test/images`
- 缺少标注的图片：`0`
- 多余标注项：`0`

可用下面的命令重新生成本节：

```powershell
uv run python scripts/summarize_test_annotations.py --write-readme
```

如果上面两项不是 `0`，先修复 `dataset/test/annotation.json`，再继续评测或提交图片。

| 身份 | 标注人数 |
| --- | ---: |
| p01 | 4 |
| p02 | 3 |
| p03 | 3 |
| p04 | 4 |
| p05 | 3 |
| p06 | 4 |
| p07 | 4 |
| p08 | 3 |
| p09 | 3 |
| p10 | 3 |
| p11 | 3 |
| p12 | 3 |
| p13 | 3 |
| p14 | 3 |
| p15 | 4 |
| p16 | 5 |
| p17 | 3 |
| p18 | 3 |
| p19 | 3 |
| p20 | 4 |
| unknown | 97 |
<!-- TEST_ANNOTATION_COUNTS:END -->

---

## 测试

```powershell
uv run pytest tests -q
```

测试覆盖：Gallery 建库与匹配、Recognizer 阈值与 unknown 判定、API 各路径与错误码、`safe_load_image` 与重试逻辑、`FakeFaceModel` 全链路、端到端/CelebA 评测与各脚本、`json2jsonl` 转换与边界钳制、层边界（后端不导入具体模型、前端不导入 `src.*`）等。

错误按三档处理：

- **致命错误**（模型加载失败、身份库无法构建或为空）：记录清晰日志并以非零状态退出。
- **瞬时错误**（IO/网络）：有限次指数退避重试，耗尽后返回友好提示，不崩进程。
- **可恢复输入错误**（坏图、空文件、非图片）：不重试，API 返回 400/413，建库时跳过坏图继续。

---

## 大文件约定

仓库不提交超过 50 MB 的权重、构建好的身份库或大数组文件。`.gitignore` 已排除 `models/`、`data/`、`*.onnx`、`*.pkl`、`*.npy`、`.venv/` 及缓存日志。模型目录由使用者自行下载（见[快速开始](#快速开始)）。`dataset/` 与 CelebA 子集（裁剪脸、单图较小）已纳入版本控制。

---

## 后续可改进

- 基于注册集内部相似度分布进一步标定 unknown 阈值（当前使用默认 `0.30`，且严禁用测试集调参）。
- 扩充自采数据集的身份数与每类样本量，提升评测的统计可靠性。

---

## 小组成员

本项目由以下成员共同完成：

- [@PhSeCl](https://github.com/PhSeCl)
- [@Antii-claude](https://github.com/Antii-claude)
- [@Anony-mous-210](https://github.com/Anony-mous-210)
