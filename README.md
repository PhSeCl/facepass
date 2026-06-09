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
`-- frontend/
    |-- app.py           # 旧 Gradio 界面（已弃用，当前不启动）
    `-- static/
        `-- index.html   # 新 Web 前端，由后端 GET / 直接返回
scripts/
|-- run_dev.py           # 一键启动脚本（只启后端 + 内置前端）
|-- preannotate_test.py  # 为 dataset/test/images 生成预标注草稿
|-- summarize_test_annotations.py # 统计 test 标注人数并检查漏标/多余标注
|-- eval_self.py         # 单脸裁剪口径评测脚本
|-- analyze_threshold.py # 注册集相似度分布与阈值分析
`-- eval_end2end.py      # 多脸端到端评测与出图
dataset/
|-- identities.csv       # 身份 ID 到显示名映射，已纳入版本控制
|-- registered/          # 注册集：每个身份一个文件夹，内含注册照
`-- test/                # 自采测试图与标注目录
models/                  # gallery.pkl 等本地产物，不进 git
reports/                 # 评测 JSON / PNG 输出，本地产物，不进 git
tests/                   # 接口、边界和错误路径测试
requirements.txt         # pip 依赖列表
```

## 依赖

本项目使用 `uv` 管理依赖。默认依赖固定为 CPU 版 `onnxruntime==1.22.1`，这样 CPU-only 机器、测试环境和 CI 都能直接安装运行。较新的 `1.24.x` 在本项目的 CPython 3.10 Windows 环境没有可用 wheel。

```powershell
uv sync
```

如果你本机已经配好了 NVIDIA 驱动，可以只在**本地当前虚拟环境**里把 ORT 切到 GPU 版，而不用改仓库默认依赖。Windows 上建议把 ORT 和它依赖的 CUDA/cuDNN Python 运行库一起装进 `.venv`：

```powershell
uv pip uninstall onnxruntime
uv pip install "onnxruntime-gpu[cuda,cudnn]"
```

这一步是可选本地增强，不应直接改成项目默认依赖，否则会降低其他同学和 CI 的可移植性。不同 Python / Windows 组合下可用的 GPU wheel 版本可能不同，安装后请立刻用下面的诊断脚本确认 provider 是否真的起来。

后端默认配置在 `src/backend/config.py`，当前 `model_name="insightface"`。如果只想跑不依赖权重的后端集成测试，可以在测试里把配置切到 `fake`。

## 运行

运行 `insightface` 模型前，需要你自己准备本地 `buffalo_l` 目录。项目不会再自动下载或托管模型文件。当前适配器按"`buffalo_l` 模型目录本身"加载，不是传 `models_cache` 根目录。

```powershell
# 安装依赖
pip install -r requirements.txt

# 一键启动（后端 + Web 前端）
python scripts/run_dev.py --model-path F:\InsightFace\models_cache\models\buffalo_l
```

`run_dev.py` 只启动 FastAPI 后端，Web 前端由 `GET http://127.0.0.1:8000/` 直接返回。无需再单独启动 Gradio。

也可以手动启动后端：

```powershell
$env:FACEPASS_MODEL_PATH = "F:\InsightFace\models_cache\models\buffalo_l"
uv run uvicorn src.backend.api:app --port 8000
```

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

当前 `InsightFaceModel` 默认会优先尝试 `CUDAExecutionProvider`，不可用时自动回退到 `CPUExecutionProvider`。因此装好 GPU 版 `onnxruntime` 和匹配的 CUDA 环境后可以直接吃到 GPU；未配置时仍会按 CPU 跑通。

可以用下面的脚本检查当前环境是否真的暴露了 CUDA provider；如果同时传 `--model-path`，脚本还会把已加载 ONNX session 的 provider 打出来：

```powershell
uv run python scripts/check_runtime.py
uv run python scripts/check_runtime.py --model-path F:\InsightFace\models_cache\models\buffalo_l
```

如果输出里的 `runtime.available_providers` 和 `session_providers` 都包含 `CUDAExecutionProvider`，说明当前 FacePass 进程已经实际具备 GPU 推理能力。

如果你已经按上面的方式把 `.venv` 切到了 GPU 版 ORT，就不要再直接用会自动按锁文件回同步的默认 `uv run` 工作流；请改用当前虚拟环境里的解释器：

```powershell
.\.venv\Scripts\python.exe scripts/check_runtime.py --model-path F:\InsightFace\models_cache\models\buffalo_l
.\.venv\Scripts\python.exe scripts/run_dev.py --model-path F:\InsightFace\models_cache\models\buffalo_l
```

`InsightFaceModel` 在选择 `CUDAExecutionProvider` 时会先调用 `onnxruntime.preload_dlls(directory="")`，优先从 Python site-packages 里预加载 NVIDIA DLL，再创建 ONNX session；这样可以减少手工改系统 `PATH` 的需求，同时保持 CPU-only 默认安装不受影响。

健康检查：

```powershell
curl http://127.0.0.1:8000/health
```

## API

所有接口返回 JSON，错误时格式为 `{"message": "..."}` 。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 服务存活检查，返回 `{"status":"ok"}` |
| `GET` | `/identities` | 列出底库中所有已注册身份 |
| `POST` | `/recognize` | 上传图片，检测人脸并返回识别结果 |
| `POST` | `/register` | 上传单张图片，注册人脸到指定身份 |
| `POST` | `/register/batch` | 批量上传图片，注册到同一身份 |

（历史接口 `POST /dataset-eval/inspect`、`POST /dataset-eval/run` 保留不变，此处不展开。）

---

### `GET /health`

```
GET http://127.0.0.1:8000/health
→ {"status": "ok"}
```

---

### `GET /identities`

返回底库中全部身份信息：

```json
{
  "identities": [
    {
      "identity_id": "p01",
      "name": "成龙",
      "count": 1,
      "prototype_count": 1,
      "valid_image_count": 3
    }
  ]
}
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `identity_id` | `string` | 身份 ID |
| `name` | `string \| null` | 显示名（来自 `identities.csv`） |
| `count` | `int` | 兼容字段，等于 `prototype_count` |
| `prototype_count` | `int` | 原型向量数 |
| `valid_image_count` | `int` | 该身份的有效注册图数量 |

---

### `POST /recognize`

上传一张图片，检测其中所有人脸并返回识别结果。

**请求**：`multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `file` | `file` | 是 | 图片文件（JPG/PNG/WEBP，≤10MB） |

**成功响应 (200)**：

```json
[
  {
    "bbox": [120, 80, 200, 240],
    "identity_id": "p01",
    "name": "成龙",
    "similarity": 0.873,
    "is_unknown": false
  },
  {
    "bbox": [400, 60, 180, 220],
    "identity_id": "unknown",
    "name": null,
    "similarity": 0.215,
    "is_unknown": true
  }
]
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `bbox` | `[int,int,int,int]` | 人脸框 `[x, y, width, height]` |
| `identity_id` | `string` | 匹配到的身份 ID，未匹配时为 `"unknown"` |
| `name` | `string \| null` | 匹配身份的显示名，unknown 时为 `null` |
| `similarity` | `float` | 与底库最佳匹配的余弦相似度 |
| `is_unknown` | `bool` | 是否低于阈值被判定为 unknown |

**错误**：
- `400` — 图片格式无效或未检测到人脸
- `413` — 图片超过 10MB

---

### `POST /register`

上传单张图片，注册到指定身份。注册成功后自动重建底库，无需重启。

**请求**：`multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `file` | `file` | 是 | 人脸照片 |
| `identity_id` | `string` | 是 | 身份 ID，例如 `p21` |
| `name` | `string` | 否 | 显示名，会写入 `identities.csv` |

**成功响应 (200)**：

```json
{
  "identity_id": "p21",
  "name": "张三"
}
```

**错误**：
- `400` — 未检测到人脸
- `413` — 图片过大
- `503` — 识别器未初始化

---

### `POST /register/batch`

批量上传多张图片，全部注册到同一个身份。适合注册集整理时批量导入。

**请求**：`multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `files` | `file[]` | 是 | 多张人脸照片 |
| `identity_id` | `string` | 是 | 目标身份 ID |
| `name` | `string` | 否 | 显示名 |

**成功响应 (200)**：

```json
{
  "identity_id": "p21",
  "name": "张三",
  "saved": 3
}
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `identity_id` | `string` | 身份 ID |
| `name` | `string` | 显示名 |
| `saved` | `int` | 成功录入的图片数量 |

**错误**：
- `400` — 没有一张图片成功录入（全部没人脸或格式错误）
- `413` — 有图片超过 10MB
- `503` — 识别器未初始化

---

## 注册新身份（给负责增加数据的同学）

### 方式一：Web 界面（推荐）

启动服务后打开 `http://127.0.0.1:8000`，切换到「录入」标签页：
- 拖入或选择一人多张照片（支持批量）
- 填写身份 ID 和姓名
- 点击「录入底库」

### 方式二：Python 脚本批量注册

可以用 Python 脚本调用 `/register/batch` 接口，适合整理大量身份时使用：

```python
import requests
from pathlib import Path

API = "http://127.0.0.1:8000"

# 为 identity p21 注册 3 张照片
identity_dir = Path("dataset/registered/p21")
files = [
    ("files", (p.name, p.read_bytes(), "image/jpeg"))
    for p in identity_dir.glob("*.jpg")
]
resp = requests.post(
    f"{API}/register/batch",
    data={"identity_id": "p21", "name": "张三"},
    files=files,
)
print(resp.json())  # → {"identity_id": "p21", "name": "张三", "saved": 3}
```

### 注册集目录结构约定

```
dataset/
|-- identities.csv              # 身份映射表（注册脚本自动更新）
`-- registered/
    |-- p01/
    |   |-- p01_r01.jpg
    |   |-- p01_r02.jpg
    |   `-- p01_r03.jpg
    |-- p02/
    |   `-- ...
    `-- p21/                     # 新身份
        |-- p21_r01.jpg
        `-- p21_r02.jpg
```

`identities.csv` 格式为 `identity_id,name,domain`，每注册一个新身份时后端自动追加一行。

---

## 通过 API 编写批量测试脚本（给负责评测的同学）

### 核心思路

1. **启动服务** → `python scripts/run_dev.py --model-path <buffalo_l路径>`
2. **注册数据** → 通过 `/register/batch` 导入所有注册照
3. **遍历测试图** → 逐张调 `/recognize` 获取检测结果
4. **对比标注** → 拿 API 返回的 `bbox` / `identity_id` / `similarity` 与 `annotation.json` 做配对
5. **计算指标** → 统计 top-1 准确率、检测召回率、unknown 准确率等

### 简单示例：遍历测试图并收集结果

```python
import requests, json
from pathlib import Path

API = "http://127.0.0.1:8000"
results = {}

for img_path in Path("dataset/test/images").glob("*.jpg"):
    with open(img_path, "rb") as f:
        resp = requests.post(f"{API}/recognize", files={"file": f})
    results[img_path.name] = resp.json()

with open("results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
```

### 检查服务状态

```python
health = requests.get(f"{API}/health").json()
print(health)  # → {"status": "ok"}

# 确认底库中已有多少身份
identities = requests.get(f"{API}/identities").json()
print(len(identities["identities"]))  # → 已注册身份数
```

## 数据与评测

当前仓库内 `dataset/test` 的正式标注文件是 [dataset/test/annotation.json](/F:/facepass/dataset/test/annotation.json)，格式是“按图片名分组”的 JSON 对象：

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

说明：

- 顶层 key 必须等于图片文件名，例如 `group_01.jpg`、`p01_t01.jpg`。
- 每个 value 是该图里所有人脸的列表。
- `bbox` 固定为 `[x, y, w, h]`。
- `identity` 只能写 `p01`..`p20` 或 `unknown`。
- `score` 是预标注阶段保留的相似度参考值；评测会忽略它，但人工核对时应该保留。
- 现有加载器同时兼容 `.json` 和旧 `.jsonl`，但当前仓库标准是 `annotation.json`，后续不要再把 `dataset/test` 主标注写回 JSONL。

### 向 `dataset/test` 添加图片的必做流程

这是当前最容易出错的地方：**往 `dataset/test/images` 增加图片时，必须同步维护 `dataset/test/annotation.json`。只加图片、不加标注，后面任何人都不知道这张图是谁，也没法直接评测。**

推荐流程：

1. 把图片放进 [dataset/test/images](/F:/facepass/dataset/test/images)。
   - 单人照命名为 `pXX_tNN.jpg`，例如 `p06_t02.jpg`。
   - 合照命名为 `group_NN.jpg`。
   - 不要提交自己都看不懂含义的文件名。
2. 同一个提交里更新 [dataset/test/annotation.json](/F:/facepass/dataset/test/annotation.json)。
   - 顶层 key 必须和图片文件名完全一致。
   - 单人照也必须显式写 bbox 和 identity，不要靠文件名猜。
   - 不确定身份就先写 `unknown`，不要为了“看起来完整”硬写某个 `pXX`。
3. 如果是新加了一批图片，先跑一次预标注，再人工核对。

```powershell
uv run python scripts/preannotate_test.py `
  --images-dir dataset/test/images `
  --out dataset/test/annotation.json `
  --registered-root dataset/registered `
  --overwrite
```

4. 预标注只是草稿，必须人工检查后再提交。
   - 重点看低分项、单人照多脸、多人照漏脸。
   - 当前规则里，低于 `0.25` 的草稿身份会直接写成 `unknown`。
5. 提交前一定跑下面这个检查。

```powershell
uv run python scripts/summarize_test_annotations.py
```

如果输出里“缺少标注的图片”或“多余标注项”不是 `0`，先修 annotation，再提交图片。

目录约定需要特别说明：

- `dataset/`：真实实验数据目录，当前已纳入版本控制；其中 `dataset/registered` 已准备完毕，`dataset/test` 当前主要包含测试图。
- `data/`：仓库内测试与脚本 fixture 目录，主要放临时合成样本和 `.gitkeep`，不是正式数据集根目录。

Gradio 前端的“数据集演示”页现在支持两种输入：

- `ZIP`：上传较小的 `test.zip`。
- `文件夹`：直接输入本机绝对路径，例如 `F:\datasets\my_eval`。

文件夹模式不会弹系统原生资源管理器。这不是业务限制，而是纯浏览器 + Gradio 方案无法可靠暴露任意本机绝对目录路径；因此前端只把路径字符串发给后端，再由后端直接从本机磁盘读取。

文件夹模式只接受两种明确布局，不再递归猜目录：

- 选中的目录本身就是测试目录：目录下直接有 `images/` 与一个标注文件。
- 选中的是数据集根目录：目录下直接有 `test/images/` 与对应标注文件。

目前仓库内有三类相关入口：

- `scripts/preannotate_test.py`：对 `dataset/test/images` 跑预标注，生成待人工核对的 `annotation.json` 草稿。
- `scripts/summarize_test_annotations.py`：统计当前 test 标注里每个 `pXX` 的出现次数，并检查图片与标注是否一一对应。
- `scripts/eval_self.py`：复用标注框裁剪后的单脸口径，适合先看识别本身是否区分开。
- `scripts/eval_end2end.py`：复用真实 `Recognizer.recognize_image()` 整图链路，做“检测框 ↔ 标注框 IoU 贪心配对 + 端到端 top-1”评测。
- `scripts/analyze_threshold.py`：只看注册集内部相似度分布，用来给 unknown 阈值找候选值。

### 当前 `dataset/test` 标注人数统计

<!-- TEST_ANNOTATION_COUNTS:START -->
当前 [dataset/test/annotation.json](dataset/test/annotation.json) 统计如下：

- 标注图片数：`64`
- 标注人脸总数：`163`
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
| p15 | 3 |
| p16 | 4 |
| p17 | 3 |
| p18 | 3 |
| p19 | 3 |
| p20 | 4 |
| unknown | 97 |
<!-- TEST_ANNOTATION_COUNTS:END -->

单脸评测：

```powershell
uv run python scripts/eval_self.py `
  --annotations-path dataset/test/annotation.json `
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
  --annotations-path dataset/test/annotation.json `
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

`dataset/` 与上述模型产物不同：当前仓库已经提交了 `dataset/identities.csv`、`dataset/registered/`、`dataset/test/images/` 和 `dataset/test/annotation.json`。

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

- 持续维护 `dataset/test/annotation.json`，确保新增图片和标注始终同步提交。
- 基于注册集内部相似度分布确定最终 unknown 阈值，不能用测试集调参。
- 最终提交前按课程要求补充打包脚本和报告。

## 小组成员

| 姓名 | 学号 | 分工 |
| --- | --- | --- |
|  |  |  |
|  |  |  |
|  |  |  |
