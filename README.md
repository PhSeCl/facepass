# FacePass — 人脸识别系统

## 简介

FacePass 是中山大学人工智能学院《人工智能原理实验》期末团队大作业，目标是实现一个可在本地运行的人脸识别系统。

## 项目结构

```text
facepass/
|-- celeba_100_identities_3reg_3test/
|   |-- register/
|   `-- test/
|-- dataset/
|   |-- annotations.jsonl
|   |-- identities.csv
|   |-- registered/
|   |   |-- p01/
|   |   |-- p02/
|   |   |-- p03/
|   |   |-- p04/
|   |   |-- p05/
|   |   |-- p06/
|   |   |-- p07/
|   |   |-- p08/
|   |   |-- p09/
|   |   |-- p10/
|   |   |-- p11/
|   |   |-- p12/
|   |   |-- p13/
|   |   |-- p14/
|   |   |-- p15/
|   |   |-- p16/
|   |   |-- p17/
|   |   |-- p18/
|   |   |-- p19/
|   |   `-- p20/
|   `-- test/
|       `-- images/
|-- docs/
|-- scripts/
|   `-- model/
|-- .gitignore
|-- main.py
|-- pyproject.toml
|-- README.md
`-- uv.lock
```

`scripts/model/` 用于存放本地模型权重文件，下载链接待补充。

`celeba_100_identities_3reg_3test/` 为课程说明中提供的 100 类 CelebA 裁剪人脸基准数据，可用于独立的 100 类识别测试。

## 环境配置

本项目使用 `uv` 管理 Python 依赖。

```powershell
uv sync
.venv\\Scripts\\Activate.ps1
```

如果使用 `cmd`，可改为：

```bat
.venv\\Scripts\\activate.bat
```

## 使用方法

待补充。

## 小组成员

| 姓名 | 学号 | 分工 |
| --- | --- | --- |
|  |  |  |
|  |  |  |
|  |  |  |
