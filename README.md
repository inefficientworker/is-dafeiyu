---
license: cc-by-nc-4.0
pipeline_tag: image-classification
library_name: timm
tags:
  - image-classification
  - binary-classification
  - dinov2
  - mobilenetv3
  - convnext
  - onnx
  - onnxruntime-web
  - anime
  - dafeiyu
  - deepseek-mascot
---

# is-dafeiyu

- **GitHub**：<https://github.com/inefficientworker/is-dafeiyu>
- **Hugging Face**：<https://huggingface.co/inefficentworker/is-dafeiyu>
- **浏览器在线测试（GitHub Pages）**：<https://inefficientworker.github.io/is-dafeiyu/>

用于检测图片中是否包含 **大肥鱼（鲸鱼娘 / DeepSeek 吉祥物）** 的二分类图像识别模型，支持静态图片、GIF/WebP 动图、目录批量推理以及**浏览器端 WASM 离线推理**，覆盖合影及 Q 版/Logo 变体。

初始版本使用 DINOv2 教师与蒸馏学生模型。当前流程采用独立的 train / val / calibration / test 划分、整图增强、冻结模块状态保护与高精度约束；真实鱼鲸照片并入无关负类，训练默认保留约 60 张，重点评估易混淆角色。具体训练方法见 [训练流程](docs/training-pipeline.md)。

## 当前候选模型

`artifacts/mobilenetv3_user_20261123/` 提供使用新增 66 张人工标签图片训练的候选版本：`best.pt`、`calibration.json`、`model_browser_fp16.onnx` 与 `model_browser_fp16.calibration.json`。数据与权重可从 Hugging Face 下载，GitHub 保存代码。

FP16 ONNX 阈值为 `0.42624813318252563`，必须使用配套校准文件。相同的 265 张测试图上，误报 1/225、漏报 0/40，Precision 97.56%、Recall 100%；已知 X 回归集仍有 1/4 负样本误报。该版本未达到 98% Precision 目标，另有新样本回退，作为候选发布，不设为浏览器默认模型。

```bash
uv run python -m is_dafeiyu.predict images/example.png \
  --checkpoint artifacts/mobilenetv3_user_20261123/best.pt --json
```

## 历史浏览器轻量 ONNX 模型（`onnx/`）

以下指标来自初始版本的历史测试集，不代表当前扩展数据集或实际使用中的误报率。

所有浏览器模型均为**单文件 `.onnx`**（无 `.onnx.data`），并已将 **RGBA `uint8 [batch, 224, 224, 4]` 白底透明合成 + `/255` + ImageNet `mean/std` 归一化 + `NHWC → NCHW` + 温度缩放 Sigmoid** 全部烘焙进计算图，浏览器端直接传入 Canvas `ImageData` 即可推理：

| 模型 | 文件 | 体积 | 推荐阈值 | CPU 单图耗时 | Precision | Recall | F1 | PR-AUC | FP / FN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Large (FP16)** *(推荐)* | `onnx/mobilenetv3_large_100_fp16.onnx` | **8.05 MB** | `0.70` | **3.1 ms** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0 / 0** |
| **ConvNeXt-Atto (INT8 静态量化)** *(极致超小)* | `onnx/convnext_atto_int8_static.onnx` | **3.59 MB** | `0.50` | 5.9 ms | **1.0000** | 0.9730 | 0.9863 | 0.9993 | **0 / 1** |
| ConvNeXt-Atto (FP16) | `onnx/convnext_atto_fp16.onnx` | 6.52 MB | `0.50` | 7.7 ms | 1.0000 | 0.9730 | 0.9863 | 1.0000 | 0 / 1 |
| ConvNeXt-Atto (FP32) | `onnx/convnext_atto_fp32.onnx` | 12.93 MB | `0.50` | 5.6 ms | 1.0000 | 0.9730 | 0.9863 | 1.0000 | 0 / 1 |
| MobileNetV3-Large (FP32) | `onnx/mobilenetv3_large_100_fp32.onnx` | 16.03 MB | `0.70` | 3.0 ms | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 / 0 |

## 教师模型产物（`artifacts/dinov2_base/`）

| 文件 | 说明 |
| :--- | :--- |
| `best.pt` | PyTorch 训练检查点（验证集 F0.5 / 高精度校准最优模型） |
| `model.onnx` + `model.onnx.data` | 导出的 ONNX 模型（已内置温度缩放 Sigmoid） |
| `calibration.json` | 校准温度（`0.2470`）与分类阈值（`0.5`） |
| `test_report.json` | 测试集评估报告 |
| `history.json` | 第一、二阶段训练指标记录 |

### 教师模型测试集评估结果（`n = 251`）

| 指标 | 数值 |
| :--- | :--- |
| **Accuracy** | `0.9960` |
| **Precision** | `1.0000`（214 张负样本 `0` 误报） |
| **Recall** | `0.9730`（37 张正样本检出 `36` 张） |
| **F1** | `0.9863` |
| **F0.5** | `0.9945` |
| **PR-AUC / ROC-AUC** | `1.0000` / `1.0000` |

各类别测试集明细：
- `negative_blue_chars`（`n = 84`）：`0` 误报（`FPR = 0.0`）
- `negative_fish`（`n = 46`）：`0` 误报（`FPR = 0.0`）
- `negative_unrelated`（`n = 84`）：`0` 误报（`FPR = 0.0`）
- `positive_dafeiyu`（`n = 37`）：检出 `36 / 37`（`Recall = 0.9730`）

## 快速开始

```bash
# 使用 uv 安装依赖（Python 3.12）
uv sync --python 3.12

# 对单张图片、GIF/WebP 动图或目录进行推理
uv run python -m is_dafeiyu.predict images/
uv run python -m is_dafeiyu.predict images/example.png --threshold 0.75 --json
```

### MobileNetV3 困难负样本验证

`configs/mobilenetv3_large.yaml` 使用硬标签训练，产物写入 `artifacts/mobilenetv3_large_100_hard_v2/`。重建划分后，在固定划分上训练教师，再用 `configs/distill_mobilenetv3_large.yaml` 对相同增强图像做在线蒸馏对照。对于已审核为 `negative_gemini` 的样本，如果教师判为正类，蒸馏版也只使用硬标签。

本地人工核对的样本通过 `data/curated/meta.jsonl` 加入划分。每行包含 `path`、`label`、`category`、`subcategory` 和 `group_id`，同一原图的变体共用 `group_id`。纯 Gemini 图使用 `label: 0`、`category: negative_blue_chars` 和 `subcategory: negative_gemini`。

检查 `calibration.json` 的 `calibration_by_subcategory` 和评估输出的 `by_subcategory`，分别看 Gemini 在独立校准集和测试集上的样本数与误报数。重新导出 FP16 后，用实际网页回归 Gemini 梗图、大肥鱼 Q 版和合影。

## 数据集与版权声明

当前数据集在 Hugging Face 保存，共 2,429 条划分样本（`train` 1,642 / `val` 262 / `calibration` 260 / `test` 265），其中本轮新增 66 张（31 正 / 35 负）。另保留未抽入训练的原图和 5 张已知 X 回归图；回归图不参与模型训练。划分及来源记录均使用相对路径；以 `data/splits/` 中的 CSV 为准。

数据集中的所有图片**仅限非商业研究、学习与分类器评估使用**，完整来源与版权声明见 [DATASET_NOTICE.md](./DATASET_NOTICE.md)：
- **正样本与 AI 拟人角色负样本**：来自 [EDMOK/blue-fish-archive](https://github.com/EDMOK/blue-fish-archive)（[fisharchive.cc](https://fisharchive.cc/)）。鲸鱼娘原设作者为 **上善无形**，DeepSeek 女仆鲸鱼娘二设作者为 **ZipZipPipe**，依 **CC BY-NC-SA 4.0** 协议授权。
- **二次元困难负样本与插画**：来自 [Safebooru.org](https://safebooru.org/)，版权归各自画师与原 IP 权利方所有。
- **真实鱼类/鲸类与自然照片**：来自 [Wikimedia Commons](https://commons.wikimedia.org/)（各自 CC / 公共领域协议）及 [Lorem Picsum](https://picsum.photos/)（Unsplash License）。
- **合成文本表情包**：由 `scripts/fetch_negatives.py` 本地生成（CC0 / MIT）。
