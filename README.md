---
license: cc-by-nc-4.0
pipeline_tag: image-classification
library_name: timm
tags:
  - image-classification
  - binary-classification
  - dinov2
  - onnx
  - anime
  - dafeiyu
  - deepseek-mascot
---

# is-dafeiyu

- **GitHub**：<https://github.com/inefficientworker/is-dafeiyu>
- **Hugging Face**：<https://huggingface.co/inefficentworker/is-dafeiyu>

用于检测图片中是否包含 **大肥鱼（鲸鱼娘 / DeepSeek 吉祥物）** 的二分类图像识别模型，支持静态图片、GIF/WebP 动图及目录批量推理，覆盖合影及 Q 版/Logo 变体。

模型基于 `vit_base_patch14_reg4_dinov2.lvd142m`（`timm`）进行两阶段微调，并在验证集上完成温度缩放（Temperature Scaling）与高精度阈值校准，针对三类困难负样本（蓝发二次元角色、真实/插画鱼类鲸类、文本/聊天表情包）进行了低误报优化。

## 模型产物（`artifacts/dinov2_base/`）

| 文件 | 说明 |
| :--- | :--- |
| `best.pt` | PyTorch 训练检查点（验证集 F0.5 / 高精度校准最优模型） |
| `model.onnx` + `model.onnx.data` | 导出的 ONNX 模型（已内置温度缩放 Sigmoid） |
| `calibration.json` | 校准温度（`0.2470`）与分类阈值（`0.5`） |
| `test_report.json` | 测试集评估报告 |
| `history.json` | 第一、二阶段训练指标记录 |

## 测试集评估结果（`n = 251`）

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
uv run python -m is_dafeiyu.predict /path/to/image_or_dir
uv run python -m is_dafeiyu.predict /path/to/image.png --threshold 0.75 --json
```

## 数据集与版权声明

仓库中的清洗数据集（`data/processed/` 与 `data/splits/`）共包含 2,506 条无泄漏划分样本（`train` 2,004 / `val` 251 / `test` 251），已通过 LFS 存储。

数据集中的所有图片**仅限非商业研究、学习与分类器评估使用**，完整来源与版权声明见 [DATASET_NOTICE.md](./DATASET_NOTICE.md)：
- **正样本与 AI 拟人角色负样本**：来自 [EDMOK/blue-fish-archive](https://github.com/EDMOK/blue-fish-archive)（[fisharchive.cc](https://fisharchive.cc/)）。鲸鱼娘原设作者为 **上善无形**，DeepSeek 女仆鲸鱼娘二设作者为 **ZipZipPipe**，依 **CC BY-NC-SA 4.0** 协议授权。
- **二次元困难负样本与插画**：来自 [Safebooru.org](https://safebooru.org/)，版权归各自画师与原 IP 权利方所有。
- **真实鱼类/鲸类与自然照片**：来自 [Wikimedia Commons](https://commons.wikimedia.org/)（各自 CC / 公共领域协议）及 [Lorem Picsum](https://picsum.photos/)（Unsplash License）。
- **合成文本表情包**：由 `scripts/fetch_negatives.py` 本地生成（CC0 / MIT）。
