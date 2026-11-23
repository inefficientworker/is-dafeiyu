---
license: cc-by-nc-4.0
pipeline_tag: image-classification
library_name: timm
tags:
  - image-classification
  - binary-classification
  - mobilenetv3
  - onnx
  - onnxruntime-web
  - dafeiyu
  - deepseek-mascot
---

# is-dafeiyu

检测图片中是否出现 **大肥鱼（鲸鱼娘 / DeepSeek 吉祥物）**，支持表情包、合影、Q 版和 Logo。

**直接使用 [在线检测器](https://inefficientworker.github.io/is-dafeiyu/)**：点击下载模型后即可检测，图片在浏览器内处理。

- [GitHub 代码](https://github.com/inefficientworker/is-dafeiyu)
- [Hugging Face 模型与数据](https://huggingface.co/inefficentworker/is-dafeiyu)

## 模型

只维护一个默认模型：**MobileNetV3-Large，8.05 MB FP16 ONNX**。网页和命令行使用同一组训练权重。

文件位于 Hugging Face 的 `artifacts/mobilenetv3_user_20261123/`：

| 文件 | 用途 |
| --- | --- |
| `model_browser_fp16.onnx` | 浏览器或 ONNX Runtime 推理 |
| `model_browser_fp16.calibration.json` | ONNX 配套阈值与文件哈希 |
| `best.pt` | Python 推理和继续训练 |
| `calibration.json` | PyTorch 配套温度与阈值 |

ONNX 输入为 `uint8 [batch, 224, 224, 4]` RGBA，透明合成、归一化和温度缩放已内置。默认阈值 **0.42624813318252563**，网页自动读取配套校准文件。

当前 265 张测试图：负样本 **1/225 误报**，正样本 **0/40 漏报**，Precision **97.56%**，Recall **100%**。已知 X 回归集有 1 张误报；持续补充易混淆角色进行改进。

## Python 使用

```bash
uv sync --python 3.12
uv run hf download inefficentworker/is-dafeiyu \
  --include 'artifacts/mobilenetv3_user_20261123/*' --local-dir .
uv run python -m is_dafeiyu.predict images/ --json
```

默认加载上述 `best.pt` 及其校准文件，无需选模型。

## 训练

只有一个配置文件 `configs/default.yaml`。下载数据后运行：

```bash
uv run hf download inefficentworker/is-dafeiyu --include 'data/*' --local-dir .
uv run python scripts/build_dataset.py
uv run python -m is_dafeiyu.train
uv run python -m is_dafeiyu.evaluate
uv run python -m is_dafeiyu.export \
  --checkpoint artifacts/mobilenetv3_v2/best.pt --browser --quantize fp16
```

训练产物写入 `artifacts/mobilenetv3_v2/`。训练使用独立的 train / val / calibration / test 划分、整图增强、冻结模块状态保护与高精度约束，详情见 [训练流程](docs/training-pipeline.md)。

## 数据集与版权声明

当前数据集在 Hugging Face 保存，共 2,429 条划分样本（`train` 1,642 / `val` 262 / `calibration` 260 / `test` 265），其中本轮新增 66 张（31 正 / 35 负）。另保留未抽入训练的原图和 5 张已知 X 回归图；回归图不参与模型训练。划分及来源记录均使用相对路径；以 `data/splits/` 中的 CSV 为准。

数据集中的所有图片**仅限非商业研究、学习与分类器评估使用**，完整来源与版权声明见 [DATASET_NOTICE.md](./DATASET_NOTICE.md)：
- **正样本与 AI 拟人角色负样本**：来自 [EDMOK/blue-fish-archive](https://github.com/EDMOK/blue-fish-archive)（[fisharchive.cc](https://fisharchive.cc/)）。鲸鱼娘原设作者为 **上善无形**，DeepSeek 女仆鲸鱼娘二设作者为 **ZipZipPipe**，依 **CC BY-NC-SA 4.0** 协议授权。
- **二次元困难负样本与插画**：来自 [Safebooru.org](https://safebooru.org/)，版权归各自画师与原 IP 权利方所有。
- **真实鱼类/鲸类与自然照片**：来自 [Wikimedia Commons](https://commons.wikimedia.org/)（各自 CC / 公共领域协议）及 [Lorem Picsum](https://picsum.photos/)（Unsplash License）。
- **合成文本表情包**：由 `scripts/fetch_negatives.py` 本地生成（CC0 / MIT）。
