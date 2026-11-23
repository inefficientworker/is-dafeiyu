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

Binary image classifier that detects **Dafeiyu (大肥鱼 / DeepSeek whale-girl mascot)** in static images, animated GIF/WebP files, and directories — including group shots and chibi/logo variants.

Built on `vit_base_patch14_reg4_dinov2.lvd142m` (`timm`) with two-stage fine-tuning and temperature-scaled calibration optimized for **high precision** against hard negatives (blue-haired anime characters, fish/whales, and text/chat memes).

## Model Artifacts (`artifacts/dinov2_base/`)

| File | Description |
| :--- | :--- |
| `best.pt` | PyTorch checkpoint (best validation F0.5 / precision-calibrated) |
| `model.onnx` + `model.onnx.data` | Exported ONNX model with baked-in temperature-scaled sigmoid |
| `calibration.json` | Temperature (`0.2470`) and calibrated threshold (`0.5`) |
| `test_report.json` | Held-out test split evaluation metrics |
| `history.json` | Stage 1 & Stage 2 training metrics |

## Evaluation Results (Held-out Test Split, `n = 251`)

| Metric | Value |
| :--- | :--- |
| **Accuracy** | `0.9960` |
| **Precision** | `1.0000` (`0` false positives across 214 negatives) |
| **Recall** | `0.9730` (`36 / 37` positives) |
| **F1** | `0.9863` |
| **F0.5** | `0.9945` |
| **PR-AUC / ROC-AUC** | `1.0000` / `1.0000` |

Per-category test breakdown:
- `negative_blue_chars` (`n = 84`): `0` false positives (`FPR = 0.0`)
- `negative_fish` (`n = 46`): `0` false positives (`FPR = 0.0`)
- `negative_unrelated` (`n = 84`): `0` false positives (`FPR = 0.0`)
- `positive_dafeiyu` (`n = 37`): `36 / 37` detected (`Recall = 0.9730`)

## Quick Start

```bash
# Install dependencies (Python 3.12 via uv)
uv sync --python 3.12

# Run CLI inference on a single image, GIF/WebP, or directory
uv run python -m is_dafeiyu.predict /path/to/image_or_dir
uv run python -m is_dafeiyu.predict /path/to/image.png --threshold 0.75 --json
```

## Dataset & Copyright Notice

The curated dataset (`data/processed/` and `data/splits/`) contains 2,506 leak-free split samples (`2,004` train / `251` val / `251` test) stored in LFS.

All dataset images are included **for non-commercial research and educational use only**, with full source attribution and license details documented in [DATASET_NOTICE.md](./DATASET_NOTICE.md):
- **Positive & AI Mascot images**: From [EDMOK/blue-fish-archive](https://github.com/EDMOK/blue-fish-archive) ([fisharchive.cc](https://fisharchive.cc/)). Original whale-girl design by **上善无形**; DeepSeek maid whale-girl derivative design by **ZipZipPipe** under **CC BY-NC-SA 4.0**.
- **Anime hard negatives & illustrations**: From [Safebooru.org](https://safebooru.org/), rights belonging to respective artists and IP holders.
- **Real-world fish/whales & general images**: From [Wikimedia Commons](https://commons.wikimedia.org/) (respective CC / Public Domain licenses) and [Lorem Picsum](https://picsum.photos/) (Unsplash License).
- **Synthetic text memes**: Locally generated (`scripts/fetch_negatives.py`), CC0 / MIT.
