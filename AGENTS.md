# AGENTS.md

## Project Overview

`is-dafeiyu` is a binary image classifier that detects whether an image contains **Dafeiyu (大肥鱼 / DeepSeek whale-girl mascot)**, including group shots and chibi/logo whale variants.

- **Label definition**: `1` (positive) if Dafeiyu appears anywhere in the image; `0` (negative) otherwise.
- **Optimization priority**: **Low false positives (high Precision)** against three explicit negative buckets:
  - `negative_blue_chars`: blue-haired anime/cartoon characters (e.g. Gawr Gura, Suisei, Miku, Rem, Furina, Aqua, Cirno, Arona, blue chibis, and other AI-mascot characters).
  - `negative_fish`: real fish, blue tangs, blue whales, sharks, dolphins, and illustrated fish/whales.
  - `negative_unrelated`: real-world photos, non-blue anime/chibis, and text/chat memes (including tricky keywords like "DeepSeek" / "已思考 13 秒" to prevent OCR shortcuts).

## Environment & Hardware

- **Package manager & Python**: Managed by `uv` with **Python 3.12** (`>=3.12,<3.13`).
- **Accelerator**: Local Intel Arc XPU (`torch` + `torchvision` from `https://download.pytorch.org/whl/xpu`, native `torch.xpu` — no legacy IPEX required). Training uses `bfloat16` autocast (`torch.amp.autocast(device_type="xpu", dtype=torch.bfloat16)`).
- Verify XPU availability:
  ```bash
  uv run python -c "import torch; assert torch.xpu.is_available(); print(torch.xpu.get_device_name(0))"
  ```

## Repository Layout

- `configs/default.yaml`: Training, model, data, and calibration configuration.
- `scripts/fetch_positives.py`: Clones `EDMOK/blue-fish-archive` (`--depth=1`) and extracts deduplicated positive frames + in-repo other AI-mascot negatives.
- `scripts/fetch_negatives.py`: Collects negative images from Wikimedia Commons, `safebooru.org`, Picsum, and local synthetic text-meme generation.
- `scripts/build_dataset.py`: Computes `phash`, drops exact/cross-label duplicates, and builds leak-free 80/10/10 `StratifiedGroupKFold` splits under `data/splits/`.
- `src/is_dafeiyu/`:
  - `dataset.py`: PyTorch `Dataset`, alpha-channel background compositing, and augmentations.
  - `model.py`: `timm` backbone wrapper, XPU device resolution, and two-stage freeze/unfreeze helpers.
  - `train.py`: Stage 1 (linear probe) + Stage 2 (partial unfreeze) training loop on XPU.
  - `evaluate.py`: Temperature scaling + high-precision threshold calibration on `val`, and per-category FPR reporting on `test`.
  - `predict.py`: CLI inference for static images, animated GIF/WebP (max probability across frames), and directories.
  - `export.py`: ONNX export with baked-in temperature-scaled sigmoid + numerical parity check.
- `data/` and `artifacts/`: Git-ignored datasets, splits, checkpoints (`best.pt`), calibration (`calibration.json`), reports (`test_report.json`), and exported ONNX (`model.onnx`).

## Common Commands

```bash
# Install / sync dependencies (Python 3.12 + PyTorch XPU)
uv sync --python 3.12

# Rebuild dataset from scratch
uv run python scripts/fetch_positives.py
uv run python scripts/fetch_negatives.py
uv run python scripts/build_dataset.py

# Train on Intel XPU, calibrate/evaluate, and export ONNX
uv run python -m is_dafeiyu.train --config configs/default.yaml
uv run python -m is_dafeiyu.evaluate --config configs/default.yaml
uv run python -m is_dafeiyu.export

# Run CLI inference (single image or directory)
uv run python -m is_dafeiyu.predict /path/to/image_or_dir
uv run python -m is_dafeiyu.predict /path/to/image.png --threshold 0.75 --json
```

## Key Invariants & Pitfalls

1. **No multi-resolution leakage from `EDMOK/blue-fish-archive`**: Only ingest `media/` and `character-art/v1/download/` (plus `logo/`, `原始立绘/`, `art/v1/`, `memes/`). Never include `large/`, `previews/`, `card/`, `small/`, `thumb/`, or `view/`.
2. **Filter non-Dafeiyu characters in `blue-fish-archive`**: `stickers/manifest.json` and `character-art/v1/download/` contain other AI mascots (ChatGPT, Claude, Gemini, GLM, Grok, Kimi, Qwen). Route non-Dafeiyu entries to `negative_blue_chars`, never to positives.
3. **Group-aware splitting**: All extracted frames from an animated GIF/WebP and near-duplicate `phash` clusters must share the same `cluster_id` and never cross `train`/`val`/`test` boundaries.
4. **Color & transparency handling**: Keep hue jitter tiny (`<= 0.02`) since blue is a core character feature. Composite RGBA transparency onto varied solid backgrounds during training so the model does not memorize white sticker backgrounds.
5. **DINOv2 `img_size` argument**: `vit_base_patch14_reg4_dinov2.lvd142m` defaults to `518`. Always pass `img_size=cfg["data"]["image_size"]` (multiple of 14, e.g. `224`) to `timm.create_model`.
6. **Code style**: Keep `src/is_dafeiyu/` free of regular expressions unless strictly required; prefer plain string operations.
