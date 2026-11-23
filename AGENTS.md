# AGENTS.md

## Project Overview

`is-dafeiyu` is a binary image classifier that detects whether an image contains **Dafeiyu (大肥鱼 / DeepSeek whale-girl mascot)**, including group shots and chibi/logo whale variants.

- **Label definition**: `1` (positive) if Dafeiyu appears anywhere in the image; `0` (negative) otherwise.
- **Optimization priority**: **Low false positives (high Precision)** against three explicit negative buckets:
  - `negative_blue_chars`: blue-haired anime/cartoon characters (e.g. Gawr Gura, Suisei, Miku, Rem, Furina, Aqua, Cirno, Arona, blue chibis, and other AI-mascot characters).
  - `negative_fish`: illustrated fish/whales that remain visually confusing.
  - `negative_unrelated`: real-world photos (including real fish/whales/sharks/dolphins), non-blue anime/chibis, and text/chat memes (including tricky keywords like "DeepSeek" / "已思考 13 秒" to prevent OCR shortcuts). Keep about 60 real aquatic training images, balanced across species, plus any inseparable mixed source groups; preserve existing holdout assignments.

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
- `scripts/build_dataset.py`: Groups sources, optional author/template/series IDs and pHash neighbors; deduplicates identical pixels and persists train/val/calibration/test assignments under `data/splits/`. Fresh data targets 70/10/10/10; legacy val/test assignments are preserved while a calibration set is reserved from train once.
- `src/is_dafeiyu/`:
  - `dataset.py`: PyTorch `Dataset`, alpha-channel background compositing, and augmentations.
  - `model.py`: `timm` backbone wrapper, XPU device resolution, and two-stage freeze/unfreeze helpers.
  - `train.py`: Stage 1 (linear probe) + Stage 2 (partial unfreeze) MobileNetV3 training loop on XPU.
  - `evaluate.py`: FP32 temperature/threshold calibration on the independent `calibration` split, strict operating constraints, and per-category FPR reporting on `test`.
  - `predict.py`: CLI inference for static images, animated GIF/WebP (max probability across frames), and directories.
  - `export.py`: ONNX export with baked-in temperature-scaled sigmoid + numerical parity check.
- `data/` and `artifacts/`: Git-ignored datasets, splits, checkpoints (`best.pt`), calibration (`calibration.json`), reports (`test_report.json`), and exported ONNX (`model.onnx`).

## Common Commands

```bash
# Install / sync dependencies (Python 3.12 + PyTorch XPU)
uv sync --python 3.12

# Fetch data and incrementally rebuild persistent splits
uv run python scripts/fetch_positives.py
uv run python scripts/fetch_negatives.py
uv run python scripts/build_dataset.py

# Train on Intel XPU, calibrate/evaluate, and export ONNX
uv run python -m is_dafeiyu.train --config configs/default.yaml
uv run python -m is_dafeiyu.evaluate --config configs/default.yaml
uv run python -m is_dafeiyu.export --checkpoint artifacts/mobilenetv3_v2/best.pt --browser --quantize fp16

# Run CLI inference (single image or directory)
uv run python -m is_dafeiyu.predict images/
uv run python -m is_dafeiyu.predict images/example.png --threshold 0.75 --json
```

## Key Invariants & Pitfalls

1. **No multi-resolution leakage from `EDMOK/blue-fish-archive`**: Only ingest `media/` and `character-art/v1/download/` (plus `logo/`, `原始立绘/`, `art/v1/`, `memes/`). Never include `large/`, `previews/`, `card/`, `small/`, `thumb/`, or `view/`.
2. **Filter non-Dafeiyu characters in `blue-fish-archive`**: `stickers/manifest.json` and `character-art/v1/download/` contain other AI mascots (ChatGPT, Claude, Gemini, GLM, Grok, Kimi, Qwen). Route non-Dafeiyu entries to `negative_blue_chars`, never to positives.
3. **Group-aware splitting**: All extracted frames from an animated GIF/WebP and near-duplicate `phash` clusters must share the same `cluster_id` and never cross `train`/`val`/`test` boundaries.
4. **Color & transparency handling**: Keep hue jitter tiny (`<= 0.02`) since blue is a core character feature. Composite RGBA transparency onto varied solid backgrounds during training so the model does not memorize white sticker backgrounds.
5. **DINOv2 `img_size` argument**: `vit_base_patch14_reg4_dinov2.lvd142m` defaults to `518`. Always pass `img_size=cfg["data"]["image_size"]` (multiple of 14, e.g. `224`) to `timm.create_model`.
6. **Code style**: Keep `src/is_dafeiyu/` free of regular expressions unless strictly required; prefer plain string operations.
7. **Small-data training**: Freeze parameters AND module state in frozen subtrees. Whole-image labels must use whole-image augmentation; do not introduce random crops/erasing without target annotations. `pos_weight` is explicit, default 1. Source groups have equal sampling contribution.
8. **Separate selection and calibration**: Use FP32 val logits for checkpoint selection. Fit calibration only on the separate calibration split. Constraints are empirical precision, recall, total FPR and worst negative-subcategory FPR; infeasible runs save diagnostic reports and fail instead of relaxing requirements. Confidence bounds assume independent observations and do not prove unseen-domain accuracy.
9. **Versioned artifacts**: The training config writes to `artifacts/mobilenetv3_v2/`. Training refuses to overwrite a previous run and records split hashes. Legacy checkpoints without provenance remain usable for inference but cannot be recalibrated as if they had never seen the new holdouts. Fix conflicting template/label metadata before rebuilding; never delete split state to hide a conflict.
10. **Distillation/deployment**: Use `distillation.teacher_checkpoint`, with aligned augmented pixels and normalization; cached whole-image logits are no longer training targets. Export validates all calibration images plus batches on the final ONNX artifact, then writes `<model>.calibration.json` with its own threshold and hash. Consume that sidecar rather than the PyTorch threshold. See `docs/training-pipeline.md` for the workflow and real-data E2E runner.
11. **Aquatic negatives**: `data_policy.py` defines the real aquatic subcategories shared by collection and splitting. `build_dataset.py --real-aquatic-train-limit 60` caps eligible whole training groups and records excluded paths; `-1` disables the cap for diagnostic comparisons. Files and historical split assignments remain intact. Illustrated fish/whales stay in `negative_fish`.
