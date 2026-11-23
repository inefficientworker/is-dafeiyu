# Browser Demo (`is-dafeiyu`)

Minimal browser-side test page powered by **ONNX Runtime Web (WASM)** and the single-file browser `.onnx` model (with RGBA `uint8 [1, 224, 224, 4]` input, alpha compositing, normalization, and calibrated sigmoid baked into the ONNX graph).

## Quick Start

Start a local static HTTP server from the **repository root** (so both `examples/browser/` and `artifacts/` are served over HTTP, avoiding `file://` CORS restrictions):

```bash
python3 -m http.server 8000
```

Then open in your browser:

- `http://localhost:8000/examples/browser/`

## Usage

1. The page automatically attempts to load the recommended default `.onnx` model from `../../artifacts/`. You can also pick any local `.onnx` file via the file picker.
2. Adjust the decision threshold slider if needed (pre-filled with the calibrated high-precision threshold).
3. Drag and drop or select one or more images (`.png`, `.jpg`, `.webp`, `.gif`) to inspect thumbnail previews, predicted probabilities, classification labels, and per-image WASM inference latency (`ms`).
