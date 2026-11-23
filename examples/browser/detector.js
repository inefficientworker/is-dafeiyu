/**
 * Browser-side Dafeiyu image detector using ONNX Runtime Web.
 * Feeds raw RGBA uint8 [1, 224, 224, 4] directly into the ONNX graph
 * (alpha compositing, normalization, and calibrated sigmoid are baked in).
 */

const DEFAULT_ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

export class DafeiyuDetector {
  /**
   * @param {object} options
   * @param {import('onnxruntime-web').InferenceSession} options.session
   * @param {typeof import('onnxruntime-web')} options.ort
   * @param {number} [options.threshold=0.5]
   * @param {number} [options.imageSize=224]
   */
  constructor({ session, ort, threshold = 0.5, imageSize = 224 }) {
    this.session = session;
    this.ort = ort;
    this.threshold = threshold;
    this.imageSize = imageSize;
    this.inputName = session.inputNames?.[0] || "image";
    this.outputName = session.outputNames?.[0] || "probability";
  }

  /**
   * Load an ONNX model from URL, ArrayBuffer, or Uint8Array.
   * @param {string | ArrayBuffer | Uint8Array} modelSource
   * @param {object} [options]
   * @param {number} [options.threshold=0.5]
   * @param {number} [options.imageSize=224]
   * @param {string[]} [options.executionProviders=['wasm']]
   * @param {string} [options.wasmPaths]
   * @param {typeof import('onnxruntime-web')} [options.ort]
   */
  static async create(modelSource, options = {}) {
    const ort = options.ort || globalThis.ort;
    if (!ort) {
      throw new Error(
        "onnxruntime-web (`ort`) is not loaded. Include ort.min.js before creating DafeiyuDetector."
      );
    }

    if (options.wasmPaths || !ort.env.wasm.wasmPaths) {
      ort.env.wasm.wasmPaths = options.wasmPaths || DEFAULT_ORT_CDN;
    }

    const sessionOptions = {
      executionProviders: options.executionProviders || ["wasm"],
      graphOptimizationLevel: "all",
    };

    const session = await ort.InferenceSession.create(modelSource, sessionOptions);
    return new DafeiyuDetector({
      session,
      ort,
      threshold: options.threshold ?? 0.5,
      imageSize: options.imageSize ?? 224,
    });
  }

  /**
   * Draw an image source onto a 224x224 canvas preserving alpha and return RGBA Uint8Array.
   * @param {CanvasImageSource | Blob} source
   * @returns {Promise<Uint8Array>}
   */
  async extractRgbaBytes(source) {
    const size = this.imageSize;
    let drawable = source;
    let ownedBitmap = null;

    if (typeof Blob !== "undefined" && source instanceof Blob) {
      if (typeof createImageBitmap === "function") {
        ownedBitmap = await createImageBitmap(source);
        drawable = ownedBitmap;
      } else {
        drawable = await loadHtmlImageFromBlob(source);
      }
    }

    try {
      const canvas =
        typeof OffscreenCanvas !== "undefined"
          ? new OffscreenCanvas(size, size)
          : Object.assign(document.createElement("canvas"), { width: size, height: size });

      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.clearRect(0, 0, size, size);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(drawable, 0, 0, size, size);

      const { data } = ctx.getImageData(0, 0, size, size);
      return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    } finally {
      if (ownedBitmap && typeof ownedBitmap.close === "function") {
        ownedBitmap.close();
      }
    }
  }

  /**
   * Run inference on an image Blob/File/ImageBitmap/HTMLImageElement/Canvas.
   * @param {CanvasImageSource | Blob} source
   * @param {number} [thresholdOverride]
   * @returns {Promise<{ probability: number, isDafeiyu: boolean, latencyMs: number }>}
   */
  async detect(source, thresholdOverride) {
    const threshold = thresholdOverride ?? this.threshold;
    const rgba = await this.extractRgbaBytes(source);
    const size = this.imageSize;
    const inputTensor = new this.ort.Tensor("uint8", rgba, [1, size, size, 4]);

    const t0 = performance.now();
    const outputs = await this.session.run({ [this.inputName]: inputTensor });
    const latencyMs = performance.now() - t0;

    const prob = Number(outputs[this.outputName].data[0]);
    return {
      probability: prob,
      isDafeiyu: prob >= threshold,
      latencyMs,
    };
  }
}

function loadHtmlImageFromBlob(blob) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = (err) => {
      URL.revokeObjectURL(url);
      reject(err);
    };
    img.src = url;
  });
}
