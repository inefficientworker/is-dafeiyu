# Browser demo

The example opens the same detector as the published site, with one default model.

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/docs/` and click the download button. The page downloads the model and its calibration from Hugging Face and caches them for later use.

For embedding, import `DafeiyuDetector` from `docs/detector.js` and pass the threshold from the model's calibration JSON to `create`.
