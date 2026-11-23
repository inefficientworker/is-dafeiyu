/** Real browser default-model download, inference and cache acceptance. */
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const assert = require("node:assert/strict");
const { chromium } = require(process.env.DAFEIYU_PLAYWRIGHT || "playwright");

const output = process.argv[2];
if (!output) throw new Error("Provide an output directory outside the checkout");
fs.mkdirSync(output, { recursive: true });
const repo = process.cwd();
const catalog = JSON.parse(fs.readFileSync("docs/models.json", "utf8"));
assert.equal(catalog.models.length, 1);
const model = catalog.models[0];
const calibration = JSON.parse(fs.readFileSync(model.calibration, "utf8"));
assert.equal(model.threshold, calibration.threshold);
const fixtures = [
  ...fs.readdirSync("data/regression/x").filter(name => name.endsWith(".jpg")).map(name => ({
    path: `data/regression/x/${name}`, label: Number(name === "HXzHaTpbIAAzk9B.jpg"),
  })),
  ...[11, 17, 21].map(n => ({ path: `data/curated/user_20261123/positive/${n}.png`, label: 1 })),
  ...[10, 20, 22, 35, 36].map(n => ({ path: `data/curated/user_20261123/negative/${n}.png`, label: 0 })),
];
const server = http.createServer((req, res) => {
  let name = new URL(req.url, "http://localhost").pathname;
  if (name.endsWith("/")) name += "index.html";
  const file = path.resolve(repo, "." + decodeURIComponent(name));
  if (!file.startsWith(repo + path.sep) || !fs.existsSync(file)) return res.writeHead(404).end();
  const type = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json" }[path.extname(file)] || "application/octet-stream";
  res.writeHead(200, { "Content-Type": type, "Content-Length": fs.statSync(file).size });
  fs.createReadStream(file).pipe(res);
});

(async () => {
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const browser = await chromium.launch({ executablePath: process.env.DAFEIYU_CHROME });
  const report = { model: model.filename, threshold: calibration.threshold, passed: false };
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    const requests = [];
    page.on("pageerror", error => errors.push(String(error)));
    page.on("request", request => requests.push(request.url()));
    const url = process.argv[3] || `http://127.0.0.1:${server.address().port}/docs/`;
    await page.goto(url);
    await page.locator(".model-card").waitFor();
    assert.equal(await page.locator(".model-card").count(), 1);
    assert.equal(requests.filter(url => url.endsWith(".onnx")).length, 0);
    await page.evaluate(async () => {
      const cache = await caches.open("is-dafeiyu-models-v1");
      await cache.put("https://example.invalid/retired.onnx", new Response("retired cache entry"));
    });
    await page.reload();
    await page.locator(".model-card").waitFor();
    assert.equal(await page.evaluate(async () => (await caches.keys()).includes("is-dafeiyu-models-v1")), false);
    await page.locator('button[data-act="load"]').click();
    await page.waitForFunction(() => document.getElementById("statusDot").classList.contains("ready"), { timeout: 120000 });
    assert.equal(Number(await page.locator("#thresholdValue").getAttribute("title")), calibration.threshold);
    assert(requests.some(url => url.endsWith(model.filename)));
    assert(requests.some(url => url.endsWith(model.calibration)));
    await page.locator("#imageInput").setInputFiles(fixtures.map(row => row.path));
    await page.waitForFunction(count => document.querySelectorAll("#resultsGrid > .card").length === count, fixtures.length);
    const cards = await page.locator("#resultsGrid > .card").evaluateAll(nodes => nodes.map(node => ({
      name: node.querySelector(".filename").textContent,
      predicted: Number(node.querySelector(".badge").classList.contains("pos")),
      probability_display: node.querySelector(".meta-row strong").textContent,
    })).reverse());
    report.samples = fixtures.map((row, i) => ({ ...row, ...cards[i] }));
    for (const row of report.samples) assert.equal(row.name, path.basename(row.path));
    assert.equal(report.samples.find(row => row.name === "HXzHaTpbIAAzk9B.jpg").predicted, 1);
    assert.equal(report.samples.find(row => row.name === "HXzHga8bAAAd7va.jpg").predicted, 0);
    for (const row of report.samples.filter(row => row.path.includes("user_20261123") && row.name !== "36.png")) {
      assert.equal(row.predicted, row.label, row.path);
    }
    await page.screenshot({ path: path.join(output, "browser.png"), fullPage: true });
    await page.reload();
    await page.locator(".model-card").waitFor();
    await page.route("https://huggingface.co/**", route => route.abort());
    await page.route("https://hf-mirror.com/**", route => route.abort());
    await page.locator('button[data-act="load"]').click();
    await page.waitForFunction(() => document.getElementById("statusDot").classList.contains("ready"));
    assert((await page.locator("#statusText").innerText()).includes("来自本地缓存"));
    assert.equal(Number(await page.locator("#thresholdValue").getAttribute("title")), calibration.threshold);
    await page.locator("#imageInput").setInputFiles("data/regression/x/HXzHaTpbIAAzk9B.jpg");
    await page.locator("#resultsGrid .badge.pos").waitFor();
    assert.deepEqual(errors, []);
    report.passed = true;
    report.model_count = 1;
    report.retired_cache_removed = true;
    report.cache_reload_with_hub_blocked = true;
    report.page_errors = errors;
    console.log(JSON.stringify(report));
  } finally {
    fs.writeFileSync(path.join(output, "report.json"), JSON.stringify(report, null, 2));
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; server.close(); });
