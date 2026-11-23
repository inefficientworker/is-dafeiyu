"""Fetch three negative image categories:
1. negative_fish: real fish, blue fish/whales (Wikimedia), and illustrated fish/whales (Safebooru.org)
2. negative_blue_chars: blue-haired anime characters (Gura, Suisei, Miku, Furina, etc.) & blue chibis (Safebooru.org)
3. negative_unrelated: diverse real-world photos (Picsum + Wikimedia), non-blue anime/chibis (Safebooru.org), and synthetic text memes
"""

from __future__ import annotations

import io
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

OUT_ROOT = Path("data/processed")
META_PATH = OUT_ROOT / "negatives_meta.jsonl"

USER_AGENT = "is-dafeiyu-dataset-builder/0.1 (research)"
SAFEBOORU_ORG_API = "https://safebooru.org/index.php"
PICSUM_API = "https://picsum.photos/v2/list"

SAFEBOORU_QUERIES: list[tuple[str, str, str, int]] = [
    # (category, subcategory, tags, limit)
    # --- Blue anime characters (Hard Negatives) ---
    ("negative_blue_chars", "gawr_gura", "gawr_gura solo", 120),
    ("negative_blue_chars", "hoshimachi_suisei", "hoshimachi_suisei solo", 80),
    ("negative_blue_chars", "hatsune_miku", "hatsune_miku solo", 80),
    ("negative_blue_chars", "rem_rezero", "rem_(re:zero) solo", 70),
    ("negative_blue_chars", "furina", "furina_(genshin_impact) solo", 80),
    ("negative_blue_chars", "aqua_konosuba", "aqua_(konosuba) solo", 70),
    ("negative_blue_chars", "cirno", "cirno solo", 70),
    ("negative_blue_chars", "arona_ba", "arona_(blue_archive) solo", 70),
    ("negative_blue_chars", "blue_hair_chibi", "blue_hair chibi", 150),
    ("negative_blue_chars", "blue_hair_shark", "blue_hair shark_tail", 60),
    ("negative_blue_chars", "blue_hair_fish", "blue_hair fish", 50),
    # --- Illustrated / cartoon fish & whales ---
    ("negative_fish", "safebooru_fish", "fish -1girl -1boy", 60),
    ("negative_fish", "safebooru_whale", "whale -1girl -1boy", 50),
    # --- Unrelated non-blue anime & chibis ---
    ("negative_unrelated", "nonblue_chibi", "chibi -blue_hair", 160),
    ("negative_unrelated", "nonblue_anime", "1girl -blue_hair", 160),
]


def query_safebooru_org(client: httpx.Client, tags: str, limit: int) -> list[tuple[str, str]]:
    """Return list of (id, url) pairs from safebooru.org DAPI."""
    results: list[tuple[str, str]] = []
    pid = 0
    while len(results) < limit and pid < 5:
        try:
            resp = client.get(
                SAFEBOORU_ORG_API,
                params={
                    "page": "dapi",
                    "s": "post",
                    "q": "index",
                    "json": "1",
                    "tags": tags,
                    "limit": min(100, limit - len(results) + 20),
                    "pid": pid,
                },
                timeout=20.0,
            )
            resp.raise_for_status()
            if not resp.text.strip():
                break
            posts = resp.json()
        except Exception as exc:
            print(f"[warn] safebooru.org query failed ({tags} pid={pid}): {exc}")
            break
        if not posts:
            break
        for post in posts:
            url = post.get("sample_url") or post.get("file_url")
            if not url:
                continue
            low = url.lower()
            if not any(low.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
                continue
            results.append((f"sbo_{post['id']}", url))
            if len(results) >= limit:
                break
        pid += 1
        time.sleep(0.2)
    return results


def query_picsum(client: httpx.Client, count: int = 450) -> list[tuple[str, str]]:
    """Return list of (id, url) pairs of diverse real-world photography from Picsum."""
    results: list[tuple[str, str]] = []
    page = 1
    while len(results) < count and page <= 8:
        try:
            resp = client.get(PICSUM_API, params={"page": page, "limit": 100}, timeout=20.0)
            resp.raise_for_status()
            items = resp.json()
        except Exception as exc:
            print(f"[warn] picsum query failed (page={page}): {exc}")
            break
        if not items:
            break
        for item in items:
            pid = item["id"]
            results.append((f"picsum_{pid}", f"https://picsum.photos/id/{pid}/512/512"))
            if len(results) >= count:
                break
        page += 1
    return results


def download_and_save(url: str, out_path: Path, headers: dict[str, str]) -> bool:
    if out_path.exists():
        return True
    try:
        with httpx.Client(headers=headers, follow_redirects=True, timeout=20.0) as c:
            resp = c.get(url)
            resp.raise_for_status()
            raw = resp.content
        with Image.open(io.BytesIO(raw)) as img:
            if img.width < 64 or img.height < 64:
                return False
            img = img.convert("RGB")
            img.thumbnail((768, 768), Image.Resampling.LANCZOS)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path, format="JPEG", quality=92)
        return True
    except Exception:
        return False


def generate_synthetic_text_memes(out_dir: Path, count: int = 80) -> list[dict]:
    """Generate text-only / chat-like memes (including tricky keywords) so the model does not rely on OCR."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    phrases = [
        "已思考 13 秒",
        "DeepSeek",
        "服务器繁忙，请稍后再试",
        "思考中...",
        "今日班味已加载",
        "拿捏了",
        "破防了",
        "Loading 99%",
        "ERROR 503",
        "Hello World",
        "Token 用完了",
        "蓝色大肥鱼",
        "AI Assistant",
        "Chat Log 2026",
        "OK 收到",
        "哈哈哈哈哈",
    ]
    font_candidates = [
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    font_path = next((p for p in font_candidates if Path(p).exists()), None)

    records: list[dict] = []
    for i in range(count):
        out_path = out_dir / f"synth_text_{i:03d}.jpg"
        if not out_path.exists():
            w = rng.choice([320, 400, 480, 512])
            h = rng.choice([240, 320, 400, 512])
            bg = rng.choice(
                [
                    (255, 255, 255),
                    (245, 247, 250),
                    (30, 34, 42),
                    (235, 245, 255),
                    (200, 225, 250),
                    (250, 248, 235),
                ]
            )
            img = Image.new("RGB", (w, h), bg)
            draw = ImageDraw.Draw(img)
            for _ in range(rng.randint(1, 3)):
                x0 = rng.randint(10, w // 3)
                y0 = rng.randint(10, h // 2)
                x1 = rng.randint(w // 2, w - 10)
                y1 = rng.randint(y0 + 30, min(h - 10, y0 + 120))
                box_color = rng.choice([(255, 255, 255), (220, 238, 255), (50, 56, 70), (240, 240, 240)])
                draw.rounded_rectangle([x0, y0, x1, y1], radius=12, fill=box_color, outline=(180, 190, 205), width=2)

            text_color = (240, 240, 240) if sum(bg) < 300 else (30, 35, 45)
            try:
                font = ImageFont.truetype(font_path, rng.randint(20, 36)) if font_path else ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()

            lines = rng.sample(phrases, k=rng.randint(1, 3))
            y_cursor = rng.randint(24, h // 3)
            for line in lines:
                draw.text((rng.randint(20, 60), y_cursor), line, fill=text_color, font=font)
                y_cursor += rng.randint(42, 68)

            img.save(out_path, format="JPEG", quality=90)
        records.append(
            {
                "path": str(out_path),
                "label": 0,
                "category": "negative_unrelated",
                "subcategory": "synthetic_text_meme",
                "group_id": f"synth_text_{i:03d}",
                "source": "synthetic",
            }
        )
    return records


def collect_existing_wikimedia_records() -> list[dict]:
    """Preserve already-downloaded Wikimedia images from data/processed/negative_*/wiki_*."""
    records: list[dict] = []
    for cat in ("negative_fish", "negative_blue_chars", "negative_unrelated"):
        cat_dir = OUT_ROOT / cat
        if not cat_dir.exists():
            continue
        for sub_dir in sorted(cat_dir.iterdir()):
            if not sub_dir.is_dir() or not sub_dir.name.startswith("wiki_"):
                continue
            for img_path in sorted(sub_dir.glob("*.jpg")):
                records.append(
                    {
                        "path": str(img_path),
                        "label": 0,
                        "category": cat,
                        "subcategory": sub_dir.name,
                        "group_id": img_path.stem,
                        "source": "wikimedia",
                    }
                )
    return records


def main() -> None:
    headers = {"User-Agent": USER_AGENT}
    jobs: list[tuple[str, str, str, str, Path]] = []
    seen_ids: set[str] = set()

    with httpx.Client(headers=headers, follow_redirects=True) as client:
        print("[fetch] Querying safebooru.org API ...")
        for category, subcat, tags, limit in SAFEBOORU_QUERIES:
            items = query_safebooru_org(client, tags, limit)
            added = 0
            for item_id, url in items:
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)
                out_path = OUT_ROOT / category / subcat / f"{item_id}.jpg"
                jobs.append((category, subcat, item_id, url, out_path))
                added += 1
            print(f"  - safebooru.org [{category}/{subcat}]: {added} candidates")

        print("[fetch] Querying Picsum real-world photos ...")
        picsum_items = query_picsum(client, count=450)
        for item_id, url in picsum_items:
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            out_path = OUT_ROOT / "negative_unrelated" / "picsum_real" / f"{item_id}.jpg"
            jobs.append(("negative_unrelated", "picsum_real", item_id, url, out_path))
        print(f"  - picsum [negative_unrelated/picsum_real]: {len(picsum_items)} candidates")

    print(f"[fetch] Downloading {len(jobs)} images with 10 workers ...")
    records: list[dict] = collect_existing_wikimedia_records()
    print(f"[fetch] Preserved {len(records)} existing Wikimedia images.")

    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=10) as pool:
        future_map = {
            pool.submit(download_and_save, url, out_path, headers): (category, subcat, item_id, url, out_path)
            for (category, subcat, item_id, url, out_path) in jobs
        }
        for i, fut in enumerate(as_completed(future_map), 1):
            category, subcat, item_id, url, out_path = future_map[fut]
            if fut.result():
                ok += 1
                records.append(
                    {
                        "path": str(out_path),
                        "label": 0,
                        "category": category,
                        "subcategory": subcat,
                        "group_id": item_id,
                        "source": url,
                    }
                )
            else:
                fail += 1
            if i % 250 == 0 or i == len(jobs):
                print(f"  progress: {i}/{len(jobs)} (ok={ok}, fail={fail})")

    records.extend(generate_synthetic_text_memes(OUT_ROOT / "negative_unrelated" / "synthetic_text_meme", count=80))

    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with META_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_cat: dict[str, int] = {}
    for r in records:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print(f"[done] Total negatives saved: {len(records)} -> {by_cat}")
    print(f"[done] Metadata written to {META_PATH}")


if __name__ == "__main__":
    main()
