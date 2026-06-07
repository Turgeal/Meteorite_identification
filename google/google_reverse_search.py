#!/usr/bin/env python3
"""
Google 以圖搜圖 批量腳本 (v2)
==============================
對 test_images_stage2/test_images 下的每張圖片：
  1. 上傳至 https://images.google.com/?hl=zh-tw
  2. 記錄搜尋結果頁面的完整文本
  3. 若存在「完全相符的結果」，導航至該頁面並記錄文本
  4. 所有結果寫入一個 txt 檔案，供下一輪 LLM 判定使用

用法：
  python google_reverse_search.py
  python google_reverse_search.py --headless
  python google_reverse_search.py --single 000001.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── 路徑設定 ──────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
CODE_DIR    = SCRIPT_DIR.parent
IMAGES_DIR  = CODE_DIR / "test_images_stage2" / "test_images"
OUTPUT_DIR  = SCRIPT_DIR / "output"
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"
PAGE_HTML_DIR  = OUTPUT_DIR / "page_html"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
PAGE_HTML_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_FILE = OUTPUT_DIR / "progress.json"
RESULTS_FILE  = OUTPUT_DIR / "all_results.txt"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
except ImportError:
    print("[FATAL] pip install playwright && playwright install chromium")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════════════

def load_progress() -> tuple[set, dict]:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            progress = json.load(f)
        return set(progress.get("processed", [])), progress
    return set(), {"processed": []}


def save_progress(processed: set):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"processed": sorted(processed),
                    "updated": datetime.now().isoformat()}, f)


def append_result(img_name: str, result: dict):
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"圖片: {img_name}\n")
        f.write(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
        for key in ["results_url", "has_exact_match", "exact_match_label",
                     "exact_match_url", "exact_match_title",
                     "exact_match_snippet", "exact_page_text",
                     "top10_titles", "full_page_text_snippet",
                     "screenshot", "error"]:
            val = result.get(key)
            if val is None:
                continue
            f.write(f"\n[{key}]\n")
            if isinstance(val, list):
                for i, item in enumerate(val, 1):
                    f.write(f"  {i}. {item}\n")
            else:
                s = str(val)
                f.write(s[:5000])
                if len(s) > 5000:
                    f.write("\n... (已截斷)")
            f.write("\n")
        f.write("\n\n")


def close_cookie_dialog(page) -> bool:
    for sel in ['button:has-text("接受全部")', '#L2AGLb',
                'button:has-text("Accept all")']:
        try:
            el = page.locator(sel).first
            if el.is_visible(timeout=2000):
                el.click()
                time.sleep(1)
                return True
        except Exception:
            continue
    return False


# ══════════════════════════════════════════════════════════════════
# 核心步驟
# ══════════════════════════════════════════════════════════════════

def click_camera_icon(page) -> bool:
    """點擊 Google Images 搜尋欄中的相機圖示。
    zh-tw 頁面 aria-label 為「以圖搜尋」(非「以圖搜圖」)。"""
    selectors = [
        '[aria-label="以圖搜尋"]',
        'div[role="button"][aria-label="以圖搜尋"]',
        '[aria-label="以圖搜圖"]',
        'div[role="button"][aria-label="以圖搜圖"]',
        '[aria-label="Search by image"]',
        '[aria-label*="搜尋"]',
        '[aria-label*="搜圖"]',
        'div[jsname="ZtOxCb"]',
        '[jsaction*="searchByImage"]',
        'div.etxtjc[role="button"]',
    ]
    for sel in selectors:
        try:
            els = page.locator(sel)
            for i in range(els.count()):
                el = els.nth(i)
                try:
                    if el.is_visible(timeout=2000):
                        el.click(timeout=3000, force=True)
                        time.sleep(1.5)
                        return True
                except Exception:
                    continue
        except Exception:
            continue

    # JS 最後手段
    print("  [warn] 用 JS 嘗試點擊相機按鈕...")
    try:
        page.evaluate("""() => {
            const all = document.querySelectorAll('[role="button"]');
            for (const el of all) {
                const style = window.getComputedStyle(el);
                if (style.cursor === 'pointer') {
                    const label = el.getAttribute('aria-label') || '';
                    if (label.includes('\u641c') || label.includes('search') ||
                        label.includes('image') || label.includes('lens')) {
                        el.click(); return true;
                    }
                }
            }
            return false;
        }""")
        time.sleep(1.5)
        return True
    except Exception:
        pass

    print("  [warn] 找不到相機按鈕")
    return False


def upload_image_file(page, img_path: str) -> bool:
    """在彈出的對話框中上傳圖片檔案"""
    time.sleep(1.5)
    current_url = page.url
    if any(kw in current_url for kw in
           ["search", "searchbyimage", "lens.google.com/search"]):
        print("    已跳轉至結果頁面，跳過上傳")
        return True

    for sel in ['text=上傳圖片', 'text=Upload an image',
                'div:has-text("上傳圖片")', 'span:has-text("上傳圖片")']:
        try:
            el = page.locator(sel).first
            if el.count() > 0 and el.is_visible(timeout=2000):
                el.click()
                time.sleep(0.5)
                break
        except Exception:
            continue

    for sel in ['input[type="file"][accept*="image"]', 'input[type="file"]']:
        try:
            fi = page.locator(sel).first
            if fi.count() > 0:
                fi.set_input_files(img_path, timeout=10000)
                time.sleep(2)
                for _ in range(5):
                    time.sleep(1)
                    if page.url != current_url and "search" in page.url:
                        return True
                return True
        except Exception as e:
            print(f"  [debug] set_input_files: {e}")
            continue

    try:
        with page.expect_file_chooser(timeout=5000) as fc_info:
            for sel in ['text=選擇檔案', 'text=Choose File',
                        'button:has-text("選擇")', 'div:has-text("上傳")']:
                try:
                    el = page.locator(sel).first
                    if el.count() > 0 and el.is_visible(timeout=2000):
                        el.click(); break
                except Exception:
                    continue
            else:
                page.keyboard.press("Enter")
        fc_info.value.set_files(img_path)
        time.sleep(2)
        return True
    except Exception as e:
        print(f"  [debug] file_chooser: {e}")

    return False


def wait_for_results(page, timeout: int = 45) -> bool:
    """等待搜尋結果頁面加載完成。
    支援兩種模式：
    1. URL 跳轉（傳統 Google Images / Lens 導航到結果頁）
    2. DOM 變化（Google Images 內嵌 Lens widget，不跳轉）
    """
    current_url = page.url

    # 先檢查是否已經在結果頁
    if any(kw in current_url for kw in
           ["search", "searchbyimage", "lens.google.com/search"]):
        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass
        time.sleep(2)
        return True

    # 策略 A：等待 URL 改變（傳統模式）
    url_changed = False
    try:
        page.wait_for_url(
            lambda url: any(kw in url for kw in
                            ["search", "searchbyimage",
                             "/search?", "lens.google.com/search"]),
            timeout=timeout * 1000)
        url_changed = True
    except PWTimeout:
        pass
    except Exception:
        pass

    if url_changed:
        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass
        time.sleep(3)
        return True

    # 策略 B：URL 沒變，等待 DOM 中出現結果內容（內嵌 Lens widget）
    print("    等待 DOM 結果...")
    dom_indicators = [
        'text=完全相符',
        'text=視覺搜尋結果',
        'text=Visual matches',
        'text=Find image source',
        'h3',  # 搜尋結果標題
        '[data-result]',
        'a[href^="http"]:has(h3)',
    ]
    for indicator in dom_indicators:
        try:
            el = page.locator(indicator).first
            if el.count() > 0:
                el.wait_for(state="visible", timeout=timeout * 1000)
                print(f"    偵測到結果元素: {indicator}")
                time.sleep(2)
                return True
        except Exception:
            continue

    # 策略 C：URL 沒變但可能已有結果，等待足夠時間後強制繼續
    print("    等待 15 秒讓結果渲染...")
    time.sleep(15)

    # 最後檢查：是否有任何 h3 或結果連結
    try:
        if page.locator("h3").count() > 2 or \
           page.locator('a[href^="http"]').count() > 5:
            print("    偵測到頁面內容，繼續")
            return True
    except Exception:
        pass

    # 完全失敗
    if "search" in page.url:
        print("  [warn] URL 含有 search，強制繼續")
        time.sleep(2)
        return True

    print("  [warn] 等待結果頁面超時")
    return False


def parse_results_page(page) -> dict:
    """解析 Google 以圖搜圖 / Google Lens 結果頁面"""
    result = {}
    result["results_url"] = page.url
    result["page_title"] = page.title()
    is_lens = "lens.google.com" in page.url

    exact_texts = [
        "完全相符的結果", "完全相符的结果", "Exact match",
        "完全一致的结果", "完全相符",
    ]
    lens_exact_texts = [
        "完全相符的圖片", "Exact matches",
        "尋找圖片來源", "Find image source", "視覺相符",
    ]
    search_texts = exact_texts + (lens_exact_texts if is_lens else [])

    has_exact = False
    for txt in search_texts:
        try:
            el = page.locator(f'text="{txt}"').first
            if el.count() > 0:
                has_exact = True
                result["has_exact_match"] = True
                result["exact_match_label"] = txt
                break
        except Exception:
            continue

    if not has_exact:
        for tag in ["h2", "h3", "div"]:
            for txt in search_texts:
                try:
                    sel = f'{tag}:has-text("{txt}")'
                    el = page.locator(sel).first
                    if el.count() > 0:
                        has_exact = True
                        result["has_exact_match"] = True
                        result["exact_match_label"] = txt
                        break
                except Exception:
                    continue
            if has_exact:
                break

    if has_exact:
        result["exact_match_url"] = None
        result["exact_match_title"] = None
        try:
            for link in page.locator("a").all():
                try:
                    href = link.get_attribute("href", timeout=500) or ""
                    text = link.inner_text(timeout=500).strip()
                    if href.startswith("http") and "google" not in href \
                       and len(text) > 5:
                        result["exact_match_url"] = href
                        result["exact_match_title"] = text[:200]
                        break
                except Exception:
                    continue
        except Exception:
            pass

        try:
            for txt in search_texts:
                el = page.locator(f'text="{txt}"').first
                if el.count() > 0:
                    next_sib = el.locator("xpath=following-sibling::div[1]")
                    if next_sib.count() > 0:
                        result["exact_match_snippet"] = next_sib.inner_text(timeout=1000)[:1000]
                    else:
                        parent = el.locator("xpath=..")
                        if parent.count() > 0:
                            result["exact_match_snippet"] = parent.inner_text(timeout=1000)[:1000]
                    break
        except Exception:
            pass
    else:
        result["has_exact_match"] = False

    # ── 前 15 個網頁標題 ──
    titles = []
    try:
        for h3 in page.locator("h3").all()[:25]:
            try:
                t = h3.inner_text(timeout=500).strip()
                if t and len(t) > 2 and t not in titles:
                    titles.append(t)
            except Exception:
                continue
    except Exception:
        pass

    if len(titles) < 8:
        try:
            for a in page.locator("a").all()[:50]:
                try:
                    t = a.inner_text(timeout=300).strip()
                    if t and len(t) > 5 and t not in titles:
                        titles.append(t)
                except Exception:
                    continue
        except Exception:
            pass

    if len(titles) < 5:
        try:
            for el in page.locator('[data-result] a, [jsname] a').all()[:20]:
                try:
                    t = el.inner_text(timeout=300).strip()
                    if t and len(t) > 3 and t not in titles:
                        titles.append(t)
                except Exception:
                    continue
        except Exception:
            pass

    result["top10_titles"] = titles[:15]

    try:
        body_text = page.locator("body").inner_text(timeout=5000)
        result["full_page_text_snippet"] = body_text[:8000]
    except Exception:
        result["full_page_text_snippet"] = "（無法取得頁面文字）"

    return result


def navigate_to_exact_page(page, exact_url: str) -> Optional[str]:
    if not exact_url:
        return None
    try:
        page.goto(exact_url, wait_until="domcontentloaded", timeout=20000)
        time.sleep(2)
        close_cookie_dialog(page)
        return page.locator("body").inner_text(timeout=5000)[:5000]
    except Exception as e:
        return f"（導航失敗: {e}）"


# ══════════════════════════════════════════════════════════════════
# 單張圖片處理
# ══════════════════════════════════════════════════════════════════

def process_single_image(page, img_path: Path, headless: bool) -> dict:
    result = {}
    img_name = img_path.name

    # ── Step 1: 導航 ──
    print(f"  [1/6] 導航至 Google Images...")
    page.goto("https://images.google.com/?hl=zh-tw",
              wait_until="domcontentloaded", timeout=30000)
    time.sleep(2)
    close_cookie_dialog(page)
    time.sleep(0.5)

    # ── Step 2: 點擊相機圖示 ──
    print(f"  [2/6] 點擊相機圖示...")
    camera_ok = click_camera_icon(page)
    if not camera_ok:
        print(f"  [warn] 嘗試 imghp 頁面...")
        page.goto("https://www.google.com/imghp?hl=zh-tw",
                  wait_until="domcontentloaded", timeout=20000)
        time.sleep(2)
        close_cookie_dialog(page)
        camera_ok = click_camera_icon(page)
    if not camera_ok:
        print(f"  [warn] 改用 Google Lens 上傳頁面...")
        page.goto("https://lens.google.com/upload?hl=zh-tw",
                  wait_until="domcontentloaded", timeout=20000)
        time.sleep(2)

    # ── Step 3: 上傳圖片 ──
    print(f"  [3/6] 上傳圖片: {img_name}")
    upload_ok = upload_image_file(page, str(img_path))
    if not upload_ok:
        print(f"  [warn] 備用上傳...")
        try:
            page.locator('input[type="file"]').first.set_input_files(
                str(img_path), timeout=5000)
            time.sleep(2)
            upload_ok = True
        except Exception:
            pass
    if not upload_ok:
        result["error"] = "無法上傳圖片"
        return result

    # ── Step 4: 等待結果 ──
    print(f"  [4/6] 等待搜尋結果...")
    if not wait_for_results(page, timeout=45):
        sp = SCREENSHOT_DIR / f"{img_path.stem}_error.png"
        page.screenshot(path=str(sp))
        result["error"] = f"等待結果頁面超時 (URL: {page.url[:100]})"
        result["screenshot"] = str(sp)
        return result

    # ── Step 5: 解析 ──
    print(f"  [5/6] 解析結果頁面...")
    sp = SCREENSHOT_DIR / f"{img_path.stem}.png"
    try:
        page.screenshot(path=str(sp))
        result["screenshot"] = str(sp)
    except Exception:
        pass
    try:
        hp = PAGE_HTML_DIR / f"{img_path.stem}.html"
        with open(hp, "w", encoding="utf-8") as f:
            f.write(page.content())
    except Exception:
        pass
    parsed = parse_results_page(page)
    result.update(parsed)

    # ── Step 6: 原圖頁面 ──
    print(f"  [6/6] 檢查完全相符結果...")
    if result.get("has_exact_match") and result.get("exact_match_url"):
        print(f"    發現完全相符: {result['exact_match_url'][:80]}...")
        result["exact_page_text"] = navigate_to_exact_page(
            page, result["exact_match_url"])
    else:
        result["exact_page_text"] = None

    return result


# ══════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Google 以圖搜圖批量腳本")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--single", type=str, default=None)
    parser.add_argument("--min-delay", type=float, default=4.0)
    parser.add_argument("--max-delay", type=float, default=10.0)
    args = parser.parse_args()

    if args.single:
        img_path = IMAGES_DIR / args.single
        if not img_path.exists():
            print(f"[ERROR] 圖片不存在: {img_path}")
            sys.exit(1)
        images = [img_path]
    else:
        images = sorted(p for p in IMAGES_DIR.iterdir()
                        if p.suffix.lower() in IMG_EXTS and p.is_file())
        if args.start > 0:
            images = images[args.start:]
        if args.limit > 0:
            images = images[:args.limit]

    total = len(images)
    print(f"共 {total} 張圖片待處理")
    print(f"輸出目錄: {OUTPUT_DIR}")
    print(f"結果檔案: {RESULTS_FILE}\n")

    processed, _ = load_progress()
    print(f"已處理: {len(processed)} 張")
    print(f"啟動瀏覽器 (headless={args.headless})...")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=args.headless,
            args=["--disable-blink-features=AutomationControlled",
                  "--disable-dev-shm-usage", "--no-sandbox"])
        context = browser.new_context(
            viewport={"width": 1366, "height": 900},
            locale="zh-TW", timezone_id="Asia/Taipei",
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/125.0.0.0 Safari/537.36"))
        page = context.new_page()
        page.set_default_timeout(15000)

        success = fail = skip = 0
        for i, img_path in enumerate(images):
            img_name = img_path.name
            if img_name in processed:
                skip += 1
                if skip <= 5 or skip % 20 == 0:
                    print(f"[{i+1}/{total}] 跳過 {img_name}")
                continue

            print(f"\n{'─' * 60}")
            print(f"[{i+1}/{total}] 處理: {img_name}")
            print(f"  檔案大小: {img_path.stat().st_size / 1024:.1f} KB")

            try:
                result = process_single_image(page, img_path, args.headless)
                if result.get("error"):
                    fail += 1
                    print(f"  ❌ 失敗: {result['error']}")
                else:
                    success += 1
                    print(f"  ✅ 成功")
                append_result(img_name, result)
                processed.add(img_name)
                save_progress(processed)
            except Exception as e:
                fail += 1
                print(f"  ❌ 例外: {e}")
                traceback.print_exc()
                append_result(img_name, {"error": str(e)})
                processed.add(img_name)
                save_progress(processed)

            if i < total - 1:
                delay = random.uniform(args.min_delay, args.max_delay)
                print(f"  ⏳ 等待 {delay:.1f} 秒...")
                time.sleep(delay)

        browser.close()

    print(f"\n{'=' * 60}")
    print(f"處理完成！成功: {success}  失敗: {fail}  跳過: {skip}")
    print(f"結果檔案: {RESULTS_FILE}")
    print(f"截圖目錄: {SCREENSHOT_DIR}")


if __name__ == "__main__":
    main()
