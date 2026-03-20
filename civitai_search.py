"""
civitai_search.py — Civitai LoRA Search for DiffuseCraft

Based on confirmed Reddit dev findings (r/civitai):
- query= is BROKEN (fails ~70% of the time) — never use it
- tag= works 100% reliably
- username= works for creator search
- Page-based pagination is broken — use cursor from metadata.nextPage/prevPage
- Large limits (20+) cause timeouts — keep at 10 max
- No sort param to avoid 400s
"""

import requests
import subprocess
import os
import re

CIVITAI_API  = "https://civitai.com/api/v1/models"
PAGE_SIZE    = 10


def compress_image_url(url, width=120):
    """
    Compress Civitai preview images via their CDN.
    Works for both imagecache.civitai.com and image.civitai.com.
    Width=120 gives fast-loading thumbnails good enough for the card grid.
    """
    if not url or "civitai.com" not in url:
        return url
    # Strip any existing width or original directive
    url = re.sub(r'/width=\d+', '', url)
    url = re.sub(r'/original=true', '', url)
    # Insert /width=N before the filename (last path segment)
    parts = url.rsplit('/', 1)
    if len(parts) == 2:
        url = f"{parts[0]}/width={width}/{parts[1]}"
    return url


def search_civitai_loras(query, search_by, api_key="", cursor=None, page_size=PAGE_SIZE):
    """
    Search Civitai LoRAs using cursor-based pagination (reliable).
    - Name search: tag= param (100% reliable per Reddit devs)
    - Creator search: username= param
    - Returns (results, status_msg, next_cursor, prev_cursors_stack)
    """
    headers = {}
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    q = query.strip()

    params = {
        "types": "LORA",
        "limit": page_size,
        "nsfw":  "true",
    }

    if search_by == "Creator":
        params["username"] = q
    else:
        params["tag"] = q

    # Use cursor for pagination if provided
    if cursor:
        params["cursor"] = cursor

    try:
        resp = requests.get(CIVITAI_API, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        return [], f"❌ Search failed ({e.response.status_code}): {e.response.text[:200]}", None
    except Exception as e:
        return [], f"❌ Search failed: {e}", None

    items    = data.get("items", [])
    metadata = data.get("metadata", {})

    # Extract next cursor from nextPage URL
    next_cursor = metadata.get("nextCursor")
    total_items = metadata.get("totalItems", len(items))

    if not items:
        return [], f"No LoRAs found for '{q}'. Try different keywords.", None

    results = []
    for item in items:
        versions = item.get("modelVersions", [])
        if not versions:
            continue
        latest = versions[0]
        files  = latest.get("files", [])

        download_url  = None
        file_size_kb  = 0
        orig_filename = ""
        for f in files:
            if f.get("primary") or f.get("name", "").endswith(".safetensors"):
                download_url  = f.get("downloadUrl") or latest.get("downloadUrl")
                file_size_kb  = f.get("sizeKB", 0)
                orig_filename = f.get("name", "")
                break
        if not download_url:
            download_url = latest.get("downloadUrl", "")

        images      = latest.get("images", [])
        raw_preview = next((img["url"] for img in images if img.get("url")), "")
        preview_url = compress_image_url(raw_preview, width=180)

        desc = re.sub(r"<[^>]+>", "", item.get("description") or "").strip()
        desc = desc[:300] + "..." if len(desc) > 300 else desc

        stats = item.get("stats", {})
        results.append({
            "id":            item.get("id"),
            "name":          item.get("name", "Unknown"),
            "creator":       item.get("creator", {}).get("username", "Unknown"),
            "description":   desc or "No description.",
            "downloads":     stats.get("downloadCount", 0),
            "rating":        round(stats.get("rating", 0), 1),
            "rating_count":  stats.get("ratingCount", 0),
            "base_model":    latest.get("baseModel", "Unknown"),
            "trigger_words": latest.get("trainedWords", []),
            "download_url":  download_url,
            "orig_filename": orig_filename,
            "preview_url":   preview_url,
            "model_url":     f"https://civitai.com/models/{item.get('id')}",
            "version_name":  latest.get("name", ""),
            "file_size_kb":  file_size_kb,
            "nsfw":          item.get("nsfw", False),
        })

    msg = f"✅ Found {len(results)} LoRAs for '{q}'" + (f" — {total_items:,} total" if total_items else "")
    return results, msg, next_cursor


def download_lora_from_result(result, directory="./loras", civitai_api_key=""):
    os.makedirs(directory, exist_ok=True)
    url = result.get("download_url", "")
    if not url:
        return None, "❌ No download URL."

    if civitai_api_key and civitai_api_key.strip() and "token=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}token={civitai_api_key.strip()}"

    if result.get("orig_filename") and result["orig_filename"].endswith(".safetensors"):
        filename = result["orig_filename"]
    else:
        safe_name = re.sub(r'[^\w\-.]', '_', result["name"])[:60]
        filename  = f"{safe_name}.safetensors"

    output_path = os.path.join(directory, filename)
    print(f"[CivitaiSearch] Downloading {filename} → {directory}")

    try:
        cmd = ["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16",
               "-k", "1M", "--out", filename, url, "-d", directory]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and os.path.exists(output_path):
            return output_path, f"✅ Downloaded: {filename}\nPath: {output_path}"
    except FileNotFoundError:
        pass

    try:
        with requests.get(url, headers={"User-Agent": "DiffuseCraft/1.0"},
                          stream=True, timeout=120) as r:
            r.raise_for_status()
            cd = r.headers.get("Content-Disposition", "")
            if "filename=" in cd:
                import cgi
                _, p = cgi.parse_header(cd)
                srv = p.get("filename", "").strip('"')
                if srv.endswith(".safetensors"):
                    filename    = srv
                    output_path = os.path.join(directory, filename)
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
        return output_path, f"✅ Downloaded: {filename}\nPath: {output_path}"
    except Exception as e:
        return None, f"❌ Download failed: {e}"


def format_results_html(results, page_num=1):
    if not results:
        return "<p style='color:#aaa;padding:10px;'>No results.</p>"

    cards = ""
    for i, r in enumerate(results):
        num     = (page_num - 1) * PAGE_SIZE + i + 1
        img_src = r["preview_url"] or "https://placehold.co/140x185/1a1a2e/888?text=No+Preview"
        nsfw_badge = (
            "<span style='background:#c0392b;color:#fff;font-size:0.6em;"
            "padding:1px 4px;border-radius:3px;margin-left:3px;'>NSFW</span>"
            if r["nsfw"] else ""
        )
        base_colors = {
            "Illustrious": "#4a9eff", "SDXL 1.0": "#7c6af7",
            "Pony": "#f7a06a", "SD 1.5": "#6af7a0", "FLUX": "#f76a6a",
        }
        base_color = base_colors.get(r["base_model"], "#7c9")

        cards += f"""
        <div style="width:calc(33.33% - 10px);min-width:120px;max-width:175px;
                    border-radius:10px;overflow:hidden;background:#1e1e2e;
                    border:2px solid #2a2a3e;font-family:sans-serif;cursor:pointer;
                    transition:border-color 0.15s,transform 0.15s;"
             onmouseover="this.style.borderColor='#7c6af7';this.style.transform='scale(1.03)';"
             onmouseout="this.style.borderColor='#2a2a3e';this.style.transform='scale(1)';">
            <div style="position:relative;">
                <img src="{img_src}" loading="lazy"
                     style="width:100%;height:185px;object-fit:cover;display:block;"
                     onerror="this.src='https://placehold.co/140x185/1a1a2e/888?text=No+Image'"/>
                <div style="position:absolute;top:5px;left:5px;background:rgba(0,0,0,0.8);
                            color:#fff;font-size:0.72em;font-weight:bold;
                            padding:2px 6px;border-radius:6px;">#{num}</div>
            </div>
            <div style="padding:7px;">
                <div style="font-weight:bold;font-size:0.78em;color:#e0e0e0;
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                     title="{r['name']}">{r['name']}{nsfw_badge}</div>
                <div style="font-size:0.68em;color:#aaa;margin-top:1px;">👤 {r['creator']}</div>
                <div style="font-size:0.67em;margin-top:1px;color:{base_color};">🏗️ {r['base_model']}</div>
                <div style="font-size:0.65em;color:#888;margin-top:2px;">
                    ⬇️ {r['downloads']:,} &nbsp; ⭐ {r['rating']}
                </div>
            </div>
        </div>"""

    return f"""
    <div style="background:#0f1117;padding:12px;border-radius:12px;">
        <div style="color:#555;font-size:0.75em;margin-bottom:8px;">
            📄 Page {page_num} &nbsp;·&nbsp;
            💡 Type a result number below to see details &amp; trigger words
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:8px;
                    max-height:600px;overflow-y:auto;padding-right:4px;">
            {cards}
        </div>
    </div>"""


def format_detail_html(result):
    if not result:
        return ""

    if result["trigger_words"]:
        tags = "".join([
            f"<span style='background:#2d2d4e;color:#c9b8ff;padding:3px 10px;"
            f"border-radius:12px;margin:3px 2px;display:inline-block;"
            f"font-size:0.85em;border:1px solid #4a4a6e;'>{t}</span>"
            for t in result["trigger_words"]
        ])
        trigger_section = f"""
        <div style='margin-top:12px;'>
            <div style='color:#888;font-size:0.8em;font-weight:bold;margin-bottom:5px;'>
                🎯 TRIGGER WORDS</div>
            <div>{tags}</div>
        </div>"""
    else:
        trigger_section = "<div style='color:#555;font-size:0.8em;margin-top:10px;'>🎯 No trigger words listed.</div>"

    img_src    = result["preview_url"] or "https://placehold.co/150x200/1a1a2e/888?text=No+Image"
    nsfw_badge = (
        "<span style='background:#c0392b;color:#fff;font-size:0.72em;"
        "padding:2px 6px;border-radius:4px;margin-right:6px;'>NSFW</span>"
        if result["nsfw"] else ""
    )
    size_str = f"{result['file_size_kb']/1024:.1f} MB" if result["file_size_kb"] else "Unknown"
    fname    = result.get("orig_filename") or f"{result['name'][:40]}.safetensors"

    return f"""
    <div style='background:#13131f;border:1px solid #333;border-radius:12px;
                padding:14px;display:flex;gap:14px;flex-wrap:wrap;
                font-family:sans-serif;margin-top:6px;'>
        <img src="{img_src}"
             style='width:150px;height:200px;object-fit:cover;border-radius:8px;flex-shrink:0;'
             onerror="this.src='https://placehold.co/150x200/1a1a2e/888?text=No+Image'"/>
        <div style='flex:1;min-width:160px;'>
            <div style='font-size:1.05em;font-weight:bold;color:#e0e0e0;margin-bottom:2px;'>
                {nsfw_badge}{result["name"]}</div>
            <div style='font-size:0.78em;color:#555;margin-bottom:10px;'>{result["version_name"]}</div>
            <div style='display:grid;grid-template-columns:auto 1fr;gap:5px 12px;font-size:0.82em;'>
                <span style='color:#666;'>👤 Creator</span>
                <span style='color:#ccc;'>{result["creator"]}</span>
                <span style='color:#666;'>🏗️ Base Model</span>
                <span style='background:#1a2a3a;color:#7ab;padding:1px 7px;border-radius:6px;font-size:0.88em;'>
                    {result["base_model"]}</span>
                <span style='color:#666;'>⬇️ Downloads</span>
                <span style='color:#ccc;'>{result["downloads"]:,}</span>
                <span style='color:#666;'>⭐ Rating</span>
                <span style='color:#ccc;'>{result["rating"]}
                    <span style='color:#555;font-size:0.88em;'>({result["rating_count"]} reviews)</span></span>
                <span style='color:#666;'>💾 Size</span>
                <span style='color:#ccc;'>{size_str}</span>
                <span style='color:#666;'>📄 File</span>
                <span style='color:#999;font-size:0.82em;word-break:break-all;'>{fname}</span>
            </div>
            {trigger_section}
            <div style='margin-top:10px;font-size:0.78em;color:#666;line-height:1.6;'>
                {result["description"]}</div>
            <a href='{result["model_url"]}' target='_blank'
               style='display:inline-block;margin-top:12px;color:#7c6af7;font-size:0.83em;text-decoration:none;'>
                🔗 View on Civitai →</a>
        </div>
    </div>"""
                
