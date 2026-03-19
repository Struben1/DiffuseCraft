"""
civitai_search.py — Civitai LoRA Search for DiffuseCraft
Adds a search-by-name or search-by-creator feature to the LoRA section.
"""

import requests
import urllib.request
import os
from datetime import datetime


CIVITAI_API = "https://civitai.com/api/v1/models"


def search_civitai_loras(query, search_by, api_key="", limit=10):
    """
    Search Civitai for LoRAs by name or creator.

    Returns a list of dicts with:
        name, creator, description, downloads, rating,
        base_model, trigger_words, download_url, preview_url, model_url
    """
    params = {
        "types": "LORA",
        "limit": limit,
        "sort": "Highest Rated",
    }

    if search_by == "Creator":
        params["username"] = query.strip()
    else:
        params["query"] = query.strip()

    headers = {}
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    try:
        resp = requests.get(CIVITAI_API, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return [], f"❌ Search failed: {e}"

    items = data.get("items", [])
    if not items:
        return [], "No LoRAs found. Try a different search term."

    results = []
    for item in items:
        versions = item.get("modelVersions", [])
        if not versions:
            continue

        latest = versions[0]
        files  = latest.get("files", [])

        # Get the primary safetensors file download URL
        download_url = None
        for f in files:
            if f.get("primary") or f.get("name", "").endswith(".safetensors"):
                download_url = f.get("downloadUrl") or latest.get("downloadUrl")
                break
        if not download_url:
            download_url = latest.get("downloadUrl", "")

        # Add API key to download URL if needed
        if download_url and api_key and api_key.strip():
            sep = "&" if "?" in download_url else "?"
            download_url = f"{download_url}{sep}token={api_key.strip()}"

        # Preview image
        images      = latest.get("images", [])
        preview_url = images[0].get("url", "") if images else ""

        # Trigger words
        trigger_words = ", ".join(latest.get("trainedWords", [])) or "None listed"

        # Base model
        base_model = latest.get("baseModel", "Unknown")

        # Stats
        stats     = item.get("stats", {})
        downloads = stats.get("downloadCount", 0)
        rating    = round(stats.get("rating", 0), 1)

        # Description — strip HTML tags simply
        desc = item.get("description") or ""
        import re
        desc = re.sub(r"<[^>]+>", "", desc).strip()
        desc = desc[:200] + "..." if len(desc) > 200 else desc

        results.append({
            "id":            item.get("id"),
            "name":          item.get("name", "Unknown"),
            "creator":       item.get("creator", {}).get("username", "Unknown"),
            "description":   desc or "No description.",
            "downloads":     downloads,
            "rating":        rating,
            "base_model":    base_model,
            "trigger_words": trigger_words,
            "download_url":  download_url,
            "preview_url":   preview_url,
            "model_url":     f"https://civitai.com/models/{item.get('id')}",
            "version_name":  latest.get("name", ""),
        })

    return results, f"✅ Found {len(results)} LoRA(s)."


def download_lora_from_result(result, directory="./loras"):
    """Download a LoRA from a search result dict."""
    os.makedirs(directory, exist_ok=True)
    url      = result["download_url"]
    name     = result["name"].replace(" ", "_").replace("/", "_")[:40]
    filename = f"{name}.safetensors"
    path     = os.path.join(directory, filename)

    print(f"[CivitaiSearch] Downloading {filename}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DiffuseCraft/1.0"})
        with urllib.request.urlopen(req) as response, open(path, "wb") as out_file:
            out_file.write(response.read())
        print(f"[CivitaiSearch] Saved to {path}")
        return path, f"✅ Downloaded: {filename}\nPath: {path}"
    except Exception as e:
        return None, f"❌ Download failed: {e}"


def format_results_html(results):
    """Format search results as an HTML string for display in Gradio."""
    if not results:
        return "<p>No results.</p>"

    html = ""
    for i, r in enumerate(results):
        stars = "⭐" * int(r["rating"]) if r["rating"] else "No rating"
        html += f"""
        <div style='border:1px solid #444; border-radius:8px; padding:12px; margin-bottom:12px;'>
            <b style='font-size:1.1em'>#{i+1} {r['name']}</b>
            <span style='color:#aaa; font-size:0.85em'> — {r['version_name']}</span><br>
            <span>👤 <b>{r['creator']}</b> &nbsp;|&nbsp;
                  🏗️ {r['base_model']} &nbsp;|&nbsp;
                  ⬇️ {r['downloads']:,} &nbsp;|&nbsp;
                  {stars} {r['rating']}</span><br>
            <span style='color:#ccc; font-size:0.9em'>🎯 Trigger words: <code>{r['trigger_words']}</code></span><br>
            <span style='color:#bbb; font-size:0.85em'>{r['description']}</span><br>
            <a href='{r['model_url']}' target='_blank' style='color:#58a6ff'>🔗 View on Civitai</a>
        </div>
        """
    return html
