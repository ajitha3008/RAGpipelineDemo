"""Web scraping utilities for the RAG pipeline."""

import hashlib
from pathlib import Path

import requests
from bs4 import BeautifulSoup

URLS = {
    "ajithayasmin.com": [
        "https://ajithayasmin.com/",
        "https://ajithayasmin.com/about",
    ],
    "ajithayasmin.wordpress.com": [
        "https://ajithayasmin.wordpress.com/",
    ],
}

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def scrape_url(url: str) -> str:
    """Fetch a URL and return cleaned text content (cached locally)."""
    cache_file = CACHE_DIR / f"{hashlib.md5(url.encode()).hexdigest()}.txt"
    if cache_file.exists():
        return cache_file.read_text()

    try:
        resp = requests.get(url, headers={"User-Agent": "RAGPipelineDemo/1.0"}, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [!] Failed to fetch {url}: {e}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    cleaned = "\n".join(line.strip() for line in soup.get_text(separator="\n", strip=True).splitlines() if line.strip())
    cache_file.write_text(cleaned)
    return cleaned


def scrape_all_sources() -> dict[str, str]:
    """Scrape all configured URLs and return {source_name: combined_text}."""
    results = {}
    for source, urls in URLS.items():
        parts = []
        for url in urls:
            print(f"  Scraping {url} ...")
            text = scrape_url(url)
            if text:
                parts.append(text)
        results[source] = "\n\n".join(parts)
    return results
