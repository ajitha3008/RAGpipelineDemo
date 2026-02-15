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
    "psychxgalore.wordpress.com": [
        "https://psychxgalore.wordpress.com/",
        "https://psychxgalore.wordpress.com/about/",
    ],
}

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def scrape_url(url: str) -> str:
    """Fetch a URL and return cleaned text content (cached locally)."""
    cache_file = CACHE_DIR / f"{_url_hash(url)}.txt"
    if cache_file.exists():
        return cache_file.read_text()

    headers = {"User-Agent": "RAGPipelineDemo/1.0 (educational project)"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [!] Failed to fetch {url}: {e}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n".join(lines)

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
                parts.append(f"--- Content from {url} ---\n{text}")
        results[source] = "\n\n".join(parts)
    return results
