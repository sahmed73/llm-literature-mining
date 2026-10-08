import os
import re
import json
from urllib.parse import urlparse

import requests

# Edit this DOI link directly in the file.
DOI_LINK = "10.3390/lubricants12040115"

# Output folder for downloaded full text.
OUT_DIR = "test_downloads"

# Try in this order: JSON -> XML -> PDF.
PREFERRED_FORMATS = [
    ("json", "application/json"),
    ("xml", "application/xml, text/xml"),
    ("pdf", "application/pdf"),
]


session = requests.Session()
session.headers.update({"User-Agent": "SimpleDOIFullTextTester/1.0"})


def normalize_doi(value: str) -> str:
    value = value.strip()
    if value.startswith("http://") or value.startswith("https://"):
        p = urlparse(value)
        path = p.path.strip("/")
        if path.lower().startswith("doi.org/"):
            return path[8:]
        if p.netloc.lower().endswith("doi.org"):
            return path
        return value
    return value


def safe_name(doi_or_url: str, ext: str) -> str:
    base = re.sub(r'[<>:"/\\|?*]', "_", doi_or_url)
    base = base.replace("/", "__")
    return f"{base[:180]}.{ext}"


def looks_like_format(content: bytes, fmt: str) -> bool:
    if not content or len(content) < 100:
        return False
    head = content[:500].lower()
    if fmt == "pdf":
        return content.startswith(b"%PDF")
    if fmt == "xml":
        return b"<?xml" in head or b"<article" in head or b"<html" not in head
    if fmt == "json":
        return head.lstrip().startswith(b"{") or head.lstrip().startswith(b"[")
    return False


def is_metadata_only_json(content: bytes) -> bool:
    try:
        obj = json.loads(content.decode("utf-8", errors="ignore"))
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False

    # Typical DOI/Crossref metadata keys; this is not full-text content.
    metadata_signals = {"DOI", "title", "publisher", "author", "reference"}
    fulltext_signals = {"body", "full_text", "fullText", "sections", "content"}
    return bool(metadata_signals.intersection(obj.keys())) and not bool(
        fulltext_signals.intersection(obj.keys())
    )


def collect_publisher_urls(doi: str):
    urls = []
    try:
        r = session.get(
            f"https://doi.org/{doi}",
            headers={"Accept": "application/json"},
            timeout=30,
            allow_redirects=True,
        )
    except requests.RequestException:
        return urls

    if r.status_code != 200:
        return urls

    try:
        obj = r.json()
    except Exception:
        return urls

    if isinstance(obj, dict):
        for item in obj.get("link", []):
            if isinstance(item, dict) and item.get("URL"):
                urls.append(item["URL"])
        primary = obj.get("resource", {}).get("primary", {}).get("URL")
        if isinstance(primary, str):
            urls.extend([primary, primary.rstrip("/") + "/pdf", primary.rstrip("/") + "/xml"])

    # Keep order, remove duplicates.
    out = []
    seen = set()
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def try_download(url: str, fmt: str, accept: str):
    try:
        r = session.get(
            url,
            headers={"Accept": accept},
            timeout=30,
            allow_redirects=True,
        )
    except requests.RequestException:
        return None

    if r.status_code != 200:
        return None

    content = r.content
    ctype = (r.headers.get("Content-Type") or "").lower()

    if fmt == "pdf" and "pdf" in ctype:
        return content
    if fmt == "xml" and ("xml" in ctype or looks_like_format(content, "xml")):
        return content
    if fmt == "json" and ("json" in ctype or looks_like_format(content, "json")):
        if is_metadata_only_json(content):
            return None
        return content

    if looks_like_format(content, fmt):
        return content

    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    doi = normalize_doi(DOI_LINK)
    doi_url = f"https://doi.org/{doi}"
    extra_urls = collect_publisher_urls(doi)
    candidate_urls = [doi_url] + extra_urls

    for fmt, accept in PREFERRED_FORMATS:
        for url in candidate_urls:
            data = try_download(url, fmt, accept)
            if data:
                path = os.path.join(OUT_DIR, safe_name(doi, fmt))
                with open(path, "wb") as f:
                    f.write(data)
                print(f"found: {fmt} -> {path}")
                return

    print("not found")


if __name__ == "__main__":
    main()
