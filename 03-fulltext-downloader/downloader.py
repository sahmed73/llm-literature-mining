"""
Downloader for full-text files and DOI landing HTML.
"""

import logging
import os
import re
import time
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests

import config as cfg

log = logging.getLogger(__name__)

_session = requests.Session()
_session.headers.update({"User-Agent": cfg.USER_AGENT})

_domain_ts: dict = {}


def _doi_to_filename(doi: str, fmt: str, year: str = "") -> str:
    safe = re.sub(r'[<>:"/\\|?*]', '_', doi.replace('/', '__'))
    safe = safe[:200]
    ext = {"pdf": ".pdf", "xml": ".xml", "html": ".html"}.get(fmt, ".bin")
    yr = str(year).strip() if year else "unknown"
    return f"{yr}_{safe}{ext}"


def _rate_limit(domain: str):
    delay = cfg.RATE_LIMIT_DEFAULT
    dl = domain.lower()
    if "unpaywall" in dl:
        delay = cfg.RATE_LIMIT_UNPAYWALL
    elif "ncbi" in dl or "nih.gov" in dl:
        delay = cfg.RATE_LIMIT_PMC
    elif "europepmc" in dl or "ebi.ac.uk" in dl:
        delay = cfg.RATE_LIMIT_EPMC
    elif "semanticscholar" in dl:
        delay = cfg.RATE_LIMIT_S2
    elif "core.ac.uk" in dl:
        delay = cfg.RATE_LIMIT_CORE
    else:
        delay = cfg.RATE_LIMIT_PUBLISHER

    now = time.time()
    last = _domain_ts.get(domain, 0.0)
    wait = delay - (now - last)
    if wait > 0:
        time.sleep(wait)
    _domain_ts[domain] = time.time()


def _looks_like_html(data: bytes) -> bool:
    head = data[:300].lower()
    return b"<html" in head or b"<!doctype html" in head


def _is_small_or_error_body(data: bytes) -> bool:
    if len(data) < 120:
        return True
    txt = data[:5000].decode("utf-8", errors="ignore").lower()
    bad = [
        "access denied", "captcha", "enable javascript", "page not found",
        "error", "temporarily unavailable", "forbidden",
    ]
    return any(token in txt for token in bad)


def download_file(url: str, doi: str, fmt: str, year: str = "") -> Optional[str]:
    filename = _doi_to_filename(doi, fmt, year)
    filepath = os.path.join(cfg.PAPERS_DIR, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) > 100:
        return filepath

    domain = urlparse(url).netloc
    accept_map = {
        "pdf": "application/pdf",
        "xml": "application/xml, text/xml",
        "html": "text/html",
    }
    headers = {"Accept": accept_map.get(fmt, "*/*")}

    for attempt in range(1, cfg.MAX_RETRIES + 1):
        try:
            _rate_limit(domain)
            r = _session.get(
                url,
                headers=headers,
                timeout=cfg.REQUEST_TIMEOUT,
                stream=True,
                allow_redirects=True,
            )

            if r.status_code == 200:
                cl = r.headers.get("Content-Length")
                if cl and int(cl) > cfg.MAX_FILE_SIZE:
                    return None

                ct = (r.headers.get("Content-Type") or "").lower()

                # Read once into memory to validate response before saving
                content = r.content
                if len(content) > cfg.MAX_FILE_SIZE:
                    return None
                if _is_small_or_error_body(content):
                    return None

                if fmt == "pdf":
                    if not content.startswith(b"%PDF"):
                        if _looks_like_html(content):
                            return None
                    # force pdf extension if publisher mislabeled
                    filepath = os.path.join(cfg.PAPERS_DIR, _doi_to_filename(doi, "pdf", year))

                elif fmt == "xml":
                    if "pdf" in ct or content.startswith(b"%PDF"):
                        filepath = os.path.join(cfg.PAPERS_DIR, _doi_to_filename(doi, "pdf", year))
                        fmt = "pdf"
                    elif _looks_like_html(content):
                        filepath = os.path.join(cfg.PAPERS_DIR, _doi_to_filename(doi, "html", year))
                        fmt = "html"

                elif fmt == "html":
                    if content.startswith(b"%PDF"):
                        filepath = os.path.join(cfg.PAPERS_DIR, _doi_to_filename(doi, "pdf", year))
                        fmt = "pdf"
                    elif b"<?xml" in content[:300] or "xml" in ct:
                        filepath = os.path.join(cfg.PAPERS_DIR, _doi_to_filename(doi, "xml", year))
                        fmt = "xml"

                with open(filepath, "wb") as f:
                    f.write(content)

                if os.path.getsize(filepath) < 120:
                    os.remove(filepath)
                    return None
                return filepath

            if r.status_code == 429:
                wait = min(30, cfg.RETRY_BACKOFF ** attempt * 5)
                time.sleep(wait)
                continue
            if r.status_code in (403, 404, 451):
                return None

            time.sleep(cfg.RETRY_BACKOFF ** attempt)

        except requests.exceptions.Timeout:
            time.sleep(cfg.RETRY_BACKOFF ** attempt)
        except Exception:
            time.sleep(cfg.RETRY_BACKOFF ** attempt)

    return None


def download_doi_html(doi: str, year: str = "") -> Tuple[Optional[str], Optional[str]]:
    """Resolve DOI landing page and save final HTML snapshot.

    Returns: (filepath, final_url)
    """
    if not doi:
        return None, None

    url = f"https://doi.org/{doi}"
    domain = "doi.org"

    for attempt in range(1, cfg.MAX_RETRIES + 1):
        try:
            _rate_limit(domain)
            r = _session.get(
                url,
                headers={"Accept": "text/html,application/xhtml+xml"},
                timeout=cfg.REQUEST_TIMEOUT,
                allow_redirects=True,
            )

            if r.status_code != 200:
                if r.status_code in (403, 404, 451):
                    return None, None
                time.sleep(cfg.RETRY_BACKOFF ** attempt)
                continue

            content = r.content
            if _is_small_or_error_body(content):
                return None, r.url
            if not _looks_like_html(content):
                return None, r.url

            filepath = os.path.join(cfg.PAPERS_DIR, _doi_to_filename(doi, "html", year))
            with open(filepath, "wb") as f:
                f.write(content)

            if os.path.getsize(filepath) < 200:
                return None, r.url

            return filepath, r.url

        except requests.exceptions.Timeout:
            time.sleep(cfg.RETRY_BACKOFF ** attempt)
        except Exception:
            time.sleep(cfg.RETRY_BACKOFF ** attempt)

    return None, None
