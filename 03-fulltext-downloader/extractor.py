"""
Extractor utilities for XML/PDF/HTML and standardized labeled text output.
"""

import json
import logging
import os
import re
from typing import Dict, Optional

import config as cfg

log = logging.getLogger(__name__)


def _import_fitz():
    try:
        import fitz
        return fitz
    except ImportError:
        log.warning("PyMuPDF (fitz) not installed; PDF extraction disabled")
        return None


def _import_lxml():
    try:
        from lxml import etree
        return etree
    except ImportError:
        log.warning("lxml not installed; XML extraction disabled")
        return None


def _import_bs4():
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup
    except ImportError:
        log.warning("beautifulsoup4 not installed; HTML extraction disabled")
        return None


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"\u00a0", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _safe_text(value: Optional[str]) -> str:
    text = _clean_text(value or "")
    return text if text else "N/A"


def _extract_abstract_from_text(text: str) -> str:
    if not text:
        return ""
    match = re.search(
        r"(?is)\babstract\b\s*[:\-]?\s*(.{120,2500}?)(?:\n\n|\bkeywords?\b|\bintroduction\b)",
        text,
    )
    if not match:
        return ""
    return _clean_text(match.group(1))


def _detect_language(text: str) -> str:
    txt = (text or "").strip()
    if len(txt) < 80:
        return "unknown"

    sample = txt[:6000]
    letters = re.findall(r"[A-Za-z]", sample)
    ascii_ratio = len(letters) / max(len(sample), 1)

    tokens = re.findall(r"[A-Za-z]+", sample.lower())
    if len(tokens) < 40:
        return "unknown"

    stop = {
        "the", "and", "of", "to", "in", "for", "with", "from", "that", "this",
        "is", "are", "was", "were", "on", "as", "by", "an", "be", "or", "we",
        "at", "which", "can", "have", "has", "these", "our", "their", "using",
    }
    stop_hits = sum(1 for t in tokens if t in stop)
    stop_ratio = stop_hits / max(len(tokens), 1)

    if ascii_ratio > 0.60 and stop_ratio > 0.04:
        return "en"
    if ascii_ratio < 0.35:
        return "non-en"
    return "unknown"


def _non_article_signal(text: str, title: str, abstract: str) -> bool:
    merged = f"{title}\n{abstract}\n{text}".lower()
    bad_patterns = [
        "cookie", "captcha", "sign in", "log in", "access through your institution",
        "buy article", "purchase", "for rights and permissions", "404", "page not found",
        "javascript is disabled", "not authorized", "institutional access",
    ]
    hits = sum(1 for p in bad_patterns if p in merged)
    if hits >= 2 and len(text) < 1800:
        return True
    if len(text) < 300 and len(abstract) < 120:
        return True
    return False


def _quality_penalties(text: str, abstract: str) -> dict:
    """Penalize noisy/truncated extractions that should not be labeled full text."""
    merged = f"{abstract}\n{text}"
    lower = merged.lower()
    reasons = []
    score = 0

    boilerplate_markers = [
        "click here to view figure",
        "article metrics",
        "cite this article",
        "journal impact factor",
        "journal is indexed in",
        "copy the following to cite",
        "share",
    ]
    marker_hits = sum(1 for m in boilerplate_markers if m in lower)
    if marker_hits >= 2:
        score += 2
        reasons.append("boilerplate_heavy")

    mojibake_hits = len(re.findall(r"(â€|â€“|â€™|Â°|Â)", merged))
    if mojibake_hits >= 5:
        score += 1
        reasons.append("encoding_noise")

    lines = [ln.strip() for ln in merged.splitlines() if ln.strip()]
    very_short = sum(1 for ln in lines if len(ln) <= 3)
    if lines and (very_short / len(lines)) > 0.08:
        score += 1
        reasons.append("truncation_fragments")

    if re.search(r"\bfigure\s+\d+\b.{0,120}click here to view figure", lower, flags=re.I | re.S):
        score += 1
        reasons.append("figure_placeholders")

    return {"score": score, "reasons": reasons}


def _classify_label(title: str, abstract: str, full_text: str) -> str:
    body_len = len(full_text)
    abs_len = len(abstract)
    title_ok = title and title != "N/A"

    if body_len >= cfg.MIN_COMPLETE_FULLTEXT_CHARS:
        return "complete_full_text"
    if body_len >= cfg.MIN_PARTIAL_FULLTEXT_CHARS:
        return "partial_full_text"
    if abs_len >= 120:
        return "abstract_only"
    if title_ok:
        return "title_only"
    return "metadata_only"


def classify_extraction(extracted: Optional[dict]) -> dict:
    if not extracted:
        return {
            "language": "unknown",
            "label": "failed",
            "extraction_status": "failed",
            "is_english": False,
            "reasons": ["no_extraction"],
        }

    title = _safe_text(extracted.get("title"))
    abstract = _safe_text(extracted.get("abstract"))
    full_text = _safe_text(extracted.get("full_text"))

    if title == "N/A" and abstract == "N/A" and full_text != "N/A":
        guess_title = full_text.split("\n", 1)[0][:300].strip()
        title = _safe_text(guess_title)

    language = _detect_language("\n".join([title, abstract, full_text]))
    non_article = _non_article_signal(full_text, title, abstract)
    penalties = _quality_penalties(full_text, abstract)

    if non_article:
        label = "non_article"
        status = "failed"
    else:
        label = _classify_label(title, abstract, full_text)
        if penalties["score"] >= 2 and label == "complete_full_text":
            label = "partial_full_text"
        if penalties["score"] >= 3 and label in {"complete_full_text", "partial_full_text"}:
            label = "abstract_only" if abstract != "N/A" else "title_only"
        status = "ok" if label == "complete_full_text" else "partial"
        if label in {"metadata_only", "title_only"}:
            status = "failed"

    is_english = language == "en"

    return {
        "language": language,
        "label": label,
        "extraction_status": status,
        "is_english": is_english,
        "title": title,
        "abstract": abstract,
        "full_text": full_text,
    }


def _extract_jats_xml(filepath: str) -> Optional[dict]:
    etree = _import_lxml()
    if etree is None:
        return None

    try:
        parser = etree.XMLParser(recover=True, remove_blank_text=True)
        tree = etree.parse(filepath, parser)
        root = tree.getroot()

        def all_text(el):
            return _clean_text(etree.tostring(el, method="text", encoding="unicode"))

        title = ""
        title_els = root.xpath(".//article-title")
        if title_els:
            title = all_text(title_els[0])

        abstract = ""
        abs_els = root.xpath(".//abstract")
        if abs_els:
            abstract = all_text(abs_els[0])

        sections = []
        full_text = ""
        body_els = root.xpath(".//body")
        if body_els:
            body = body_els[0]
            sec_els = body.xpath(".//sec")
            if sec_els:
                for sec in sec_els:
                    heading = all_text(sec.find("title")) if sec.find("title") is not None else ""
                    paras = sec.findall("p")
                    txt = "\n".join(all_text(p) for p in paras if all_text(p))
                    if txt:
                        sections.append({"heading": heading, "text": txt})
            if not sections:
                paras = body.findall(".//p")
                txt = "\n".join(all_text(p) for p in paras if all_text(p))
                if txt:
                    sections.append({"heading": "", "text": txt})

            full_text = "\n\n".join(
                (f"## {s['heading']}\n{s['text']}" if s["heading"] else s["text"])
                for s in sections
            )

        if not full_text:
            full_text = all_text(root)

        if len(full_text) < 200 and len(abstract) < 120:
            return None

        return {
            "format": "xml",
            "title": title,
            "abstract": abstract,
            "full_text": full_text,
            "sections": sections,
        }
    except Exception as e:
        log.debug("JATS extraction failed for %s: %s", filepath, e)
        return None


def _extract_generic_xml(filepath: str) -> Optional[dict]:
    etree = _import_lxml()
    if etree is None:
        return None

    try:
        parser = etree.XMLParser(recover=True, remove_blank_text=True)
        tree = etree.parse(filepath, parser)
        root = tree.getroot()
        text = _clean_text(etree.tostring(tree, method="text", encoding="unicode"))

        title = ""
        tnodes = root.xpath(".//*[local-name()='title']")
        if tnodes:
            title = _clean_text(etree.tostring(tnodes[0], method="text", encoding="unicode"))

        abstract = ""
        anodes = root.xpath(".//*[contains(local-name(),'abstract')]")
        if anodes:
            abstract = _clean_text(etree.tostring(anodes[0], method="text", encoding="unicode"))

        if not abstract:
            abstract = _extract_abstract_from_text(text[:6000])

        if len(text) < 200 and len(abstract) < 120:
            return None

        return {
            "format": "xml",
            "title": title,
            "abstract": abstract,
            "full_text": text,
            "sections": [],
        }
    except Exception as e:
        log.debug("Generic XML extraction failed for %s: %s", filepath, e)
        return None


def _extract_pdf(filepath: str) -> Optional[dict]:
    fitz = _import_fitz()
    if fitz is None:
        return None

    try:
        doc = fitz.open(filepath)
        pages = []
        for page_num in range(len(doc)):
            txt = doc[page_num].get_text("text")
            if txt and txt.strip():
                pages.append(txt.strip())
        doc.close()

        if not pages:
            return None

        full_text = _clean_text("\n\n".join(pages))
        if len(full_text) < 200:
            return None

        lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
        title = lines[0][:300] if lines else ""
        abstract = _extract_abstract_from_text("\n".join(lines[:200]))

        return {
            "format": "pdf",
            "title": title,
            "abstract": abstract,
            "full_text": full_text,
            "sections": [],
            "pages": len(pages),
        }
    except Exception as e:
        log.debug("PDF extraction failed for %s: %s", filepath, e)
        return None


def _extract_html(filepath: str) -> Optional[dict]:
    BeautifulSoup = _import_bs4()
    if BeautifulSoup is None:
        return None

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        title = ""
        meta_title = soup.find("meta", attrs={"name": "citation_title"})
        if meta_title and meta_title.get("content"):
            title = meta_title.get("content")
        if not title:
            og = soup.find("meta", attrs={"property": "og:title"})
            if og and og.get("content"):
                title = og.get("content")
        if not title and soup.title:
            title = soup.title.get_text(" ", strip=True)
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(" ", strip=True)

        abstract = ""
        mdesc = soup.find("meta", attrs={"name": "description"})
        if mdesc and mdesc.get("content"):
            abstract = mdesc.get("content")
        if not abstract:
            cabs = soup.find("meta", attrs={"name": "citation_abstract"})
            if cabs and cabs.get("content"):
                abstract = cabs.get("content")

        article = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_=re.compile(r"article|content|body", re.I))
        )
        target = article if article else (soup.body or soup)

        full_text = _clean_text(target.get_text(separator="\n", strip=True))
        if not abstract:
            abstract = _extract_abstract_from_text(full_text[:8000])

        if len(full_text) < 200 and len(abstract) < 120:
            return None

        return {
            "format": "html",
            "title": title,
            "abstract": abstract,
            "full_text": full_text,
            "sections": [],
        }
    except Exception as e:
        log.debug("HTML extraction failed for %s: %s", filepath, e)
        return None


def extract_text(filepath: str) -> Optional[dict]:
    if not os.path.exists(filepath):
        return None

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".xml":
        return _extract_jats_xml(filepath) or _extract_generic_xml(filepath)
    if ext == ".pdf":
        return _extract_pdf(filepath)
    if ext in {".html", ".htm"}:
        return _extract_html(filepath)

    try:
        with open(filepath, "rb") as f:
            header = f.read(300)
        if header.startswith(b"%PDF"):
            return _extract_pdf(filepath)
        if b"<?xml" in header or b"<article" in header:
            return _extract_jats_xml(filepath) or _extract_generic_xml(filepath)
        if b"<html" in header.lower():
            return _extract_html(filepath)
    except Exception:
        return None
    return None


def save_labeled_text(
    doi: str,
    year: str,
    source_url: str,
    source_format: str,
    classified: dict,
    output_dir: str,
) -> str:
    safe_doi = re.sub(r'[<>:"/\\|?*]', '_', doi.replace('/', '__'))[:200]
    yr = str(year).strip() if year else "unknown"
    outpath = os.path.join(output_dir, f"{yr}_{safe_doi}.txt")

    lines = [
        f"### DOI: {doi}",
        f"### YEAR: {yr}",
        f"### SOURCE_URL: {source_url or 'N/A'}",
        f"### SOURCE_FORMAT: {source_format or 'unknown'}",
        f"### LANGUAGE: {classified.get('language', 'unknown')}",
        f"### LABEL: {classified.get('label', 'failed')}",
        f"### EXTRACTION_STATUS: {classified.get('extraction_status', 'failed')}",
        "",
        "### TITLE",
        _safe_text(classified.get("title")),
        "",
        "### ABSTRACT",
        _safe_text(classified.get("abstract")),
        "",
        "### FULL_TEXT",
        _safe_text(classified.get("full_text")),
        "",
    ]

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return outpath


# Backward-compat wrapper

def save_extracted(doi: str, extracted: Dict, output_dir: str, year: str = "") -> str:
    classified = classify_extraction(extracted)
    return save_labeled_text(doi, year, "", extracted.get("format", "unknown"), classified, output_dir)
