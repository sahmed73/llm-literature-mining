"""
Read PDF/XML papers, merge optional metadata, and cache chunked text.
"""

import json
import logging
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import config as cfg

log = logging.getLogger(__name__)


def _import_fitz():
    try:
        import fitz
        return fitz
    except ImportError:
        return None


def _import_lxml():
    try:
        from lxml import etree
        return etree
    except ImportError:
        return None


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"\u00a0", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


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


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value or "").strip("_").lower()


def _derive_doi_and_year(stem: str) -> Tuple[str, str]:
    year = ""
    base = stem
    match = re.match(r"^(\d{4}|unknown)_(.+)$", stem)
    if match:
        year = "" if match.group(1) == "unknown" else match.group(1)
        base = match.group(2)
    doi = base.replace("__", "/", 1) if "__" in base else ""
    return doi, year


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _load_metadata(path: Optional[str]) -> Tuple[Dict[str, dict], Dict[str, dict], List[dict]]:
    by_doi: Dict[str, dict] = {}
    by_slug: Dict[str, dict] = {}
    records: List[dict] = []

    if not path:
        return by_doi, by_slug, records
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if isinstance(data.get("papers"), list):
            records = data["papers"]
        elif isinstance(data.get("items"), list):
            records = data["items"]
        else:
            records = [data]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("Metadata JSON must be a list or an object with papers/items.")

    for rec in records:
        if not isinstance(rec, dict):
            continue
        doi = (rec.get("doi") or "").strip()
        slug = _safe_slug(doi or rec.get("paper_id") or rec.get("file_name") or rec.get("title", ""))
        if doi:
            by_doi[doi] = rec
        if slug:
            by_slug[slug] = rec

    return by_doi, by_slug, records


def _extract_pdf(filepath: str) -> Optional[dict]:
    fitz = _import_fitz()
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF extraction.")

    doc = fitz.open(filepath)
    try:
        pages = []
        for page_num in range(len(doc)):
            txt = doc[page_num].get_text("text")
            if txt and txt.strip():
                pages.append(txt.strip())
    finally:
        doc.close()

    if not pages:
        return None

    full_text = _clean_text("\n\n".join(pages))
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    title = lines[0][:300] if lines else ""
    abstract = _extract_abstract_from_text("\n".join(lines[:200]))

    return {
        "title": title,
        "abstract": abstract,
        "full_text": full_text,
        "source_format": "pdf",
    }


def _extract_xml(filepath: str) -> Optional[dict]:
    etree = _import_lxml()
    if etree is None:
        raise RuntimeError("lxml is required for XML extraction.")

    parser = etree.XMLParser(recover=True, remove_blank_text=True)
    tree = etree.parse(filepath, parser)
    root = tree.getroot()

    def all_text(el) -> str:
        return _clean_text(etree.tostring(el, method="text", encoding="unicode"))

    title = ""
    title_els = root.xpath(".//article-title")
    if title_els:
        title = all_text(title_els[0])
    if not title:
        generic_titles = root.xpath(".//*[local-name()='title']")
        if generic_titles:
            title = all_text(generic_titles[0])

    abstract = ""
    abs_els = root.xpath(".//abstract")
    if abs_els:
        abstract = all_text(abs_els[0])

    sections = []
    body_els = root.xpath(".//body")
    if body_els:
        body = body_els[0]
        sec_els = body.xpath(".//sec")
        if sec_els:
            for sec in sec_els:
                heading = all_text(sec.find("title")) if sec.find("title") is not None else ""
                paras = sec.findall("p")
                txt = "\n".join(all_text(p) for p in paras if all_text(p))
                section = f"{heading}\n{txt}".strip()
                if section:
                    sections.append(section)

    full_text = "\n\n".join(sections).strip()
    if not full_text:
        full_text = all_text(root)
    if not abstract:
        abstract = _extract_abstract_from_text(full_text[:6000])

    return {
        "title": title,
        "abstract": abstract,
        "full_text": _clean_text(full_text),
        "source_format": "xml",
    }


def extract_text(filepath: str) -> dict:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        extracted = _extract_pdf(filepath)
    elif ext == ".xml":
        extracted = _extract_xml(filepath)
    else:
        raise ValueError(
            f"Unsupported file type '{ext or 'unknown'}' for {filepath}. "
            f"Only {sorted(cfg.SUPPORTED_FILE_TYPES)} are supported."
        )

    if not extracted:
        raise ValueError(f"Could not extract usable text from {filepath}")
    return extracted


def _merge_texts(abstract: str, full_text: str) -> str:
    abstract = _clean_text(abstract)
    full_text = _clean_text(full_text)
    if not abstract:
        return full_text
    if not full_text:
        return abstract
    if abstract[:120] and abstract[:120] in full_text[:1200]:
        return full_text
    return f"{abstract}\n\n{full_text}"


def chunk_text(text: str, paper_id: str = "") -> List[dict]:
    """
    Split text into overlapping chunks without dropping any text.
    """
    max_len = cfg.MAX_CHUNK_CHARS
    overlap = cfg.CHUNK_OVERLAP_CHARS

    if len(text) <= max_len:
        return [{
            "chunk_id": f"{paper_id}:chunk-1" if paper_id else "chunk-1",
            "text": text,
            "char_start": 0,
            "char_end": len(text),
        }]

    chunks = []
    start = 0
    chunk_num = 0

    while start < len(text):
        end = min(start + max_len, len(text))

        if end < len(text):
            para_break = text.rfind("\n\n", max(start, end - 500), end)
            if para_break > start:
                end = para_break
            else:
                sent_break = text.rfind(". ", max(start, end - 300), end)
                if sent_break > start:
                    end = sent_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunk_num += 1
            chunks.append({
                "chunk_id": f"{paper_id}:chunk-{chunk_num}" if paper_id else f"chunk-{chunk_num}",
                "text": chunk,
                "char_start": start,
                "char_end": end,
            })

        start = end - overlap if end < len(text) else len(text)

    return chunks


def _chunk_cache_path(paper_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id)[:240]
    return os.path.join(cfg.CHUNKS_DIR, f"{safe_id}.json")


def _build_metadata_only_record(rec: dict) -> Optional[dict]:
    abstract = _clean_text(rec.get("abstract") or "")
    if not abstract:
        return None

    doi = (rec.get("doi") or "").strip()
    title = _clean_text(rec.get("title") or "")
    year = str(rec.get("year") or "").strip()
    paper_id = doi or _safe_slug(rec.get("paper_id") or title or abstract[:80])
    text = abstract

    return {
        "paper_id": paper_id,
        "doi": doi,
        "title": title,
        "year": year,
        "abstract": abstract,
        "full_text": "",
        "text": text,
        "source": "metadata_only",
        "source_format": "metadata",
        "source_filepath": "",
        "metadata_found": True,
        "chunks": chunk_text(text, paper_id),
    }


def _load_cached_record(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception as e:
        log.warning("Could not inspect existing cache %s: %s", path, e)
        return None


def _enrich_cached_fulltext_record(record: dict, meta: dict) -> Tuple[dict, bool]:
    if not meta:
        return record, False

    updated = False
    enriched = dict(record)

    meta_title = _clean_text(meta.get("title") or "")
    meta_abstract = _clean_text(meta.get("abstract") or "")
    meta_year = str(meta.get("year") or "").strip()
    meta_doi = (meta.get("doi") or "").strip()

    if not _clean_text(enriched.get("title") or "") and meta_title:
        enriched["title"] = meta_title
        updated = True
    if not _clean_text(enriched.get("abstract") or "") and meta_abstract:
        enriched["abstract"] = meta_abstract
        updated = True
    if not str(enriched.get("year") or "").strip() and meta_year:
        enriched["year"] = meta_year
        updated = True
    if not (enriched.get("doi") or "").strip() and meta_doi:
        enriched["doi"] = meta_doi
        updated = True

    if meta and enriched.get("source") == "paper_only":
        enriched["source"] = "paper+metadata"
        updated = True

    if bool(enriched.get("metadata_found")) != bool(meta):
        enriched["metadata_found"] = bool(meta)
        updated = True

    return enriched, updated


def _process_paper_file(task: dict) -> dict:
    filepath = task["filepath"]
    filename = task["filename"]
    stem = task["stem"]
    doi = task["doi"]
    year = task["year"]
    meta = task["metadata"]

    extracted = extract_text(filepath)

    title = _clean_text(meta.get("title") or extracted.get("title") or "")
    abstract = _clean_text(meta.get("abstract") or extracted.get("abstract") or "")
    full_text = _clean_text(extracted.get("full_text") or "")
    combined_text = _merge_texts(abstract, full_text)

    slug = _safe_slug(doi or stem or filename)
    paper_id = doi or slug or stem

    record = {
        "paper_id": paper_id,
        "doi": doi,
        "title": title,
        "year": str(meta.get("year") or year or "").strip(),
        "abstract": abstract,
        "full_text": full_text,
        "text": combined_text,
        "source": "paper+metadata" if meta else "paper_only",
        "source_format": extracted.get("source_format", os.path.splitext(filename)[1].lstrip(".")),
        "source_filepath": filepath,
        "metadata_found": bool(meta),
    }
    record["chunks"] = chunk_text(record["text"], paper_id)
    record["chunk_cache_path"] = _chunk_cache_path(paper_id)
    return record


def build_chunk_cache() -> Tuple[List[dict], List[dict]]:
    """
    Build or reuse cached chunk files for all papers and metadata-only abstracts.
    Returns (paper_records, warnings).
    """
    if not os.path.isdir(cfg.INPUT_PAPERS_DIR):
        raise NotADirectoryError(f"Paper folder not found: {cfg.INPUT_PAPERS_DIR}")

    t_phase = time.time()
    log.info("Scanning input papers in %s", cfg.INPUT_PAPERS_DIR)
    metadata_by_doi, metadata_by_slug, metadata_records = _load_metadata(cfg.INPUT_METADATA_JSON)
    log.info(
        "Metadata loaded: %d DOI entries, %d slug entries, %d total records",
        len(metadata_by_doi), len(metadata_by_slug), len(metadata_records),
    )

    metadata_used = set()
    warnings: List[dict] = []
    tasks: List[dict] = []
    total_files_seen = 0
    supported_files = 0
    cached_files = 0
    cached_fulltext_files = 0
    cached_metadata_only_files = 0
    metadata_enriched_files = 0
    fresh_files = 0
    upgraded_files = 0
    metadata_matches = 0

    for filename in sorted(os.listdir(cfg.INPUT_PAPERS_DIR)):
        filepath = os.path.join(cfg.INPUT_PAPERS_DIR, filename)
        if not os.path.isfile(filepath):
            continue

        total_files_seen += 1
        stem, ext = os.path.splitext(filename)
        ext = ext.lower()
        doi, year = _derive_doi_and_year(stem)
        slug = _safe_slug(doi or stem)

        if ext not in cfg.SUPPORTED_FILE_TYPES:
            warnings.append({
                "file": filepath,
                "warning": f"unsupported_file_type:{ext or 'none'}",
            })
            continue

        supported_files += 1
        meta = metadata_by_doi.get(doi) or metadata_by_slug.get(slug) or {}
        if meta:
            metadata_matches += 1
            key = (meta.get("doi") or "").strip() or _safe_slug(meta.get("title") or meta.get("paper_id") or "")
            if key:
                metadata_used.add(key)

        paper_id = doi or slug or stem
        cache_path = _chunk_cache_path(paper_id)
        cached_record = _load_cached_record(cache_path)
        if cached_record:
            cached_source = cached_record.get("source", "")
            if cached_source == "metadata_only":
                upgraded_files += 1
                log.info(
                    "Upgrading metadata-only cache to full text for %s (%s)",
                    filename, cache_path,
                )
            else:
                enriched_record, updated = _enrich_cached_fulltext_record(cached_record, meta)
                if updated:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(enriched_record, f, ensure_ascii=False, indent=2)
                    metadata_enriched_files += 1
                    log.info(
                        "Enriched cached full-text record with metadata for %s (%s) without rechunking",
                        filename, cache_path,
                    )
                cached_files += 1
                cached_fulltext_files += 1
                log.info(
                    "Using cached full-text chunks for %s (%s)",
                    filename, cache_path,
                )
                continue
        elif os.path.exists(cache_path):
            cached_files += 1
            log.info(
                "Cache file exists but could not be inspected; reusing %s (%s)",
                filename, cache_path,
            )
            continue

        fresh_files += 1
        log.info(
            "Fresh chunking required for %s | doi=%s | year=%s | metadata=%s | upgrade=%s",
            filename, doi or "N/A", year or "N/A", "yes" if meta else "no",
            "yes" if cached_record and cached_record.get("source") == "metadata_only" else "no",
        )
        tasks.append({
            "filepath": filepath,
            "filename": filename,
            "stem": stem,
            "doi": doi,
            "year": year,
            "metadata": meta,
        })

    log.info(
        "Chunk scan summary: files_seen=%d | supported=%d | cached_fulltext=%d | metadata_enriched=%d | upgraded_from_metadata=%d | fresh_total=%d | metadata_matches=%d | unsupported=%d",
        total_files_seen, supported_files, cached_fulltext_files, metadata_enriched_files, upgraded_files, fresh_files, metadata_matches, len(warnings),
    )

    if tasks:
        log.info(
            "Starting fresh chunk generation for %d paper(s) with %d worker(s)",
            len(tasks), cfg.MAX_WORKERS,
        )
        completed = 0
        t_fresh = time.time()
        with ProcessPoolExecutor(max_workers=cfg.MAX_WORKERS) as pool:
            futures = {pool.submit(_process_paper_file, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    record = future.result()
                    with open(record["chunk_cache_path"], "w", encoding="utf-8") as f:
                        json.dump(record, f, ensure_ascii=False, indent=2)
                    completed += 1
                    elapsed = time.time() - t_fresh
                    rate = completed / max(elapsed, 0.1)
                    remaining = len(tasks) - completed
                    eta = remaining / max(rate, 0.001)
                    log.info(
                        "Chunked %d/%d: %s | format=%s | chunks=%d | text_chars=%d | metadata=%s | elapsed=%s | ETA=%s",
                        completed, len(tasks), task["filename"],
                        record.get("source_format", "unknown"),
                        len(record.get("chunks", [])),
                        len(record.get("text", "")),
                        "yes" if record.get("metadata_found") else "no",
                        _fmt_elapsed(elapsed),
                        _fmt_elapsed(eta),
                    )
                except Exception as e:
                    warnings.append({
                        "file": task["filepath"],
                        "warning": f"parse_failed:{e}",
                    })
                    log.warning("Chunking failed for %s: %s", task["filename"], e)
        log.info(
            "Fresh chunk generation finished in %s",
            _fmt_elapsed(time.time() - t_fresh),
        )
    else:
        log.info("No fresh chunking needed. All supported papers already have cached chunk files.")

    # Add metadata-only records for papers lacking full text.
    metadata_only_added = 0
    metadata_only_cached = 0
    for rec in metadata_records:
        if not isinstance(rec, dict):
            continue
        key = (rec.get("doi") or "").strip() or _safe_slug(rec.get("title") or rec.get("paper_id") or "")
        if key in metadata_used:
            continue
        record = _build_metadata_only_record(rec)
        if not record:
            continue
        cache_path = _chunk_cache_path(record["paper_id"])
        record["chunk_cache_path"] = cache_path
        existing_cache = _load_cached_record(cache_path)
        if existing_cache and existing_cache.get("source") != "metadata_only":
            continue
        if not os.path.exists(cache_path):
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            metadata_only_added += 1
            log.info(
                "Added metadata-only cache for %s | chunks=%d",
                record["paper_id"], len(record.get("chunks", [])),
            )
        else:
            metadata_only_cached += 1
            cached_metadata_only_files += 1

    paper_records = load_cached_papers()
    log.info(
        "Chunk cache complete: total_cached_records=%d | fresh_chunked=%d | reused_cached_fulltext=%d | metadata_enriched=%d | upgraded_from_metadata=%d | metadata_only_added=%d | metadata_only_already_cached=%d | warnings=%d | elapsed=%s",
        len(paper_records), fresh_files, cached_fulltext_files, metadata_enriched_files, upgraded_files, metadata_only_added, metadata_only_cached, len(warnings),
        _fmt_elapsed(time.time() - t_phase),
    )
    return paper_records, warnings


def load_cached_papers() -> List[dict]:
    papers = []
    if not os.path.isdir(cfg.CHUNKS_DIR):
        return papers

    for filename in sorted(os.listdir(cfg.CHUNKS_DIR)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(cfg.CHUNKS_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                papers.append(json.load(f))
        except Exception as e:
            log.warning("Could not read chunk cache %s: %s", path, e)

    log.info("Loaded %d cached paper record(s) from %s", len(papers), cfg.CHUNKS_DIR)
    return papers
