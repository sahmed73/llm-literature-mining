#!/usr/bin/env python3
"""
Full-text downloader orchestrator.

Pipeline:
  1) Resolve candidate OA URLs (checkpointed)
  2) Per DOI fallback chain: HTML (DOI landing) -> XML -> PDF
  3) Extract title/abstract/full_text, classify quality, enforce English filter
  4) Save standardized labeled TXT output
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import config as cfg
from downloader import download_doi_html, download_file
from extractor import classify_extraction, extract_text, save_labeled_text
from resolver import resolve_paper, resolve_paper_by_format

LOG_FMT = "%(asctime)s │ %(levelname)-7s │ %(message)s"
LOG_DATE = "%Y-%m-%d %H:%M:%S"

log_file = os.path.join(cfg.OUTPUT_DIR, "downloader.log")
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FMT,
    datefmt=LOG_DATE,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a"),
    ],
)
log = logging.getLogger(__name__)


def _load_papers(path: str) -> list:
    log.info("Loading papers from %s ...", path)
    with open(path, "r") as f:
        papers = json.load(f)
    log.info("Loaded %d papers.", len(papers))
    return papers


def _load_checkpoint(path: str) -> Dict[str, dict]:
    records = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    doi = rec.get("doi")
                    if doi:
                        records[doi] = rec
                except json.JSONDecodeError:
                    continue
    return records


def _append_checkpoint(path: str, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    if seconds < 86400:
        return f"{seconds/3600:.1f}h"
    return f"{seconds/86400:.1f}d"


def phase_resolve(papers: list, resume: bool = True) -> list:
    log.info("=" * 72)
    log.info("PHASE 1: Resolving candidate URLs")
    log.info("=" * 72)

    resolve_ckpt = os.path.join(cfg.OUTPUT_DIR, "resolve_checkpoint.jsonl")
    existing = _load_checkpoint(resolve_ckpt) if resume else {}
    if existing:
        log.info("Loaded %d previously resolved papers.", len(existing))

    resolved = []
    skipped = 0
    found = 0
    not_found = 0
    t0 = time.time()

    for i, paper in enumerate(papers, 1):
        doi = paper.get("doi", "")
        if doi in existing:
            resolved.append(existing[doi])
            skipped += 1
            continue

        result = resolve_paper(paper)
        record = {
            "doi": doi,
            "title": paper.get("title", ""),
            "year": paper.get("year", ""),
            "resolved": result is not None,
            "url": result["url"] if result else None,
            "format": result["format"] if result else None,
            "source": result["source"] if result else None,
        }
        resolved.append(record)
        _append_checkpoint(resolve_ckpt, record)

        if result:
            found += 1
        else:
            not_found += 1

        total_processed = found + not_found
        if total_processed % 25 == 0 or i == len(papers):
            elapsed = time.time() - t0
            rate = total_processed / max(elapsed, 0.1)
            remaining = len(papers) - skipped - total_processed
            eta = remaining / max(rate, 0.001)
            log.info(
                "Resolve: %d/%d (skipped=%d, found=%d, not_found=%d) | %.1f papers/sec | ETA: %s",
                skipped + total_processed,
                len(papers),
                skipped,
                found,
                not_found,
                rate,
                _fmt_time(eta),
            )

    with open(cfg.RESOLVED_JSON, "w") as f:
        json.dump(resolved, f, indent=2, default=str)
    log.info("Saved resolved URLs to %s", cfg.RESOLVED_JSON)
    return resolved


def _label_rank(label: str) -> int:
    order = {
        "complete_full_text": 6,
        "partial_full_text": 5,
        "abstract_only": 4,
        "title_only": 3,
        "metadata_only": 2,
        "non_article": 1,
        "failed": 0,
    }
    return order.get(label, 0)


def _is_full_text_label(label: str) -> bool:
    return label in {"partial_full_text", "complete_full_text"}


def _merge_field(candidates: list, key: str, fallback: str = "N/A") -> str:
    best = ""
    for c in candidates:
        val = (c.get(key) or "").strip()
        if val and val != "N/A" and len(val) > len(best):
            best = val
    return best or fallback


def _attempt_extract(filepath: Optional[str], source_url: str, source_format: str) -> Optional[dict]:
    if not filepath:
        return None
    extracted = extract_text(filepath)
    classified = classify_extraction(extracted)
    classified["source_format"] = source_format
    classified["source_url"] = source_url or "N/A"
    classified["source_filepath"] = filepath
    return classified


def _process_paper(paper: dict, resolved_by_doi: Dict[str, dict]) -> dict:
    doi = (paper.get("doi") or "").strip()
    year = str(paper.get("year", "")).strip()
    input_title = (paper.get("title") or "").strip()

    record = {
        "doi": doi,
        "year": year,
        "input_title": input_title,
        "html_attempt": False,
        "xml_attempt": False,
        "pdf_attempt": False,
        "chosen_source": None,
        "source_url": None,
        "language": "unknown",
        "label": "failed",
        "extraction_status": "failed",
        "output_filepath": None,
        "failure_reason": None,
    }

    if not doi:
        record["failure_reason"] = "missing_doi"
        return record

    candidates = []

    # Step A: HTML from DOI landing page
    record["html_attempt"] = True
    html_path, final_url = download_doi_html(doi, year)
    html_candidate = _attempt_extract(html_path, final_url or f"https://doi.org/{doi}", "html")
    if html_candidate:
        candidates.append(html_candidate)

    # Step B: XML fallback
    needs_more = (not html_candidate) or (_label_rank(html_candidate["label"]) < _label_rank("partial_full_text"))
    if needs_more:
        record["xml_attempt"] = True
        xml_hit = resolve_paper_by_format(paper, "xml")
        if xml_hit:
            xml_path = download_file(xml_hit["url"], doi, "xml", year)
            xml_candidate = _attempt_extract(xml_path, xml_hit["url"], "xml")
            if xml_candidate:
                candidates.append(xml_candidate)

    # Step C: PDF fallback
    best_so_far = max(candidates, key=lambda c: _label_rank(c["label"]), default=None)
    needs_pdf = (best_so_far is None) or (_label_rank(best_so_far["label"]) < _label_rank("partial_full_text"))
    if needs_pdf:
        record["pdf_attempt"] = True
        pdf_hit = resolve_paper_by_format(paper, "pdf")
        if not pdf_hit:
            r = resolved_by_doi.get(doi)
            if r and r.get("format") == "pdf" and r.get("url"):
                pdf_hit = {"url": r["url"], "format": "pdf", "source": r.get("source", "resolved_pdf")}
        if pdf_hit:
            pdf_path = download_file(pdf_hit["url"], doi, "pdf", year)
            pdf_candidate = _attempt_extract(pdf_path, pdf_hit["url"], "pdf")
            if pdf_candidate:
                candidates.append(pdf_candidate)

    if not candidates:
        record["failure_reason"] = "no_extractable_content"
        return record

    # Prefer English candidates when English-only mode is enabled
    candidate_pool = candidates
    if cfg.ENGLISH_ONLY:
        en_candidates = [c for c in candidates if c.get("language") == "en"]
        if en_candidates:
            candidate_pool = en_candidates

    best = max(candidate_pool, key=lambda c: _label_rank(c.get("label", "failed")))

    # Enforce full-text only policy for final output.
    if not _is_full_text_label(best.get("label", "")):
        record.update(
            {
                "chosen_source": best.get("source_format"),
                "source_url": best.get("source_url"),
                "language": best.get("language", "unknown"),
                "label": "failed",
                "extraction_status": "failed",
                "failure_reason": "not_found_full_text",
            }
        )
        return record

    merged_title = _merge_field(candidates, "title", fallback=input_title or "N/A")
    merged_abstract = _merge_field(candidates, "abstract", fallback="N/A")

    final = {
        "title": merged_title,
        "abstract": merged_abstract,
        "full_text": best.get("full_text", "N/A"),
        "language": best.get("language", "unknown"),
        "label": best.get("label", "failed"),
        "extraction_status": best.get("extraction_status", "failed"),
    }

    if cfg.ENGLISH_ONLY and final["language"] != "en":
        final["label"] = "failed"
        final["extraction_status"] = "failed"
        record["failure_reason"] = "non_english"
        if not cfg.SAVE_NON_ENGLISH:
            record.update({
                "chosen_source": best.get("source_format"),
                "source_url": best.get("source_url"),
                "language": final["language"],
                "label": final["label"],
                "extraction_status": final["extraction_status"],
            })
            return record

    outpath = save_labeled_text(
        doi=doi,
        year=year,
        source_url=best.get("source_url", "N/A"),
        source_format=best.get("source_format", "unknown"),
        classified=final,
        output_dir=cfg.FINAL_TEXT_DIR,
    )

    record.update(
        {
            "chosen_source": best.get("source_format"),
            "source_url": best.get("source_url"),
            "language": final["language"],
            "label": final["label"],
            "extraction_status": final["extraction_status"],
            "output_filepath": outpath,
        }
    )

    return record


def phase_text_pipeline(papers: list, resolved: list, resume: bool = True) -> list:
    log.info("=" * 72)
    log.info("PHASE 2: HTML -> XML -> PDF extraction pipeline")
    log.info("=" * 72)

    existing = _load_checkpoint(cfg.TEXT_PIPELINE_LOG) if resume else {}
    if existing:
        log.info("Loaded %d previous text pipeline records.", len(existing))

    resolved_by_doi = {r.get("doi"): r for r in resolved if r.get("doi")}

    to_process = []
    done = []
    for p in papers:
        doi = (p.get("doi") or "").strip()
        if doi in existing:
            done.append(existing[doi])
        else:
            to_process.append(p)

    log.info("Need to process: %d (skipping %d already done)", len(to_process), len(done))

    results = list(done)
    if not to_process:
        return results

    t0 = time.time()
    ok = 0
    partial = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as pool:
        futures = {pool.submit(_process_paper, p, resolved_by_doi): p for p in to_process}

        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception as e:
                p = futures[fut]
                rec = {
                    "doi": (p.get("doi") or "").strip(),
                    "year": str(p.get("year", "")).strip(),
                    "label": "failed",
                    "extraction_status": "failed",
                    "failure_reason": f"exception:{e}",
                    "output_filepath": None,
                }

            results.append(rec)
            _append_checkpoint(cfg.TEXT_PIPELINE_LOG, rec)

            st = rec.get("extraction_status")
            if st == "ok":
                ok += 1
            elif st == "partial":
                partial += 1
            else:
                failed += 1

            if i % 25 == 0 or i == len(to_process):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 0.1)
                remaining = len(to_process) - i
                eta = remaining / max(rate, 0.001)
                log.info(
                    "Text pipeline: %d/%d (ok=%d, partial=%d, failed=%d) | %.1f papers/sec | ETA: %s",
                    i,
                    len(to_process),
                    ok,
                    partial,
                    failed,
                    rate,
                    _fmt_time(eta),
                )

    return results


def print_summary(papers: list, resolved: list, text_results: list):
    log.info("=" * 72)
    log.info("FINAL SUMMARY")
    log.info("=" * 72)

    total = len(papers)
    n_resolved = sum(1 for r in resolved if r.get("resolved"))
    n_saved = sum(1 for r in text_results if r.get("output_filepath"))

    label_counts = {}
    lang_counts = {}
    for r in text_results:
        label = r.get("label", "unknown")
        lang = r.get("language", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    log.info("Total papers:        %d", total)
    log.info("URLs resolved:       %d (%.1f%%)", n_resolved, 100 * n_resolved / max(total, 1))
    log.info("Final TXT saved:     %d (%.1f%%)", n_saved, 100 * n_saved / max(total, 1))

    log.info("")
    log.info("Label breakdown:")
    for k, v in sorted(label_counts.items(), key=lambda x: (-x[1], x[0])):
        log.info("  %-20s: %d", k, v)

    log.info("")
    log.info("Language breakdown:")
    for k, v in sorted(lang_counts.items(), key=lambda x: (-x[1], x[0])):
        log.info("  %-20s: %d", k, v)

    summary = {
        "total_papers": total,
        "urls_resolved": n_resolved,
        "final_txt_saved": n_saved,
        "label_breakdown": label_counts,
        "language_breakdown": lang_counts,
    }
    summary_path = os.path.join(cfg.OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved to %s", summary_path)


def main():
    parser = argparse.ArgumentParser(description="Full-text downloader")
    parser.add_argument("--input", default=cfg.INPUT_JSON, help="Input JSON file")
    parser.add_argument("--workers", type=int, default=cfg.MAX_WORKERS, help="Parallel workers")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh (ignore checkpoints)")
    parser.add_argument("--resolve-only", action="store_true", help="Only resolve URLs")
    parser.add_argument("--text-only", action="store_true", help="Skip resolve; use existing resolved_urls.json")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N papers (validation)")
    args = parser.parse_args()

    cfg.MAX_WORKERS = args.workers
    resume = not args.no_resume

    log.info("=" * 72)
    log.info("Full-text downloader – starting")
    log.info("=" * 72)
    log.info("Input   : %s", args.input)
    log.info("Workers : %d", cfg.MAX_WORKERS)
    log.info("Resume  : %s", resume)

    papers = _load_papers(args.input)
    if args.limit and args.limit > 0:
        papers = papers[: args.limit]
        log.info("Validation mode: limiting to first %d papers", len(papers))

    if args.text_only:
        if not os.path.exists(cfg.RESOLVED_JSON):
            log.error("No resolved_urls.json found; run resolve first.")
            return
        with open(cfg.RESOLVED_JSON, "r") as f:
            resolved = json.load(f)
    else:
        resolved = phase_resolve(papers, resume=resume)
        if args.resolve_only:
            return

    text_results = phase_text_pipeline(papers, resolved, resume=resume)
    print_summary(papers, resolved, text_results)


if __name__ == "__main__":
    main()
