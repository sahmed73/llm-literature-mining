#!/usr/bin/env python3
"""
Compound extractor.

Pipeline:
  Phase 1: Build or reuse cached chunks from local PDF/XML papers + metadata
  Phase 2: Run bounded paper-level LLM extraction over cached chunks

The pipeline automatically resumes using cached chunk files and an extraction
checkpoint. Paths are configured in config.py.
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import requests

import config as cfg
from chunker import build_chunk_cache
from extractor import extract_from_paper, get_llm_stats, reset_llm_stats

LOG_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"
LOG_DATE = "%Y-%m-%d %H:%M:%S"

log_file = os.path.join(cfg.OUTPUT_DIR, "extractor.log")
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


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def _load_checkpoint(path: str) -> Dict[str, dict]:
    records = {}
    if not os.path.exists(path):
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = rec.get("paper_id") or rec.get("doi")
            if key:
                records[key] = rec
    return records


def _append_checkpoint(path: str, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def _save_json(path: str, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _checkpoint_matches_paper(checkpoint_rec: dict, paper: dict) -> bool:
    old_source = checkpoint_rec.get("source")
    new_source = paper.get("source")
    compatible_sources = {
        ("paper_only", "paper+metadata"),
        ("paper+metadata", "paper_only"),
    }
    if old_source != new_source and (old_source, new_source) not in compatible_sources:
        return False
    if checkpoint_rec.get("source_format") != paper.get("source_format"):
        return False
    if int(checkpoint_rec.get("num_chunks", 0)) != len(paper.get("chunks", [])):
        return False
    return True


def _ollama_hosts() -> List[str]:
    hosts = getattr(cfg, "OLLAMA_HOSTS", None) or [cfg.OLLAMA_HOST]
    deduped = []
    for host in hosts:
        if host and host not in deduped:
            deduped.append(host)
    return deduped


def _check_ollama() -> bool:
    hosts = _ollama_hosts()
    model_base = cfg.MODEL_NAME.split(":")[0].lower()
    for host in hosts:
        log.info("Checking Ollama at %s ...", host)
        try:
            r = requests.get(f"{host}/api/tags", timeout=10)
            if r.status_code != 200:
                log.error("Ollama at %s returned %d", host, r.status_code)
                return False
            models = [m["name"] for m in r.json().get("models", [])]
            log.info("Ollama ready at %s. Models: %s", host, models)
            if not any(model_base in model.lower() for model in models):
                log.info("Pulling model '%s' on %s ...", cfg.MODEL_NAME, host)
                pr = requests.post(
                    f"{host}/api/pull",
                    json={"name": cfg.MODEL_NAME, "stream": False},
                    timeout=600,
                )
                log.info("Model pull on %s: %d", host, pr.status_code)
        except Exception as e:
            log.error("Cannot reach Ollama at %s: %s", host, e)
            return False
    return True


def _extract_paper_task(paper: dict, ollama_host: str) -> dict:
    t0 = time.time()
    result = extract_from_paper(paper, ollama_host=ollama_host)
    return {
        "result": result,
        "host": ollama_host,
        "elapsed_seconds": time.time() - t0,
    }


def _log_extract_progress(
    total_papers: int,
    skipped: int,
    processed: int,
    total_compounds: int,
    total_chunk_calls: int,
    total_pending_chunks: int,
    total_llm_wall: float,
    t0: float,
    worker_count: int,
    host_count: int,
) -> None:
    elapsed = time.time() - t0
    paper_rate = processed / max(elapsed, 0.1)
    chunk_rate = total_chunk_calls / max(elapsed, 0.1)
    remaining_chunks = max(total_pending_chunks - total_chunk_calls, 0)
    eta_chunks = remaining_chunks / max(chunk_rate, 0.001)
    avg_chunk = total_llm_wall / max(total_chunk_calls, 1)
    log.info(
        "Extract: %d/%d (skipped=%d, processed=%d, workers=%d, hosts=%d) | chunks=%d/%d | additives=%d | papers/sec=%.2f | chunks/sec=%.2f | avg_chunk=%.2fs | ETA(chunks): %s",
        skipped + processed,
        total_papers,
        skipped,
        processed,
        worker_count,
        host_count,
        total_chunk_calls,
        total_pending_chunks,
        total_compounds,
        paper_rate,
        chunk_rate,
        avg_chunk,
        _fmt_time(eta_chunks),
    )


def phase_extract(papers: List[dict]) -> List[dict]:
    log.info("=" * 72)
    log.info("PHASE 2: Paper-Level LLM Antioxidant Extraction")
    log.info("=" * 72)
    log.info("Model: %s", cfg.MODEL_NAME)
    log.info("Papers available: %d", len(papers))
    log.info("LLM workers: %d", cfg.LLM_WORKERS)
    log.info("Ollama hosts: %s", _ollama_hosts())

    existing = _load_checkpoint(cfg.EXTRACTION_CHECKPOINT)
    if existing:
        log.info("Auto-resume found %d previously extracted papers.", len(existing))

    reset_llm_stats()
    results_by_key: Dict[str, dict] = {}
    pending = []
    skipped = 0
    processed = 0
    total_compounds = 0
    total_chunk_calls = 0
    total_llm_wall = 0.0
    t0 = time.time()

    for paper in papers:
        key = paper.get("paper_id") or paper.get("doi") or ""
        if key in existing and _checkpoint_matches_paper(existing[key], paper):
            results_by_key[key] = existing[key]
            skipped += 1
            total_compounds += existing[key].get("num_compounds", 0)
            continue
        if key in existing:
            old = existing[key]
            log.info(
                "Re-extracting %s because cache changed | old_source=%s old_chunks=%s -> new_source=%s new_chunks=%d",
                key,
                old.get("source", "unknown"),
                old.get("num_chunks", "unknown"),
                paper.get("source", "unknown"),
                len(paper.get("chunks", [])),
            )
        pending.append((key, paper))

    total_pending_chunks = sum(len(paper.get("chunks", [])) for _, paper in pending)

    log.info(
        "Extraction queue prepared: skipped=%d | pending=%d | pending_chunks=%d | mode=%s",
        skipped,
        len(pending),
        total_pending_chunks,
        "serial" if cfg.LLM_WORKERS <= 1 else "multiprocessing (one worker per host)",
    )

    if cfg.LLM_WORKERS <= 1:
        host = _ollama_hosts()[0]
        for idx, (key, paper) in enumerate(pending, 1):
            payload = _extract_paper_task(paper, host)
            result = payload["result"]
            results_by_key[key] = result
            _append_checkpoint(cfg.EXTRACTION_CHECKPOINT, result)
            processed += 1
            total_compounds += result.get("num_compounds", 0)
            total_chunk_calls += result.get("num_chunks", 0)
            total_llm_wall += payload["elapsed_seconds"]

            if processed % 5 == 0 or idx == len(pending):
                _log_extract_progress(
                    total_papers=len(papers),
                    skipped=skipped,
                    processed=processed,
                    total_compounds=total_compounds,
                    total_chunk_calls=total_chunk_calls,
                    total_pending_chunks=total_pending_chunks,
                    total_llm_wall=total_llm_wall,
                    t0=t0,
                    worker_count=1,
                    host_count=1,
                )
    else:
        hosts = _ollama_hosts()
        if not hosts:
            raise RuntimeError("No Ollama hosts configured.")
        worker_count = min(cfg.LLM_WORKERS, len(hosts))
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(_extract_paper_task, paper, hosts[idx % len(hosts)]): key
                for idx, (key, paper) in enumerate(pending)
            }
            for future in as_completed(futures):
                key = futures[future]
                payload = future.result()
                result = payload["result"]
                results_by_key[key] = result
                _append_checkpoint(cfg.EXTRACTION_CHECKPOINT, result)
                processed += 1
                total_compounds += result.get("num_compounds", 0)
                total_chunk_calls += result.get("num_chunks", 0)
                total_llm_wall += payload["elapsed_seconds"]

                if processed % 5 == 0 or processed == len(pending):
                    _log_extract_progress(
                        total_papers=len(papers),
                        skipped=skipped,
                        processed=processed,
                        total_compounds=total_compounds,
                        total_chunk_calls=total_chunk_calls,
                        total_pending_chunks=total_pending_chunks,
                        total_llm_wall=total_llm_wall,
                        t0=t0,
                        worker_count=worker_count,
                        host_count=len(hosts),
                    )

    if total_chunk_calls:
        log.info(
            "LLM timing summary: processed_papers=%d | total_chunks=%d | total_worker_wall=%s | avg_per_chunk=%.2fs",
            processed,
            total_chunk_calls,
            _fmt_time(total_llm_wall),
            total_llm_wall / max(total_chunk_calls, 1),
        )
    else:
        log.info("LLM timing summary: no new chunk calls were made during this run.")

    return [results_by_key[paper.get("paper_id") or paper.get("doi") or ""] for paper in papers]


def main():
    parser = argparse.ArgumentParser(
        description="Extract antioxidant additives from papers with a local LLM (Ollama)."
    )
    parser.add_argument(
        "--model", default=None,
        help=f"Ollama model to use (default: LLM_MODEL env var or {cfg.MODEL_NAME}).",
    )
    args = parser.parse_args()
    if args.model:
        cfg.MODEL_NAME = args.model

    log.info("=" * 72)
    log.info("Compound extractor – starting")
    log.info("=" * 72)
    log.info("Papers dir     : %s", cfg.INPUT_PAPERS_DIR)
    log.info("Metadata JSON  : %s", cfg.INPUT_METADATA_JSON or "None")
    log.info("Chunk workers  : %d", cfg.MAX_WORKERS)
    log.info("LLM workers    : %d", cfg.LLM_WORKERS)
    log.info("Resume         : auto")
    log.info("")

    t_start = time.time()

    log.info("=" * 72)
    log.info("PHASE 1: Build / Reuse Chunk Cache")
    log.info("=" * 72)
    papers, warnings = build_chunk_cache()
    _save_json(cfg.INGEST_WARNINGS_JSON, warnings)
    _save_json(
        cfg.CHUNK_MANIFEST_JSON,
        {
            "papers_cached": len(papers),
            "warnings": len(warnings),
            "chunks_dir": cfg.CHUNKS_DIR,
        },
    )

    if not papers:
        log.error("No cached papers are available. Check your configured inputs.")
        return

    log.info("Chunk cache ready for %d papers.", len(papers))
    if warnings:
        log.info("Warnings written to %s", cfg.INGEST_WARNINGS_JSON)

    if not _check_ollama():
        log.error("Ollama not available. Exiting.")
        return

    results = phase_extract(papers)
    _save_json(cfg.EXTRACTED_COMPOUNDS_JSON, results)
    with open(cfg.EXTRACTED_COMPOUNDS_JSONL, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    papers_with_hits = sum(1 for r in results if r.get("num_compounds", 0) > 0)
    total_hits = sum(r.get("num_compounds", 0) for r in results)

    log.info("")
    log.info("Output files:")
    log.info("  Chunk manifest:      %s", cfg.CHUNK_MANIFEST_JSON)
    log.info("  Ingest warnings:     %s", cfg.INGEST_WARNINGS_JSON)
    log.info("  Extraction checkpoint: %s", cfg.EXTRACTION_CHECKPOINT)
    log.info("  Final JSONL:         %s", cfg.EXTRACTED_COMPOUNDS_JSONL)
    log.info("  Final JSON:          %s", cfg.EXTRACTED_COMPOUNDS_JSON)
    log.info("")
    log.info("Papers with additives: %d/%d", papers_with_hits, len(results))
    log.info("Total additive entries: %d", total_hits)
    log.info("Total elapsed: %s", _fmt_time(time.time() - t_start))


if __name__ == "__main__":
    main()
