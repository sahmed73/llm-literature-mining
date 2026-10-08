#!/usr/bin/env python3
"""
Metadata miner – main orchestrator.

Harvests paper metadata (DOI, title, abstract, year, keywords, …)
from multiple academic APIs in parallel, deduplicates, and writes
a single JSON output file.

Usage:
    python main.py                      # run all sources
    python main.py --sources openalex semantic_scholar
    python main.py --dry-run            # show query plan only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import config
from api_clients import ALL_CLIENTS, BaseClient
from dedup import deduplicate

# Path for incremental raw-paper checkpoint
_RAW_CHECKPOINT = os.path.join("output", "raw_papers_checkpoint.jsonl")


def _json_default(obj):
    """Fallback serialiser for json.dump – converts unknown types to str."""
    return str(obj)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging() -> logging.Logger:
    """Configure root + file + console logging."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(config.OUTPUT_DIR, config.LOG_FILE)

    fmt = "%(asctime)s │ %(levelname)-7s │ %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger("litmine")
    logger.setLevel(logging.DEBUG)

    # File handler – detailed
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(fh)

    # Console handler – info+
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Worker: harvest one (client, intent) pair
# ---------------------------------------------------------------------------

def _harvest_one(client: BaseClient, intent: Dict[str, Any],
                 logger: logging.Logger) -> List[Dict[str, Any]]:
    """Run a single client × intent harvest.  Returns list of paper dicts."""
    tag = f"{client.name}::{intent['id']}"
    try:
        papers = client.harvest(intent, logger)
        logger.info("✓ %s → %d papers", tag, len(papers))
        return papers
    except Exception as exc:
        logger.error("✗ %s FAILED: %s", tag, exc, exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(sources: List[str] | None = None, dry_run: bool = False):
    logger = _setup_logging()
    logger.info("=" * 72)
    logger.info("Metadata miner – starting")
    logger.info("=" * 72)

    # Filter clients
    clients = ALL_CLIENTS
    if sources:
        clients = [c for c in ALL_CLIENTS if c.name in sources]
        skipped = set(sources) - {c.name for c in clients}
        if skipped:
            logger.warning("Unknown source names (skipped): %s", skipped)

    intents = config.SEARCH_INTENTS
    total_tasks = len(clients) * len(intents)

    logger.info("Sources  : %s", [c.name for c in clients])
    logger.info("Intents  : %d search queries", len(intents))
    logger.info("Tasks    : %d  (sources × intents)", total_tasks)
    logger.info("Workers  : %d", config.MAX_WORKERS)
    logger.info("Output   : %s/%s", config.OUTPUT_DIR, config.OUTPUT_JSON)
    logger.info("-" * 72)

    if dry_run:
        logger.info("DRY RUN – printing query plan only.\n")
        from query_translator import TRANSLATORS
        for client in clients:
            for intent in intents:
                translator = TRANSLATORS.get(client.name)
                if translator:
                    q = translator(intent)
                    logger.info("  [%s] intent=%-35s  query=%s",
                                client.name, intent["id"], q)
        logger.info("\nDry run complete. No API calls were made.")
        return

    # ── Parallel harvest (with incremental checkpoint) ───────────────
    all_papers: List[Dict[str, Any]] = []
    source_stats: Dict[str, Dict[str, int]] = {}  # source → {ok, fail, papers}
    t_start = time.time()

    for client in clients:
        source_stats[client.name] = {"ok": 0, "fail": 0, "papers": 0}

    # Open checkpoint file for incremental writes (one JSON object per line)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    checkpoint_fh = open(_RAW_CHECKPOINT, "w", encoding="utf-8")
    logger.info("Incremental checkpoint: %s", _RAW_CHECKPOINT)

    completed = 0
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        future_map = {}
        for client in clients:
            for intent in intents:
                fut = pool.submit(_harvest_one, client, intent, logger)
                future_map[fut] = (client.name, intent["id"])

        for fut in as_completed(future_map):
            src_name, intent_id = future_map[fut]
            completed += 1
            try:
                papers = fut.result()
                all_papers.extend(papers)
                source_stats[src_name]["ok"] += 1
                source_stats[src_name]["papers"] += len(papers)

                # Stream raw papers to checkpoint file
                for p in papers:
                    checkpoint_fh.write(
                        json.dumps(p, ensure_ascii=False, default=_json_default) + "\n"
                    )
                checkpoint_fh.flush()

            except Exception as exc:
                logger.error("Future error [%s::%s]: %s", src_name, intent_id, exc)
                source_stats[src_name]["fail"] += 1

            if completed % 10 == 0 or completed == total_tasks:
                logger.info(
                    "Progress: %d/%d tasks done  |  %d raw papers so far",
                    completed, total_tasks, len(all_papers),
                )

    checkpoint_fh.close()
    logger.info("Checkpoint saved: %d raw papers in %s",
                len(all_papers), _RAW_CHECKPOINT)

    harvest_time = time.time() - t_start

    # ── Summary per source ────────────────────────────────────────────
    logger.info("-" * 72)
    logger.info("HARVEST SUMMARY  (%.1f min)", harvest_time / 60)
    logger.info("-" * 72)
    logger.info("%-20s %8s %8s %10s", "Source", "OK", "Failed", "Papers")
    logger.info("-" * 50)
    total_raw = 0
    for src, st in source_stats.items():
        logger.info("%-20s %8d %8d %10d", src, st["ok"], st["fail"], st["papers"])
        total_raw += st["papers"]
    logger.info("-" * 50)
    logger.info("%-20s %8s %8s %10d", "TOTAL (raw)", "", "", total_raw)

    # ── Deduplication ─────────────────────────────────────────────────
    logger.info("-" * 72)
    logger.info("DEDUPLICATION")
    logger.info("-" * 72)
    t_dedup = time.time()
    unique_papers = deduplicate(all_papers, logger)
    dedup_time = time.time() - t_dedup
    logger.info("Dedup took %.1f s", dedup_time)

    # ── Write output ──────────────────────────────────────────────────
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(config.OUTPUT_DIR, config.OUTPUT_JSON)

    output = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sources_used": [c.name for c in clients],
            "search_intents_count": len(intents),
            "raw_papers_total": total_raw,
            "unique_papers_total": len(unique_papers),
            "with_abstract": sum(1 for p in unique_papers if p.get("abstract")),
            "with_doi": sum(1 for p in unique_papers if p.get("doi")),
            "harvest_time_sec": round(harvest_time, 1),
            "source_stats": source_stats,
        },
        "papers": unique_papers,
    }

    logger.info("Writing %d unique papers to %s …", len(unique_papers), out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=_json_default)
    file_mb = os.path.getsize(out_path) / (1024 * 1024)
    logger.info("Saved %s (%.1f MB)", out_path, file_mb)

    # ── Final report ──────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("FINAL REPORT")
    logger.info("=" * 72)
    logger.info("  Raw papers collected : %d", total_raw)
    logger.info("  Unique papers        : %d", len(unique_papers))
    logger.info("  With abstract        : %d (%.1f%%)",
                output["metadata"]["with_abstract"],
                100 * output["metadata"]["with_abstract"] / max(len(unique_papers), 1))
    logger.info("  With DOI             : %d (%.1f%%)",
                output["metadata"]["with_doi"],
                100 * output["metadata"]["with_doi"] / max(len(unique_papers), 1))
    logger.info("  Harvest time         : %.1f min", harvest_time / 60)
    logger.info("  Output file          : %s", out_path)
    logger.info("=" * 72)
    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Metadata miner – harvest lubricant-antioxidant "
                    "paper metadata from multiple academic APIs."
    )
    parser.add_argument(
        "--sources", nargs="+", default=None,
        help="Subset of sources to run (default: all). "
             "Options: openalex semantic_scholar crossref pubmed "
             "europe_pmc core lens",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print query plan without making API calls.",
    )
    args = parser.parse_args()
    run(sources=args.sources, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
