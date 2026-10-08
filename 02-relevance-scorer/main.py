#!/usr/bin/env python3
"""
Relevance scorer – main orchestrator.

Loads papers from the metadata miner output, scores each with LLaMA via
Ollama, filters by relevance threshold, and saves a filtered JSON.

Usage:
    python main.py                          # score all papers
    python main.py --threshold 7            # custom threshold
    python main.py --resume                 # resume from checkpoint
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
from typing import Any, Dict, List

import requests

import config
from scorer import score_paper


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> logging.Logger:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    fmt = "%(asctime)s │ %(levelname)-7s │ %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger("scorer")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(config.LOG_FILE, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(ch)

    return logger


def _json_default(obj):
    return str(obj)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(logger: logging.Logger) -> Dict[str, Dict[str, Any]]:
    """Load previously scored papers from checkpoint (keyed by DOI or title)."""
    scored = {}
    if not os.path.exists(config.CHECKPOINT_FILE):
        return scored
    with open(config.CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = rec.get("doi") or rec.get("title", "")
                if key:
                    scored[key] = rec
            except json.JSONDecodeError:
                continue
    logger.info("Loaded %d previously scored papers from checkpoint.", len(scored))
    return scored


def _append_checkpoint(paper: Dict[str, Any], fh):
    """Append one scored paper to checkpoint file."""
    fh.write(json.dumps(paper, ensure_ascii=False, default=_json_default) + "\n")
    fh.flush()


# ---------------------------------------------------------------------------
# Ollama health check
# ---------------------------------------------------------------------------

def _wait_for_ollama(logger: logging.Logger, max_wait: int = 120):
    """Wait until Ollama server is responsive."""
    logger.info("Waiting for Ollama at %s ...", config.OLLAMA_URL)
    start = time.time()
    while time.time() - start < max_wait:
        try:
            r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                logger.info("Ollama ready. Available models: %s", models)
                return True
        except Exception:
            pass
        time.sleep(3)
    logger.error("Ollama not responding after %d seconds!", max_wait)
    return False


def _ensure_model(logger: logging.Logger):
    """Pull the model if not already available."""
    try:
        r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=10)
        models = [m["name"] for m in r.json().get("models", [])]
        if config.MODEL_NAME in models:
            logger.info("Model '%s' is available.", config.MODEL_NAME)
            return True
        # Try pulling
        logger.info("Pulling model '%s'...", config.MODEL_NAME)
        r = requests.post(
            f"{config.OLLAMA_URL}/api/pull",
            json={"name": config.MODEL_NAME},
            timeout=600,
        )
        logger.info("Model pull complete.")
        return True
    except Exception as exc:
        logger.error("Failed to ensure model: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(threshold: int | None = None, resume: bool = False):
    logger = _setup_logging()
    logger.info("=" * 72)
    logger.info("Relevance scorer – starting")
    logger.info("=" * 72)

    threshold = threshold or config.SCORE_THRESHOLD
    logger.info("Threshold  : %d / 10", threshold)
    logger.info("Model      : %s", config.MODEL_NAME)
    logger.info("Input      : %s", config.INPUT_JSON)
    logger.info("Output     : %s", config.OUTPUT_JSON)

    # Wait for Ollama
    if not _wait_for_ollama(logger):
        logger.error("Aborting – Ollama not available.")
        sys.exit(1)
    _ensure_model(logger)

    # Load input papers
    logger.info("Loading papers from %s ...", config.INPUT_JSON)
    with open(config.INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    papers = data.get("papers", [])
    input_metadata = data.get("metadata", {})
    logger.info("Loaded %d papers.", len(papers))

    # Load checkpoint for resume
    already_scored: Dict[str, Dict[str, Any]] = {}
    if resume:
        already_scored = _load_checkpoint(logger)

    # Open checkpoint for appending
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    checkpoint_fh = open(config.CHECKPOINT_FILE, "a", encoding="utf-8")

    # Score papers
    scored_papers: List[Dict[str, Any]] = []
    skipped = 0
    failed = 0
    t_start = time.time()

    total = len(papers)
    logger.info("Scoring %d papers (workers=%d) ...", total, config.MAX_WORKERS)
    logger.info("-" * 72)

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        future_map = {}
        for i, paper in enumerate(papers):
            key = paper.get("doi") or paper.get("title", "")

            # Skip if already scored (resume mode)
            if key and key in already_scored:
                scored_papers.append(already_scored[key])
                skipped += 1
                continue

            fut = pool.submit(score_paper, paper, logger)
            future_map[fut] = (i, paper)

        new_scored = 0
        for fut in as_completed(future_map):
            idx, original = future_map[fut]
            new_scored += 1
            try:
                result = fut.result()
                scored_papers.append(result)

                # Checkpoint
                _append_checkpoint(result, checkpoint_fh)

                score = result.get("relevance_score")
                if score is None:
                    failed += 1

                # Progress log
                if new_scored % 50 == 0 or new_scored == len(future_map):
                    elapsed = time.time() - t_start
                    rate = new_scored / elapsed if elapsed > 0 else 0
                    eta_min = (len(future_map) - new_scored) / rate / 60 if rate > 0 else 0
                    logger.info(
                        "Progress: %d/%d scored (skipped=%d, failed=%d) | "
                        "%.1f papers/sec | ETA: %.1f min",
                        new_scored + skipped, total, skipped, failed, rate, eta_min,
                    )

            except Exception as exc:
                logger.error("Scoring error for paper %d: %s", idx, exc)
                failed += 1
                scored_papers.append(dict(original, relevance_score=None,
                                          relevance_reason="error", score_confidence="none"))

    checkpoint_fh.close()
    score_time = time.time() - t_start

    # Filter by threshold
    filtered = [p for p in scored_papers
                if p.get("relevance_score") is not None
                and p["relevance_score"] >= threshold]

    # Score distribution
    score_dist = {}
    for p in scored_papers:
        s = p.get("relevance_score")
        if s is not None:
            score_dist[s] = score_dist.get(s, 0) + 1

    logger.info("-" * 72)
    logger.info("SCORING SUMMARY")
    logger.info("-" * 72)
    logger.info("  Total papers       : %d", total)
    logger.info("  Successfully scored: %d", total - failed)
    logger.info("  Failed to score    : %d", failed)
    logger.info("  Skipped (resumed)  : %d", skipped)
    logger.info("  Above threshold (%d): %d", threshold, len(filtered))
    logger.info("  Below threshold    : %d", total - failed - len(filtered))
    logger.info("  Scoring time       : %.1f min", score_time / 60)
    logger.info("  Score distribution :")
    for s in sorted(score_dist.keys()):
        bar = "█" * score_dist[s]
        logger.info("    %2d : %5d  %s", s, score_dist[s], bar[:80])

    # Save all scores (for analysis)
    all_scores_output = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_file": config.INPUT_JSON,
            "model": config.MODEL_NAME,
            "threshold": threshold,
            "total_papers": total,
            "scored_ok": total - failed,
            "scored_fail": failed,
            "above_threshold": len(filtered),
            "score_distribution": {str(k): v for k, v in sorted(score_dist.items())},
            "scoring_time_sec": round(score_time, 1),
            "input_metadata": input_metadata,
        },
        "papers": scored_papers,
    }
    with open(config.SCORES_JSON, "w", encoding="utf-8") as f:
        json.dump(all_scores_output, f, ensure_ascii=False, indent=2, default=_json_default)
    logger.info("All scores saved to %s", config.SCORES_JSON)

    # Save filtered output
    filtered_output = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_file": config.INPUT_JSON,
            "model": config.MODEL_NAME,
            "threshold": threshold,
            "total_input_papers": total,
            "filtered_papers": len(filtered),
            "score_distribution": {str(k): v for k, v in sorted(score_dist.items())},
            "input_metadata": input_metadata,
        },
        "papers": filtered,
    }
    with open(config.OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(filtered_output, f, ensure_ascii=False, indent=2, default=_json_default)

    file_mb = os.path.getsize(config.OUTPUT_JSON) / (1024 * 1024)
    logger.info("Filtered JSON saved to %s (%.1f MB)", config.OUTPUT_JSON, file_mb)

    logger.info("=" * 72)
    logger.info("FINAL: %d / %d papers passed (threshold >= %d)",
                len(filtered), total, threshold)
    logger.info("=" * 72)
    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score papers for relevance to lubricant antioxidants using a local LLM (Ollama)."
    )
    parser.add_argument(
        "--threshold", type=int, default=None,
        help=f"Score threshold (0-10, default: {config.SCORE_THRESHOLD}). "
             "Papers scoring >= threshold are kept.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint (skip already-scored papers).",
    )
    parser.add_argument(
        "--model", default=None,
        help=f"Ollama model to use (default: LLM_MODEL env var or {config.MODEL_NAME}).",
    )
    args = parser.parse_args()
    if args.model:
        config.MODEL_NAME = args.model
    run(threshold=args.threshold, resume=args.resume)


if __name__ == "__main__":
    main()
