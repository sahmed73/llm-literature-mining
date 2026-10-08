#!/usr/bin/env python3
"""
Deduplicate resolved compounds by SMILES and keep only priority antioxidant types.
"""

import csv
import json
import logging
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional

import config as cfg
from classifier import canonicalize_smiles, choose_smiles, classify_smiles, heavy_atom_count

LOG_FMT = "%(asctime)s │ %(levelname)-7s │ %(message)s"
LOG_DATE = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FMT,
    datefmt=LOG_DATE,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(cfg.LOG_FILE, mode="a"),
    ],
)
log = logging.getLogger(__name__)


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _read_csv(path: str) -> List[dict]:
    nul_lines = 0
    nul_bytes = 0

    def _cleaned_lines():
        nonlocal nul_lines, nul_bytes
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            for line in f:
                count = line.count("\x00")
                if count:
                    nul_lines += 1
                    nul_bytes += count
                    line = line.replace("\x00", "")
                yield line

    rows = list(csv.DictReader(_cleaned_lines()))
    if nul_bytes:
        log.warning(
            "Removed %d NUL byte(s) across %d line(s) from %s",
            nul_bytes,
            nul_lines,
            path,
        )
    return rows


def _write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _safe_name(row: dict) -> str:
    return (
        row.get("resolved_name")
        or row.get("compound_name")
        or row.get("iupac_name")
        or "NOT FOUND"
    ).strip() or "NOT FOUND"


def _final_type(class_info: Dict[str, str]) -> str:
    family = class_info["ao_family"]
    if family == "phenolic":
        return "phenolic"
    if family == "aminic_nh":
        return "aminic"
    return "other"


def _select_better(existing: Optional[dict], candidate: dict) -> dict:
    if existing is None:
        return candidate

    existing_priority = 1 if existing["type"] in {"phenolic", "aminic"} else 0
    candidate_priority = 1 if candidate["type"] in {"phenolic", "aminic"} else 0
    if candidate_priority > existing_priority:
        return candidate

    existing_name = existing["name"]
    candidate_name = candidate["name"]
    if existing_name == "NOT FOUND" and candidate_name != "NOT FOUND":
        return candidate
    if candidate_name != "NOT FOUND" and len(candidate_name) > len(existing_name):
        return candidate

    existing_cid = existing["cid"]
    candidate_cid = candidate["cid"]
    if existing_cid == "NOT FOUND" and candidate_cid != "NOT FOUND":
        return candidate

    return existing


def _build_smiles_rows(rows: List[dict]) -> List[dict]:
    by_smiles: Dict[str, dict] = {}
    invalid_or_missing = 0

    for row in rows:
        smiles_info = choose_smiles(row)
        smiles_used = smiles_info["smiles_used"]
        canonical_smiles = canonicalize_smiles(smiles_used)
        class_info = classify_smiles(smiles_used)
        out = {
            "cid": (row.get("cid") or "NOT FOUND").strip() or "NOT FOUND",
            "name": _safe_name(row),
            "smiles": canonical_smiles or "NOT FOUND",
            "type": _final_type(class_info),
            "heavy_atoms": heavy_atom_count(canonical_smiles or smiles_used),
        }

        if canonical_smiles is None:
            invalid_or_missing += 1
            out["smiles"] = "NOT FOUND"
            out["type"] = "other"
            dedup_key = f"missing::{out['name'].lower()}::{out['cid']}"
        else:
            dedup_key = canonical_smiles

        by_smiles[dedup_key] = _select_better(by_smiles.get(dedup_key), out)

    final_rows = list(by_smiles.values())
    final_rows.sort(key=lambda row: (row["type"], row["name"].lower(), row["cid"], row["smiles"]))
    log.info("Rows lacking usable canonical SMILES: %d", invalid_or_missing)
    return final_rows


def main() -> None:
    log.info("=" * 72)
    log.info("SMILES filter – starting")
    log.info("=" * 72)
    log.info("Input CSV       : %s", cfg.INPUT_CSV)
    log.info("Output priority : %s", cfg.PRIORITY_CSV)
    log.info("Output other    : %s", cfg.OTHER_CSV)
    log.info("")

    t_start = time.time()
    rows = _read_csv(cfg.INPUT_CSV)
    log.info("Input rows loaded: %d", len(rows))

    smiles_rows = _build_smiles_rows(rows)
    type_counts = Counter(row["type"] for row in smiles_rows)
    priority_rows = [row for row in smiles_rows if row["type"] in {"phenolic", "aminic"}]
    other_rows = [row for row in smiles_rows if row["type"] not in {"phenolic", "aminic"}]
    heavy_atom_values = [row["heavy_atoms"] for row in smiles_rows if row.get("heavy_atoms") is not None]
    max_heavy_atoms = max(heavy_atom_values) if heavy_atom_values else 0

    fieldnames = ["cid", "name", "smiles", "type"]
    _write_csv(cfg.PRIORITY_CSV, priority_rows, fieldnames)
    _write_csv(cfg.OTHER_CSV, other_rows, fieldnames)

    summary = {
        "input_csv": cfg.INPUT_CSV,
        "input_rows": len(rows),
        "unique_smiles_rows": len(smiles_rows),
        "priority_rows": len(priority_rows),
        "other_rows": len(other_rows),
        "max_heavy_atoms": max_heavy_atoms,
        "type_counts": dict(type_counts),
    }
    with open(cfg.SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("Unique rows after SMILES deduplication: %d", len(smiles_rows))
    log.info("Priority rows (phenolic or aminic): %d", len(priority_rows))
    log.info("Other or unclassified rows: %d", len(other_rows))
    log.info("Max heavy atoms: %d", max_heavy_atoms)
    log.info("Type counts: %s", dict(type_counts))
    log.info("")
    log.info("Output files:")
    log.info("  Priority CSV: %s", cfg.PRIORITY_CSV)
    log.info("  Other CSV:    %s", cfg.OTHER_CSV)
    log.info("  Summary JSON: %s", cfg.SUMMARY_JSON)
    log.info("  Log file:     %s", cfg.LOG_FILE)
    log.info("Total elapsed: %s", _fmt_time(time.time() - t_start))


if __name__ == "__main__":
    main()
