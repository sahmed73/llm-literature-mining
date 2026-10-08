#!/usr/bin/env python3
"""
Resolve extracted antioxidant names against PubChem and write clean CSV outputs.
"""

import csv
import json
import logging
import os
import sys
import time
from typing import Dict, List

import config as cfg
from resolver import get_api_stats, normalized_key, reset_api_stats, resolve_name

LOG_FMT = "%(asctime)s │ %(levelname)-7s │ %(message)s"
LOG_DATE = "%Y-%m-%d %H:%M:%S"

log_file = os.path.join(cfg.OUTPUT_DIR, "resolver.log")
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


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
            key = rec.get("normalized_name")
            if key:
                records[key] = rec
    return records


def _append_checkpoint(path: str, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def _safe_str(value) -> str:
    text = str(value or "").strip()
    return text if text else "NOT FOUND"


def _extract_mentions(papers: List[dict]) -> List[dict]:
    mentions = []
    for paper in papers:
        compounds = paper.get("compounds", [])
        for compound in compounds:
            compound_name = (
                compound.get("compound_name")
                or compound.get("name")
                or ""
            ).strip()
            if not compound_name:
                continue
            mentions.append({
                "paper_id": _safe_str(paper.get("paper_id"),),
                "doi": _safe_str(paper.get("doi")),
                "title": _safe_str(paper.get("title")),
                "year": _safe_str(paper.get("year")),
                "source": _safe_str(paper.get("source")),
                "source_format": _safe_str(paper.get("source_format")),
                "source_filepath": _safe_str(paper.get("source_filepath")),
                "compound_name": compound_name,
                "paper_smiles": _safe_str(compound.get("smiles")),
                "paper_formula": _safe_str(compound.get("formula")),
                "paper_structural_info": _safe_str(
                    compound.get("structural_info") or compound.get("structural_description")
                ),
            })
    return mentions


def _estimate_api_calls(num_pending: int) -> Dict[str, int]:
    # Per unique name, PubChem usage can vary by strategy and candidate CID count.
    # This is a rough planning range, not a guarantee.
    return {
        "min_calls": num_pending,
        "rough_max_calls": num_pending * (5 + cfg.PUBCHEM_MAX_CIDS),
    }


def _resolve_unique_names(mentions: List[dict]) -> Dict[str, dict]:
    unique_names = {}
    for mention in mentions:
        key = normalized_key(mention["compound_name"])
        if key and key not in unique_names:
            unique_names[key] = mention["compound_name"]

    checkpoint = _load_checkpoint(cfg.RESOLUTION_CHECKPOINT)
    if checkpoint:
        log.info("Auto-resume found %d previously resolved unique compound names.", len(checkpoint))

    results = dict(checkpoint)
    pending = [(key, name) for key, name in unique_names.items() if key not in results]
    estimate = _estimate_api_calls(len(pending))
    log.info(
        "Unique compound names: %d │ Already checkpointed: %d │ To resolve now: %d",
        len(unique_names), len(checkpoint), len(pending),
    )
    log.info(
        "Estimated PubChem API calls: min=%d │ rough_max=%d",
        estimate["min_calls"], estimate["rough_max_calls"],
    )

    reset_api_stats()
    t0 = time.time()
    resolved_now = 0
    unresolved_now = 0
    for idx, (key, name) in enumerate(pending, 1):
        result = resolve_name(name)
        results[key] = result
        _append_checkpoint(cfg.RESOLUTION_CHECKPOINT, result)
        if result.get("lookup_status") == "resolved":
            resolved_now += 1
        else:
            unresolved_now += 1

        if idx % 25 == 0 or idx == len(pending):
            elapsed = time.time() - t0
            rate = idx / max(elapsed, 0.1)
            remaining = len(pending) - idx
            eta = remaining / max(rate, 0.001)
            api_calls = get_api_stats()["api_calls"]
            log.info(
                "Resolve: %d/%d │ resolved=%d │ unresolved=%d │ api_calls=%d │ avg_api/name=%.1f │ %.2f names/sec │ ETA: %s",
                idx,
                len(pending),
                resolved_now,
                unresolved_now,
                api_calls,
                api_calls / max(idx, 1),
                rate,
                _fmt_time(eta),
            )

    return results


def _build_master_rows(mentions: List[dict], resolved_map: Dict[str, dict]) -> List[dict]:
    grouped = {}
    for mention in mentions:
        key = normalized_key(mention["compound_name"])
        resolved = resolved_map.get(key, {})
        cid = resolved.get("cid", "NOT FOUND")
        group_key = f"cid:{cid}" if cid != "NOT FOUND" else f"name:{key}"

        if group_key not in grouped:
            grouped[group_key] = {
                "cid": cid,
                "resolved_name": resolved.get("resolved_name", "NOT FOUND"),
                "lookup_status": resolved.get("lookup_status", "not_found"),
                "strategy": resolved.get("strategy", "none"),
                "pubchem_canonical_smiles": resolved.get("pubchem_canonical_smiles", "NOT FOUND"),
                "pubchem_isomeric_smiles": resolved.get("pubchem_isomeric_smiles", "NOT FOUND"),
                "iupac_name": resolved.get("iupac_name", "NOT FOUND"),
                "aliases": set(),
                "paper_smiles_values": set(),
                "paper_count": set(),
                "mention_count": 0,
            }

        rec = grouped[group_key]
        rec["aliases"].add(mention["compound_name"])
        if mention["paper_smiles"] != "NOT FOUND":
            rec["paper_smiles_values"].add(mention["paper_smiles"])
        rec["paper_count"].add(f"{mention['paper_id']}|{mention['doi']}")
        rec["mention_count"] += 1

    rows = []
    for rec in grouped.values():
        rows.append({
            "cid": rec["cid"],
            "resolved_name": rec["resolved_name"],
            "lookup_status": rec["lookup_status"],
            "strategy": rec["strategy"],
            "pubchem_canonical_smiles": rec["pubchem_canonical_smiles"],
            "pubchem_isomeric_smiles": rec["pubchem_isomeric_smiles"],
            "paper_smiles": "; ".join(sorted(rec["paper_smiles_values"])) or "NOT FOUND",
            "iupac_name": rec["iupac_name"],
            "aliases": "; ".join(sorted(rec["aliases"])),
            "paper_count": len(rec["paper_count"]),
            "mention_count": rec["mention_count"],
        })
    rows.sort(key=lambda row: (row["cid"] == "NOT FOUND", row["cid"], row["resolved_name"]))
    return rows


def _build_link_rows(mentions: List[dict], resolved_map: Dict[str, dict]) -> List[dict]:
    seen = set()
    rows = []
    for mention in mentions:
        key = normalized_key(mention["compound_name"])
        resolved = resolved_map.get(key, {})
        row = {
            "paper_id": mention["paper_id"],
            "doi": mention["doi"],
            "title": mention["title"],
            "year": mention["year"],
            "source": mention["source"],
            "source_format": mention["source_format"],
            "source_filepath": mention["source_filepath"],
            "compound_name": mention["compound_name"],
            "cid": resolved.get("cid", "NOT FOUND"),
            "resolved_name": resolved.get("resolved_name", "NOT FOUND"),
            "lookup_status": resolved.get("lookup_status", "not_found"),
            "strategy": resolved.get("strategy", "none"),
            "paper_smiles": mention["paper_smiles"],
            "pubchem_canonical_smiles": resolved.get("pubchem_canonical_smiles", "NOT FOUND"),
            "pubchem_isomeric_smiles": resolved.get("pubchem_isomeric_smiles", "NOT FOUND"),
            "paper_formula": mention["paper_formula"],
            "paper_structural_info": mention["paper_structural_info"],
        }
        dedup_key = (
            row["paper_id"], row["doi"], row["compound_name"], row["cid"],
            row["paper_smiles"], row["pubchem_canonical_smiles"],
        )
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        rows.append(row)
    return rows


def _write_csv(path: str, rows: List[dict], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    log.info("=" * 72)
    log.info("PubChem resolver – starting")
    log.info("=" * 72)
    log.info("Input JSON      : %s", cfg.INPUT_EXTRACTED_JSON)
    log.info("Resume          : auto")
    log.info("")

    t_start = time.time()
    papers = _load_json(cfg.INPUT_EXTRACTED_JSON)
    mentions = _extract_mentions(papers)
    log.info("Paper records loaded: %d", len(papers))
    log.info("Compound mentions found: %d", len(mentions))
    log.info("Resume scope: compound-dependent (unique normalized compound names), not paper-dependent")

    resolved_map = _resolve_unique_names(mentions)
    master_rows = _build_master_rows(mentions, resolved_map)
    link_rows = _build_link_rows(mentions, resolved_map)
    api_stats = get_api_stats()

    unresolved = [
        rec for rec in resolved_map.values()
        if rec.get("lookup_status") != "resolved"
    ]

    _write_csv(
        cfg.RESOLVED_COMPOUNDS_CSV,
        master_rows,
        [
            "cid", "resolved_name", "lookup_status", "strategy",
            "pubchem_canonical_smiles", "pubchem_isomeric_smiles",
            "paper_smiles", "iupac_name", "aliases", "paper_count", "mention_count",
        ],
    )
    _write_csv(
        cfg.PAPER_COMPOUND_LINKS_CSV,
        link_rows,
        [
            "paper_id", "doi", "title", "year", "source", "source_format",
            "source_filepath", "compound_name", "cid", "resolved_name",
            "lookup_status", "strategy", "paper_smiles",
            "pubchem_canonical_smiles", "pubchem_isomeric_smiles",
            "paper_formula", "paper_structural_info",
        ],
    )
    with open(cfg.UNRESOLVED_NAMES_JSON, "w", encoding="utf-8") as f:
        json.dump(unresolved, f, indent=2, ensure_ascii=False)
    with open(cfg.SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "papers": len(papers),
                "compound_mentions": len(mentions),
                "unique_names": len({normalized_key(m["compound_name"]) for m in mentions}),
                "master_rows": len(master_rows),
                "paper_compound_links": len(link_rows),
                "unresolved_names": len(unresolved),
                "pubchem_api_calls_this_run": api_stats["api_calls"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    log.info("")
    log.info("Output files:")
    log.info("  Checkpoint:          %s", cfg.RESOLUTION_CHECKPOINT)
    log.info("  Resolved compounds:  %s", cfg.RESOLVED_COMPOUNDS_CSV)
    log.info("  Paper links:         %s", cfg.PAPER_COMPOUND_LINKS_CSV)
    log.info("  Unresolved names:    %s", cfg.UNRESOLVED_NAMES_JSON)
    log.info("  Summary:             %s", cfg.SUMMARY_JSON)
    log.info("PubChem API calls this run: %d", api_stats["api_calls"])
    log.info("Total elapsed: %s", _fmt_time(time.time() - t_start))


if __name__ == "__main__":
    main()
