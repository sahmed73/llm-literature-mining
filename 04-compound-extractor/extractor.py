"""
Serial LLM extraction of antioxidant additives from cached text chunks.
"""

import json
import logging
import re
import time
from typing import Dict, List, Optional

import requests

import config as cfg

log = logging.getLogger(__name__)

_session = requests.Session()
_LLM_STATS = {
    "chunks": 0,
    "total_seconds": 0.0,
    "successes": 0,
    "failures": 0,
}


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def reset_llm_stats() -> None:
    _LLM_STATS["chunks"] = 0
    _LLM_STATS["total_seconds"] = 0.0
    _LLM_STATS["successes"] = 0
    _LLM_STATS["failures"] = 0


def get_llm_stats() -> Dict[str, float]:
    chunks = int(_LLM_STATS["chunks"])
    total_seconds = float(_LLM_STATS["total_seconds"])
    return {
        "chunks": chunks,
        "total_seconds": total_seconds,
        "successes": int(_LLM_STATS["successes"]),
        "failures": int(_LLM_STATS["failures"]),
        "avg_seconds_per_chunk": (total_seconds / chunks) if chunks else 0.0,
    }


def _call_ollama(prompt: str, ollama_host: Optional[str] = None) -> Optional[str]:
    host = ollama_host or cfg.OLLAMA_HOST
    payload = {
        "model": cfg.MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": cfg.TEMPERATURE,
            "num_predict": 2048,
        },
    }
    if cfg.NUM_CTX:
        payload["options"]["num_ctx"] = cfg.NUM_CTX
    if cfg.THINK is not None:
        payload["think"] = cfg.THINK

    for attempt in range(1, cfg.MAX_RETRIES + 1):
        try:
            r = _session.post(
                f"{host}/api/chat",
                json=payload,
                timeout=cfg.REQUEST_TIMEOUT,
            )
            if r.status_code == 200:
                data = r.json()
                return data.get("message", {}).get("content", "")
            log.warning("Ollama returned %d from %s (attempt %d)", r.status_code, host, attempt)
        except requests.exceptions.Timeout:
            log.warning("Ollama timeout from %s (attempt %d)", host, attempt)
        except Exception as e:
            log.warning("Ollama error from %s: %s (attempt %d)", host, e, attempt)
        time.sleep(2 ** attempt)

    return None


def _parse_response(raw: str) -> List[dict]:
    if not raw:
        return []

    try:
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get("compounds"), list):
            return [item for item in data["compounds"] if isinstance(item, dict)]
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return [item for item in data["items"] if isinstance(item, dict)]
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[^{}]*"compounds"\s*:\s*\[.*?\]\s*\}', raw, re.DOTALL)
    if match:
        try:
            return _parse_response(match.group())
        except Exception:
            return []

    log.debug("Failed to parse LLM response: %s", raw[:200])
    return []


def _clean_field(value: object, default: str = "NOT FOUND") -> str:
    text = str(value or "").strip()
    return text if text else default


def _is_excluded(name: str) -> bool:
    return name.lower().strip() in cfg.EXCLUDE_COMPOUNDS


def _normalize_compound(item: dict, chunk_id: str) -> Optional[dict]:
    name = str(item.get("compound_name") or item.get("name") or "").strip()
    if not name or _is_excluded(name):
        return None

    return {
        "compound_name": name,
        "smiles": _clean_field(item.get("smiles")),
        "formula": _clean_field(item.get("formula")),
        "structural_info": _clean_field(item.get("structural_info")),
        "role": _clean_field(item.get("role")),
        "evidence": _clean_field(item.get("evidence"), default=""),
        "chunk_id": chunk_id,
    }


def extract_from_chunk(chunk: dict, ollama_host: Optional[str] = None) -> List[dict]:
    chunk_id = chunk.get("chunk_id", "")
    t0 = time.time()
    prompt = cfg.EXTRACTION_PROMPT.format(text=chunk["text"])
    host = ollama_host or cfg.OLLAMA_HOST
    raw = _call_ollama(prompt, ollama_host=host)
    elapsed = time.time() - t0
    _LLM_STATS["chunks"] += 1
    _LLM_STATS["total_seconds"] += elapsed

    if raw is None:
        _LLM_STATS["failures"] += 1
        avg = _LLM_STATS["total_seconds"] / max(_LLM_STATS["chunks"], 1)
        log.warning(
            "LLM chunk failed: %s | host=%s | chars=%d | elapsed=%s | running_avg=%s",
            chunk_id or "unknown",
            host,
            len(chunk.get("text", "")),
            _fmt_elapsed(elapsed),
            _fmt_elapsed(avg),
        )
        return []

    compounds = []
    for item in _parse_response(raw):
        normalized = _normalize_compound(item, chunk.get("chunk_id", ""))
        if normalized:
            compounds.append(normalized)
    _LLM_STATS["successes"] += 1
    avg = _LLM_STATS["total_seconds"] / max(_LLM_STATS["chunks"], 1)
    log.info(
        "LLM chunk complete: %s | host=%s | chars=%d | compounds=%d | elapsed=%s | running_avg=%s",
        chunk_id or "unknown",
        host,
        len(chunk.get("text", "")),
        len(compounds),
        _fmt_elapsed(elapsed),
        _fmt_elapsed(avg),
    )
    return compounds


def extract_from_paper(paper: dict, ollama_host: Optional[str] = None) -> dict:
    """
    Extract antioxidant additives from all cached chunks for one paper.
    """
    aggregated = {}
    chunks = paper.get("chunks", [])

    for chunk in chunks:
        compounds = extract_from_chunk(chunk, ollama_host=ollama_host)
        for compound in compounds:
            key = compound["compound_name"].lower().strip()
            if key not in aggregated:
                aggregated[key] = {
                    "compound_name": compound["compound_name"],
                    "smiles": compound["smiles"],
                    "formula": compound["formula"],
                    "structural_info": compound["structural_info"],
                    "role": compound["role"],
                    "evidence": [],
                    "source_chunk_ids": [],
                }
            rec = aggregated[key]
            if rec["smiles"] == "NOT FOUND" and compound["smiles"] != "NOT FOUND":
                rec["smiles"] = compound["smiles"]
            if rec["formula"] == "NOT FOUND" and compound["formula"] != "NOT FOUND":
                rec["formula"] = compound["formula"]
            if rec["structural_info"] == "NOT FOUND" and compound["structural_info"] != "NOT FOUND":
                rec["structural_info"] = compound["structural_info"]
            if rec["role"] == "NOT FOUND" and compound["role"] != "NOT FOUND":
                rec["role"] = compound["role"]
            if compound["evidence"] and compound["evidence"] not in rec["evidence"]:
                rec["evidence"].append(compound["evidence"])
            if compound["chunk_id"] and compound["chunk_id"] not in rec["source_chunk_ids"]:
                rec["source_chunk_ids"].append(compound["chunk_id"])

    return {
        "paper_id": paper.get("paper_id", ""),
        "doi": paper.get("doi", ""),
        "title": paper.get("title", ""),
        "year": paper.get("year", ""),
        "source": paper.get("source", ""),
        "source_format": paper.get("source_format", ""),
        "source_filepath": paper.get("source_filepath", ""),
        "chunk_cache_path": paper.get("chunk_cache_path", ""),
        "abstract_used": bool(paper.get("abstract")),
        "num_chunks": len(chunks),
        "num_compounds": len(aggregated),
        "compounds": sorted(aggregated.values(), key=lambda x: x["compound_name"].lower()),
    }
