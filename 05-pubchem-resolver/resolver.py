"""
PubChem lookup helpers.
"""

from difflib import SequenceMatcher
import logging
import re
import time
import unicodedata
from typing import Dict, List, Optional

import requests

import config as cfg

log = logging.getLogger(__name__)

_session = requests.Session()
_session.headers.update({"User-Agent": "llm-lit-mining-antioxidants/1.0 (academic research)"})
_last_request_ts = 0.0
_api_call_count = 0

_GREEK_MAP = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "ε": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta",
    "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu",
    "ν": "nu", "ξ": "xi", "ο": "omicron", "π": "pi",
    "ρ": "rho", "σ": "sigma", "τ": "tau", "υ": "upsilon",
    "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
}


def clean_name(name: str) -> str:
    name = (name or "").strip()
    for greek, latin in _GREEK_MAP.items():
        name = name.replace(greek, latin)
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r"[^\x20-\x7E]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def normalized_key(name: str) -> str:
    cleaned = clean_name(name).lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _compare_key(name: str) -> str:
    cleaned = normalized_key(name)
    cleaned = re.sub(r"[^a-z0-9]+", "", cleaned)
    return cleaned


def _is_short_or_risky_name(name: str) -> bool:
    key = _compare_key(name)
    return len(key) <= 3


def _names_similar(query: str, candidate: str, threshold: float = 0.72) -> bool:
    q = _compare_key(query)
    c = _compare_key(candidate)
    if not q or not c:
        return False
    if q == c or q in c or c in q:
        return True
    return SequenceMatcher(None, q, c).ratio() >= threshold


def _rate_limit():
    global _last_request_ts
    now = time.time()
    wait = cfg.PUBCHEM_DELAY - (now - _last_request_ts)
    if wait > 0:
        time.sleep(wait)
    _last_request_ts = time.time()


def reset_api_stats() -> None:
    global _api_call_count
    _api_call_count = 0


def get_api_stats() -> Dict[str, int]:
    return {
        "api_calls": int(_api_call_count),
    }


def _get_json(url: str) -> Optional[dict]:
    global _api_call_count
    _rate_limit()
    _api_call_count += 1
    try:
        r = _session.get(url, timeout=cfg.PUBCHEM_TIMEOUT)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        if r.status_code == 429:
            log.warning("PubChem rate-limited; sleeping briefly")
            time.sleep(5)
            return None
        log.debug("PubChem returned %d for %s", r.status_code, url[:120])
        return None
    except Exception as e:
        log.debug("PubChem request failed: %s", e)
        return None


def _name_to_cids(name: str) -> List[int]:
    url = f"{cfg.PUBCHEM_BASE}/compound/name/{requests.utils.quote(name)}/cids/JSON"
    data = _get_json(url)
    if data and "IdentifierList" in data:
        return data["IdentifierList"].get("CID", [])[:cfg.PUBCHEM_MAX_CIDS]
    return []


def _synonym_to_cids(name: str) -> List[int]:
    url = (
        f"{cfg.PUBCHEM_BASE}/compound/name/{requests.utils.quote(name)}"
        f"/cids/JSON?name_type=complete"
    )
    data = _get_json(url)
    if data and "IdentifierList" in data:
        return data["IdentifierList"].get("CID", [])[:cfg.PUBCHEM_MAX_CIDS]
    return []


def _autocomplete_to_cids(name: str) -> List[int]:
    url = f"{cfg.PUBCHEM_AUTOCOMPLETE}/{requests.utils.quote(name)}/JSON?limit=3"
    data = _get_json(url)
    if not data:
        return []
    suggestions = data.get("dictionary_terms", {}).get("compound", [])
    if not suggestions:
        return []
    for suggestion in suggestions:
        if _names_similar(name, suggestion):
            return _name_to_cids(suggestion)
    return []


def _cid_properties(cid: int) -> Dict[str, str]:
    url = (
        f"{cfg.PUBCHEM_BASE}/compound/cid/{cid}"
        f"/property/SMILES,ConnectivitySMILES,IUPACName,Title/JSON"
    )
    data = _get_json(url)
    props = (data or {}).get("PropertyTable", {}).get("Properties", [])
    if not props:
        return {}
    prop = props[0]
    return {
        "pubchem_canonical_smiles": prop.get("SMILES") or "NOT FOUND",
        "pubchem_isomeric_smiles": prop.get("ConnectivitySMILES") or "NOT FOUND",
        "iupac_name": prop.get("IUPACName") or "NOT FOUND",
        "resolved_name": prop.get("Title") or "NOT FOUND",
    }


def resolve_name(name: str) -> dict:
    cleaned = clean_name(name)
    key = normalized_key(name)
    if not cleaned:
        return {
            "input_name": name,
            "normalized_name": key,
            "lookup_status": "invalid_name",
            "strategy": "none",
            "cid": "NOT FOUND",
            "resolved_name": "NOT FOUND",
            "pubchem_canonical_smiles": "NOT FOUND",
            "pubchem_isomeric_smiles": "NOT FOUND",
            "iupac_name": "NOT FOUND",
        }

    strategies = [
        ("exact", cleaned),
        ("abbreviation", cfg.ABBREVIATIONS.get(cleaned.lower(), "")),
        ("synonym", cleaned),
        ("autocomplete", cleaned if not _is_short_or_risky_name(cleaned) else ""),
    ]

    for strategy, value in strategies:
        if not value:
            continue
        if strategy == "exact":
            cids = _name_to_cids(value)
        elif strategy == "abbreviation":
            cids = _name_to_cids(value)
        elif strategy == "synonym":
            cids = _synonym_to_cids(value)
        else:
            cids = _autocomplete_to_cids(value)

        if not cids:
            continue

        for cid in cids:
            props = _cid_properties(cid)
            if props:
                if strategy == "autocomplete" and not _names_similar(name, props.get("resolved_name", "")):
                    continue
                return {
                    "input_name": name,
                    "normalized_name": key,
                    "lookup_status": "resolved",
                    "strategy": strategy,
                    "cid": str(cid),
                    "resolved_name": props["resolved_name"],
                    "pubchem_canonical_smiles": props["pubchem_canonical_smiles"],
                    "pubchem_isomeric_smiles": props["pubchem_isomeric_smiles"],
                    "iupac_name": props["iupac_name"],
                }

    return {
        "input_name": name,
        "normalized_name": key,
        "lookup_status": "not_found",
        "strategy": "none",
        "cid": "NOT FOUND",
        "resolved_name": "NOT FOUND",
        "pubchem_canonical_smiles": "NOT FOUND",
        "pubchem_isomeric_smiles": "NOT FOUND",
        "iupac_name": "NOT FOUND",
    }
