"""
De-duplication and merging of paper records from multiple API sources.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional


def _normalise_doi(doi: Optional[str]) -> Optional[str]:
    """Lower-case, strip URL prefix."""
    if not doi:
        return None
    doi = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi or None


def _normalise_title(title: Optional[str]) -> Optional[str]:
    """Lower, strip punctuation, collapse whitespace – for fuzzy matching."""
    if not title:
        return None
    t = unicodedata.normalize("NFKD", title)
    t = t.lower()
    t = re.sub(r"[^a-z0-9 ]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) > 10 else None  # skip very short titles


# Priority order for choosing the "best" abstract (higher = preferred)
_ABSTRACT_PRIORITY = {
    "semantic_scholar": 6,
    "europe_pmc": 5,
    "pubmed": 4,
    "openalex": 3,
    "crossref": 2,
    "core": 1,
    "lens": 1,
    "dimensions": 1,
}


def _pick_best(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two records for the same paper, keeping the richest data."""
    merged = dict(existing)

    # Abstract: prefer higher-priority source
    if incoming.get("abstract") and (
        not merged.get("abstract")
        or _ABSTRACT_PRIORITY.get(incoming["source_api"], 0)
           > _ABSTRACT_PRIORITY.get(merged.get("_abstract_source", merged.get("source_api", "")), 0)
    ):
        merged["abstract"] = incoming["abstract"]
        merged["_abstract_source"] = incoming["source_api"]

    # DOI: fill if missing
    if not merged.get("doi") and incoming.get("doi"):
        merged["doi"] = incoming["doi"]

    # Year: fill if missing
    if not merged.get("year") and incoming.get("year"):
        merged["year"] = incoming["year"]

    # Keywords: union (filter out None values)
    kw_set = set(merged.get("keywords") or [])
    kw_set.update(incoming.get("keywords") or [])
    kw_set.discard(None)
    merged["keywords"] = sorted(kw_set)

    # Authors: keep longer list
    if len(incoming.get("authors") or []) > len(merged.get("authors") or []):
        merged["authors"] = incoming["authors"]

    # Journal: fill if missing
    if not merged.get("journal") and incoming.get("journal"):
        merged["journal"] = incoming["journal"]

    # Citation count: keep max
    inc_cc = incoming.get("citation_count")
    cur_cc = merged.get("citation_count")
    if inc_cc is not None:
        if cur_cc is None or inc_cc > cur_cc:
            merged["citation_count"] = inc_cc

    # PMID: fill if missing
    if not merged.get("pmid") and incoming.get("pmid"):
        merged["pmid"] = incoming["pmid"]

    # Open access: fill if missing
    if merged.get("open_access") is None and incoming.get("open_access") is not None:
        merged["open_access"] = incoming["open_access"]

    # Track which APIs contributed
    sources = set(merged.get("source_apis", [merged.get("source_api", "")]))
    sources.add(incoming.get("source_api", ""))
    sources.discard(None)
    sources.discard("")
    merged["source_apis"] = sorted(sources)

    return merged


def deduplicate(papers: List[Dict[str, Any]],
                logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    De-duplicate a flat list of paper records.

    Strategy:
      1. Match by normalised DOI  (exact)
      2. Match by normalised title (exact on cleaned string)
      3. Everything else kept as-is

    Returns a list of unique, merged records.
    """
    by_doi: Dict[str, Dict[str, Any]] = {}
    by_title: Dict[str, Dict[str, Any]] = {}
    no_key: List[Dict[str, Any]] = []

    doi_hits = 0
    title_hits = 0

    for p in papers:
        norm_doi = _normalise_doi(p.get("doi"))
        norm_title = _normalise_title(p.get("title"))

        # Ensure source_apis list
        if "source_apis" not in p:
            p["source_apis"] = [p.get("source_api", "")]

        matched = False

        # 1. DOI match
        if norm_doi:
            if norm_doi in by_doi:
                by_doi[norm_doi] = _pick_best(by_doi[norm_doi], p)
                doi_hits += 1
                matched = True
            else:
                by_doi[norm_doi] = p
                matched = True

        # 2. Title match (only if no DOI match found this record)
        if not matched and norm_title:
            if norm_title in by_title:
                by_title[norm_title] = _pick_best(by_title[norm_title], p)
                title_hits += 1
                matched = True
            else:
                by_title[norm_title] = p
                matched = True

        if not matched:
            no_key.append(p)

    # Merge DOI-keyed and title-keyed (some DOI records may also appear
    # in title dict with a slightly different title; reconcile)
    final: Dict[str, Dict[str, Any]] = {}
    for ndoi, rec in by_doi.items():
        final[ndoi] = rec

    for ntitle, rec in by_title.items():
        ndoi = _normalise_doi(rec.get("doi"))
        if ndoi and ndoi in final:
            final[ndoi] = _pick_best(final[ndoi], rec)
        else:
            key = ndoi or f"_title_{ntitle[:80]}"
            final[key] = rec

    # Add no-key records
    for i, rec in enumerate(no_key):
        final[f"_nokey_{i}"] = rec

    unique = list(final.values())

    # Clean up internal helper fields
    for rec in unique:
        rec.pop("_abstract_source", None)
        rec.pop("source_api", None)  # replaced by source_apis list

    logger.info(
        "Deduplication: %d raw → %d unique  "
        "(DOI merges: %d, title merges: %d, no-key: %d)",
        len(papers), len(unique), doi_hits, title_hits, len(no_key),
    )

    # Stats
    with_abstract = sum(1 for r in unique if r.get("abstract"))
    with_doi = sum(1 for r in unique if r.get("doi"))
    logger.info(
        "Coverage: %d with DOI (%.1f%%), %d with abstract (%.1f%%)",
        with_doi, 100 * with_doi / max(len(unique), 1),
        with_abstract, 100 * with_abstract / max(len(unique), 1),
    )

    return unique
