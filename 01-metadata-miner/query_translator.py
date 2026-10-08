"""
Translate a generic search intent into the native query syntax
required by each academic API.
"""
from typing import Dict, List, Any


def _join_or(terms: List[str]) -> str:
    """Join terms with OR for boolean-supporting APIs."""
    return " OR ".join(f'"{t}"' for t in terms)


def _join_and(terms: List[str]) -> str:
    """Join terms with AND."""
    return " AND ".join(f'"{t}"' for t in terms)


def _plain(intent: Dict[str, Any], max_any: int = 3) -> str:
    """Plain keyword string (no booleans)."""
    parts = list(intent["must_contain"])
    parts.extend(intent["any_of"][:max_any])
    return " ".join(parts)


# ── OpenAlex ──────────────────────────────────────────────────────────────
def to_openalex(intent: Dict[str, Any]) -> str:
    """OpenAlex search parameter – plain keyword string."""
    return _plain(intent, max_any=3)


# ── Semantic Scholar ──────────────────────────────────────────────────────
def to_semantic_scholar(intent: Dict[str, Any]) -> str:
    """Semantic Scholar – plain keyword string."""
    return _plain(intent, max_any=2)


# ── CrossRef ──────────────────────────────────────────────────────────────
def to_crossref(intent: Dict[str, Any]) -> Dict[str, str]:
    """CrossRef – returns dict with query + optional query.bibliographic."""
    params = {"query": " ".join(intent["must_contain"])}
    if intent["any_of"]:
        params["query.bibliographic"] = " ".join(intent["any_of"][:3])
    return params


# ── PubMed (Entrez) ──────────────────────────────────────────────────────
def to_pubmed(intent: Dict[str, Any]) -> str:
    """PubMed Entrez – full boolean with field tags."""
    must = " AND ".join(
        f'"{t}"[Title/Abstract]' for t in intent["must_contain"]
    )
    if intent["any_of"]:
        any_part = " OR ".join(
            f'"{t}"[Title/Abstract]' for t in intent["any_of"]
        )
        return f"{must} AND ({any_part})"
    return must


# ── Europe PMC ────────────────────────────────────────────────────────────
def to_europe_pmc(intent: Dict[str, Any]) -> str:
    """Europe PMC – Lucene-style booleans."""
    must = " AND ".join(
        f'(TITLE:"{t}" OR ABSTRACT:"{t}")' for t in intent["must_contain"]
    )
    if intent["any_of"]:
        any_part = " OR ".join(
            f'TITLE:"{t}" OR ABSTRACT:"{t}"' for t in intent["any_of"]
        )
        return f"{must} AND ({any_part})"
    return must


# ── CORE ──────────────────────────────────────────────────────────────────
def to_core(intent: Dict[str, Any]) -> str:
    """CORE API – plain keyword string."""
    return _plain(intent, max_any=2)


# ── Lens.org ──────────────────────────────────────────────────────────────
def to_lens(intent: Dict[str, Any]) -> Dict[str, Any]:
    """Lens.org – Elasticsearch-style JSON body."""
    must_clauses = [
        {"match_phrase": {"title_abstract_keyword": t}}
        for t in intent["must_contain"]
    ]
    should_clauses = [
        {"match_phrase": {"title_abstract_keyword": t}}
        for t in intent["any_of"]
    ]
    bool_q: Dict[str, Any] = {"must": must_clauses}
    if should_clauses:
        bool_q["should"] = should_clauses
        bool_q["minimum_should_match"] = 1
    return {"query": {"bool": bool_q}}


# ── Dimensions ────────────────────────────────────────────────────────────
def to_dimensions(intent: Dict[str, Any]) -> str:
    """Dimensions DSL query string."""
    all_terms = intent["must_contain"] + intent["any_of"]
    phrase = " ".join(f'"{t}"' for t in all_terms)
    return (
        f'search publications for {phrase} '
        f'return publications [doi+title+abstract+year+keywords+journal+authors] '
        f'limit 1000'
    )


# ── Convenience mapping ──────────────────────────────────────────────────
TRANSLATORS = {
    "openalex": to_openalex,
    "semantic_scholar": to_semantic_scholar,
    "crossref": to_crossref,
    "pubmed": to_pubmed,
    "europe_pmc": to_europe_pmc,
    "core": to_core,
    "lens": to_lens,
    "dimensions": to_dimensions,
}
