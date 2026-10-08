"""
API clients for each academic metadata source.

Every client exposes ``harvest(intent, logger) -> List[dict]``
that returns normalised paper records.

Normalised record schema
-------------------------
{
    "doi":            str | None,
    "title":          str | None,
    "abstract":       str | None,
    "year":           int | None,
    "keywords":       list[str],
    "authors":        list[str],
    "journal":        str | None,
    "citation_count": int | None,
    "open_access":    bool | None,
    "pmid":           str | None,
    "source_api":     str,
}
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests

import config
import query_translator as qt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_get(url: str, params: dict | None = None,
              headers: dict | None = None,
              delay: float = 0.1,
              timeout: int = 60,
              logger: logging.Logger | None = None) -> Optional[dict]:
    """GET with retry (x3), delay, and JSON parsing."""
    for attempt in range(1, 4):
        try:
            time.sleep(delay)
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if logger:
                logger.warning("  GET %s attempt %d failed: %s", url[:120], attempt, exc)
            if attempt == 3:
                return None
    return None


def _safe_post(url: str, json_body: dict | None = None,
               params: dict | None = None,
               headers: dict | None = None,
               delay: float = 0.1,
               timeout: int = 60,
               logger: logging.Logger | None = None) -> Optional[dict | list]:
    """POST with retry (x3), delay, and JSON parsing."""
    for attempt in range(1, 4):
        try:
            time.sleep(delay)
            resp = requests.post(url, json=json_body, params=params,
                                 headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if logger:
                logger.warning("  POST %s attempt %d failed: %s", url[:120], attempt, exc)
            if attempt == 3:
                return None
    return None


def _reconstruct_abstract(inv_index: dict | None) -> Optional[str]:
    """Reconstruct OpenAlex inverted-index abstract to plain text."""
    if not inv_index:
        return None
    pairs: list[tuple[int, str]] = []
    for word, positions in inv_index.items():
        for pos in positions:
            pairs.append((pos, word))
    pairs.sort()
    return " ".join(w for _, w in pairs)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseClient(ABC):
    name: str = "base"

    @abstractmethod
    def harvest(self, intent: Dict[str, Any],
                logger: logging.Logger) -> List[Dict[str, Any]]:
        ...


# ═══════════════════════════════════════════════════════════════════════════
#  1. OpenAlex
# ═══════════════════════════════════════════════════════════════════════════

class OpenAlexClient(BaseClient):
    name = "openalex"
    BASE = "https://api.openalex.org/works"

    def harvest(self, intent, logger):
        query = qt.to_openalex(intent)
        logger.info("[OpenAlex] query='%s'  (intent=%s)", query, intent["id"])
        delay = config.REQUEST_DELAY.get(self.name, 0.11)

        papers: List[Dict[str, Any]] = []
        cursor = "*"
        page = 0
        while cursor:
            page += 1
            data = _safe_get(
                self.BASE,
                params={
                    "search": query,
                    "filter": "type:article|review|proceedings-article",
                    "select": "id,doi,title,abstract_inverted_index,"
                              "publication_year,keywords,authorships,"
                              "primary_location,cited_by_count,open_access,"
                              "concepts",
                    "per_page": 200,
                    "cursor": cursor,
                    "mailto": config.MAILTO,
                },
                delay=delay,
                logger=logger,
            )
            if not data or "results" not in data:
                logger.warning("[OpenAlex] No data on page %d – stopping.", page)
                break

            for w in data["results"]:
                kw = [k.get("display_name", "") for k in (w.get("keywords") or [])]
                kw += [c.get("display_name", "") for c in (w.get("concepts") or [])
                       if c.get("score", 0) > 0.3]
                kw = [k for k in kw if k]  # drop empty strings / None
                authors = [
                    a["author"]["display_name"]
                    for a in (w.get("authorships") or [])
                    if a.get("author", {}).get("display_name")
                ]
                loc = (w.get("primary_location") or {})
                journal = (loc.get("source") or {}).get("display_name")

                papers.append({
                    "doi": (w.get("doi") or "").replace("https://doi.org/", "") or None,
                    "title": w.get("title"),
                    "abstract": _reconstruct_abstract(w.get("abstract_inverted_index")),
                    "year": w.get("publication_year"),
                    "keywords": list(set(kw)),
                    "authors": authors,
                    "journal": journal,
                    "citation_count": w.get("cited_by_count"),
                    "open_access": (w.get("open_access") or {}).get("is_oa"),
                    "pmid": None,
                    "source_api": self.name,
                })

            cursor = data.get("meta", {}).get("next_cursor")
            if page % 20 == 0:
                logger.info("[OpenAlex] intent=%s  pages=%d  papers=%d",
                            intent["id"], page, len(papers))

        logger.info("[OpenAlex] intent=%s DONE – %d papers collected.",
                    intent["id"], len(papers))
        return papers


# ═══════════════════════════════════════════════════════════════════════════
#  2. Semantic Scholar
# ═══════════════════════════════════════════════════════════════════════════

class SemanticScholarClient(BaseClient):
    name = "semantic_scholar"
    SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = "title,abstract,year,externalIds,s2FieldsOfStudy,authors,citationCount"

    def _headers(self):
        h = {}
        key = config.API_KEYS.get("semantic_scholar")
        if key:
            h["x-api-key"] = key
        return h

    def harvest(self, intent, logger):
        query = qt.to_semantic_scholar(intent)
        logger.info("[SemanticScholar] query='%s'  (intent=%s)", query, intent["id"])
        delay = config.REQUEST_DELAY.get(self.name, 0.12)
        headers = self._headers()

        papers: List[Dict[str, Any]] = []
        offset = 0
        limit = 100
        while True:
            data = _safe_get(
                self.SEARCH_URL,
                params={
                    "query": query,
                    "limit": limit,
                    "offset": offset,
                    "fields": self.FIELDS,
                },
                headers=headers,
                delay=delay,
                logger=logger,
            )
            if not data or "data" not in data:
                break

            for p in data["data"]:
                ext = p.get("externalIds") or {}
                doi = ext.get("DOI")
                pmid = ext.get("PubMed")
                kw = [f["category"] for f in (p.get("s2FieldsOfStudy") or [])
                      if f.get("category")]
                authors = [a.get("name", "") for a in (p.get("authors") or [])]
                papers.append({
                    "doi": doi,
                    "title": p.get("title"),
                    "abstract": p.get("abstract"),
                    "year": p.get("year"),
                    "keywords": kw,
                    "authors": authors,
                    "journal": None,
                    "citation_count": p.get("citationCount"),
                    "open_access": None,
                    "pmid": pmid,
                    "source_api": self.name,
                })

            total = data.get("total", 0)
            offset += limit
            if offset >= total or offset >= 10000:  # S2 hard cap
                break
            if offset % 500 == 0:
                logger.info("[SemanticScholar] intent=%s  offset=%d/%d",
                            intent["id"], offset, total)

        logger.info("[SemanticScholar] intent=%s DONE – %d papers collected.",
                    intent["id"], len(papers))
        return papers


# ═══════════════════════════════════════════════════════════════════════════
#  3. CrossRef
# ═══════════════════════════════════════════════════════════════════════════

class CrossRefClient(BaseClient):
    name = "crossref"
    BASE = "https://api.crossref.org/works"

    def harvest(self, intent, logger):
        qparams = qt.to_crossref(intent)
        logger.info("[CrossRef] params=%s  (intent=%s)", qparams, intent["id"])
        delay = config.REQUEST_DELAY.get(self.name, 0.05)

        papers: List[Dict[str, Any]] = []
        cursor = "*"
        page = 0
        while cursor:
            page += 1
            params = {
                **qparams,
                "filter": "type:journal-article",
                "select": "DOI,title,abstract,published,subject,author,"
                          "container-title,is-referenced-by-count",
                "rows": 1000,
                "cursor": cursor,
                "mailto": config.MAILTO,
            }
            data = _safe_get(self.BASE, params=params, delay=delay, logger=logger)
            if not data:
                break

            msg = data.get("message", {})
            items = msg.get("items", [])
            if not items:
                break

            for it in items:
                doi = it.get("DOI")
                title_list = it.get("title", [])
                title = title_list[0] if title_list else None
                abstract = it.get("abstract")  # may contain HTML
                if abstract:
                    # Strip simple HTML tags
                    import re
                    abstract = re.sub(r"<[^>]+>", "", abstract).strip()
                pub = it.get("published") or it.get("published-print") or it.get("published-online") or {}
                parts = pub.get("date-parts", [[None]])[0]
                year = None
                try:
                    year = int(parts[0]) if parts and parts[0] else None
                except (ValueError, TypeError, IndexError):
                    year = None
                kw = it.get("subject", [])
                authors = [
                    f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in (it.get("author") or [])
                ]
                journal_list = it.get("container-title", [])
                journal = journal_list[0] if journal_list else None

                papers.append({
                    "doi": doi,
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "keywords": [k for k in kw if k],
                    "authors": authors,
                    "journal": journal,
                    "citation_count": it.get("is-referenced-by-count"),
                    "open_access": None,
                    "pmid": None,
                    "source_api": self.name,
                })

            cursor = msg.get("next-cursor")
            # CrossRef returns empty items when exhausted
            if len(items) < 20:
                break
            if page % 10 == 0:
                logger.info("[CrossRef] intent=%s  pages=%d  papers=%d",
                            intent["id"], page, len(papers))

        logger.info("[CrossRef] intent=%s DONE – %d papers collected.",
                    intent["id"], len(papers))
        return papers


# ═══════════════════════════════════════════════════════════════════════════
#  4. PubMed (Entrez via REST – no biopython dependency)
# ═══════════════════════════════════════════════════════════════════════════

class PubMedClient(BaseClient):
    name = "pubmed"
    ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def harvest(self, intent, logger):
        query = qt.to_pubmed(intent)
        logger.info("[PubMed] query='%s'  (intent=%s)", query, intent["id"])
        delay = config.REQUEST_DELAY.get(self.name, 0.34)

        # Step 1: search for PMIDs
        data = _safe_get(
            self.ESEARCH,
            params={
                "db": "pubmed", "term": query,
                "retmax": 0, "retmode": "json",
                "email": config.MAILTO,
            },
            delay=delay, logger=logger,
        )
        if not data:
            logger.warning("[PubMed] esearch failed for intent=%s", intent["id"])
            return []

        total = int(data.get("esearchresult", {}).get("count", 0))
        logger.info("[PubMed] intent=%s  total hits=%d", intent["id"], total)
        if total == 0:
            return []

        # Collect all PMIDs (batches of 10 000)
        all_pmids: List[str] = []
        for start in range(0, min(total, 100_000), 10_000):
            d = _safe_get(
                self.ESEARCH,
                params={
                    "db": "pubmed", "term": query,
                    "retstart": start, "retmax": 10_000,
                    "retmode": "json", "email": config.MAILTO,
                },
                delay=delay, logger=logger,
            )
            if d:
                all_pmids.extend(d.get("esearchresult", {}).get("idlist", []))

        logger.info("[PubMed] intent=%s  fetched %d PMIDs.", intent["id"], len(all_pmids))

        # Step 2: fetch XML metadata in batches of 200
        import xml.etree.ElementTree as ET

        papers: List[Dict[str, Any]] = []
        for i in range(0, len(all_pmids), 200):
            batch = all_pmids[i:i + 200]
            time.sleep(delay)
            try:
                resp = requests.post(
                    self.EFETCH,
                    data={
                        "db": "pubmed", "id": ",".join(batch),
                        "retmode": "xml", "email": config.MAILTO,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
            except Exception as exc:
                logger.warning("[PubMed] efetch batch %d failed: %s", i, exc)
                continue

            try:
                root = ET.fromstring(resp.content)
            except ET.ParseError:
                logger.warning("[PubMed] XML parse error at batch %d", i)
                continue

            for article in root.iter("PubmedArticle"):
                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else None

                title_el = article.find(".//ArticleTitle")
                title = title_el.text if title_el is not None else None

                abs_parts = article.findall(".//AbstractText")
                abstract = " ".join(
                    (a.text or "") for a in abs_parts
                ).strip() or None

                year_el = article.find(".//PubDate/Year")
                year = None
                try:
                    year = int(year_el.text) if year_el is not None and year_el.text else None
                except (ValueError, TypeError):
                    year = None

                # DOI
                doi = None
                for aid in article.findall(".//ArticleId"):
                    if aid.get("IdType") == "doi":
                        doi = aid.text
                        break

                # MeSH
                mesh = [
                    m.text for m in article.findall(".//MeshHeading/DescriptorName")
                    if m.text
                ]

                authors_el = article.findall(".//Author")
                authors = []
                for au in authors_el:
                    ln = (au.findtext("LastName") or "")
                    fn = (au.findtext("ForeName") or "")
                    name = f"{fn} {ln}".strip()
                    if name:
                        authors.append(name)

                journal_el = article.find(".//Journal/Title")
                journal = journal_el.text if journal_el is not None else None

                papers.append({
                    "doi": doi,
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "keywords": mesh,
                    "authors": authors,
                    "journal": journal,
                    "citation_count": None,
                    "open_access": None,
                    "pmid": pmid,
                    "source_api": self.name,
                })

            if (i // 200) % 20 == 0 and i > 0:
                logger.info("[PubMed] intent=%s  fetched %d/%d",
                            intent["id"], len(papers), len(all_pmids))

        logger.info("[PubMed] intent=%s DONE – %d papers collected.",
                    intent["id"], len(papers))
        return papers


# ═══════════════════════════════════════════════════════════════════════════
#  5. Europe PMC
# ═══════════════════════════════════════════════════════════════════════════

class EuropePMCClient(BaseClient):
    name = "europe_pmc"
    BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def harvest(self, intent, logger):
        query = qt.to_europe_pmc(intent)
        logger.info("[EuropePMC] query='%s'  (intent=%s)", query, intent["id"])
        delay = config.REQUEST_DELAY.get(self.name, 0.12)

        papers: List[Dict[str, Any]] = []
        cursor = "*"
        page = 0
        while cursor:
            page += 1
            data = _safe_get(
                self.BASE,
                params={
                    "query": query,
                    "resultType": "core",
                    "pageSize": 1000,
                    "cursorMark": cursor,
                    "format": "json",
                },
                delay=delay, logger=logger,
            )
            if not data:
                break

            results = data.get("resultList", {}).get("result", [])
            if not results:
                break

            for r in results:
                doi = r.get("doi")
                kw_list = r.get("keywordList", {}).get("keyword", [])
                authors_str = r.get("authorString", "")
                authors = [a.strip() for a in authors_str.split(",") if a.strip()] if authors_str else []

                papers.append({
                    "doi": doi,
                    "title": r.get("title"),
                    "abstract": r.get("abstractText"),
                    "year": int(r["pubYear"]) if r.get("pubYear") and str(r["pubYear"]).isdigit() else None,
                    "keywords": kw_list if isinstance(kw_list, list) else [],
                    "authors": authors,
                    "journal": r.get("journalTitle"),
                    "citation_count": r.get("citedByCount"),
                    "open_access": r.get("isOpenAccess") == "Y",
                    "pmid": r.get("pmid"),
                    "source_api": self.name,
                })

            next_cursor = data.get("nextCursorMark")
            if next_cursor == cursor:
                break
            cursor = next_cursor

            if page % 10 == 0:
                logger.info("[EuropePMC] intent=%s  pages=%d  papers=%d",
                            intent["id"], page, len(papers))

        logger.info("[EuropePMC] intent=%s DONE – %d papers collected.",
                    intent["id"], len(papers))
        return papers


# ═══════════════════════════════════════════════════════════════════════════
#  6. CORE
# ═══════════════════════════════════════════════════════════════════════════

class COREClient(BaseClient):
    name = "core"
    BASE = "https://api.core.ac.uk/v3/search/works"

    def _headers(self):
        key = config.API_KEYS.get("core")
        if not key:
            return {}
        return {"Authorization": f"Bearer {key}"}

    def harvest(self, intent, logger):
        key = config.API_KEYS.get("core")
        if not key:
            logger.info("[CORE] Skipped – no API key configured.")
            return []

        query = qt.to_core(intent)
        logger.info("[CORE] query='%s'  (intent=%s)", query, intent["id"])
        delay = config.REQUEST_DELAY.get(self.name, 0.20)
        headers = self._headers()

        papers: List[Dict[str, Any]] = []
        offset = 0
        limit = 100
        while True:
            body = {"q": query, "limit": limit, "offset": offset}
            data = _safe_post(self.BASE, json_body=body, headers=headers,
                              delay=delay, logger=logger)
            if not data or "results" not in data:
                break

            results = data["results"]
            if not results:
                break

            for r in results:
                doi_val = None
                for ident in (r.get("identifiers") or []):
                    if isinstance(ident, str) and "doi.org" in ident:
                        doi_val = ident.replace("https://doi.org/", "")
                        break

                papers.append({
                    "doi": doi_val,
                    "title": r.get("title"),
                    "abstract": r.get("abstract"),
                    "year": r.get("yearPublished"),
                    "keywords": r.get("fieldOfStudy") or [],
                    "authors": [a.get("name", "") for a in (r.get("authors") or [])],
                    "journal": (r.get("journal") or {}).get("title") if isinstance(r.get("journal"), dict) else None,
                    "citation_count": r.get("citationCount"),
                    "open_access": None,
                    "pmid": None,
                    "source_api": self.name,
                })

            total = data.get("totalHits", 0)
            offset += limit
            if offset >= total or offset >= 10000:
                break
            if offset % 500 == 0:
                logger.info("[CORE] intent=%s  offset=%d/%d",
                            intent["id"], offset, total)

        logger.info("[CORE] intent=%s DONE – %d papers collected.",
                    intent["id"], len(papers))
        return papers


# ═══════════════════════════════════════════════════════════════════════════
#  7. Lens.org
# ═══════════════════════════════════════════════════════════════════════════

class LensClient(BaseClient):
    name = "lens"
    BASE = "https://api.lens.org/scholarly/search"

    def _headers(self):
        key = config.API_KEYS.get("lens")
        if not key:
            return {}
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def harvest(self, intent, logger):
        key = config.API_KEYS.get("lens")
        if not key:
            logger.info("[Lens] Skipped – no API key configured.")
            return []

        body = qt.to_lens(intent)
        logger.info("[Lens] intent=%s", intent["id"])
        delay = config.REQUEST_DELAY.get(self.name, 0.20)
        headers = self._headers()

        papers: List[Dict[str, Any]] = []
        scroll_id = None
        page = 0
        body["size"] = 1000
        body["include"] = [
            "doi", "title", "abstract", "year_published",
            "keywords", "authors", "source", "scholarly_citations_count",
            "external_ids",
        ]

        while True:
            page += 1
            if scroll_id:
                body["scroll_id"] = scroll_id

            data = _safe_post(self.BASE, json_body=body, headers=headers,
                              delay=delay, logger=logger)
            if not data or "data" not in data:
                break

            results = data["data"]
            if not results:
                break

            for r in results:
                doi_val = None
                for eid in (r.get("external_ids") or []):
                    if eid.get("type") == "doi":
                        doi_val = eid.get("value")
                        break
                if not doi_val:
                    doi_val = r.get("doi")

                authors = []
                for a in (r.get("authors") or []):
                    fn = a.get("first_name", "")
                    ln = a.get("last_name", "")
                    name = f"{fn} {ln}".strip()
                    if name:
                        authors.append(name)

                papers.append({
                    "doi": doi_val,
                    "title": r.get("title"),
                    "abstract": r.get("abstract"),
                    "year": r.get("year_published"),
                    "keywords": r.get("keywords") or [],
                    "authors": authors,
                    "journal": (r.get("source") or {}).get("title"),
                    "citation_count": r.get("scholarly_citations_count"),
                    "open_access": None,
                    "pmid": None,
                    "source_api": self.name,
                })

            scroll_id = data.get("scroll_id")
            total = data.get("total", 0)
            if not scroll_id or len(papers) >= total:
                break
            if page % 5 == 0:
                logger.info("[Lens] intent=%s  pages=%d  papers=%d",
                            intent["id"], page, len(papers))

        logger.info("[Lens] intent=%s DONE – %d papers collected.",
                    intent["id"], len(papers))
        return papers


# ═══════════════════════════════════════════════════════════════════════════
#  Registry
# ═══════════════════════════════════════════════════════════════════════════

ALL_CLIENTS: List[BaseClient] = [
    OpenAlexClient(),
    SemanticScholarClient(),
    CrossRefClient(),
    PubMedClient(),
    EuropePMCClient(),
    COREClient(),
    LensClient(),
    # DimensionsClient can be added later if needed
]
