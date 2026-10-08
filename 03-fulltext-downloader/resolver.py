"""
Resolver – for each paper, find the best available full-text URL.

Priority:  PMC XML  >  Europe PMC XML  >  Unpaywall  >  Semantic Scholar
           >  CORE  >  CrossRef TDM links

Each resolver function returns:
    {
        "url": str,
        "format": "xml" | "pdf" | "html",
        "source": str,           # which resolver found it
    }
    or None if nothing found.
"""

import logging
import re
import time
import requests
from typing import Optional, Dict, List

import config as cfg

log = logging.getLogger(__name__)

# ── Shared session ────────────────────────────────────────────────────────────
_session = requests.Session()
_session.headers.update({
    "User-Agent": cfg.USER_AGENT,
    "Accept": "application/json",
})

# Per-domain last-request timestamp for rate limiting
_domain_ts: Dict[str, float] = {}


def _rate_limit(domain: str, delay: float):
    """Block until at least `delay` seconds since last request to `domain`."""
    now = time.time()
    last = _domain_ts.get(domain, 0.0)
    wait = delay - (now - last)
    if wait > 0:
        time.sleep(wait)
    _domain_ts[domain] = time.time()


def _get_json(url: str, domain: str, delay: float,
              headers: Optional[dict] = None, timeout: int = 30) -> Optional[dict]:
    """GET with rate limiting, returns parsed JSON or None."""
    _rate_limit(domain, delay)
    try:
        r = _session.get(url, headers=headers or {}, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 429:
            log.warning("Rate-limited by %s, backing off 5s", domain)
            time.sleep(5)
        else:
            log.debug("%s returned %s for %s", domain, r.status_code, url)
    except Exception as e:
        log.debug("Error fetching %s: %s", url, e)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Individual resolvers
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_pmc_xml(paper: dict) -> Optional[dict]:
    """
    Check if paper has a PMC ID and get JATS XML from NCBI.
    PMC XML URL pattern: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=PMCID&rettype=xml
    Or: https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:PMCID&metadataPrefix=pmc
    """
    # Try to find PMCID — might be in paper or need to convert from DOI/PMID
    pmcid = paper.get("pmcid") or paper.get("pmc_id")

    if not pmcid:
        # Try converter: DOI → PMCID via NCBI ID Converter
        doi = paper.get("doi")
        if doi:
            url = (
                f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
                f"?ids={doi}&format=json&email={cfg.EMAIL}"
            )
            data = _get_json(url, "ncbi_idconv", cfg.RATE_LIMIT_PMC)
            if data and data.get("records"):
                rec = data["records"][0]
                pmcid = rec.get("pmcid")

    if not pmcid:
        # Try PMID → PMCID
        pmid = paper.get("pmid")
        if pmid:
            url = (
                f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
                f"?ids={pmid}&format=json&email={cfg.EMAIL}"
            )
            data = _get_json(url, "ncbi_idconv", cfg.RATE_LIMIT_PMC)
            if data and data.get("records"):
                rec = data["records"][0]
                pmcid = rec.get("pmcid")

    if not pmcid:
        return None

    # Clean PMCID (ensure it starts with "PMC")
    pmcid = str(pmcid).strip()
    if not pmcid.upper().startswith("PMC"):
        pmcid = "PMC" + pmcid

    # Fetch XML via efetch
    xml_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pmc&id={pmcid}&rettype=xml&email={cfg.EMAIL}"
    )
    return {"url": xml_url, "format": "xml", "source": "pmc_xml", "pmcid": pmcid}


def _resolve_epmc_xml(paper: dict) -> Optional[dict]:
    """
    Europe PMC: check if full text XML is available.
    API: https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:xxx&resultType=core&format=json
    Then check fullTextUrlList for xml.
    """
    doi = paper.get("doi")
    if not doi:
        return None

    url = (
        f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        f"?query=DOI%3A%22{doi}%22&resultType=core&format=json"
    )
    data = _get_json(url, "europepmc", cfg.RATE_LIMIT_EPMC)
    if not data:
        return None

    results = data.get("resultList", {}).get("result", [])
    if not results:
        return None

    rec = results[0]
    pmcid = rec.get("pmcid")

    # Check fullTextUrlList for XML
    ft_urls = rec.get("fullTextUrlList", {}).get("fullTextUrl", [])
    for ft in ft_urls:
        doc_style = ft.get("documentStyle", "").lower()
        avail = ft.get("availability", "").lower()
        if doc_style == "xml" and "open" in avail:
            return {
                "url": ft["url"],
                "format": "xml",
                "source": "epmc_xml",
                "pmcid": pmcid,
            }

    # Fallback: if we got a PMCID, we can build the XML URL directly
    if pmcid:
        xml_url = (
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/"
            f"{pmcid}/fullTextXML"
        )
        return {"url": xml_url, "format": "xml", "source": "epmc_xml", "pmcid": pmcid}

    # Check for PDF in fullTextUrlList
    for ft in ft_urls:
        doc_style = ft.get("documentStyle", "").lower()
        avail = ft.get("availability", "").lower()
        if doc_style == "pdf" and "open" in avail:
            return {
                "url": ft["url"],
                "format": "pdf",
                "source": "epmc_pdf",
            }

    return None


def _resolve_unpaywall(paper: dict) -> Optional[dict]:
    """
    Unpaywall API: https://api.unpaywall.org/v2/{doi}?email=xxx
    Returns best_oa_location with url_for_pdf or url_for_landing_page.
    """
    doi = paper.get("doi")
    if not doi:
        return None

    url = f"https://api.unpaywall.org/v2/{doi}?email={cfg.EMAIL}"
    data = _get_json(url, "unpaywall", cfg.RATE_LIMIT_UNPAYWALL)
    if not data or not data.get("is_oa"):
        return None

    # Check all OA locations, prefer PDF
    locations = data.get("oa_locations", [])
    if not locations:
        return None

    # Sort: prefer repository/PMC/publisher versions
    pdf_url = None
    landing_url = None

    for loc in locations:
        u_pdf = loc.get("url_for_pdf")
        u_land = loc.get("url_for_landing_page") or loc.get("url")
        version = loc.get("version", "")

        if u_pdf:
            # Check if it's actually an XML source (some PMC links)
            if "pmc" in (u_pdf or "").lower() and u_pdf.endswith(".xml"):
                return {"url": u_pdf, "format": "xml", "source": "unpaywall_xml"}
            if not pdf_url:
                pdf_url = u_pdf
        if u_land and not landing_url:
            landing_url = u_land

    if pdf_url:
        return {"url": pdf_url, "format": "pdf", "source": "unpaywall"}
    if landing_url:
        return {"url": landing_url, "format": "html", "source": "unpaywall_html"}

    return None


def _resolve_semantic_scholar(paper: dict) -> Optional[dict]:
    """
    Semantic Scholar: GET /graph/v1/paper/DOI:{doi}?fields=openAccessPdf
    """
    doi = paper.get("doi")
    if not doi:
        return None

    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=openAccessPdf"
    headers = {}
    if cfg.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = cfg.SEMANTIC_SCHOLAR_API_KEY

    data = _get_json(url, "semanticscholar", cfg.RATE_LIMIT_S2, headers=headers)
    if not data:
        return None

    oa_pdf = data.get("openAccessPdf")
    if oa_pdf and oa_pdf.get("url"):
        return {"url": oa_pdf["url"], "format": "pdf", "source": "semantic_scholar"}

    return None


def _resolve_core(paper: dict) -> Optional[dict]:
    """
    CORE API v3: search by DOI, get downloadUrl.
    """
    if not cfg.CORE_API_KEY:
        return None

    doi = paper.get("doi")
    if not doi:
        return None

    url = f"https://api.core.ac.uk/v3/search/works?q=doi%3A%22{doi}%22&limit=1"
    headers = {"Authorization": f"Bearer {cfg.CORE_API_KEY}"}
    data = _get_json(url, "core", cfg.RATE_LIMIT_CORE, headers=headers)
    if not data:
        return None

    results = data.get("results", [])
    if not results:
        return None

    dl_url = results[0].get("downloadUrl")
    if dl_url:
        return {"url": dl_url, "format": "pdf", "source": "core"}

    return None


def _resolve_crossref(paper: dict) -> Optional[dict]:
    """
    CrossRef: check for TDM links in the work metadata.
    GET https://api.crossref.org/works/{doi}
    Look at link[] for application/pdf or application/xml content-type.
    """
    doi = paper.get("doi")
    if not doi:
        return None

    url = f"https://api.crossref.org/works/{doi}"
    headers = {"User-Agent": cfg.USER_AGENT}
    data = _get_json(url, "crossref", cfg.RATE_LIMIT_DEFAULT, headers=headers)
    if not data:
        return None

    work = data.get("message", {})
    links = work.get("link", [])

    # Prefer XML over PDF
    xml_link = None
    pdf_link = None
    html_link = None
    for link in links:
        ct = (link.get("content-type") or "").lower()
        link_url = link.get("URL")
        if not link_url:
            continue
        if "xml" in ct and not xml_link:
            xml_link = link_url
        elif "pdf" in ct and not pdf_link:
            pdf_link = link_url
        elif "html" in ct and not html_link:
            html_link = link_url
        elif not ct:
            lurl = link_url.lower()
            if (".xml" in lurl or "/xml" in lurl) and not xml_link:
                xml_link = link_url
            elif (".pdf" in lurl or "/pdf" in lurl) and not pdf_link:
                pdf_link = link_url
            elif not html_link:
                html_link = link_url

    # Crossref primary resource is often a landing page; keep as last resort.
    primary = (work.get("resource", {}) or {}).get("primary", {})
    primary_url = primary.get("URL") if isinstance(primary, dict) else None
    if primary_url and not html_link:
        html_link = primary_url

    if xml_link:
        return {"url": xml_link, "format": "xml", "source": "crossref_xml"}
    if pdf_link:
        return {"url": pdf_link, "format": "pdf", "source": "crossref_pdf"}
    if html_link:
        return {"url": html_link, "format": "html", "source": "crossref_html"}

    return None


def _resolve_crossref_by_format(paper: dict, wanted_format: str) -> Optional[dict]:
    """Format-aware Crossref resolver for resolve_paper_by_format."""
    doi = paper.get("doi")
    if not doi:
        return None

    url = f"https://api.crossref.org/works/{doi}"
    headers = {"User-Agent": cfg.USER_AGENT}
    data = _get_json(url, "crossref", cfg.RATE_LIMIT_DEFAULT, headers=headers)
    if not data:
        return None

    work = data.get("message", {})
    links = work.get("link", [])
    wanted = (wanted_format or "").lower()

    for link in links:
        link_url = link.get("URL")
        if not link_url:
            continue
        ct = (link.get("content-type") or "").lower()
        lurl = link_url.lower()

        if wanted == "xml" and ("xml" in ct or ".xml" in lurl or "/xml" in lurl):
            return {"url": link_url, "format": "xml", "source": "crossref_xml"}
        if wanted == "pdf" and ("pdf" in ct or ".pdf" in lurl or "/pdf" in lurl):
            return {"url": link_url, "format": "pdf", "source": "crossref_pdf"}
        if wanted == "html" and ("html" in ct):
            return {"url": link_url, "format": "html", "source": "crossref_html"}

    primary = (work.get("resource", {}) or {}).get("primary", {})
    primary_url = primary.get("URL") if isinstance(primary, dict) else None
    if primary_url:
        base = primary_url.rstrip("/")
        if wanted == "pdf":
            return {"url": f"{base}/pdf", "format": "pdf", "source": "crossref_pdf_guess"}
        if wanted == "xml":
            return {"url": f"{base}/xml", "format": "xml", "source": "crossref_xml_guess"}
        if wanted == "html":
            return {"url": primary_url, "format": "html", "source": "crossref_html"}

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Main resolve function
# ═══════════════════════════════════════════════════════════════════════════════

_RESOLVER_MAP = {
    "pmc_xml":           _resolve_pmc_xml,
    "epmc_xml":          _resolve_epmc_xml,
    "unpaywall":         _resolve_unpaywall,
    "semantic_scholar":  _resolve_semantic_scholar,
    "core":              _resolve_core,
    "crossref_link":     _resolve_crossref,
}


def resolve_paper(paper: dict) -> Optional[dict]:
    """
    Try each resolver in priority order. Return first successful result,
    or None if no full text is available.
    """
    doi = paper.get("doi", "unknown")

    for source_name in cfg.RESOLVER_PRIORITY:
        fn = _RESOLVER_MAP.get(source_name)
        if fn is None:
            continue
        try:
            result = fn(paper)
            if result:
                log.debug("Resolved %s via %s → %s", doi, source_name, result["format"])
                return result
        except Exception as e:
            log.debug("Resolver %s failed for %s: %s", source_name, doi, e)
            continue

    log.debug("No full text found for %s", doi)
    return None


def resolve_paper_by_format(paper: dict, wanted_format: str) -> Optional[dict]:
    """
    Resolve a paper and return the first source matching wanted_format.
    wanted_format: \"xml\" | \"pdf\" | \"html\"
    """
    wanted = (wanted_format or "").strip().lower()
    doi = paper.get("doi", "unknown")

    if wanted not in {"xml", "pdf", "html"}:
        return None

    for source_name in cfg.RESOLVER_PRIORITY:
        if source_name == "crossref_link":
            try:
                result = _resolve_crossref_by_format(paper, wanted)
                if result:
                    log.debug("Resolved %s via %s -> %s", doi, source_name, wanted)
                    return result
            except Exception as e:
                log.debug("Resolver %s failed for %s: %s", source_name, doi, e)
            continue

        fn = _RESOLVER_MAP.get(source_name)
        if fn is None:
            continue
        try:
            result = fn(paper)
            if result and (result.get("format", "").lower() == wanted):
                log.debug("Resolved %s via %s -> %s", doi, source_name, wanted)
                return result
        except Exception as e:
            log.debug("Resolver %s failed for %s: %s", source_name, doi, e)
            continue
    return None
