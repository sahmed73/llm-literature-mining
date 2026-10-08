#!/usr/bin/env python3
"""Export papers from workflow JSON to a BibTeX file."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

INPUT_JSON = "output/peek_filtered_ge6.json"
OUTPUT_BIB = "output/all_papers.bib"
UNIQUE_ONLY = True


def load_papers(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        papers = data.get("papers", [])
        if isinstance(papers, list):
            return papers

    raise ValueError(f"Unsupported JSON structure in {path}")


def sanitize_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "\\{").replace("}", "\\}")
    return " ".join(text.split())


def sanitize_key(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", text)
    return cleaned[:60] or "entry"


def make_entry_key(paper: dict, index: int) -> str:
    doi = (paper.get("doi") or "").strip()
    if doi:
        return sanitize_key(doi.replace("/", "_"))

    authors = paper.get("authors") or []
    first_author = authors[0].split()[-1] if authors else "unknown"
    year = paper.get("year") or "nd"
    title = (paper.get("title") or "untitled").split()
    title_token = title[0] if title else "untitled"
    return sanitize_key(f"{first_author}{year}{title_token}{index}")


def format_authors(authors) -> str:
    if not authors:
        return ""
    if isinstance(authors, list):
        return " and ".join(sanitize_text(author) for author in authors if str(author).strip())
    return sanitize_text(authors)


def format_keywords(keywords) -> str:
    if not keywords:
        return ""
    if isinstance(keywords, list):
        return "; ".join(sanitize_text(keyword) for keyword in keywords if str(keyword).strip())
    return sanitize_text(keywords)


def format_source_apis(source_apis) -> str:
    if not source_apis:
        return ""
    if isinstance(source_apis, list):
        return ", ".join(sanitize_text(source) for source in source_apis if str(source).strip())
    return sanitize_text(source_apis)


def infer_entry_type(paper: dict) -> str:
    return "article" if paper.get("journal") else "misc"


def build_bibtex_entry(paper: dict, entry_key: str) -> str:
    fields: list[tuple[str, str]] = []

    title = sanitize_text(paper.get("title", ""))
    authors = format_authors(paper.get("authors"))
    journal = sanitize_text(paper.get("journal", ""))
    year = paper.get("year")
    doi = sanitize_text(paper.get("doi", ""))
    abstract = sanitize_text(paper.get("abstract", ""))
    keywords = format_keywords(paper.get("keywords"))
    pmid = sanitize_text(paper.get("pmid", ""))
    citation_count = paper.get("citation_count")
    open_access = paper.get("open_access")
    source_apis = format_source_apis(paper.get("source_apis"))
    relevance_score = paper.get("relevance_score")
    relevance_reason = sanitize_text(paper.get("relevance_reason", ""))
    score_confidence = sanitize_text(paper.get("score_confidence", ""))

    if title:
        fields.append(("title", title))
    if authors:
        fields.append(("author", authors))
    if journal:
        fields.append(("journal", journal))
    if year not in (None, ""):
        fields.append(("year", str(year).strip()))
    if doi:
        fields.append(("doi", doi))
    if pmid:
        fields.append(("pmid", pmid))
    if abstract:
        fields.append(("abstract", abstract))
    if keywords:
        fields.append(("keywords", keywords))
    if citation_count not in (None, ""):
        fields.append(("citation_count", str(citation_count)))
    if open_access is not None:
        fields.append(("open_access", str(bool(open_access)).lower()))
    if source_apis:
        fields.append(("source_apis", source_apis))
    if relevance_score not in (None, ""):
        fields.append(("relevance_score", str(relevance_score)))
    if relevance_reason:
        fields.append(("relevance_reason", relevance_reason))
    if score_confidence:
        fields.append(("score_confidence", score_confidence))

    lines = [f"@{infer_entry_type(paper)}{{{entry_key},"]
    for name, value in fields:
        lines.append(f"  {name} = {{{value}}},")
    lines.append("}")
    return "\n".join(lines)


def export_bibtex(input_json: Path, output_bib: Path, unique_only: bool) -> int:
    papers = load_papers(input_json)
    output_bib.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    entries = []

    for index, paper in enumerate(papers, start=1):
        dedupe_key = ((paper.get("doi") or "").strip() or (paper.get("title") or "").strip()).lower()
        if unique_only and dedupe_key:
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

        entry_key = make_entry_key(paper, index)
        entries.append(build_bibtex_entry(paper, entry_key))

    with output_bib.open("w", encoding="utf-8") as f:
        f.write("\n\n".join(entries))
        if entries:
            f.write("\n")

    return len(entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=INPUT_JSON, help="Input JSON file")
    parser.add_argument("--output", default=OUTPUT_BIB, help="Output BibTeX file")
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Keep duplicate DOI/title records instead of deduplicating them",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    count = export_bibtex(
        input_json=Path(args.input),
        output_bib=Path(args.output),
        unique_only=not args.allow_duplicates,
    )
    print(f"wrote {count} BibTeX entries -> {args.output}")


if __name__ == "__main__":
    main()
