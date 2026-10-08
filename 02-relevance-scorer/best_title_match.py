#!/usr/bin/env python3
"""Find best matching paper title from a JSON file."""

import json
import difflib

INPUT_JSON = "output/peek_filtered_ge6.json"
QUERY_TITLE = "Effect of the natural antioxidant ferulic acid and its ester derivatives on the oxidative stability of ester-based lubricants: Experimental and molecular simulation investigations"
MIN_MATCH_SCORE = 0.60  # 0.0 to 1.0


def normalize(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        papers = json.load(f)

    query = normalize(QUERY_TITLE)
    if not query:
        print("no match")
        return

    best_paper = None
    best_score = -1.0

    for p in papers:
        title = p.get("title", "")
        ntitle = normalize(title)
        if not ntitle:
            continue
        score = difflib.SequenceMatcher(None, query, ntitle).ratio()
        if score > best_score:
            best_score = score
            best_paper = p

    if best_paper is None or best_score < MIN_MATCH_SCORE:
        print("no match")
        return

    print(f"match_score: {best_score:.3f}")
    print(f"title: {best_paper.get('title', 'N/A')}")
    print(f"doi: {best_paper.get('doi', 'N/A')}")
    print(f"year: {best_paper.get('year', 'N/A')}")


if __name__ == "__main__":
    main()
