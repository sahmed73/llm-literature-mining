#!/usr/bin/env python3
"""Print random paper titles using a fixed seed."""

import json
import random

INPUT_JSON = "output/peek_filtered_ge8.json"
SEED = 42
NUM_TITLES = 50


def load_titles(path: str):
    with open(path, "r", encoding="utf-8") as f:
        papers = json.load(f)
    titles = []
    for p in papers:
        t = (p.get("title") or "").strip()
        if t:
            titles.append(t)
    return titles


def main():
    titles = load_titles(INPUT_JSON)
    if not titles:
        print("No titles found.")
        return

    n = min(NUM_TITLES, len(titles))
    rng = random.Random(SEED)
    chosen = rng.sample(titles, n)

    print(f"seed={SEED} | requested={NUM_TITLES} | printed={n}")
    for i, title in enumerate(chosen, 1):
        print(f"{i}. {title}")


if __name__ == "__main__":
    main()
