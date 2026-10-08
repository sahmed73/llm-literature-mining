#!/usr/bin/env python3
"""
Quick script to extract papers from the scoring checkpoint that meet the threshold.
Run while the scorer job is still running to preview results.

Usage:
    python peek_checkpoint.py                      # default threshold=6
    python peek_checkpoint.py --threshold 7        # custom threshold
    python peek_checkpoint.py --threshold 5 --out my_filtered.json
"""

import argparse
import json
import os
from collections import Counter

CHECKPOINT = os.path.join("output", "scoring_checkpoint.jsonl")

def main():
    parser = argparse.ArgumentParser(description="Preview scored papers from checkpoint")
    parser.add_argument("--checkpoint", default=CHECKPOINT, help="Path to checkpoint JSONL")
    parser.add_argument("--threshold", type=int, default=6, help="Minimum relevance score (default: 6)")
    parser.add_argument("--out", default=None, help="Output JSON file for filtered papers (optional)")
    parser.add_argument("--top", type=int, default=20, help="Show top N highest-scored papers (default: 20)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    # --- Load checkpoint ---
    papers = []
    with open(args.checkpoint, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                papers.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Total scored papers in checkpoint: {len(papers)}")

    # --- Score distribution ---
    scores = [p.get("relevance_score", -1) for p in papers]
    dist = Counter(scores)
    print("\n--- Score Distribution ---")
    for s in sorted(dist.keys()):
        bar = "█" * dist[s] if dist[s] < 80 else "█" * 80 + f"...(x{dist[s]})"
        print(f"  Score {s:>2}: {dist[s]:>6}  {bar}")

    # --- Filter ---
    filtered = [p for p in papers if p.get("relevance_score", 0) >= args.threshold]
    print(f"\nPapers scoring >= {args.threshold}: {len(filtered)} / {len(papers)}  ({100*len(filtered)/max(len(papers),1):.1f}%)")

    # --- Show top N ---
    filtered_sorted = sorted(filtered, key=lambda p: p.get("relevance_score", 0), reverse=True)
    print(f"\n--- Top {min(args.top, len(filtered_sorted))} Papers (score >= {args.threshold}) ---")
    for i, p in enumerate(filtered_sorted[:args.top], 1):
        title = p.get("title", "No title")[:100]
        score = p.get("relevance_score", "?")
        reason = p.get("relevance_reason", "")[:120]
        doi = p.get("doi", "")
        print(f"\n  [{i}] Score: {score}/10")
        print(f"      Title: {title}")
        if doi:
            print(f"      DOI:   {doi}")
        if reason:
            print(f"      Reason: {reason}")

    # --- Save filtered ---
    if args.out:
        with open(args.out, "w") as f:
            json.dump(filtered_sorted, f, indent=2, default=str)
        print(f"\nSaved {len(filtered_sorted)} filtered papers to {args.out}")
    elif filtered:
        default_out = f"output/peek_filtered_ge{args.threshold}.json"
        os.makedirs("output", exist_ok=True)
        with open(default_out, "w") as f:
            json.dump(filtered_sorted, f, indent=2, default=str)
        print(f"\nSaved {len(filtered_sorted)} filtered papers to {default_out}")


if __name__ == "__main__":
    main()
