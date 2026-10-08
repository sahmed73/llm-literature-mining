"""
Run the pipeline's two LLM steps on built-in examples with any Ollama model.

    python llm/demo.py --model llama3:8b
    python llm/demo.py --model llama3:70b
    python llm/demo.py --model qwen3:30b-a3b

Step 02 (relevance scoring) and step 04 (compound extraction) are called through
the pipeline's own functions, so the prompts, generation settings, and response
parsing are exactly those of a full run. An Ollama server must be running and the
model must already be pulled (`ollama pull <model>`).
"""

import argparse
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]

# Abstract of Ahmed et al., ACS Omega 11, 16886 (2026), CC BY 4.0. Should score high.
RELEVANT_PAPER = {
    "title": "Reactive MD Screening of Antioxidants for Substituent-Dependent Phenoxyl Radical Stability",
    "abstract": (
        "Oxidation limits the performance and lifetime of lubricants, and phenolic antioxidants are "
        "commonly used to slow this process by scavenging hydrocarbon peroxyl radicals. The performance "
        "of phenolic antioxidants is largely determined by the stability of the antioxidant radical that "
        "remains after hydrogen donation. To explore the relationship between antioxidant chemical "
        "structure and radical stability, we used REACTER-based reactive molecular dynamics simulations "
        "to model the reverse hydrogen transfer reaction from polyalphaolefin hydroperoxides to phenoxyl "
        "radicals. Simulations were run for 718 distinct single-ring phenoxyl radicals with varied "
        "substituent types and positions in a polyalphaolefin hydroperoxide environment."
    ),
}

# Invented example from food science. Should score low.
UNRELATED_PAPER = {
    "title": "Polyphenol content and antioxidant capacity of green tea extracts during storage",
    "abstract": (
        "Green tea extracts were stored for 12 weeks at 4 and 25 °C, and total polyphenols, catechins, "
        "and DPPH radical scavenging capacity were measured every two weeks. Antioxidant capacity "
        "decreased by 18% at 25 °C, mainly through catechin degradation."
    ),
}

# Invented paragraph written for this demo, not taken from a paper.
EXTRACTION_TEXT = (
    "The oxidation stability of a polyalphaolefin (PAO 4) base oil was evaluated by RPVOT and PDSC. "
    "Adding 0.5 wt% 2,6-di-tert-butyl-4-methylphenol (BHT) extended the oxidation induction time from "
    "12 to 41 min, while 0.5 wt% N-phenyl-1-naphthylamine (PANA) extended it to 58 min. A 1:1 blend of "
    "BHT and 4,4'-dioctyldiphenylamine showed a synergistic effect. Vitamin E was tested as a "
    "bio-based reference but was less effective at 150 °C."
)


def import_stage(stage: str, module: str):
    """Import a stage module so that its own `config.py` is the one it sees."""
    stage_dir = str(REPO / stage)
    sys.modules.pop("config", None)
    sys.path.insert(0, stage_dir)
    try:
        return importlib.import_module(module)
    finally:
        sys.path.remove(stage_dir)


def check_server(url: str, model: str) -> None:
    try:
        tags = requests.get(f"{url}/api/tags", timeout=10).json()
    except requests.RequestException as exc:
        sys.exit(f"No Ollama server at {url} ({exc}). Start one with `ollama serve`.")
    names = {m["name"] for m in tags.get("models", [])}
    wanted = {model.lower(), f"{model}:latest".lower()}  # Ollama names are case-insensitive
    if not wanted & {n.lower() for n in names}:
        sys.exit(f"Model '{model}' is not pulled on {url}. Run `ollama pull {model}`. "
                 f"Available: {', '.join(sorted(names)) or 'none'}")


def context_info(url: str, model: str) -> str:
    """Report the model's maximum context and the context Ollama actually loaded."""
    parts = []
    try:
        info = requests.post(f"{url}/api/show", json={"model": model}, timeout=30).json()
        maximum = next((v for k, v in info.get("model_info", {}).items() if k.endswith(".context_length")), None)
        if maximum:
            parts.append(f"model supports {maximum:,} tokens")
        loaded = next((m for m in requests.get(f"{url}/api/ps", timeout=10).json().get("models", [])
                       if m.get("name", "").lower().startswith(model.lower())), None)
        if loaded and loaded.get("context_length"):
            parts.append(f"loaded with {loaded['context_length']:,}")
    except (requests.RequestException, ValueError):
        pass
    return "; ".join(parts) or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "llama3:8b"),
                        help="Ollama model name (default: LLM_MODEL env var or llama3:8b)")
    parser.add_argument("--url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
                        help="Ollama server URL (default: OLLAMA_URL env var or http://localhost:11434)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    log = logging.getLogger("demo")

    check_server(args.url, args.model)
    scorer = import_stage("02-relevance-scorer", "scorer")
    extractor = import_stage("04-compound-extractor", "extractor")
    scorer.config.MODEL_NAME = args.model
    scorer.config.OLLAMA_URL = args.url
    extractor.cfg.MODEL_NAME = args.model

    print(f"Model: {args.model}  |  server: {args.url}\n")

    print("Step 02 - relevance scoring (0-10)")
    for label, paper in [("lubricant antioxidant paper", RELEVANT_PAPER), ("food-science paper", UNRELATED_PAPER)]:
        t0 = time.time()
        result = scorer.score_paper(paper, log)
        print(f"  {label:<28} score={result['relevance_score']}  ({time.time() - t0:.1f}s)")
        print(f"    reason: {result['relevance_reason']}")

    print("\nStep 04 - compound extraction")
    t0 = time.time()
    compounds = extractor.extract_from_chunk({"chunk_id": "demo", "text": EXTRACTION_TEXT}, ollama_host=args.url)
    print(f"  {len(compounds)} compound(s) in {time.time() - t0:.1f}s")
    for c in compounds:
        print(f"    - {c['compound_name']}  [{c['role']}]")
    print("\n  Raw records:")
    print("  " + json.dumps(compounds, indent=2, ensure_ascii=False).replace("\n", "\n  "))

    print(f"\nContext window: {context_info(args.url, args.model)}")


if __name__ == "__main__":
    main()
