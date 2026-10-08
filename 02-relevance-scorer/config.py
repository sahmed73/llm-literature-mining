"""
Configuration for Relevance scorer.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(BASE_DIR, "..", "01-metadata-miner", "output", "papers_metadata.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "filtered_metadata.json")
SCORES_JSON = os.path.join(OUTPUT_DIR, "all_scores.json")   # all papers + scores (even rejected)
LOG_FILE = os.path.join(OUTPUT_DIR, "scorer.log")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "scoring_checkpoint.jsonl")

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
# Any model served by Ollama works (llama3:8b, llama3:70b, qwen3:30b-a3b, ...).
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("LLM_MODEL", "llama3:8b")
NUM_CTX = int(os.environ.get("LLM_NUM_CTX", "0"))  # 0 = Ollama's default context window

# Reasoning models such as Qwen 3 "think" before answering by default, which can use
# up the answer budget. LLM_THINK=false turns thinking off; unset = model default
# (leave unset for models without a thinking mode, such as Llama 3).
_think = os.environ.get("LLM_THINK", "").strip().lower()
THINK = None if not _think else _think in ("1", "true", "yes", "on")

# LLM_FORCE_JSON=true makes Ollama constrain the answer to {"score", "reason"} JSON
# (structured outputs). Useful for models that write commentary before the JSON, such
# as Qwen 3. Off by default, which is how the full Llama 3 runs were made.
FORCE_JSON = os.environ.get("LLM_FORCE_JSON", "").strip().lower() in ("1", "true", "yes", "on")
SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer"}, "reason": {"type": "string"}},
    "required": ["score", "reason"],
}

# ---------------------------------------------------------------------------
# Scoring settings
# ---------------------------------------------------------------------------
SCORE_THRESHOLD = 6          # papers with score >= this are kept (0-10 scale)
MAX_ABSTRACT_CHARS = 2000    # truncate long abstracts to save tokens
MAX_WORKERS = 16             # concurrent Ollama requests
REQUEST_TIMEOUT = 300        # seconds per LLM call
MAX_RETRIES = 3              # retries on LLM failure per paper

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert in tribology, lubricant chemistry, and antioxidant additives. 
Your task is to score a scientific paper's relevance to the topic: "antioxidant additives for lubricants".

Score from 0 to 10:
- 10: Directly about antioxidant additives in lubricants, lubricating oils, greases or any lubricant formulation, with strong focus on antioxidant chemistry, performance, or testing.
- 8-9: Closely related (oxidation stability/inhibition of lubricants, specific antioxidant chemistry like hindered phenols or aromatic amines in oils, ZDDP as antioxidant)
- 6-7: Related (lubricant degradation/aging with oxidation focus, antioxidant testing methods like RPVOT/PDSC for oils, deposit/sludge formation from oil oxidation)
- 4-5: Partially related (general lubricant additives without antioxidant focus, general oil chemistry, antioxidants but for food/biology/polymers not lubricants)
- 2-3: Weakly related (mentions lubricant or antioxidant but paper's focus is elsewhere)
- 0-1: Not related at all (e.g., biological antioxidants, food science, unrelated chemistry)

Respond with ONLY a valid JSON object, no other text:
{"score": <integer 0-10>, "reason": "<one brief sentence explaining the score>"}"""

USER_PROMPT_TEMPLATE = """Title: {title}
Abstract: {abstract}"""
