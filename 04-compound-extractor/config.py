"""
Configuration for Compound extractor.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CHUNKS_DIR = os.path.join(OUTPUT_DIR, "chunks")

# Inputs: a folder of paper PDFs/XMLs (not distributed with this repo) and the
# relevance-scored metadata from step 02.
INPUT_PAPERS_DIR = os.environ.get("PAPERS_DIR", os.path.join(BASE_DIR, "..", "data", "papers"))
INPUT_METADATA_JSON = os.path.join(
    BASE_DIR, "..", "02-relevance-scorer", "output", "peek_filtered_ge6.json"
)

# Output files
EXTRACTION_CHECKPOINT = os.path.join(OUTPUT_DIR, "extraction_checkpoint.jsonl")
EXTRACTED_COMPOUNDS_JSONL = os.path.join(OUTPUT_DIR, "extracted_additives.jsonl")
EXTRACTED_COMPOUNDS_JSON = os.path.join(OUTPUT_DIR, "extracted_additives.json")
INGEST_WARNINGS_JSON = os.path.join(OUTPUT_DIR, "ingest_warnings.json")
CHUNK_MANIFEST_JSON = os.path.join(OUTPUT_DIR, "chunk_manifest.json")

# ── Ollama / LLM ─────────────────────────────────────────────────────────────
OLLAMA_BASE_HOST = "http://localhost"
OLLAMA_BASE_PORT = 11434
# One Ollama server per worker. 4 suits an 8B model on one GPU; use 1 for 70B models.
NUM_OLLAMA_SERVERS = int(os.environ.get("OLLAMA_SERVERS", "4"))
OLLAMA_HOSTS    = [
    f"{OLLAMA_BASE_HOST}:{OLLAMA_BASE_PORT + i}"
    for i in range(NUM_OLLAMA_SERVERS)
]
OLLAMA_HOST     = OLLAMA_HOSTS[0]
OLLAMA_BIN      = os.environ.get("OLLAMA_BIN", "ollama")
# Any model served by Ollama works (llama3:8b, llama3:70b, qwen3:30b-a3b, ...).
MODEL_NAME      = os.environ.get("LLM_MODEL", "llama3:8b")
REQUEST_TIMEOUT = 300          # seconds per LLM call
MAX_RETRIES     = 3
TEMPERATURE     = 0.1
MAX_WORKERS     = 56
LLM_WORKERS     = NUM_OLLAMA_SERVERS  # one paper-level worker per Ollama server

# Context window requested from Ollama (tokens). 0 = Ollama's default, which can
# be smaller than what the model supports. Llama 3 supports 8192; Qwen 2.5/3 and
# Llama 3.1+ support 32k or more.
NUM_CTX         = int(os.environ.get("LLM_NUM_CTX", "0"))

# Reasoning models such as Qwen 3 "think" before answering by default, which can use
# up the answer budget. LLM_THINK=false turns thinking off; unset = model default
# (leave unset for models without a thinking mode, such as Llama 3).
_think = os.environ.get("LLM_THINK", "").strip().lower()
THINK = None if not _think else _think in ("1", "true", "yes", "on")

# ── Chunking ─────────────────────────────────────────────────────────────────
# Papers longer than MAX_CHUNK_CHARS are split into overlapping chunks so each
# LLM call fits the context window (~4 characters per token). The default of
# 10,000 characters (~2,500 tokens) suits Llama 3's 8k window. With a
# long-context model, raise CHUNK_CHARS (e.g. 100000) and LLM_NUM_CTX (e.g.
# 32768) to send most papers in a single call. Delete output/chunks after
# changing these so the chunk cache is rebuilt.
MAX_CHUNK_CHARS   = int(os.environ.get("CHUNK_CHARS", "10000"))
CHUNK_OVERLAP_CHARS = int(os.environ.get("CHUNK_OVERLAP_CHARS", "800"))
SUPPORTED_FILE_TYPES = {".pdf", ".xml"}

# ── Known false-positive exclusion list ──────────────────────────────────────
EXCLUDE_COMPOUNDS = {
    "ascorbic acid", "vitamin c", "vitamin e", "alpha-tocopherol",
    "quercetin", "catechin", "epicatechin", "gallic acid", "resveratrol",
    "glutathione", "superoxide dismutase", "sod", "catalase",
    "curcumin", "lycopene", "beta-carotene", "retinol",
    "green tea", "ginger", "moringa", "cbd",
}

# ── LLM Prompt ───────────────────────────────────────────────────────────────
EXTRACTION_PROMPT = """You are an expert in lubricant additive chemistry.

Task: Extract all specific antioxidant additives or oxidation inhibitors
explicitly mentioned in this text for lubricants, base oils, engine oils,
gear oils, greases, biolubricants, or tribological systems.

Include:
- Specific antioxidant molecules
- Trade names and abbreviations when the paper uses them as antioxidant additives
- Any explicit structural or formula information written in the text

Exclude:
- Food, dietary, biological, or medical antioxidants
- Generic classes without a specific compound
- Base oils, viscosity modifiers, detergents, dispersants, and other non-antioxidant additives
- Any chemistry that must be inferred from the name alone

Output strictly valid JSON with no markdown, no backticks, no commentary:
{{
  "compounds": [
    {{
      "compound_name": "compound name exactly as written in the text",
      "smiles": "SMILES only if explicitly written in the text, else NOT FOUND",
      "formula": "molecular formula only if explicitly written in the text, else NOT FOUND",
      "structural_info": "structural wording only if explicitly written in the text, else NOT FOUND",
      "role": "brief role such as phenolic antioxidant, aminic antioxidant, oxidation inhibitor, else NOT FOUND",
      "evidence": "brief supporting phrase copied from the text"
    }}
  ]
}}
If no relevant lubricant antioxidant is found, output exactly:
{{"compounds": []}}

Rules:
- Do not infer SMILES or formulas.
- Do not extract a generic class without a specific named additive.
- If a value is not explicitly present in the text, return NOT FOUND.

TEXT:
<<<{text}>>>"""

# Ensure output dirs exist
for path in [OUTPUT_DIR, CHUNKS_DIR]:
    os.makedirs(path, exist_ok=True)
