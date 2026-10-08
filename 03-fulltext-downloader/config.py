"""
Configuration for Full-text downloader.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PAPERS_DIR = os.path.join(OUTPUT_DIR, "papers")          # raw PDFs / XMLs / HTMLs
FULLTEXT_DIR = os.path.join(OUTPUT_DIR, "fulltext")      # legacy extracted text
FINAL_TEXT_DIR = os.path.join(OUTPUT_DIR, "final_text")  # standardized final TXT output

RESOLVED_JSON = os.path.join(OUTPUT_DIR, "resolved_urls.json")
DOWNLOAD_LOG = os.path.join(OUTPUT_DIR, "download_log.jsonl")
EXTRACTION_LOG = os.path.join(OUTPUT_DIR, "extraction_log.jsonl")
TEXT_PIPELINE_LOG = os.path.join(OUTPUT_DIR, "text_pipeline_log.jsonl")

# Input
INPUT_JSON = os.path.join(BASE_DIR, "..", "02-relevance-scorer", "output", "peek_filtered_ge8.json")

# Identity
EMAIL = os.environ.get("CONTACT_EMAIL", "your_email@university.edu")
USER_AGENT = (
    "llm-lit-mining-antioxidants/1.0 "
    f"(mailto:{EMAIL}; academic research)"
)

# API keys
# API keys come from environment variables (see .env.example); unset = skipped
CORE_API_KEY = os.environ.get("CORE_API_KEY") or None
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None

# Rate limits (seconds between requests per domain)
RATE_LIMIT_DEFAULT = 1.0
RATE_LIMIT_UNPAYWALL = 0.1
RATE_LIMIT_PMC = 0.35
RATE_LIMIT_EPMC = 0.2
RATE_LIMIT_S2 = 0.1
RATE_LIMIT_CORE = 1.0
RATE_LIMIT_PUBLISHER = 2.0

# Download settings
MAX_WORKERS = 8
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
MAX_FILE_SIZE = 100 * 1024 * 1024

# Extraction / labeling controls
ENGLISH_ONLY = True
SAVE_NON_ENGLISH = False
MIN_PARTIAL_FULLTEXT_CHARS = 1200
MIN_COMPLETE_FULLTEXT_CHARS = 4000

# Resolver priority
RESOLVER_PRIORITY = [
    "pmc_xml",
    "epmc_xml",
    "unpaywall",
    "semantic_scholar",
    "core",
    "crossref_link",
]

for d in [OUTPUT_DIR, PAPERS_DIR, FULLTEXT_DIR, FINAL_TEXT_DIR]:
    os.makedirs(d, exist_ok=True)
