"""
Configuration for Metadata miner.
Central place for search queries, API keys, and settings.
API keys and the contact email come from environment variables (see .env.example).
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# API Keys  (unset environment variable = that source is skipped)
# ---------------------------------------------------------------------------
API_KEYS = {
    "semantic_scholar": os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None,  # https://api.semanticscholar.org/
    "core": os.environ.get("CORE_API_KEY") or None,                          # https://core.ac.uk/services/api
    "lens": os.environ.get("LENS_API_KEY") or None,                          # https://www.lens.org/
    "dimensions": os.environ.get("DIMENSIONS_API_KEY") or None,              # https://app.dimensions.ai/
}

# Polite-pool email – used by OpenAlex, CrossRef, PubMed
MAILTO = os.environ.get("CONTACT_EMAIL", "your_email@university.edu")

# ---------------------------------------------------------------------------
# Parallel / rate-limit settings
# ---------------------------------------------------------------------------
MAX_WORKERS = 6                 # concurrent API source workers
REQUEST_DELAY = {               # seconds between requests per source
    "openalex": 0.11,           # ~9 req/s  (polite pool)
    "semantic_scholar": 0.11,   # ~9 req/s  (with key)
    "crossref": 0.05,           # ~20 req/s (polite pool)
    "pubmed": 0.34,             # ~3 req/s  (NCBI guideline)
    "europe_pmc": 0.12,         # ~8 req/s
    "core": 0.20,               # ~5 req/s
    "lens": 0.20,               # ~5 req/s
    "dimensions": 0.34,         # ~3 req/s
}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_JSON = "papers_metadata.json"
LOG_FILE = "metadata_miner.log"

# ---------------------------------------------------------------------------
# Search intents  (broad but relevant)
# ---------------------------------------------------------------------------
# Each intent: must_contain (AND), any_of (OR), fields hint
SEARCH_INTENTS = [
    # --- Core: lubricant + antioxidant ---
    {
        "id": "lubricant_antioxidant",
        "must_contain": ["lubricant", "antioxidant"],
        "any_of": [],
    },
    {
        "id": "lubricating_oil_antioxidant",
        "must_contain": ["lubricating oil", "antioxidant"],
        "any_of": [],
    },
    {
        "id": "lubricant_oxidation_inhibitor",
        "must_contain": ["lubricant", "oxidation inhibitor"],
        "any_of": [],
    },
    # --- Oil types + antioxidant ---
    {
        "id": "oil_antioxidant_types",
        "must_contain": ["antioxidant"],
        "any_of": [
            "engine oil", "motor oil", "base oil",
            "synthetic oil", "turbine oil", "gear oil",
        ],
    },
    # --- Oil + oxidation concepts ---
    {
        "id": "oil_oxidation_stability",
        "must_contain": ["oxidation stability"],
        "any_of": ["lubricant", "lubricating oil", "oil", "grease"],
    },
    {
        "id": "oil_oxidative_degradation",
        "must_contain": ["oxidative degradation"],
        "any_of": ["lubricant", "oil", "grease"],
    },
    {
        "id": "oil_thermal_oxidation",
        "must_contain": ["thermal oxidation"],
        "any_of": ["lubricant", "lubricating oil", "oil"],
    },
    {
        "id": "oil_oxidation_mechanism",
        "must_contain": ["oxidation mechanism"],
        "any_of": ["lubricant", "oil"],
    },
    # --- Antioxidant additive ---
    {
        "id": "antioxidant_additive",
        "must_contain": ["antioxidant additive"],
        "any_of": ["lubricant", "oil", "grease", "lubrication"],
    },
    {
        "id": "oxidation_inhibitor_additive",
        "must_contain": ["oxidation inhibitor"],
        "any_of": ["lubricant", "oil", "grease"],
    },
    # --- Specific antioxidant chemistries (broad) ---
    {
        "id": "hindered_phenol",
        "must_contain": ["hindered phenol"],
        "any_of": ["lubricant", "oil", "oxidation", "antioxidant"],
    },
    {
        "id": "phenolic_antioxidant",
        "must_contain": ["phenolic antioxidant"],
        "any_of": ["lubricant", "oil", "oxidation"],
    },
    {
        "id": "aminic_antioxidant",
        "must_contain": ["aminic antioxidant"],
        "any_of": ["lubricant", "oil"],
    },
    {
        "id": "aromatic_amine_antioxidant",
        "must_contain": ["aromatic amine"],
        "any_of": ["lubricant", "oil", "antioxidant", "oxidation"],
    },
    {
        "id": "diphenylamine_oil",
        "must_contain": ["diphenylamine"],
        "any_of": ["lubricant", "oil", "antioxidant", "oxidation"],
    },
    {
        "id": "zddp",
        "must_contain": ["ZDDP"],
        "any_of": ["lubricant", "oil", "antioxidant", "oxidation"],
    },
    {
        "id": "zinc_dialkyldithiophosphate",
        "must_contain": ["zinc dialkyldithiophosphate"],
        "any_of": ["lubricant", "oil", "antioxidant", "oxidation"],
    },
    # --- Testing standards ---
    {
        "id": "oxidation_induction_time",
        "must_contain": ["oxidation induction time"],
        "any_of": ["lubricant", "oil"],
    },
    # --- Broader tribology-related ---
    {
        "id": "lubricant_degradation",
        "must_contain": ["lubricant degradation"],
        "any_of": ["oxidation", "antioxidant", "thermal"],
    },
    {
        "id": "oil_aging_oxidation",
        "must_contain": ["oil aging"],
        "any_of": ["oxidation", "antioxidant", "lubricant"],
    },
    {
        "id": "grease_oxidation",
        "must_contain": ["grease", "oxidation"],
        "any_of": ["antioxidant", "stability", "degradation"],
    },

    # --- Radical scavenging in oils ---
    {
        "id": "radical_scavenger_oil",
        "must_contain": ["radical scavenger"],
        "any_of": ["lubricant", "oil", "antioxidant"],
    },
    {
        "id": "free_radical_oil",
        "must_contain": ["free radical"],
        "any_of": ["lubricant", "lubricating oil", "engine oil"],
    },
    # --- Synergy / mixture studies ---
    {
        "id": "antioxidant_synergy",
        "must_contain": ["antioxidant", "synergy"],
        "any_of": ["lubricant", "oil"],
    },
    {
        "id": "antioxidant_depletion",
        "must_contain": ["antioxidant depletion"],
        "any_of": ["lubricant", "oil"],
    },
    # --- Broadest catch-all ---
    {
        "id": "lubricant_oxidation",
        "must_contain": ["lubricant", "oxidation"],
        "any_of": ["anti-oxidant", "stability", "degradation", "inhibitor"],
    },
    {
        "id": "oil_antioxidant_broad",
        "must_contain": ["oil", "antioxidant"],
        "any_of": ["lubricant", "lubrication", "additive"],
    },
]
