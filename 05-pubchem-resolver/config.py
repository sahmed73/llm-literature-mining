"""
Configuration for PubChem resolver.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

INPUT_EXTRACTED_JSON = os.path.join(
    BASE_DIR, "..", "04-compound-extractor", "output", "extracted_additives.json"
)  # Extractor output to resolve

RESOLUTION_CHECKPOINT = os.path.join(OUTPUT_DIR, "resolution_checkpoint.jsonl")  # Auto-resume file
RESOLVED_COMPOUNDS_CSV = os.path.join(OUTPUT_DIR, "resolved_compounds.csv")  # Deduplicated compound table
PAPER_COMPOUND_LINKS_CSV = os.path.join(OUTPUT_DIR, "paper_compound_links.csv")  # One row per paper-compound link
UNRESOLVED_NAMES_JSON = os.path.join(OUTPUT_DIR, "unresolved_names.json")  # Names PubChem could not resolve
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "summary.json")  # Small run summary

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"  # Main PubChem API
PUBCHEM_AUTOCOMPLETE = "https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound"
PUBCHEM_DELAY = 0.25  # Seconds between requests
PUBCHEM_TIMEOUT = 20  # Timeout per request
PUBCHEM_MAX_CIDS = 10  # Max candidate CIDs checked per name

ABBREVIATIONS = {  # Common additive short names
    "bht": "butylated hydroxytoluene",
    "bha": "butylated hydroxyanisole",
    "dppd": "N,N'-diphenyl-p-phenylenediamine",
    "pana": "phenyl-alpha-naphthylamine",
    "zddp": "zinc dialkyldithiophosphate",
    "tbhq": "tert-butylhydroquinone",
    "dtbp": "2,6-di-tert-butylphenol",
    "mbp": "2,2'-methylenebis(4-methyl-6-tert-butylphenol)",
    "odpa": "octylated diphenylamine",
}

for path in [OUTPUT_DIR]:
    os.makedirs(path, exist_ok=True)
