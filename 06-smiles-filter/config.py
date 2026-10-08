"""
Configuration for SMILES filter.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

INPUT_CSV = os.path.join(
    BASE_DIR, "..", "05-pubchem-resolver", "output", "paper_compound_links.csv"
)  # Resolver output with paper_smiles and pubchem_canonical_smiles

PRIORITY_CSV = os.path.join(OUTPUT_DIR, "phenolic_aminic_antioxidants.csv")  # Final priority output
OTHER_CSV = os.path.join(OUTPUT_DIR, "other_or_unclassified_smiles.csv")  # Secondary output
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "summary.json")  # Small run summary
LOG_FILE = os.path.join(OUTPUT_DIR, "smiles_filter.log")

for path in [OUTPUT_DIR]:
    os.makedirs(path, exist_ok=True)
