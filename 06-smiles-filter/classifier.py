"""
RDKit-based SMILES family classification.
"""

from typing import Dict, Optional

from rdkit import Chem


def _load_pattern(smarts: str):
    return Chem.MolFromSmarts(smarts)


PHENOLIC_PATTERNS = [
    _load_pattern("[c][OX2H]"),  # aromatic phenol OH
]

AMINIC_NH_PATTERNS = [
    _load_pattern("[NX3;H1][c]"),      # aryl amine with one N-H
    _load_pattern("[NX3;H2][c]"),      # primary aryl amine
    _load_pattern("[c][NX3;H1][c]"),   # diarylamine with one N-H
]

ALLOWED_ATOMIC_NUMBERS = {1, 6, 7, 8}  # H, C, N, O


def choose_smiles(row: dict) -> Dict[str, str]:
    paper_smiles = (row.get("paper_smiles") or "").strip()
    pubchem_smiles = (row.get("pubchem_canonical_smiles") or "").strip()

    if paper_smiles and paper_smiles != "NOT FOUND":
        return {"smiles_used": paper_smiles, "smiles_source": "paper"}
    if pubchem_smiles and pubchem_smiles != "NOT FOUND":
        return {"smiles_used": pubchem_smiles, "smiles_source": "pubchem"}
    return {"smiles_used": "NOT FOUND", "smiles_source": "none"}


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if not smiles or smiles == "NOT FOUND":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def heavy_atom_count(smiles: str) -> Optional[int]:
    if not smiles or smiles == "NOT FOUND":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return int(mol.GetNumHeavyAtoms())


def classify_smiles(smiles: str) -> Dict[str, str]:
    if not smiles or smiles == "NOT FOUND":
        return {
            "is_phenolic": "no",
            "is_aminic_nh": "no",
            "ao_family": "no_smiles",
            "reason": "no_smiles_available",
        }

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "is_phenolic": "no",
            "is_aminic_nh": "no",
            "ao_family": "invalid_smiles",
            "reason": "rdkit_failed_to_parse_smiles",
        }

    atomic_numbers = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
    if not atomic_numbers.issubset(ALLOWED_ATOMIC_NUMBERS):
        return {
            "is_phenolic": "no",
            "is_aminic_nh": "no",
            "ao_family": "non_chon",
            "reason": "contains_elements_outside_chon",
        }

    phenolic = any(pattern is not None and mol.HasSubstructMatch(pattern) for pattern in PHENOLIC_PATTERNS)
    aminic_nh = any(pattern is not None and mol.HasSubstructMatch(pattern) for pattern in AMINIC_NH_PATTERNS)

    if phenolic and aminic_nh:
        family = "phenolic+aminic_nh"
        reason = "matched_phenolic_and_aminic_nh_smarts"
    elif phenolic:
        family = "phenolic"
        reason = "matched_phenolic_smarts"
    elif aminic_nh:
        family = "aminic_nh"
        reason = "matched_aminic_nh_smarts"
    else:
        family = "unclassified"
        reason = "no_target_smarts_match"

    return {
        "is_phenolic": "yes" if phenolic else "no",
        "is_aminic_nh": "yes" if aminic_nh else "no",
        "ao_family": family,
        "reason": reason,
    }
