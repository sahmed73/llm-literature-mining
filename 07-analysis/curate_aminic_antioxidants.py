#!/usr/bin/env python3
"""Curate conventional aminic antioxidants from LLM role assignments.

This is deliberately conservative.  The LLM role is used only to define the
1,784-record audit universe; it is not accepted as chemical ground truth.
Specific aromatic-amine radical-trapping antioxidants are kept in the core
set.  Related hydroxylamines, aminophenols, nitroxides, HALS, and weak/simple
amines are kept in a separate adjacent set.  Mixtures, generic classes, and
trade products without a unique molecular identity are also kept separately.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "04-compound-extractor/output/extracted_additives.jsonl"
OLD_SET = ROOT / "06-smiles-filter/output/phenolic_aminic_antioxidants.csv"
OUT = Path(__file__).resolve().parent / "output/aminic_curation"

ROLE_RE = re.compile(
    r"\baminic\b|\bamine(?:[- ]type)? antioxidant|\baromatic amine|"
    r"\barylamine|\bdiarylamine",
    re.I,
)

DIRECT_EVIDENCE_RE = re.compile(
    r"antioxid|oxidation inhib|inhibit(?:s|ed|ing)? (?:the )?oxid|"
    r"oxidation stabil|radical[- ]trapp|scaveng|peroxyl|induction period|"
    r"chain break|rate constant|\bRTA\b",
    re.I,
)


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "").casefold()
    value = value.replace("α", "alpha").replace("β", "beta")
    value = re.sub(r"[′’‘`´ʹ]", "'", value)
    value = re.sub(r"[‐‑‒–—−]", "-", value)
    value = value.replace("n,n0", "n,n'").replace("n0-", "n'-")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def clean_text(value) -> str:
    """Make extracted text safe for CSV/JSON without altering visible content."""
    return str(value or "").replace("\x00", "")


def spec(
    candidate_id: str,
    name: str,
    scope: str,
    subclass: str,
    patterns: list[str],
    smiles: str = "",
    identity_note: str = "",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "canonical_name": name,
        "scope": scope,
        "subclass": subclass,
        "patterns": [re.compile(p, re.I) for p in patterns],
        "smiles": smiles,
        "identity_note": identity_note,
    }


# Ordered from more specific to more general.  SMILES are supplied only where
# the name denotes a single, unambiguous molecule; mixture/product structures
# are intentionally blank.
SPECS = [
    spec("AMN001", "4,4'-Dioctyldiphenylamine (DODPA)", "core_specific", "alkylated diarylamine",
         [r"^dodpa$", r"dioctyl.*diphenylamine", r"di-?iso-?octyl-?diphenylamine"],
         "CCCCCCCCc1ccc(Nc2ccc(CCCCCCCC)cc2)cc1"),
    spec("AMN002", "4,4'-Di-tert-butyldiphenylamine (tBu2DPA)", "core_specific", "alkylated diarylamine",
         [r"^tbu2dpa$", r"di-?tert-?butyl.*diphenylamine"],
         "CC(C)(C)c1ccc(Nc2ccc(C(C)(C)C)cc2)cc1"),
    spec("AMN003", "4-Isooctyldiphenylamine", "core_specific", "alkylated diarylamine",
         [r"^p-?isooctyldiphenylamine$"], identity_note="Reported isooctyl substituent is not a unique constitutional structure."),
    spec("AMN004", "Diphenylamine", "core_specific", "diarylamine",
         [r"^diphenylamine(?: \(dpa\))?$", r"^dpa$", r"^ph2nh \(diphenylamine\)$", r"^diphenytamine$", r"^dpa antioxidant$", r"^dpa \(aromatic amine derivative\)$"],
         "c1ccc(Nc2ccccc2)cc1"),
    spec("AMN005", "N-(4-Octylphenyl)-1-naphthylamine (octyl PANA)", "core_specific", "arylnaphthylamine",
         [r"octyl.*phenyl.*(?:alpha|a)-?naphthylamine", r"p-octyl-n-phenyl-a-naphthylamine"],
         identity_note="The literature label does not specify the octyl branching pattern."),
    spec("AMN006", "N-Phenyl-1-naphthylamine (PANA)", "core_specific", "arylnaphthylamine",
         [r"^(?:n-)?phenyl-?(?:1|alpha|a|l)[ -]?(?:naphthylamine|naphtylamine)(?: \([^)]*\))?$", r"^phenyl ?alpha ?naphthylamine$", r"^phenyl-alpha-napthylamine$", r"^phenyl-~-naphthyl-amine$", r"^phenyl-r-naphthylamine \(pan\)$", r"^-naphthyl-phenyl-amine \(npa\)$", r"^n-phenylnaphth-1-ylamine$", r"^n-alpha-naphthyl-n-phenylamine$", r"^pana(?: \(aromatic amine derivative\))?$", r"^pan$", r"^a-npa$", r"^n-phenyl-naphthylamine$"],
         "c1ccc(Nc2cccc3ccccc23)cc1"),
    spec("AMN007", "N-Phenyl-2-naphthylamine (PBNA/Neozone D)", "core_specific", "arylnaphthylamine",
         [r"^(?:n-)?phenyl-?(?:2|3|7|beta|b|fl)[ -]?(?:naphthylamine|naphthyl-amine)", r"^beta-naphthylphenylamine$"],
         "c1ccc(Nc2ccc3ccccc3c2)cc1",
         "OCR labels 3, 7, B, fl, and /3 were consolidated with the beta/2-naphthyl isomer."),
    spec("AMN008", "N,N'-Diphenyl-p-phenylenediamine (DPPD)", "core_specific", "p-phenylenediamine",
         [r"^dppd$", r"di-?phenyl.*p-?phenyl(?:ene|en|£-phenylene).*diamine", r"di-?phenyl-1,4-phenylenediamine"],
         "c1ccc(Nc2ccc(Nc3ccccc3)cc2)cc1"),
    spec("AMN009", "N-Phenyl-p-phenylenediamine (NPPD)", "core_specific", "p-phenylenediamine",
         [r"^nppd$", r"^n-ppda$", r"n-phenyl-(?:1, ?4-|p-)?phenylenediam", r"n-phenyl-p-phenylenediammine"],
         "Nc1ccc(Nc2ccccc2)cc1"),
    spec("AMN010", "N,N'-Di-sec-butyl-p-phenylenediamine (DBPDA)", "core_specific", "p-phenylenediamine",
         [r"di-sec-butyl-p-phenylenediamine", r"dibutan-2-ylbenzene-1,4-diamine"],
         "CCC(C)Nc1ccc(NC(C)CC)cc1"),
    spec("AMN011", "N-Isopropyl-N'-phenyl-p-phenylenediamine (IPPD)", "core_specific", "p-phenylenediamine",
         [r"n-phenyl-n'-isopropyl-p-phenylenediamine", r"n-isopropyl-n'-phenyl-p-phenylenediamine", r"^ippd$"],
         "CC(C)Nc1ccc(Nc2ccccc2)cc1"),
    spec("AMN012", "N-Cyclohexyl-N'-phenyl-p-phenylenediamine (CPPD/Flexzone 6H)", "core_specific", "p-phenylenediamine",
         [r"^flexzone 6h$", r"n-phenyl-n'-cyclohexyl-p-phenylenediamine", r"n-cyclohexyl-n'-phenyl"],
         "c1ccc(Nc2ccc(NC3CCCCC3)cc2)cc1"),
    spec("AMN013", "p-Phenylenediamine", "core_specific", "p-phenylenediamine",
         [r"^p-?phenylene-?diamine$", r"^p-phenylenediamine$"],
         "Nc1ccc(N)cc1"),
    spec("AMN014", "m-Phenylenediamine", "core_specific", "phenylenediamine",
         [r"^meta-phenylene-?diamine$"], "Nc1cccc(N)c1"),
    spec("AMN015", "N,N,N',N'-Tetramethyl-p-phenylenediamine", "core_specific", "tertiary aromatic diamine",
         [r"tetra-?methyl.*p-?phenylenediamine", r"^tetramethyl-p-phenylenediamine$", r"^v,ar'-tetra-methyl-/>-phenylenediamine$"],
         "CN(C)c1ccc(N(C)C)cc1",
         "No N-H bond; retained because the source explicitly evaluates it as an amine antioxidant."),
    spec("AMN016", "Phenothiazine", "core_specific", "heterocyclic diarylamine",
         [r"^ptz$", r"^phenothiazine(?: \(98 %\)| 9)?$"],
         "c1ccc2Sc3ccccc3Nc2c1"),
    spec("AMN017", "3,7-Dimethoxyphenothiazine", "core_specific", "heterocyclic diarylamine",
         [r"^3,7-dimethoxyphenothiazine$"], "COc1ccc2Sc3ccc(OC)cc3Nc2c1"),
    spec("AMN018", "Phenoxazine", "core_specific", "heterocyclic diarylamine",
         [r"^phenoxazine(?: 10)?$"], "c1ccc2Oc3ccccc3Nc2c1"),
    spec("AMN019", "Ethoxyquin", "core_specific", "dihydroquinoline",
         [r"^ethoxyquin(?: \(i\))?$", r"^emq$", r"^6-ethoxy-2,2,4-trimethyl-1,2-dihydroquinoline$"],
         "CCOc1ccc2c(c1)C(C)=CC(C)(C)N2"),
    spec("AMN020", "5-Ethyl-10,10-diphenylphenazasiline", "core_specific", "heterocyclic arylamine",
         [r"^5-ethyl-10,10-diphenylphenazasiline"], identity_note="Specific reported molecule; structure was not recoverable from the extraction."),
    spec("AMN021", "4,4'-Bis(dimethylamino)diphenylamine", "core_specific", "tertiary/secondary aromatic amine",
         [r"^4,4'-bis\(n,n-dimethylamino\)diphenylamine$"], "CN(C)c1ccc(Nc2ccc(N(C)C)cc2)cc1"),
    spec("AMN023", "Dimethyl-bis(p-phenylaminophenoxy)silane", "core_specific", "silicon-containing diarylamine",
         [r"^dimethyl-bis\(p-phenylaminophenoxy\)silane"], identity_note="Specific reported molecule; source structure check required."),

    # Conventional aminic families/products that do not identify one molecule.
    spec("AMF001", "Alkylated diphenylamines (ADPA)", "core_family", "alkylated diarylamine mixture/family",
         [r"^adpas?$", r"^adpa(?: \(alkylated diphenylamine\))?$", r"alkylated diphenyl ?amines?", r"^alkyldiphenylamine$", r"^mixed alkylated diphenyl amine$", r"^alkylated diphenylamine antioxidants$", r"^diphenylamines$", r"^odpa$"],
         identity_note="ADPA is a compositional family/mixture, not one molecule."),
    spec("AMF002", "Nonylated diphenylamine (NDPA)", "core_family", "alkylated diarylamine mixture/family",
         [r"^ndpa$", r"nonylated diphenylamine", r"diisononyldiphenylamine"], identity_note="Commercial NDPA is an isomer/substitution mixture."),
    spec("AMF003", "Styrenated diphenylamine", "core_family", "alkylated diarylamine mixture/family",
         [r"^styrenated diphenylamine$"], identity_note="Commercial styrenated DPA is a reaction-product mixture."),
    spec("AMF004", "Octylated/butylated diphenylamine (OBPA)", "core_family", "alkylated diarylamine mixture/family",
         [r"octylated.*butylated diphenyl"], identity_note="Commercial alkylation mixture."),
    spec("AMF005", "Dialkylated diphenylamine", "core_family", "alkylated diarylamine mixture/family",
         [r"^dialkylated diphenylamine$", r"^dialkyldiphenylamine; l-01$"], identity_note="Alkyl groups and substitution pattern are not uniquely specified."),
    spec("AMF006", "Octadecyl diphenylamine (ODA label)", "core_family", "alkylated diarylamine",
         [r"^oda \(octadecyl diphenylamine\)$"], identity_note="Position/branching are not specified; ODA is also ambiguously used for octadecylamine."),
    spec("AMF007", "Alkoxy/alkylamino diphenylamines", "core_family", "substituted diarylamine family",
         [r"^adpa \(n-sec-alkoxydiphenylamine\)$", r"^alkylaminodiphenylamine"], identity_note="Family-level literature label."),
    spec("AMF008", "Substituted or mixed phenyl-naphthylamines (including APANA/OPANA)", "core_family", "arylnaphthylamine mixture/family",
         [r"alkylated-?phenyl-alpha-naphthylamine", r"^phenyl-(?:alpha|a)-naphthylamines(?: \(a\))?$", r"^n-phenyl-alpha,beta-naphthylamine$", r"^opana$", r"^\[opana\]"],
         identity_note="Alkyl substitution and composition are not uniquely specified."),
    spec("AMF009", "Poly(diphenylamine) derivatives (PDPA)", "core_family", "polymeric aromatic amine",
         [r"poly\(diphenylamine\)", r"^pdpa$"], identity_note="Polymeric/reaction-product family; no unique molecular SMILES."),
    spec("AMF010", "Poly(p-phenylenediamine) (PPDA)", "core_family", "polymeric aromatic diamine",
         [r"^poly\(p-phenylenediamine\) \(ppda\)$"], identity_note="Polymer, not a single molecular structure."),
    spec("AMF011", "Schiff-base/phenolic diphenylamine antioxidants (SPD/SSPD)", "core_family", "hybrid aromatic amine family",
         [r"schiff base bridged phenolic diphenylamine", r"^sspd", r"^rsspd$", r"diphenylamine-phenol antioxidants"],
         identity_note="Multiple related structures/products are grouped because the extraction does not recover the defining figures."),
    spec("AMF012", "Pyridine-/pyrimidine-containing diarylamines", "core_family", "heteroarylamine family",
         [r"pyridine- and pyrimidine-containing diarylamine antioxidants", r"nitrogen-incorporated phenoxazines"], identity_note="Family-level label."),
    spec("AMF013", "Generic phenylenediamine antioxidants", "core_family", "aromatic diamine family",
         [r"^p-?phenylenediamines$", r"^phenylenediamines$", r"^/-phenylenediamine$", r"^ppda$", r"substituted n,n'-diphenyl-p-phenylenediamine"], identity_note="Family-level label; PPDA can also denote the polymer."),
    spec("AMF014", "Generic phenothiazines", "core_family", "heterocyclic diarylamine family",
         [r"^phenothiazines$"], identity_note="Plural class mention; not forced to the parent phenothiazine molecule."),
    spec("AMF015", "Generic diarylamine/arylamine products", "core_family", "aromatic amine family/product",
         [r"^arylamine$", r"^n-alkyl aniline$", r"^[a-z]-[a-z]-[a-z]-methylaniline$", r"^a-methylaniline$", r"^additin rc 7132$", r"^t531$", r"^tz516$", r"^ao3 \(amine-type antioxidant\)$", r"^amine one$", r"^amine two$", r"^nda$", r"^noa$", r"^sac$", r"^am \(amine antioxidant\)$"],
         identity_note="Credible aminic label, but no unique identity was recoverable."),
    spec("AMF016", "Diphenylamine/N-aryl-naphthylamine oligomer reaction products", "core_family", "oligomeric aromatic amine",
         [r"^reaction product of specific molar ratios of diphenylamines and n-aryl naphthylamines$"], identity_note="Reaction-product mixture."),

    # Chemically adjacent antioxidants: real antioxidant evidence, but outside
    # the conventional lubricant 'aminic antioxidant' core definition.
    spec("AMA001", "N,N-Diethylhydroxylamine (DEHA)", "adjacent_specific", "hydroxylamine",
         [r"diethylhydroxylamine \(deha\)"], "CCN(O)CC"),
    spec("AMA002", "Ethoxyquin nitroxide V", "adjacent_specific", "nitroxide transformation product",
         [r"^nitroxide v", r"^nitroxide free-radical scavenger$"], identity_note="Nitroxide product/scavenger, not an N-H aminic antioxidant."),
    spec("AMA003", "p-Hydroxydiphenylamine", "adjacent_specific", "aminophenol/diarylamine hybrid",
         [r"^p-hydroxydiphenylamine$"], "Oc1ccc(Nc2ccccc2)cc1"),
    spec("AMA004", "p-Aminophenol", "adjacent_specific", "aminophenol hybrid",
         [r"^p-aminophenol(?: \(x\))?$"], "Nc1ccc(O)cc1"),
    spec("AMA005", "o-Aminophenol", "adjacent_specific", "aminophenol hybrid",
         [r"^o-aminophenol$"], "Nc1ccccc1O"),
    spec("AMA006", "m-Aminophenol", "adjacent_specific", "aminophenol hybrid",
         [r"^m-aminophenol$"], "Nc1cccc(O)c1"),
    spec("AMA007", "Aniline", "adjacent_specific", "simple aromatic amine",
         [r"^aniline(?: \(anil\))?$", r"^aromatic amines с6н5nh2$"], "Nc1ccccc1"),
    spec("AMA008", "p-Anisidine", "adjacent_specific", "simple aromatic amine",
         [r"^p-anisidine \(anis\)$"], "COc1ccc(N)cc1"),
    spec("AMA009", "N,N-Dimethylaniline", "adjacent_specific", "tertiary aromatic amine",
         [r"^n,n-dimethylaniline$"], "CN(C)c1ccccc1"),
    spec("AMA010", "Diphenylguanidine (DPG)", "adjacent_specific", "guanidine antioxidant",
         [r"^diphenylguanidine$", r"diphenylguanidine \(dpg\)"], "NC(=Nc1ccccc1)Nc1ccccc1"),
    spec("AMA011", "Ethylenediamine", "adjacent_specific", "simple aliphatic diamine",
         [r"^ethylenediamine(?: \(eda\))?$"], "NCCN"),
    spec("AMA012", "N,N'-Bis(2-hydroxybenzyl)-o-phenylenediamine", "adjacent_specific", "aminophenol/diamine hybrid",
         [r"^n,n- bis\(2-hydroxybenzyl\)-o-phenylenediamine"], identity_note="Reported molecule; stereochemical/positional source check required."),
    spec("AMA013", "N,N-Bis(2-hydroxy-3,5-di-tert-butylphenyl)-o-ethylenediamine", "adjacent_specific", "aminophenol/diamine hybrid",
         [r"^n,n-bis\(2-hydroxy-3,5-di-tert-butylphenyl\)-o-ethylenediamine$"], identity_note="Reported molecule; source structure check required."),
    spec("AMA014", "N,N-Bis(4-hydroxy-3-methoxybenzyl)-o-phenylenediamine", "adjacent_specific", "aminophenol/diamine hybrid",
         [r"^n,n-bis\(4-hydroxy-3-methoxybenzyl\)-o-phenylenediamine$"], identity_note="Reported molecule; source structure check required."),
    spec("AMA015", "5-Ethyl-10,10-diphenylphenazasiline", "adjacent_specific", "silicon-containing heterocycle",
         [r"^5-ethyl-10,10-diphenylphenazasiline"], identity_note="Reported radical inhibitor; source structure check required."),
    spec("AMA016", "Tinuvin 770", "adjacent_specific", "hindered amine light stabilizer (HALS)",
         [r"^tinuvin 770$"], "CC1(CC(CC(N1)(C)C)OC(=O)CCCCCCCCC(=O)OC2CC(NC(C2)(C)C)(C)C)C",
         "HALS/photooxidation stabilizer; source says it has virtually no thermal antioxidant activity in the tested saturated polymer."),
    spec("AMA020", "alpha-Naphthylamine", "adjacent_specific", "primary aromatic amine",
         [r"^a-naphthylamine$"], "Nc1cccc2ccccc12"),
    spec("AMA021", "Dibenzylamine", "adjacent_specific", "secondary aliphatic amine",
         [r"^dibenzylamine$"], "c1ccc(CNCC2=CC=CC=C2)cc1"),
    spec("AMA022", "N-Oxydiphenylamine", "adjacent_specific", "N-oxy diarylamine",
         [r"^n-oxydiphenylamine$"], identity_note="Historical name is structurally ambiguous; source structure must be checked."),
    spec("AMA023", "4-(Dimethylaminomethyl)-2,6-di-tert-butylphenol", "adjacent_specific", "aminomethyl phenol hybrid",
         [r"^4-\(dimethylaminomethyl\)-2,6-di-tert-butylphenol$"], "CN(C)Cc1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1"),
    spec("AMA024", "Aminomethyl-2-methoxyphenol derivatives", "adjacent_family", "aminophenol hybrid family",
         [r"^4,6-di\[\(morpholin-4-yl\)methyl\]-2-methoxyphenol", r"pyrrolidin-1-yl.*vanillic acid", r"^aminomethyl derivative v$"], identity_note="Multiple aminomethyl phenol structures; retained as an adjacent family."),
    spec("AMA025", "beta-Naphthylamine", "adjacent_specific", "primary aromatic amine",
         [r"^/3-naphthylamine$"], "Nc1ccc2ccccc2c1"),
    spec("AMA017", "Toluidine (isomer unspecified)", "adjacent_family", "simple aromatic amine",
         [r"^toluidine$"], identity_note="The ortho/meta/para isomer is not specified."),
    spec("AMA018", "Aminophenols (generic)", "adjacent_family", "aminophenol hybrid family",
         [r"^aminophenol$"], identity_note="Isomer/substitution not specified."),
    spec("AMA019", "Generic aromatic/aminic antioxidants", "generic_class", "generic class",
         [r"^aminic antioxidants?$", r"^a minic antioxidant$", r"^aminic anti-oxidant(?: additives?)?$", r"^amine antioxidants?$", r"^amine-type antioxidant$", r"^amine-based antioxidant$", r"^aromatic amines?$", r"^aromatic amine antioxidants?$", r"^classic amine antioxidants$", r"^amines?$", r"^amine$", r"^secondary amines$", r"^mixed amines$", r"^aminic$", r"^aminic antioxidant system$", r"^phenolic/aminic antioxidant system$", r"^phenolic/aminic antioxidants$", r"^mixed phenol/amine antioxidant system$"],
         identity_note="Class-only mention; cannot be converted into a molecular record."),
]


FALSE_POSITIVE_REASONS = [
    (re.compile(r"tocopher|tocotrien|vitamin e", re.I), "phenolic/tocopherol antioxidant, not aminic"),
    (re.compile(r"ascorb|vitamin c", re.I), "ascorbate antioxidant, not aminic"),
    (re.compile(r"\bzdd?p\b|zndt?p|dialkyl.?dithiophosph", re.I), "organometallic/phosphorodithioate additive, not aminic"),
    (re.compile(r"\bbht\b|\bbha\b|butylated hydro|tbhq|propyl gallate|phenolic antioxidant", re.I), "phenolic antioxidant, not aminic"),
    (re.compile(r"mos2|molybdenum|graphene|graphite|nanoparticle|copper|zinc borate|\bcuo\b", re.I), "inorganic/nanomaterial or metal additive, not aminic"),
    (re.compile(r"sitosterol|sesamol|carnos|curcumin|flavon|extract|lignin|rosmarin|tocoph", re.I), "natural/phenolic antioxidant, not aminic"),
    (re.compile(r"octadecylamine|oleylamine|dodecyl amine|alkylamine salts", re.I), "simple long-chain amine used here for surface/tribological function, not supported as an aminic antioxidant"),
    (re.compile(r"phosphatidyl|lecithin|phospholipid", re.I), "phospholipid, not a conventional aminic antioxidant"),
]


def match_spec(name: str) -> dict | None:
    n = norm(name)
    for item in SPECS:
        if any(p.search(n) for p in item["patterns"]):
            return item
    return None


def work_key(record: dict) -> str:
    doi = norm(record.get("doi", ""))
    if doi and doi != "not found":
        return "doi:" + doi
    paper_id = norm(record.get("paper_id", ""))
    paper_id = re.sub(r"_\d+$", "", paper_id)
    return "id:" + paper_id


def exclude_reason(name: str) -> str:
    for pattern, reason in FALSE_POSITIVE_REASONS:
        if pattern.search(name or ""):
            return reason
    return "not supported as a conventional aminic antioxidant by name/structure in this automated audit"


def canonicalize_smiles(smiles: str) -> tuple[str, str]:
    if not smiles:
        return "", "not_applicable_or_unresolved"
    try:
        from rdkit import Chem
    except ImportError:
        return smiles, "manual_smiles_not_rdkit_checked"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, "invalid_manual_smiles"
    return Chem.MolToSmiles(mol), "manually_curated_rdkit_validated"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mentions = []
    all_papers = set()

    with RAW.open(encoding="utf-8") as handle:
        for line in handle:
            paper = json.loads(line)
            for compound in paper.get("compounds", []):
                role = clean_text(compound.get("role", ""))
                if not ROLE_RE.search(role):
                    continue
                evidence = " | ".join(clean_text(v) for v in compound.get("evidence", []))
                compound_name = clean_text(compound.get("compound_name", ""))
                item = match_spec(compound_name)
                decision = item["scope"] if item else "excluded"
                reason = item["identity_note"] if item else exclude_reason(compound_name)
                row = {
                    "paper_id": clean_text(paper.get("paper_id", "")),
                    "work_key": work_key(paper),
                    "doi": clean_text(paper.get("doi", "")),
                    "title": clean_text(paper.get("title", "")),
                    "year": clean_text(paper.get("year", "")),
                    "source": clean_text(paper.get("source", "")),
                    "source_format": clean_text(paper.get("source_format", "")),
                    "compound_name": compound_name,
                    "normalized_name": norm(compound_name),
                    "llm_role": role,
                    "paper_smiles": clean_text(compound.get("smiles", "")),
                    "paper_formula": clean_text(compound.get("formula", "")),
                    "structural_info": clean_text(compound.get("structural_info", "")),
                    "evidence": evidence,
                    "direct_evidence_keyword": bool(DIRECT_EVIDENCE_RE.search(evidence)),
                    "decision": decision,
                    "candidate_id": item["candidate_id"] if item else "",
                    "canonical_name": item["canonical_name"] if item else "",
                    "decision_note": reason,
                }
                mentions.append(row)
                all_papers.add(row["work_key"])

    audit_fields = list(mentions[0])
    with (OUT / "aminic_mentions_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(mentions)

    grouped = defaultdict(list)
    for row in mentions:
        if row["candidate_id"]:
            grouped[row["candidate_id"]].append(row)

    final_rows = []
    for item in SPECS:
        rows = grouped.get(item["candidate_id"], [])
        if not rows:
            continue
        works = sorted({r["work_key"] for r in rows})
        paper_records = sorted({r["paper_id"] for r in rows})
        aliases = sorted({r["compound_name"] for r in rows}, key=lambda x: norm(x))
        dois = sorted({r["doi"] for r in rows if r["doi"] and norm(r["doi"]) != "not found"})
        direct = sum(bool(r["direct_evidence_keyword"]) for r in rows)
        csmiles, structure_status = canonicalize_smiles(item["smiles"])
        if item["scope"] in {"core_specific", "adjacent_specific"}:
            if len(works) >= 2 and direct:
                confidence = "high"
            elif direct:
                confidence = "moderate"
            else:
                confidence = "low"
        else:
            confidence = "family_or_product_no_unique_structure"
        evidence_rows = sorted(rows, key=lambda r: (not r["direct_evidence_keyword"], -len(r["evidence"])))
        representative = evidence_rows[0]
        final_rows.append({
            "candidate_id": item["candidate_id"],
            "canonical_name": item["canonical_name"],
            "scope": item["scope"],
            "subclass": item["subclass"],
            "confidence": confidence,
            "canonical_smiles": csmiles,
            "structure_status": structure_status,
            "n_mentions": len(rows),
            "n_extraction_records": len(paper_records),
            "n_deduplicated_works": len(works),
            "n_direct_evidence_mentions": direct,
            "aliases": " | ".join(aliases),
            "dois": " | ".join(dois),
            "representative_paper_id": representative["paper_id"],
            "representative_title": representative["title"],
            "representative_evidence": representative["evidence"],
            "curation_note": item["identity_note"],
        })

    final_rows.sort(key=lambda r: (r["scope"], r["canonical_name"].casefold()))
    final_fields = list(final_rows[0])
    with (OUT / "curated_aminic_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=final_fields)
        writer.writeheader()
        writer.writerows(final_rows)

    molecular = [r for r in final_rows if r["scope"] == "core_specific"]
    with (OUT / "final_core_molecular_set.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=final_fields)
        writer.writeheader()
        writer.writerows(molecular)

    families = [r for r in final_rows if r["scope"] in {"core_family", "generic_class"}]
    with (OUT / "credible_families_products_and_generic_classes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=final_fields)
        writer.writeheader()
        writer.writerows(families)

    adjacent = [r for r in final_rows if r["scope"].startswith("adjacent")]
    with (OUT / "adjacent_amine_antioxidants.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=final_fields)
        writer.writeheader()
        writer.writerows(adjacent)

    excluded = [r for r in mentions if r["decision"] == "excluded"]
    exclusion_counts = Counter((r["normalized_name"], r["decision_note"]) for r in excluded)
    with (OUT / "excluded_name_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["normalized_name", "example_name", "reason", "n_mentions", "n_deduplicated_works"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        by_name = defaultdict(list)
        for row in excluded:
            by_name[(row["normalized_name"], row["decision_note"])].append(row)
        for (nname, reason), rows in sorted(by_name.items(), key=lambda x: (-len(x[1]), x[0][0])):
            writer.writerow({
                "normalized_name": nname,
                "example_name": rows[0]["compound_name"],
                "reason": reason,
                "n_mentions": len(rows),
                "n_deduplicated_works": len({r["work_key"] for r in rows}),
            })

    # Structure-level comparison to the earlier 34-row aminic output.  This is
    # intentionally not a name join because several of the old problems were
    # incorrect name-to-PubChem resolutions.
    old_by_smiles = {}
    if OLD_SET.exists():
        with OLD_SET.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("type") != "aminic":
                    continue
                canonical, status = canonicalize_smiles(row.get("smiles", ""))
                if canonical and status != "invalid_manual_smiles":
                    old_by_smiles[canonical] = row
    new_by_smiles = {r["canonical_smiles"]: r for r in molecular if r["canonical_smiles"]}
    comparison = []
    for smiles in sorted(set(old_by_smiles) | set(new_by_smiles)):
        old = old_by_smiles.get(smiles, {})
        new = new_by_smiles.get(smiles, {})
        status = "shared" if old and new else ("new_core_only" if new else "previous_34_only")
        comparison.append({
            "status": status,
            "canonical_smiles": smiles,
            "previous_cid": old.get("cid", ""),
            "previous_name": old.get("name", ""),
            "new_candidate_id": new.get("candidate_id", ""),
            "new_canonical_name": new.get("canonical_name", ""),
        })
    with (OUT / "comparison_to_previous_34.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["status", "canonical_smiles", "previous_cid", "previous_name", "new_candidate_id", "new_canonical_name"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison)

    scopes = Counter(r["decision"] for r in mentions)
    candidate_scopes = Counter(r["scope"] for r in final_rows)
    summary = {
        "input": str(RAW),
        "role_filter": ROLE_RE.pattern,
        "llm_flagged_mentions": len(mentions),
        "verbatim_names": len({r["compound_name"] for r in mentions}),
        "lightly_normalized_names": len({r["normalized_name"] for r in mentions}),
        "extraction_paper_records": len({r["paper_id"] for r in mentions}),
        "deduplicated_works": len(all_papers),
        "mention_decisions": dict(sorted(scopes.items())),
        "canonical_candidate_counts": dict(sorted(candidate_scopes.items())),
        "core_specific_candidates": len(molecular),
        "core_specific_with_validated_smiles": sum(r["structure_status"] == "manually_curated_rdkit_validated" for r in molecular),
        "core_family_candidates": sum(r["scope"] == "core_family" for r in final_rows),
        "adjacent_candidates": len(adjacent),
        "generic_class_candidates": sum(r["scope"] == "generic_class" for r in final_rows),
        "excluded_mentions": len(excluded),
        "previous_aminic_structures": len(old_by_smiles),
        "previous_vs_new_shared_structures": len(set(old_by_smiles) & set(new_by_smiles)),
        "new_core_structures_not_in_previous": len(set(new_by_smiles) - set(old_by_smiles)),
        "previous_structures_not_in_new_core": len(set(old_by_smiles) - set(new_by_smiles)),
        "method_note": "Conservative deterministic chemical-name curation; source figures and full text were not manually re-read for every mention.",
    }
    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
