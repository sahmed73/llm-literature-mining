import argparse
import json
import logging
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "06-smiles-filter" / "output"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def normalize_smiles(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .replace({"NOT FOUND": "", "not found": ""})
    )


def summarize_file(label: str, path: Path) -> dict:
    df = load_csv(path)
    smiles = normalize_smiles(df["smiles"])
    valid_mask = smiles.ne("")

    type_counts = (
        df["type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace({"": "missing"})
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "table": label,
        "rows": int(len(df)),
        "unique_cids": int(df["cid"].dropna().nunique()) if "cid" in df.columns else 0,
        "unique_names": int(
            df["name"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace({"": pd.NA})
            .dropna()
            .nunique()
        ),
        "unique_smiles": int(smiles[valid_mask].nunique()),
        "valid_smiles_rows": int(valid_mask.sum()),
        "missing_smiles_rows": int((~valid_mask).sum()),
        "type_counts": type_counts,
    }


def build_tables(input_dir: Path, output_dir: Path) -> None:
    log = logging.getLogger("summary")
    tables_dir = output_dir / "summary_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "priority": input_dir / "phenolic_aminic_antioxidants.csv",
        "other": input_dir / "other_or_unclassified_smiles.csv",
    }

    records = []
    compact_summary = {}

    for label, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")
        stats = summarize_file(label, path)
        records.append(stats)
        compact_summary[label] = stats
        log.info(
            "%s rows=%d | unique_smiles=%d | valid_smiles=%d",
            label,
            stats["rows"],
            stats["unique_smiles"],
            stats["valid_smiles_rows"],
        )

    summary_df = pd.DataFrame(records).sort_values("table")
    summary_df.to_csv(tables_dir / "smiles_filter_summary.csv", index=False)

    summary_json_path = input_dir / "summary.json"
    upstream_summary = {}
    if summary_json_path.exists():
        with open(summary_json_path, "r", encoding="utf-8") as f:
            upstream_summary = json.load(f)

    payload = {
        "input_dir": str(input_dir),
        "upstream_summary": upstream_summary,
        "derived_summary": compact_summary,
    }
    with open(tables_dir / "summary_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log.info("Summary tables written to %s", tables_dir)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Build compact summary tables from SMILES-filter outputs."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing phenolic_aminic_antioxidants.csv and other_or_unclassified_smiles.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "output"),
        help="Directory where summary tables will be written.",
    )
    args = parser.parse_args()

    build_tables(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
