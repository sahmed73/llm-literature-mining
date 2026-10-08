import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    umap_lib = None
    HAS_UMAP = False


PALETTE = {
    "phenolic": "#c45a1a",
    "aminic": "#1f6aa5",
    "other": "#9a9a9a",
}

DISPLAY_NAME = {
    "phenolic": "Phenolic",
    "aminic": "Aminic",
    "other": "Unclassified",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "06-smiles-filter" / "output"

DEFAULT_SCORES_FILE = REPO_ROOT / "02-relevance-scorer" / "output" / "all_scores.json"

# Set to 0 to use all papers; set to a positive integer to cap sampling.
MAX_PAPERS_DEFAULT = 0

# Set to True to (re-)generate molecule-space plots 01–05.
PLOT_MOLECULE_SPACE = True


def _score_color(score: int) -> str:
    if score <= 2:
        return "#E07070"
    if score <= 5:
        return "#E07070"
    return "#2E8B6A"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "savefig.dpi": 500,
        }
    )


def normalize_smiles(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .replace({"NOT FOUND": "", "not found": ""})
    )


def load_priority_dataset(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "phenolic_aminic_antioxidants.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["type"] = (
        df["type"].fillna("").astype(str).str.strip().str.lower().replace({"": "other"})
    )
    df["smiles"] = normalize_smiles(df["smiles"])
    df = df[df["smiles"].ne("")].copy()
    return df


def load_other_dataset(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "other_or_unclassified_smiles.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["type"] = "other"
    df["smiles"] = normalize_smiles(df["smiles"])
    df = df[df["smiles"].ne("")].copy()
    return df


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def heavy_atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return int(mol.GetNumHeavyAtoms())


def fingerprints_from_smiles(smiles_list: list[str], n_bits: int = 2048) -> np.ndarray:
    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=float)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    return np.vstack(fps)


def prepare_datasets(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    priority = load_priority_dataset(input_dir)
    other = load_other_dataset(input_dir)

    priority["canonical_smiles"] = priority["smiles"].map(canonicalize_smiles)
    other["canonical_smiles"] = other["smiles"].map(canonicalize_smiles)

    priority = priority[priority["canonical_smiles"].ne("")].copy()
    other = other[other["canonical_smiles"].ne("")].copy()

    priority = priority.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
    other = other.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)

    priority["heavy_atoms"] = priority["canonical_smiles"].map(heavy_atom_count)
    other["heavy_atoms"] = other["canonical_smiles"].map(heavy_atom_count)

    stats = {
        "priority_rows": int(len(priority)),
        "other_rows": int(len(other)),
        "priority_type_counts": priority["type"].value_counts().sort_index().to_dict(),
        "max_heavy_atoms_priority": int(priority["heavy_atoms"].max()) if not priority.empty else 0,
        "max_heavy_atoms_other": int(other["heavy_atoms"].max()) if not other.empty else 0,
    }
    return priority, other, stats


def clear_previous_outputs(output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    if figures_dir.exists():
        for path in figures_dir.glob("*"):
            if path.is_file():
                path.unlink()
    figures_dir.mkdir(parents=True, exist_ok=True)

    for filename in ["embedding_points.csv", "plot_stats.json"]:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def plot_priority_counts(priority: pd.DataFrame, figures_dir: Path) -> None:
    counts = (
        priority["type"]
        .value_counts()
        .reindex(["phenolic", "aminic"], fill_value=0)
        .astype(int)
    )
    labels = [DISPLAY_NAME[t] for t in counts.index]

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    bars = ax.bar(labels, counts.values, color=[PALETTE[t] for t in counts.index], width=0.68)
    ax.set_ylabel("Counts")
    ax.set_xlabel("")
    ax.grid(False)
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + max(1, 0.015 * max(counts.values.max(), 1)),
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=16,
        )
    fig.savefig(figures_dir / "01_priority_type_counts.png")
    plt.close(fig)


def plot_heavy_atom_distribution(priority: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    bins = np.arange(0, max(priority["heavy_atoms"].max() + 3, 8), 2)
    for family in ["phenolic", "aminic"]:
        sub = priority.loc[priority["type"] == family, "heavy_atoms"]
        if sub.empty:
            continue
        ax.hist(
            sub,
            bins=bins,
            density=False,
            histtype="step",
            linewidth=2.2,
            color=PALETTE[family],
            label=DISPLAY_NAME[family],
        )

    ax.set_xlabel("Heavy atoms")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    ax.grid(False)
    fig.savefig(figures_dir / "02_heavy_atom_distribution.png")
    plt.close(fig)


def add_pca_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    X = fingerprints_from_smiles(df["canonical_smiles"].tolist())
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    out = df.copy()
    out["pca_1"] = coords[:, 0]
    out["pca_2"] = coords[:, 1]
    return out


def plot_pca_space(priority: pd.DataFrame, other: pd.DataFrame, figures_dir: Path) -> None:
    if priority.empty:
        return

    frames = []
    if not other.empty:
        bg = other.copy()
        bg["plot_family"] = "other"
        frames.append(bg)
    fg = priority.copy()
    fg["plot_family"] = fg["type"]
    frames.append(fg)

    combined = pd.concat(frames, ignore_index=True)
    combined = add_pca_coordinates(combined)

    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)

    other_sub = combined[combined["plot_family"] == "other"]
    if not other_sub.empty:
        ax.scatter(
            other_sub["pca_1"],
            other_sub["pca_2"],
            s=18,
            c=PALETTE["other"],
            alpha=0.22,
            edgecolors="none",
            label=DISPLAY_NAME["other"],
        )

    for family in ["phenolic", "aminic"]:
        sub = combined[combined["plot_family"] == family]
        if sub.empty:
            continue
        ax.scatter(
            sub["pca_1"],
            sub["pca_2"],
            s=42,
            c=PALETTE[family],
            alpha=0.92,
            edgecolors="none",
            label=DISPLAY_NAME[family],
        )

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.grid(False)
    fig.savefig(figures_dir / "03_pca_chemical_space.png")
    plt.close(fig)

    combined.to_csv(figures_dir.parent / "embedding_points.csv", index=False)


# ── t-SNE / UMAP for antioxidant chemical space ──────────────────────────────

def add_tsne_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    X = fingerprints_from_smiles(df["canonical_smiles"].tolist())
    coords = TSNE(n_components=2, random_state=42, perplexity=8, n_iter=1000).fit_transform(X)
    out = df.copy()
    out["tsne_1"] = coords[:, 0]
    out["tsne_2"] = coords[:, 1]
    return out


def add_umap_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    X = fingerprints_from_smiles(df["canonical_smiles"].tolist())
    coords = umap_lib.UMAP(n_components=2, random_state=42).fit_transform(X)
    out = df.copy()
    out["umap_1"] = coords[:, 0]
    out["umap_2"] = coords[:, 1]
    return out


def _scatter_antioxidant_space(ax, combined: pd.DataFrame, x_col: str, y_col: str) -> None:
    """Draw other compounds in the background, then phenolic/aminic on top."""
    other_sub = combined[combined["plot_family"] == "other"]
    if not other_sub.empty:
        ax.scatter(
            other_sub[x_col], other_sub[y_col],
            s=30, c=PALETTE["other"], alpha=0.5, edgecolors="none",
            label=DISPLAY_NAME["other"],
        )
    for family in ["phenolic", "aminic"]:
        sub = combined[combined["plot_family"] == family]
        if sub.empty:
            continue
        ax.scatter(
            sub[x_col], sub[y_col],
            s=40, c=PALETTE[family], alpha=0.6, edgecolors="k", linewidths=0.3,
            label=DISPLAY_NAME[family],
        )


def _build_combined_antioxidants(priority: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not other.empty:
        bg = other.copy()
        bg["plot_family"] = "other"
        frames.append(bg)
    fg = priority.copy()
    fg["plot_family"] = fg["type"]
    frames.append(fg)
    return pd.concat(frames, ignore_index=True)


def _drop_outliers(df: pd.DataFrame, x_col: str, y_col: str, n_std: float = 3.0) -> pd.DataFrame:
    for col in (x_col, y_col):
        mean, std = df[col].mean(), df[col].std()
        df = df[df[col].between(mean - n_std * std, mean + n_std * std)]
    return df


def plot_tsne_space(priority: pd.DataFrame, other: pd.DataFrame, figures_dir: Path) -> None:
    if priority.empty:
        return
    log = logging.getLogger("plots")
    log.info("Computing t-SNE for antioxidant chemical space …")

    combined = _build_combined_antioxidants(priority, other)
    combined = add_tsne_coordinates(combined)
    before = len(combined)
    combined = _drop_outliers(combined, "tsne_1", "tsne_2")
    log.info("Dropped %d outlier point(s) from t-SNE plot", before - len(combined))

    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    _scatter_antioxidant_space(ax, combined, "tsne_1", "tsne_2")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ymin, _ = combined["tsne_2"].min(), combined["tsne_2"].max()
    ax.set_ylim(ymin-5, 70)
    ax.legend(frameon=True, loc="upper left", borderaxespad=0.0, ncol=3, markerscale=1.5,
              columnspacing=0.75, handletextpad=0.1, bbox_to_anchor=(0.05, 1.0))
    
    ax.grid(False)
    fig.savefig(figures_dir / "04_tsne_chemical_space.png", bbox_inches="tight")
    plt.close(fig)
    log.info("Saved 04_tsne_chemical_space.png")


def plot_umap_space(priority: pd.DataFrame, other: pd.DataFrame, figures_dir: Path) -> None:
    if not HAS_UMAP:
        logging.getLogger("plots").warning("umap-learn not installed — skipping UMAP plot.")
        return
    if priority.empty:
        return
    log = logging.getLogger("plots")
    log.info("Computing UMAP for antioxidant chemical space …")

    combined = _build_combined_antioxidants(priority, other)
    combined = add_umap_coordinates(combined)

    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    _scatter_antioxidant_space(ax, combined, "umap_1", "umap_2")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(False)
    fig.savefig(figures_dir / "05_umap_chemical_space.png")
    plt.close(fig)
    log.info("Saved 05_umap_chemical_space.png")


def load_papers_with_abstracts(scores_file: Path, max_papers: int, rng_seed: int = 42) -> pd.DataFrame:
    log = logging.getLogger("plots")
    log.info("Loading paper scores from %s …", scores_file)
    with open(scores_file, encoding="utf-8") as f:
        data = json.load(f)
    papers = data["papers"]

    rows = [
        {
            "doi": p.get("doi", ""),
            "title": p.get("title", ""),
            "abstract": p.get("abstract", ""),
            "year": p.get("year"),
            "relevance_score": p.get("relevance_score"),
            "score_confidence": p.get("score_confidence", ""),
        }
        for p in papers
        if p.get("abstract") and str(p["abstract"]).strip()
    ]
    df = pd.DataFrame(rows)
    df["relevance_score"] = pd.to_numeric(df["relevance_score"], errors="coerce")
    df = df.dropna(subset=["relevance_score"]).copy()
    df["relevance_score"] = df["relevance_score"].astype(int)

    log.info("Papers with abstract: %d", len(df))
    if max_papers > 0 and len(df) > max_papers:
        df = df.sample(n=max_papers, random_state=rng_seed).reset_index(drop=True)
        log.info("Sampled %d papers", max_papers)
    return df


def plot_paper_score_distribution(papers_df: pd.DataFrame, figures_dir: Path) -> dict:
    counts = papers_df["relevance_score"].value_counts().sort_index()
    score_counts = {int(k): int(v) for k, v in counts.items()}
    _draw_score_distribution(score_counts, figures_dir)
    return score_counts


def _draw_score_distribution(score_counts: dict, figures_dir: Path) -> None:
    sorted_counts = dict(sorted(score_counts.items()))
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    bar_colors = [_score_color(s) for s in sorted_counts]
    ax.bar([str(s) for s in sorted_counts], list(sorted_counts.values()),
           color=bar_colors, width=0.7, edgecolor='k')
    ax.set_xlabel("Relevance score")
    ax.set_ylabel("Number of papers")
    ax.set_yscale("log")
    ax.grid(False)
    fig.savefig(figures_dir / "06_paper_score_distribution.png")
    plt.close(fig)
    logging.getLogger("plots").info("Saved 06_paper_score_distribution.png")


def load_cached_plot_stats(output_dir: Path) -> dict:
    path = output_dir / "plot_stats.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def main() -> None:
    setup_logging()
    apply_plot_style()
    log = logging.getLogger("plots")

    parser = argparse.ArgumentParser(
        description="Create chemical-space and paper-scoring plots for the literature-mining pipeline."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing phenolic_aminic_antioxidants.csv and other_or_unclassified_smiles.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "output"),
        help="Directory where figures and embedding CSV will be written.",
    )
    parser.add_argument(
        "--scores-file",
        default=str(DEFAULT_SCORES_FILE),
        help="Path to all_scores.json from 02-relevance-scorer.",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=MAX_PAPERS_DEFAULT,
        help="Maximum number of papers (with abstract) to use for the score distribution plot.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    scores_file = Path(args.scores_file)

    log.info("Input directory : %s", input_dir)
    log.info("Output directory: %s", output_dir)
    log.info("Scores file     : %s", scores_file)

    # Read cache before clearing outputs.
    cached = load_cached_plot_stats(output_dir)
    cached_score_counts = {int(k): int(v) for k, v in cached.get("paper_score_counts", {}).items()}
    _MOL_STAT_KEYS = ("priority_rows", "other_rows", "priority_type_counts",
                      "max_heavy_atoms_priority", "max_heavy_atoms_other")
    cached_mol_stats = {k: cached[k] for k in _MOL_STAT_KEYS if k in cached}

    clear_previous_outputs(output_dir)

    # ── molecule-space plots 01–05 ────────────────────────────────────────────
    if PLOT_MOLECULE_SPACE or not cached_mol_stats:
        priority, other, stats = prepare_datasets(input_dir)
        log.info("Prepared datasets | priority=%d | other=%d",
                 stats["priority_rows"], stats["other_rows"])
        if priority.empty:
            raise SystemExit("No valid priority SMILES were found for plotting.")
    else:
        stats = cached_mol_stats
        priority = other = pd.DataFrame()
        log.info("Skipping prepare_datasets — using cached molecule stats.")

    # plot_priority_counts(priority, figures_dir)           # 01
    # plot_heavy_atom_distribution(priority, figures_dir)   # 02
    # plot_pca_space(priority, other, figures_dir)          # 03
    plot_tsne_space(priority, other, figures_dir)         # 04
    plot_umap_space(priority, other, figures_dir)         # 05

    # ── paper score plot 06 ───────────────────────────────────────────────────
    score_counts: dict = {}
    if cached_score_counts:
        log.info("Using cached paper_score_counts — skipping raw file load.")
        score_counts = cached_score_counts
        _draw_score_distribution(score_counts, figures_dir)
    elif scores_file.exists():
        papers_df = load_papers_with_abstracts(scores_file, max_papers=args.max_papers)
        score_counts = plot_paper_score_distribution(papers_df, figures_dir)  # 06
    else:
        log.warning("Scores file not found, skipping paper score plot: %s", scores_file)

    with open(output_dir / "plot_stats.json", "w", encoding="utf-8") as f:
        payload = dict(stats)
        payload["fingerprint_mode"] = "morgan_rdkit"
        payload["paper_score_counts"] = score_counts
        payload["figures"] = [
            "01_priority_type_counts.png",
            "02_heavy_atom_distribution.png",
            "03_pca_chemical_space.png",
            "04_tsne_chemical_space.png",
            "05_umap_chemical_space.png",
            "06_paper_score_distribution.png",
        ]
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log.info("Figures written to %s", figures_dir)


if __name__ == "__main__":
    main()
