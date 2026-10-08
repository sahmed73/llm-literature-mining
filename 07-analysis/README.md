# Analysis

Small analysis package for the final SMILES-filter outputs.

This folder is designed to live beside:

- `06-smiles-filter/`

Default input:

- `../06-smiles-filter/output`

Default outputs:

- `./output/summary_tables`
- `./output/figures`

## Expected input files

From `06-smiles-filter/output`:

- `phenolic_aminic_antioxidants.csv`
- `other_or_unclassified_smiles.csv`
- `summary.json`

## What it makes

`make_summary_tables.py`
- compact CSV summary for the priority and secondary files
- JSON snapshot combining upstream and derived counts

`make_smiles_space_plots.py`
- `01_priority_type_counts.png`
- `02_heavy_atom_distribution.png`
- `03_pca_chemical_space.png`, `04_tsne_chemical_space.png`, `05_umap_chemical_space.png` (UMAP needs `umap-learn`)
- `06_paper_score_distribution.png`: relevance scores of the papers that have an abstract
- `embedding_points.csv`
- `plot_stats.json`

`curate_aminic_antioxidants.py`
- conservative review of the aminic candidates in the extraction output, written to `output/aminic_curation/`

## Run

From this folder:

```bash
python make_summary_tables.py
python make_smiles_space_plots.py
```

Optional custom paths:

```bash
python make_summary_tables.py --input-dir /path/to/06-smiles-filter/output --output-dir ./output
python make_smiles_space_plots.py --input-dir /path/to/06-smiles-filter/output --output-dir ./output
```

## Notes

- The main analysis is centered on the priority CHON-only antioxidant set.
- The secondary file is still used as a faint background in PCA space for context.
- The PCA plot uses Morgan fingerprints from RDKit.
- Only valid SMILES are used for plotting.
