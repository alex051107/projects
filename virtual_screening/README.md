# "Vassar" Virtual Screening Pipeline

This package reconstructs the virtual screening workflow for the SARS-CoV-2 main protease (Mpro) described in the Vassar
project while extending it with automated ligand preparation, docking, machine learning rescoring, and ADMET filtering.
It is designed to be reproducible end-to-end, from fetching public structures and ligand libraries to ranking the top
candidates with interactive reports.

## Components

1. **Target & ligand preparation** – `virtual_screening.data.download_target` downloads a PDB structure (default 6LU7).
   `virtual_screening.prep.prepare_ligands` converts SMILES libraries from ZINC/DrugBank into 3D conformers ready for
   docking.
2. **Docking** – `virtual_screening.docking.run_vina` wraps AutoDock Vina/smina command-line tools with configuration
   files generated from YAML templates.
3. **Rescoring** – `virtual_screening.rescoring.train_models` fits RF-Score (RandomForest), XGBoost-QSAR, and gnina CNN
   predictors on docking poses and RDKit descriptors, producing calibrated consensus scores.
4. **ADMET filtering** – `virtual_screening.admet.filter_candidates` applies Lipinski, Veber, and configurable ADMET
   thresholds (cLogP, TPSA, alerts) derived from RDKit descriptors and optional pkCSM predictions.
5. **Reporting** – `virtual_screening.reporting.build_report` creates `top20_hits.csv`, Plotly visualizations of score
   distributions, and 3D ligand-protein overlays using `nglview` snapshots.

## Quick start

```bash
uv venv --python 3.11
uv pip install -e .[screening]

# Download receptor and a small ligand library
python -m virtual_screening.data.download_target --pdb-id 6LU7 --output data/targets
python -m virtual_screening.data.download_ligands --subset fragment_like --limit 500 --output data/ligands

# Prepare inputs, dock, rescore, and filter
python -m virtual_screening.scripts.run_pipeline --config configs/mpro.yaml
```

Outputs are stored under `virtual_screening/outputs` and include docking logs, rescoring summaries, ADMET tables, and
visualization artifacts (`figures/`, `nglview/`).
