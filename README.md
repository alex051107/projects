# Projects Portfolio Roadmap

This repository now includes reproducible pipelines for three interdisciplinary analytics projects alongside the planning
roadmap. Each project contains data ingestion, feature engineering, modeling, and reporting utilities implemented as
Python packages with CLI entrypoints.

## Project directories

| Project | Path | Highlights |
| --- | --- | --- |
| Bee Colony Forecasting | `bee_forecasting/` | Download Bee Informed survey tables, engineer covariates with Meteostat weather, train ARIMA/Prophet/ML/DL models, serve a Dash dashboard. |
| Hydrological Water Level Forecasting | `water_level_forecasting/` | Fetch USGS gauge series, merge hydrometeorological drivers, reuse the rolling evaluation engine, export metrics/backtests. |
| "Vassar" Virtual Screening | `virtual_screening/` | Prepare receptor & ligands, orchestrate AutoDock Vina docking, train RF/XGBoost rescoring, apply ADMET filters, and generate reports. |

Each package exposes a `scripts/run_forecasts.py` or `scripts/run_pipeline.py` module to execute the full workflow with a
YAML configuration. Refer to the project-specific README files for detailed instructions, configuration examples, and
artifact descriptions.

## Getting started

```bash
uv venv --python 3.11
uv pip install -e .[bee,water,screening]
```

Run individual projects:

```bash
# Bee forecasting
python -m bee_forecasting.data.download_bee_data --output-dir bee_forecasting/data/raw
python -m bee_forecasting.data.download_weather --station-id=725300 --start 2016-01-01 --end 2017-12-31 --output-dir bee_forecasting/data/raw
python -m bee_forecasting.scripts.run_forecasts --config bee_forecasting/configs/default.yaml

# Hydrological forecasting
python -m water_level_forecasting.data.download_usgs --site 01646500 --start 2015-01-01 --end 2020-12-31 --output water_level_forecasting/data/raw
python -m water_level_forecasting.data.download_weather --station-id=724050 --start 2015-01-01 --end 2020-12-31 --output water_level_forecasting/data/raw
python -m water_level_forecasting.scripts.run_forecasts --config water_level_forecasting/configs/potomac.yaml

# Virtual screening
python -m virtual_screening.data.download_target --pdb-id 6LU7 --output virtual_screening/data/targets
python -m virtual_screening.data.download_ligands --subset fragment_like --limit 500 --output virtual_screening/data/ligands
python -m virtual_screening.scripts.run_pipeline --config virtual_screening/configs/mpro.yaml
```

Use the project READMEs for dashboard usage, reporting, and interpretation guidance. The roadmap below remains as the
high-level milestone tracker for future enhancements.

## Overview

| Project | Focus | Key Data Sources | Baseline Models | ML/DL Enhancements | Major Deliverables |
| --- | --- | --- | --- | --- | --- |
| **P1. Bee Colony Activity Forecasting** | Time-series prediction of colony health/activity with weather covariates | Bee Informed Partnership hive metrics, NOAA/ERA5 weather archives | Naïve, ARIMA, Prophet | LightGBM, XGBoost, Elman RNN, LSTM, Temporal Convolutional Network | Metrics table, rolling forecasts (7/14/30/90 days), interactive Dash dashboard |
| **P2. Hydrological Water Level Prediction** | Multi-horizon water level forecasts with exogenous hydrometeorological drivers | USGS NWIS station levels, precipitation & temperature feeds, upstream discharge records | Naïve, SARIMA, ETS | LightGBM, XGBoost, LSTM, Temporal Fusion Transformer | Backtesting metrics by station/horizon, SHAP feature attribution, visual report |
| **P3. "Vassar" Virtual Screening Pipeline** | Molecular docking and rescoring for a selected protein target | RCSB PDB structure, ZINC15 or DrugBank ligand library | AutoDock Vina/Smina docking | RF-Score, XGBoost-QSAR, gnina CNN, ADMET filtering | top20 hits table, docking visualizations (3D/2D), reproducible screening scripts |

## High-Level Timeline

| Phase | Weeks | Cross-Project Milestones |
| --- | --- | --- |
| **Planning & Data Acquisition** | Week 1–2 | Finalize targets, acquire raw datasets, document retrieval scripts |
| **Baseline Reconstruction** | Week 3–4 | Reproduce statistical models (ARIMA/SARIMA/ETS/Prophet, Vina docking) |
| **Feature Engineering & Pipelines** | Week 5–6 | Standardize data schemas (time, target, group, features), implement preprocessing notebooks/scripts |
| **ML/DL Upgrades** | Week 7–9 | Train machine learning and deep learning models (LightGBM, LSTM, TCN, TFT, gnina) |
| **Evaluation & Visualization** | Week 10–11 | Rolling backtests, metrics.csv, dashboards, visual reports |
| **Packaging & Documentation** | Week 12 | Final README updates, environment specs, deployment instructions |
