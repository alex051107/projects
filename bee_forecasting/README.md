# Bee Colony Forecasting

This module reconstructs and extends honey bee colony activity forecasts with reproducible data ingestion, feature
engineering, and multi-model evaluation. It targets the Bee Informed Partnership colony loss survey and enriches the
observations with hourly weather covariates from NOAA/Meteostat before training traditional, machine learning, and
deep learning models across rolling forecast horizons.

## Data sources

| Dataset | Description | Retrieval script |
| --- | --- | --- |
| Bee Informed colony loss survey (2017 release) | Quarterly colony counts, losses, additions, and renovations collected by USDA NASS and Bee Informed Partnership | `python -m bee_forecasting.data.download_bee_data` |
| NOAA/Meteostat hourly weather | Temperature, precipitation, and atmospheric pressure for user-selected weather stations | `python -m bee_forecasting.data.download_weather --station-id=725300` |

The colony loss survey is hosted inside the [`tidytuesday`](https://github.com/rfordatascience/tidytuesday) project
under `data/2022/2022-01-11/bee-17`. The download helper caches the files in `bee_forecasting/data/raw` and records a
`README` with provenance metadata. Weather downloads rely on the public Meteostat API.

## Pipeline

1. **Standardize schema** – `bee_forecasting.datasets.load_colony_panel` converts the survey tables into a tidy
   `(date, state, target, features...)` DataFrame with configurable aggregation windows.
2. **Feature generation** – `bee_forecasting.features.build_feature_matrix` merges colony observations with lagged and
   rolling statistics as well as aligned weather covariates.
3. **Model zoo** – Baseline (`SeasonalNaiveModel`, `ArimaModel`, `ProphetModel`), machine learning (`RandomForestModel`,
   `LightGBMModel`, `XGBoostModel`, `CatBoostModel`, `SupportVectorRegressionModel`), and deep learning (`ElmanRNNModel`,
   `LSTMModel`, `TemporalConvNetModel`) estimators share a common
   interface and are orchestrated by `bee_forecasting.rolling.RollingForecaster`.
4. **Evaluation** – Rolling-origin forecasts for 7/14/30/90-day horizons log metrics to `metrics/metrics.csv` and
   persist diagnostic plots in `reports/`.
5. **Dashboard** – A Plotly Dash app (see `apps/dashboard.py`) visualizes historical data, model forecasts, residuals,
   and SHAP feature attributions with interactive station, horizon, and model selectors.

## Quick start

```bash
# 1. Create environment (full dependency list in pyproject.toml)
uv venv --python 3.11
uv pip install -e .[bee]

# 2. Download data
python -m bee_forecasting.data.download_bee_data --output-dir data/raw
python -m bee_forecasting.data.download_weather --station-id=725300 --start 2016-01-01 --end 2017-12-31 --output-dir data/raw

# 3. Build features and run multi-horizon forecasts
python -m bee_forecasting.scripts.run_forecasts --config configs/default.yaml

# 4. Launch dashboard
python -m bee_forecasting.apps.dashboard --config configs/default.yaml
```

Artifacts are written to `bee_forecasting/outputs` by default and include:

- `metrics/metrics.csv` – evaluation table `(model, horizon, metric, value)`
- `figures/` – forecast curves, residual diagnostics, SHAP summaries
- `artifacts/forecast_rollout.parquet` – concatenated rolling predictions

See `configs/default.yaml` for a template covering data paths, forecast horizons, feature settings, and model
hyperparameters.
