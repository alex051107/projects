# Hydrological Water Level Forecasting

This project recreates and extends classical ARIMA/SARIMA water level forecasts with machine learning and deep learning
baselines. The workflow standardizes United States Geological Survey (USGS) gauge records, enriches them with meteorological
covariates, and evaluates multi-horizon forecasts with interpretable diagnostics.

## Data sources

| Dataset | Description | Retrieval script |
| --- | --- | --- |
| USGS NWIS Daily Values | Water level, discharge, and temperature measurements for a specified site | `python -m water_level_forecasting.data.download_usgs --site 01646500` |
| Meteostat hourly weather | Local weather covariates used as exogenous drivers | `python -m water_level_forecasting.data.download_weather --station-id=724050` |

## Pipeline

1. **Ingestion** – `water_level_forecasting.datasets.load_usgs_timeseries` loads downloaded JSON/CSV into a tidy
   `(timestamp, target, features...)` dataframe. Optional weather features are merged on timestamp.
2. **Feature engineering** – `water_level_forecasting.features.build_feature_matrix` adds lagged water level terms,
   rolling hydrometric aggregates, and engineered meteorological summaries.
3. **Model evaluation** – The shared `RollingForecaster` (imported from `bee_forecasting`) orchestrates SARIMA, ETS,
   gradient boosting, and LSTM/TCN models across configurable forecast horizons. SHAP inspection utilities in
   `water_level_forecasting.explain` compute feature attributions for tree-based models.
4. **Reporting** – Metrics are persisted to `outputs/metrics.csv`, forecasts to `outputs/forecasts.parquet`, and
   visualization notebooks in `notebooks/` reproduce diagnostic plots.

## Quick start

```bash
uv venv --python 3.11
uv pip install -e .[water]

python -m water_level_forecasting.data.download_usgs --site 01646500 --start 2015-01-01 --end 2020-12-31 --output data/raw
python -m water_level_forecasting.data.download_weather --station-id=724050 --start 2015-01-01 --end 2020-12-31 --output data/raw

python -m water_level_forecasting.scripts.run_forecasts --config configs/potomac.yaml
```

Artifacts mirror the bee project for consistency: metrics table, forecast parquet, SHAP summaries, and dashboards
built with Plotly/Dash.
