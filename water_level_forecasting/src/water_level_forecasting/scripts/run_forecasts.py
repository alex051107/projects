"""CLI to run the water level forecasting experiment."""
from __future__ import annotations

import argparse
import pathlib
from typing import Any, Dict, List

import pandas as pd
import yaml

from bee_forecasting.rolling import RollingForecaster
from bee_forecasting.models.machine_learning import LightGBMModel, LSTMModel, TemporalConvNetModel, XGBoostModel
from bee_forecasting.models.statistical import ArimaModel, SeasonalNaiveModel

from .. import build_feature_matrix, load_usgs_timeseries

MODEL_REGISTRY = {
    "seasonal_naive": lambda cfg: SeasonalNaiveModel(season_length=cfg.get("season_length", 7)),
    "sarima": lambda cfg: ArimaModel(
        order=tuple(cfg.get("order", (1, 0, 1))),
        seasonal_order=tuple(cfg.get("seasonal_order", (1, 1, 1, 7))),
    ),
    "lightgbm": lambda cfg: LightGBMModel(params=cfg.get("params", {})),
    "xgboost": lambda cfg: XGBoostModel(params=cfg.get("params", {})),
    "lstm": lambda cfg: LSTMModel(
        lookback=cfg.get("lookback", 14),
        epochs=cfg.get("epochs", 150),
        lr=cfg.get("lr", 1e-3),
        hidden_size=cfg.get("hidden_size", 64),
    ),
    "tcn": lambda cfg: TemporalConvNetModel(
        lookback=cfg.get("lookback", 30),
        epochs=cfg.get("epochs", 200),
        lr=cfg.get("lr", 1e-3),
        channels=tuple(cfg.get("channels", (32, 32))),
    ),
}


def _instantiate_models(model_cfgs: List[Dict[str, Any]]):
    models = []
    for cfg in model_cfgs:
        model_type = cfg.get("type")
        factory = MODEL_REGISTRY.get(model_type)
        if factory is None:
            raise KeyError(f"Unknown model type: {model_type}")
        params = {k: v for k, v in cfg.items() if k != "type"}
        models.append(factory(params))
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())

    paths = config["paths"]
    features_cfg = config["features"]
    forecast_cfg = config["forecast"]

    raw_paths = [pathlib.Path(p) for p in paths["usgs"]]
    dataset = load_usgs_timeseries(raw_paths, rename_parameters=features_cfg.get("rename_parameters"))

    weather_df = None
    if paths.get("weather"):
        weather_path = pathlib.Path(paths["weather"])
        if weather_path.exists():
            weather_df = pd.read_parquet(weather_path)

    feature_matrix = build_feature_matrix(
        data=dataset,
        weather=weather_df,
        target_column=features_cfg.get("target_column", "00065"),
        lag_steps=features_cfg.get("lag_steps", (1, 2, 7, 30)),
        rolling_windows=features_cfg.get("rolling_windows", (7, 30)),
    )

    feature_columns = features_cfg.get("feature_columns")
    if feature_columns is None:
        excluded = {
            features_cfg.get("target_column", "00065"),
            "timestamp",
            "site",
        }
        feature_columns = [col for col in feature_matrix.columns if col not in excluded]

    models = _instantiate_models(forecast_cfg.get("models", []))

    forecaster = RollingForecaster(
        target_column=features_cfg.get("target_column", "00065"),
        time_column="timestamp",
        feature_columns=feature_columns,
        horizons=tuple(forecast_cfg.get("horizons", (7, 14, 30))),
        min_train_size=forecast_cfg.get("min_train_size", 180),
        step=forecast_cfg.get("step", 7),
        group_column="site",
    )

    metrics_df, forecasts_df = forecaster.evaluate(feature_matrix, models)

    metrics_path = pathlib.Path(paths["metrics_output"])
    forecasts_path = pathlib.Path(paths["forecasts_output"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    forecasts_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(metrics_path, index=False)
    forecasts_df.to_parquet(forecasts_path, index=False)

    print(f"Metrics saved to {metrics_path}")
    print(f"Forecasts saved to {forecasts_path}")


if __name__ == "__main__":
    main()
