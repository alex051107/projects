"""CLI entrypoint to run the bee colony forecasting pipeline."""
from __future__ import annotations

import argparse
import pathlib
from typing import Any, Dict, List

import pandas as pd
import yaml

from .. import build_feature_matrix, load_colony_panel
from ..rolling import RollingForecaster
from ..models.machine_learning import (
    CatBoostModel,
    ElmanRNNModel,
    LightGBMModel,
    LSTMModel,
    RandomForestModel,
    SupportVectorRegressionModel,
    TemporalConvNetModel,
    XGBoostModel,
)
from ..models.statistical import ArimaModel, ProphetModel, SeasonalNaiveModel


MODEL_REGISTRY = {
    "seasonal_naive": lambda cfg: SeasonalNaiveModel(
        season_length=cfg.get("season_length", 4)
    ),
    "sarima": lambda cfg: ArimaModel(
        order=tuple(cfg.get("order", (1, 0, 0))),
        seasonal_order=tuple(cfg.get("seasonal_order", (0, 1, 1, 4))),
        trend=cfg.get("trend"),
    ),
    "prophet": lambda cfg: ProphetModel(
        yearly_seasonality=cfg.get("yearly_seasonality", True),
        weekly_seasonality=cfg.get("weekly_seasonality", False),
        daily_seasonality=cfg.get("daily_seasonality", False),
        changepoint_prior_scale=cfg.get("changepoint_prior_scale", 0.05),
    ),
    "lightgbm": lambda cfg: LightGBMModel(params=cfg.get("params", {})),
    "xgboost": lambda cfg: XGBoostModel(params=cfg.get("params", {})),
    "random_forest": lambda cfg: RandomForestModel(params=cfg.get("params", {})),
    "catboost": lambda cfg: CatBoostModel(params=cfg.get("params", {})),
    "svr": lambda cfg: SupportVectorRegressionModel(params=cfg.get("params", {})),
    "elman": lambda cfg: ElmanRNNModel(
        lookback=cfg.get("lookback", 8),
        epochs=cfg.get("epochs", 200),
        lr=cfg.get("lr", 1e-3),
    ),
    "lstm": lambda cfg: LSTMModel(
        lookback=cfg.get("lookback", 8),
        epochs=cfg.get("epochs", 200),
        lr=cfg.get("lr", 1e-3),
        hidden_size=cfg.get("hidden_size", 64),
    ),
    "tcn": lambda cfg: TemporalConvNetModel(
        lookback=cfg.get("lookback", 8),
        epochs=cfg.get("epochs", 200),
        lr=cfg.get("lr", 1e-3),
        channels=tuple(cfg.get("channels", (32, 32))),
        kernel_size=cfg.get("kernel_size", 3),
    ),
}


def _instantiate_models(model_cfgs: List[Dict[str, Any]]):
    models = []
    for cfg in model_cfgs:
        model_type = cfg.get("type")
        if model_type not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model type: {model_type}")
        factory = MODEL_REGISTRY[model_type]
        params = {k: v for k, v in cfg.items() if k != "type"}
        models.append(factory(params))
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())

    paths = config["paths"]
    features_cfg = config["features"]
    forecast_cfg = config["forecast"]

    colony_dir = pathlib.Path(paths["colony_tables"])
    colony_df = load_colony_panel(colony_dir)

    weather_path = paths.get("weather")
    weather_df = None
    if weather_path:
        weather_path = pathlib.Path(weather_path)
        if weather_path.exists():
            weather_df = pd.read_parquet(weather_path)
        else:
            raise FileNotFoundError(f"Weather dataset not found: {weather_path}")

    feature_matrix = build_feature_matrix(
        panel=colony_df,
        weather=weather_df,
        target_column=features_cfg.get("target_column", "lost_percent"),
        lag_periods=tuple(features_cfg.get("lag_periods", (1, 2, 4))),
        rolling_windows=tuple(features_cfg.get("rolling_windows", (2, 4))),
    )

    feature_columns = features_cfg.get("feature_columns")
    if feature_columns is None:
        excluded = {features_cfg.get("target_column", "lost_percent"), features_cfg.get("time_column", "period_end")}
        feature_columns = [col for col in feature_matrix.columns if col not in excluded]

    time_column = features_cfg.get("time_column", "period_end")
    target_column = features_cfg.get("target_column", "lost_percent")

    models = _instantiate_models(forecast_cfg.get("models", []))

    forecaster = RollingForecaster(
        target_column=target_column,
        time_column=time_column,
        feature_columns=feature_columns,
        horizons=tuple(forecast_cfg.get("horizons", (7, 14, 30, 90))),
        min_train_size=forecast_cfg.get("min_train_size", 24),
        step=forecast_cfg.get("step", 1),
    )

    metrics_df, forecasts_df = forecaster.evaluate(feature_matrix, models)

    metrics_output = pathlib.Path(paths["metrics_output"])
    forecasts_output = pathlib.Path(paths["forecasts_output"])

    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    forecasts_output.parent.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(metrics_output, index=False)
    forecasts_df.to_parquet(forecasts_output, index=False)

    print(f"Saved metrics to {metrics_output}")
    print(f"Saved forecasts to {forecasts_output}")


if __name__ == "__main__":
    main()
