"""Feature engineering for water level forecasting."""
from __future__ import annotations

from typing import Iterable

import pandas as pd


def _aggregate_weather(weather: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    weather = weather.copy()
    if "time" not in weather.columns:
        raise KeyError("Weather dataframe must include a 'time' column")
    weather["time"] = pd.to_datetime(weather["time"])
    weather = weather.set_index("time").resample(freq)

    agg_args: dict[str, tuple[str, str]] = {}
    if "temp" in weather.obj.columns:
        agg_args["weather_temp_mean"] = ("temp", "mean")
        agg_args["weather_temp_max"] = ("temp", "max")
        agg_args["weather_temp_min"] = ("temp", "min")
    if "prcp" in weather.obj.columns:
        agg_args["weather_precip_total"] = ("prcp", "sum")
    if "pres" in weather.obj.columns:
        agg_args["weather_pressure_mean"] = ("pres", "mean")
    if "wspd" in weather.obj.columns:
        agg_args["weather_wind_mean"] = ("wspd", "mean")

    aggregated = weather.agg(**agg_args).reset_index().rename(columns={"time": "timestamp"})
    return aggregated


def build_feature_matrix(
    data: pd.DataFrame,
    weather: pd.DataFrame | None = None,
    target_column: str = "00065",
    lag_steps: Iterable[int] = (1, 2, 7, 30),
    rolling_windows: Iterable[int] = (7, 30),
    drop_na: bool = True,
) -> pd.DataFrame:
    features = data.sort_values("timestamp").copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"])

    for lag in lag_steps:
        features[f"{target_column}_lag_{lag}"] = features[target_column].shift(lag)

    for window in rolling_windows:
        features[f"{target_column}_rolling_mean_{window}"] = (
            features[target_column].shift(1).rolling(window=window, min_periods=1).mean()
        )
        features[f"{target_column}_rolling_std_{window}"] = (
            features[target_column].shift(1).rolling(window=window, min_periods=1).std()
        )

    if weather is not None:
        weather_features = _aggregate_weather(weather)
        features = features.merge(weather_features, on="timestamp", how="left")

    if drop_na:
        features = features.dropna()

    return features
