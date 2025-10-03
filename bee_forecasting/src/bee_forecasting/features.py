"""Feature engineering for bee colony forecasting."""
from __future__ import annotations

import pandas as pd

from .datasets import DATA_COLUMNS


def _merge_weather_features(features: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    if weather is None or weather.empty:
        return features

    weather = weather.copy()
    if "time" not in weather.columns:
        raise KeyError("Weather dataframe must contain a 'time' column with timestamps.")

    weather["time"] = pd.to_datetime(weather["time"])

    periods = features[["period_start", "period_end"]].drop_duplicates().reset_index(drop=True)
    intervals = pd.IntervalIndex.from_arrays(
        periods["period_start"], periods["period_end"], closed="both"
    )
    weather["period_idx"] = intervals.get_indexer(weather["time"].values)
    weather = weather[weather["period_idx"] >= 0]

    agg_spec = {}
    if "temp" in weather.columns:
        agg_spec.update(
            {
                "weather_temp_mean": ("temp", "mean"),
                "weather_temp_max": ("temp", "max"),
                "weather_temp_min": ("temp", "min"),
            }
        )
    if "prcp" in weather.columns:
        agg_spec["weather_precip_total"] = ("prcp", "sum")
    if "pres" in weather.columns:
        agg_spec["weather_pressure_mean"] = ("pres", "mean")
    if "wspd" in weather.columns:
        agg_spec["weather_wind_mean"] = ("wspd", "mean")

    if not agg_spec:
        raise ValueError("No known weather columns found to aggregate (expected temp/prcp/pres/wspd).")

    aggregated = weather.groupby("period_idx").agg(**agg_spec).reset_index()
    period_features = periods.reset_index().rename(columns={"index": "period_idx"})
    period_features = period_features.merge(aggregated, on="period_idx", how="left").drop(columns="period_idx")
    features = features.merge(period_features, on=["period_start", "period_end"], how="left")
    return features


def build_feature_matrix(
    panel: pd.DataFrame,
    weather: pd.DataFrame | None = None,
    target_column: str = "lost_percent",
    lag_periods: tuple[int, ...] = (1, 2, 4),
    rolling_windows: tuple[int, ...] = (2, 4),
    drop_na: bool = True,
) -> pd.DataFrame:
    """Create a model-ready feature matrix.

    Parameters
    ----------
    panel:
        Output of :func:`load_colony_panel`.
    weather:
        Optional weather dataframe containing a ``time`` column and Meteostat variables.
    target_column:
        Dependent variable used to generate lagged and rolling features.
    lag_periods:
        Sequence of lag offsets (measured in quarters) for autoregressive features.
    rolling_windows:
        Sequence of window lengths (quarters) for rolling mean features.
    drop_na:
        Whether to drop rows containing missing feature values after engineering.
    """

    features = panel.sort_values(["state", "period_start"]).copy()

    for col in DATA_COLUMNS:
        if col == "state":
            continue
        features[f"{col}_per_colony"] = features[col] / features["colonies"].where(features["colonies"] > 0)

    group = features.groupby("state", group_keys=False)
    for lag in lag_periods:
        features[f"{target_column}_lag_{lag}"] = group[target_column].shift(lag)

    for window in rolling_windows:
        features[f"{target_column}_rolling_mean_{window}"] = group[target_column].apply(
            lambda s, w=window: s.shift(1).rolling(window=w, min_periods=1).mean()
        )

    features = features.drop(columns=['table_id'], errors='ignore')
    if weather is not None:
        features = _merge_weather_features(features, weather)

    if drop_na:
        features = features.dropna()

    return features
