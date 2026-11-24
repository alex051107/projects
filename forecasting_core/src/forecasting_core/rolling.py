"""Rolling-origin evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from .models.base import ForecastModel


MetricFn = Callable[[np.ndarray, np.ndarray], float]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


DEFAULT_METRICS: dict[str, MetricFn] = {
    "mae": lambda y_true, y_pred: float(mean_absolute_error(y_true, y_pred)),
    "rmse": rmse,
    "mape": lambda y_true, y_pred: float(mean_absolute_percentage_error(y_true, y_pred)),
}


@dataclass
class RollingForecaster:
    target_column: str
    time_column: str
    feature_columns: Sequence[str] = field(default_factory=list)
    horizons: Sequence[int] = field(default_factory=lambda: (7, 14, 30, 90))
    metrics: dict[str, MetricFn] = field(default_factory=lambda: DEFAULT_METRICS)
    min_train_size: int = 24
    step: int = 1
    group_column: str | None = "state"

    def evaluate(
        self,
        data: pd.DataFrame,
        models: Iterable[ForecastModel],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run rolling-origin evaluation across horizons and models."""

        metrics_records = []
        forecast_records = []

        groups: list[str | None]
        if self.group_column and self.group_column in data.columns:
            groups = list(data[self.group_column].dropna().unique())
        else:
            groups = [None]

        for group in groups:
            if group is None:
                subset = data.copy()
            else:
                subset = data[data[self.group_column] == group].copy()
            subset = subset.sort_values(self.time_column).reset_index(drop=True)

            for horizon in self.horizons:
                fold = 0
                last_start = len(subset) - horizon
                for train_end in range(self.min_train_size, last_start + 1, self.step):
                    train = subset.iloc[:train_end]
                    test = subset.iloc[train_end : train_end + horizon]

                    y_train = train[self.target_column]
                    y_test = test[self.target_column].to_numpy()

                    for model in models:
                        estimator = model.clone()
                        X_train = None
                        X_test = None
                        if self.feature_columns and estimator.uses_exogenous:
                            X_train = train[list(self.feature_columns)]
                            X_test = test[list(self.feature_columns)]

                        estimator.fit(y_train, X_train)
                        y_pred = estimator.predict(horizon, X_test)

                        for metric_name, metric_fn in self.metrics.items():
                            score = metric_fn(y_test, y_pred)
                            metrics_records.append(
                                {
                                    "model": estimator.name,
                                    "horizon": horizon,
                                    "fold": fold,
                                    "metric": metric_name,
                                    "value": score,
                                    "group": group,
                                }
                            )

                        forecast_records.extend(
                            {
                                self.time_column: test[self.time_column].iloc[idx],
                                "model": estimator.name,
                                "horizon": horizon,
                                "fold": fold,
                                "forecast": float(y_pred[idx]),
                                "actual": float(y_test[idx]),
                                "group": group,
                            }
                            for idx in range(len(y_pred))
                        )
                    fold += 1

        metrics_df = pd.DataFrame(metrics_records)
        forecasts_df = pd.DataFrame(forecast_records)
        return metrics_df, forecasts_df
