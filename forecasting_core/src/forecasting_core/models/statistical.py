"""Statistical forecasting models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .base import ForecastModel


@dataclass
class SeasonalNaiveModel(ForecastModel):
    season_length: int = 4
    name: str = "seasonal_naive"

    def __post_init__(self) -> None:
        self._history: Optional[pd.Series] = None

    @property
    def uses_exogenous(self) -> bool:
        return False

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> None:
        self._history = y.dropna().astype(float)
        if len(self._history) < self.season_length:
            raise ValueError(
                f"Need at least {self.season_length} observations for seasonal naive model"
            )

    def predict(self, horizon: int, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self._history is None:
            raise RuntimeError("Model must be fit before calling predict().")
        tail = self._history.iloc[-self.season_length :].to_numpy()
        reps = int(np.ceil(horizon / self.season_length))
        forecast = np.tile(tail, reps)[:horizon]
        return forecast.astype(float)


@dataclass
class ArimaModel(ForecastModel):
    order: tuple[int, int, int] = (1, 0, 0)
    seasonal_order: tuple[int, int, int, int] = (0, 1, 1, 4)
    trend: Optional[str] = None
    name: str = "sarima"

    def __post_init__(self) -> None:
        self._results = None

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> None:
        model = SARIMAX(
            y,
            exog=X,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._results = model.fit(disp=False)

    def predict(self, horizon: int, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self._results is None:
            raise RuntimeError("Model must be fit before calling predict().")
        forecast = self._results.forecast(steps=horizon, exog=X)
        return np.asarray(forecast)


@dataclass
class ProphetModel(ForecastModel):
    yearly_seasonality: bool = True
    weekly_seasonality: bool = False
    daily_seasonality: bool = False
    changepoint_prior_scale: float = 0.05
    name: str = "prophet"

    def __post_init__(self) -> None:
        self._model = None

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> None:
        try:
            from prophet import Prophet
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "prophet is required for ProphetModel; install with `pip install prophet`."
            ) from exc

        df = pd.DataFrame({"ds": y.index, "y": y.values})
        model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )
        if X is not None and not X.empty:
            for column in X.columns:
                model.add_regressor(column)
            df = df.join(X.reset_index(drop=True))
        self._model = model.fit(df)

    def predict(self, horizon: int, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model must be fit before calling predict().")
        future = self._model.make_future_dataframe(periods=horizon, freq="Q")
        if X is not None and not X.empty:
            future = future.join(X.reset_index(drop=True))
        forecast = self._model.predict(future).tail(horizon)
        return forecast["yhat"].to_numpy()
