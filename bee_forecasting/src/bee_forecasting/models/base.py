"""Base interfaces for forecasting models."""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    predictions: np.ndarray
    horizon: int
    model_name: str
    fold: int


class ForecastModel(ABC):
    name: str

    def clone(self) -> "ForecastModel":
        return copy.deepcopy(self)

    @property
    def uses_exogenous(self) -> bool:
        return True

    @abstractmethod
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> None:
        """Fit model on historical data."""

    @abstractmethod
    def predict(self, horizon: int, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Forecast the next ``horizon`` steps."""
