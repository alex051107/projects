"""Model interfaces shared across forecasting projects."""
from .base import ForecastModel, ForecastResult
from .machine_learning import (
    CatBoostModel,
    ElmanRNNModel,
    LightGBMModel,
    LSTMModel,
    RandomForestModel,
    SupportVectorRegressionModel,
    TemporalConvNetModel,
    XGBoostModel,
)
from .statistical import ArimaModel, ProphetModel, SeasonalNaiveModel

__all__ = [
    "ForecastModel",
    "ForecastResult",
    "CatBoostModel",
    "ElmanRNNModel",
    "LightGBMModel",
    "LSTMModel",
    "RandomForestModel",
    "SupportVectorRegressionModel",
    "TemporalConvNetModel",
    "XGBoostModel",
    "ArimaModel",
    "ProphetModel",
    "SeasonalNaiveModel",
]
