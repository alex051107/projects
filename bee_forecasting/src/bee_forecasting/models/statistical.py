"""Compatibility wrapper exposing shared statistical forecasters."""
from forecasting_core.models.statistical import ArimaModel, ProphetModel, SeasonalNaiveModel

__all__ = ["ArimaModel", "ProphetModel", "SeasonalNaiveModel"]
