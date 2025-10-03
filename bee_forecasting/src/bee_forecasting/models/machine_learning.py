"""Machine learning and deep learning forecasting models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .base import ForecastModel


@dataclass
class LightGBMModel(ForecastModel):
    params: dict = field(default_factory=lambda: {"n_estimators": 500, "learning_rate": 0.05})
    name: str = "lightgbm"

    def __post_init__(self) -> None:
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "lightgbm is required for LightGBMModel; install with `pip install lightgbm`."
            ) from exc
        self._model = lgb.LGBMRegressor(**self.params)

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> None:
        if X is None:
            raise ValueError("LightGBMModel requires exogenous regressors.")
        self._model.fit(X, y)

    def predict(self, horizon: int, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if X is None:
            raise ValueError("LightGBMModel requires exogenous regressors for prediction.")
        if len(X) < horizon:
            raise ValueError("Not enough exogenous rows provided for requested horizon.")
        return self._model.predict(X.iloc[:horizon])


@dataclass
class XGBoostModel(ForecastModel):
    params: dict = field(
        default_factory=lambda: {
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
        }
    )
    name: str = "xgboost"

    def __post_init__(self) -> None:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "xgboost is required for XGBoostModel; install with `pip install xgboost`."
            ) from exc
        self._model = XGBRegressor(**self.params)

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> None:
        if X is None:
            raise ValueError("XGBoostModel requires exogenous regressors.")
        self._model.fit(X, y, verbose=False)

    def predict(self, horizon: int, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if X is None:
            raise ValueError("XGBoostModel requires exogenous regressors for prediction.")
        if len(X) < horizon:
            raise ValueError("Not enough exogenous rows provided for requested horizon.")
        return self._model.predict(X.iloc[:horizon])


@dataclass
class RandomForestModel(ForecastModel):
    params: dict = field(
        default_factory=lambda: {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        }
    )
    name: str = "random_forest"

    def __post_init__(self) -> None:
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "scikit-learn is required for RandomForestModel; install with `pip install scikit-learn`."
            ) from exc
        self._model = RandomForestRegressor(**self.params)

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> None:
        if X is None:
            raise ValueError("RandomForestModel requires exogenous regressors.")
        self._model.fit(X, y)

    def predict(self, horizon: int, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if X is None:
            raise ValueError("RandomForestModel requires exogenous regressors for prediction.")
        if len(X) < horizon:
            raise ValueError("Not enough exogenous rows provided for requested horizon.")
        return self._model.predict(X.iloc[:horizon])


@dataclass
class CatBoostModel(ForecastModel):
    params: dict = field(
        default_factory=lambda: {
            "iterations": 800,
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "RMSE",
        }
    )
    name: str = "catboost"

    def __post_init__(self) -> None:
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "catboost is required for CatBoostModel; install with `pip install catboost`."
            ) from exc
        self._model = CatBoostRegressor(verbose=False, **self.params)

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> None:
        if X is None:
            raise ValueError("CatBoostModel requires exogenous regressors.")
        self._model.fit(X, y)

    def predict(self, horizon: int, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if X is None:
            raise ValueError("CatBoostModel requires exogenous regressors for prediction.")
        if len(X) < horizon:
            raise ValueError("Not enough exogenous rows provided for requested horizon.")
        return self._model.predict(X.iloc[:horizon])


@dataclass
class SupportVectorRegressionModel(ForecastModel):
    params: dict = field(
        default_factory=lambda: {
            "C": 10.0,
            "epsilon": 0.1,
            "kernel": "rbf",
            "gamma": "scale",
        }
    )
    name: str = "svr"

    def __post_init__(self) -> None:
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVR
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "scikit-learn is required for SupportVectorRegressionModel; install with `pip install scikit-learn`."
            ) from exc
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", SVR(**self.params)),
            ]
        )

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> None:
        if X is None:
            raise ValueError("SupportVectorRegressionModel requires exogenous regressors.")
        self._pipeline.fit(X, y)

    def predict(self, horizon: int, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if X is None:
            raise ValueError("SupportVectorRegressionModel requires exogenous regressors for prediction.")
        if len(X) < horizon:
            raise ValueError("Not enough exogenous rows provided for requested horizon.")
        return self._pipeline.predict(X.iloc[:horizon])


class _TorchSequenceModel(ForecastModel):
    def __init__(self, lookback: int = 8, epochs: int = 200, lr: float = 1e-3, name: str = "torch_model") -> None:
        import torch

        self.lookback = lookback
        self.epochs = epochs
        self.lr = lr
        self.name = name
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[torch.nn.Module] = None
        self._history: Optional[pd.Series] = None

    @property
    def uses_exogenous(self) -> bool:
        return False

    def _prepare_sequences(self, series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        data = series.to_numpy(dtype=float)
        X, y = [], []
        for idx in range(self.lookback, len(data)):
            window = data[idx - self.lookback : idx]
            target = data[idx]
            if np.isnan(window).any() or np.isnan(target):
                continue
            X.append(window)
            y.append(target)
        if not X:
            raise ValueError("Insufficient history to build training sequences.")
        return np.stack(X), np.array(y)

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> None:
        import torch
        import torch.nn as nn

        features, targets = self._prepare_sequences(y.dropna())
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32).unsqueeze(-1),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        self._model = self._build_model()
        self._model.to(self._device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        self._history = y.astype(float)
        self._history.index = pd.to_datetime(self._history.index)

        self._model.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self._device)
                batch_y = batch_y.to(self._device)
                optimizer.zero_grad()
                output = self._model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, horizon: int, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        import torch

        if self._model is None or self._history is None:
            raise RuntimeError("Model must be fit before calling predict().")

        self._model.eval()
        history = self._history.dropna().to_numpy(dtype=float)
        preds = []
        window = history[-self.lookback :].copy()
        for _ in range(horizon):
            tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self._device)
            with torch.no_grad():
                value = self._model(tensor).cpu().numpy().ravel()[0]
            preds.append(float(value))
            window = np.roll(window, -1)
            window[-1] = value
        return np.array(preds)

    def _build_model(self):  # pragma: no cover - implemented in subclasses
        raise NotImplementedError


class ElmanRNNModel(_TorchSequenceModel):
    def __init__(self, hidden_size: int = 32, **kwargs) -> None:
        self.hidden_size = hidden_size
        super().__init__(name="elman_rnn", **kwargs)

    def _build_model(self):
        import torch
        import torch.nn as nn

        class SimpleRNNRegressor(nn.Module):
            def __init__(self, input_size: int, hidden_size: int) -> None:
                super().__init__()
                self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, nonlinearity="tanh", batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.unsqueeze(-1)
                out, _ = self.rnn(x)
                out = out[:, -1, :]
                return self.fc(out)

        return SimpleRNNRegressor(self.lookback, self.hidden_size)


class LSTMModel(_TorchSequenceModel):
    def __init__(self, hidden_size: int = 64, **kwargs) -> None:
        self.hidden_size = hidden_size
        super().__init__(name="lstm", **kwargs)

    def _build_model(self):
        import torch
        import torch.nn as nn

        class LSTMRegressor(nn.Module):
            def __init__(self, input_size: int, hidden_size: int) -> None:
                super().__init__()
                self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.unsqueeze(-1)
                out, _ = self.lstm(x)
                out = out[:, -1, :]
                return self.fc(out)

        return LSTMRegressor(self.lookback, self.hidden_size)


class TemporalConvNetModel(_TorchSequenceModel):
    def __init__(self, channels: tuple[int, ...] = (32, 32), kernel_size: int = 3, **kwargs) -> None:
        self.channels = channels
        self.kernel_size = kernel_size
        super().__init__(name="tcn", **kwargs)

    def _build_model(self):
        import torch
        import torch.nn as nn

        class TCNRegressor(nn.Module):
            def __init__(self, channels: tuple[int, ...], kernel_size: int) -> None:
                super().__init__()
                layers = []
                in_channels = 1
                for layer_idx, out_channels in enumerate(channels):
                    dilation = 2 ** layer_idx
                    padding = (kernel_size - 1) * dilation
                    layers.append(
                        nn.Conv1d(
                            in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding=padding,
                        )
                    )
                    layers.append(nn.ReLU())
                    in_channels = out_channels
                layers.append(nn.Conv1d(in_channels, 1, kernel_size=1))
                self.network = nn.Sequential(*layers)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.unsqueeze(1)
                out = self.network(x)
                return out[:, :, -1]

        return TCNRegressor(self.channels, self.kernel_size)
