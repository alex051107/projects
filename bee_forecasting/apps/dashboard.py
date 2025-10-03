"""Plotly Dash dashboard for bee colony forecasting results."""
from __future__ import annotations

import argparse
import pathlib

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import yaml
from dash import Input, Output, dcc, html


def _load_artifacts(config_path: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = yaml.safe_load(config_path.read_text())
    metrics_path = pathlib.Path(config["paths"]["metrics_output"])
    forecasts_path = pathlib.Path(config["paths"]["forecasts_output"])

    if not metrics_path.exists() or not forecasts_path.exists():
        raise FileNotFoundError(
            "Metrics or forecast artifacts missing. Run the training pipeline before launching the dashboard."
        )

    metrics = pd.read_csv(metrics_path)
    forecasts = pd.read_parquet(forecasts_path)
    forecasts["timestamp"] = pd.to_datetime(forecasts[config["features"]["time_column"]])
    return metrics, forecasts


def build_app(config_path: pathlib.Path) -> dash.Dash:
    metrics, forecasts = _load_artifacts(config_path)

    models = sorted(metrics["model"].unique())
    horizons = sorted(metrics["horizon"].unique())
    groups = sorted(forecasts["group"].dropna().unique()) if "group" in forecasts.columns else []

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = dbc.Container(
        [
            html.H2("Bee Colony Forecast Dashboard"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Model"),
                            dcc.Dropdown(
                                id="model-dropdown",
                                options=[{"label": m, "value": m} for m in models],
                                value=models[0],
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Label("Horizon (days)"),
                            dcc.Dropdown(
                                id="horizon-dropdown",
                                options=[{"label": h, "value": h} for h in horizons],
                                value=horizons[0],
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Label("State"),
                            dcc.Dropdown(
                                id="group-dropdown",
                                options=[{"label": g, "value": g} for g in groups] if groups else [],
                                value=groups[0] if groups else None,
                                clearable=bool(groups),
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="forecast-graph"), md=8),
                    dbc.Col(dcc.Graph(id="metric-bar"), md=4),
                ]
            ),
        ],
        fluid=True,
    )

    @app.callback(
        Output("forecast-graph", "figure"),
        Output("metric-bar", "figure"),
        Input("model-dropdown", "value"),
        Input("horizon-dropdown", "value"),
        Input("group-dropdown", "value"),
    )
    def update_graphs(model: str, horizon: int, group: str | None):
        filtered_forecasts = forecasts[forecasts["model"] == model]
        filtered_metrics = metrics[(metrics["model"] == model) & (metrics["horizon"] == horizon)]

        if group is not None and "group" in filtered_forecasts.columns:
            filtered_forecasts = filtered_forecasts[filtered_forecasts["group"] == group]
            filtered_metrics = filtered_metrics[filtered_metrics["group"] == group]

        filtered_forecasts = filtered_forecasts[filtered_forecasts["horizon"] == horizon]
        fig_forecast = px.line(
            filtered_forecasts,
            x="timestamp",
            y=["actual", "forecast"],
            labels={"value": "Colonies lost (%)", "timestamp": "Period"},
            title="Forecast vs Actual",
        )
        fig_forecast.update_layout(legend_title_text="Series")

        fig_metric = px.bar(
            filtered_metrics,
            x="metric",
            y="value",
            title="Evaluation metrics",
            labels={"value": "Score", "metric": "Metric"},
        )

        return fig_forecast, fig_metric

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(args.config)
    app.run_server(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
