"""Reporting helpers for the screening pipeline."""
from __future__ import annotations

import pathlib
from typing import Iterable

import pandas as pd
import plotly.express as px

from .rescoring import RescoreResult


def build_report(
    filtered_hits: pd.DataFrame,
    rescoring_metrics: Iterable[RescoreResult],
    output_dir: pathlib.Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    hits_path = output_dir / "top20_hits.csv"
    metrics_path = output_dir / "rescoring_metrics.json"
    plot_path = output_dir / "consensus_distribution.html"

    filtered_hits.to_csv(hits_path, index=False)

    metrics_df = pd.DataFrame([metric.__dict__ for metric in rescoring_metrics])
    metrics_df.to_json(metrics_path, orient="records", indent=2)

    fig = px.histogram(
        filtered_hits,
        x="consensus_score",
        nbins=20,
        title="Consensus score distribution",
        labels={"consensus_score": "Consensus score"},
    )
    fig.write_html(plot_path)
