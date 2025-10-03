"""Utilities for loading USGS water level data."""
from __future__ import annotations

import json
import pathlib
from typing import Iterable, Optional

import pandas as pd


INVALID_VALUES = {"", "-999999"}


def load_usgs_timeseries(
    paths: Iterable[pathlib.Path],
    rename_parameters: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Load one or more USGS NWIS JSON files into a tidy dataframe."""

    frames = []
    for path in paths:
        payload = json.loads(path.read_text())
        series_list = payload.get("value", {}).get("timeSeries", [])
        records = []
        for series in series_list:
            parameter = series["variable"]["variableCode"][0]["value"]
            site = series["sourceInfo"]["siteCode"][0]["value"]
            for entry in series.get("values", []):
                for value in entry.get("value", []):
                    raw_value = value.get("value")
                    if raw_value in INVALID_VALUES:
                        numeric = None
                    else:
                        numeric = float(raw_value)
                    records.append(
                        {
                            "timestamp": pd.to_datetime(value["dateTime"]),
                            "parameter": parameter,
                            "value": numeric,
                            "site": site,
                        }
                    )
        if not records:
            continue
        df = pd.DataFrame.from_records(records)
        df = df.pivot_table(
            index=["timestamp", "site"],
            columns="parameter",
            values="value",
        ).reset_index()
        frames.append(df)

    if not frames:
        raise ValueError("No observations found in provided USGS files.")

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values("timestamp").reset_index(drop=True)

    if rename_parameters:
        data = data.rename(columns=rename_parameters)

    return data
