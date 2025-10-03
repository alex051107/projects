"""Data ingestion helpers for Bee Informed colony loss survey."""
from __future__ import annotations

import calendar
import datetime as dt
import pathlib
import re
from typing import Iterable, Optional, Tuple

import pandas as pd

DATA_COLUMNS = [
    "state",
    "colonies",
    "maximum_colonies",
    "lost_colonies",
    "lost_percent",
    "added_colonies",
    "renovated_colonies",
    "renovated_percent",
]

TABLE_TITLE_PATTERN = re.compile(
    r":\s+(?P<start_month>[A-Za-z]+)\s+(?P<start_day>\d{1,2}),\s+(?P<start_year>\d{4})\s+and\s+"
    r"(?P<period>[A-Za-z]+-[A-Za-z]+\s+\d{4})"
)


def _parse_period(title: str) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    match = TABLE_TITLE_PATTERN.search(title)
    if not match:
        raise ValueError(f"Unable to parse period from title: {title}")

    start_month = match.group("start_month")
    start_day = int(match.group("start_day"))
    start_year = int(match.group("start_year"))
    period_text = match.group("period")  # e.g. "April-June 2016"

    period_months, period_year = period_text.rsplit(" ", 1)
    start_period_month, end_period_month = period_months.split("-")
    period_year = int(period_year)

    start_date = pd.Timestamp(dt.datetime.strptime(
        f"{start_month} {start_day} {start_year}", "%B %d %Y"
    ))
    start_window = pd.Timestamp(dt.datetime.strptime(
        f"{start_period_month} 1 {period_year}", "%B %d %Y"
    ))
    end_month_number = dt.datetime.strptime(end_period_month, "%B").month
    last_day = calendar.monthrange(period_year, end_month_number)[1]
    end_window = pd.Timestamp(
        dt.datetime(period_year, end_month_number, last_day)
    )

    return start_date, start_window, end_window


def _load_single_table(path: pathlib.Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    title = raw.loc[raw[1] == "t", 2].iloc[-1]
    snapshot_date, period_start, period_end = _parse_period(title)

    data_rows = raw[raw[1] == "d"].iloc[:, 2:].copy()
    data_rows.columns = DATA_COLUMNS
    data_rows["period_start"] = period_start
    data_rows["period_end"] = period_end
    data_rows["snapshot_date"] = snapshot_date
    data_rows["table_id"] = path.stem
    return data_rows


def load_colony_panel(
    source_dir: pathlib.Path,
    include_tables: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load and tidy Bee Informed colony loss survey tables.

    Parameters
    ----------
    source_dir:
        Directory containing raw CSV files (e.g. output of `download_bee_data`).
    include_tables:
        Optional whitelist of file stems (without extension) to include.

    Returns
    -------
    pandas.DataFrame
        Tidy panel with columns `state`, `period_start`, `period_end`, `snapshot_date`,
        and colony metrics.
    """

    files = sorted(source_dir.glob("hcny_*.csv"))
    if include_tables is not None:
        include_tables = {stem.lower() for stem in include_tables}
        files = [f for f in files if f.stem.lower() in include_tables]

    frames = [_load_single_table(path) for path in files]
    data = pd.concat(frames, ignore_index=True)

    # Normalize state names and numeric columns
    data["state"] = data["state"].str.strip()
    numeric_cols = [c for c in DATA_COLUMNS if c != "state"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["period_start"] = pd.to_datetime(data["period_start"])
    data["period_end"] = pd.to_datetime(data["period_end"])
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"])

    return data
