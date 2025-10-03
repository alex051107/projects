"""Download hourly weather covariates from Meteostat."""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib

import pandas as pd
from meteostat import Hourly, Stations


def download_hourly_weather(
    station_id: str,
    start: dt.datetime,
    end: dt.datetime,
    output_dir: pathlib.Path,
    include_station_metadata: bool = True,
) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / f"weather_{station_id}_{start:%Y%m%d}_{end:%Y%m%d}.parquet"

    hourly = Hourly(station_id, start, end)
    df = hourly.fetch()
    if df.empty:
        raise RuntimeError(
            f"Meteostat returned no records for station={station_id} between {start} and {end}"
        )

    df.reset_index().to_parquet(dataset_path, index=False)

    if include_station_metadata:
        stations = Stations().id(station_id)
        metadata = stations.fetch(1)
        if not metadata.empty:
            metadata_path = output_dir / f"weather_station_{station_id}.csv"
            metadata.to_csv(metadata_path, index=False)

    return dataset_path


def _parse_date(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station-id", required=True, help="Meteostat station identifier (e.g. 725300).")
    parser.add_argument("--start", type=_parse_date, required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=_parse_date, required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "raw",
        help="Directory where the parquet file will be stored.",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip saving the Meteostat station metadata CSV.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_path = download_hourly_weather(
        station_id=args.station_id,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        include_station_metadata=not args.no_metadata,
    )
    print(f"Weather data saved to {dataset_path}")


if __name__ == "__main__":
    main()
