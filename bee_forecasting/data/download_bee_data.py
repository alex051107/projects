"""Utilities to download Bee Informed colony loss survey tables."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
from typing import Iterable

import requests

BASE_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2022/2022-01-11"
)
DEFAULT_TABLES = [
    "bee-17/hcny_all_tables.csv",
    "bee-18/hcny_all_tables.csv",
    "bee-19/hcny_all_tables.csv",
    "bee-20/hcny_all_tables.csv",
    "bee-21/hcny_all_tables.csv",
]


def _download_file(url: str, destination: pathlib.Path) -> None:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)



def download_tables(output_dir: pathlib.Path, tables: Iterable[str] = DEFAULT_TABLES) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for table in tables:
        url = f"{BASE_URL}/{table}"
        target = output_dir / table.split("/")[-1]
        print(f"Downloading {url} -> {target}")
        _download_file(url, target)
        manifest.append(
            {
                "source": url,
                "destination": str(target.relative_to(output_dir)),
                "downloaded_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
        )

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "README.txt").write_text(
        "Bee Informed colony loss survey files sourced from the tidytuesday project\n"
        "https://github.com/rfordatascience/tidytuesday/tree/main/data/2022/2022-01-11\n",
        encoding="utf-8",
    )



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "raw",
        help="Directory where CSV files will be stored.",
    )
    parser.add_argument(
        "--tables",
        nargs="*",
        default=DEFAULT_TABLES,
        help="Subset of table paths within the tidytuesday bee release to download.",
    )
    return parser



def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    download_tables(args.output_dir, args.tables)


if __name__ == "__main__":
    main()
