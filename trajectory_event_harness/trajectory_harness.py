#!/usr/bin/env python3
"""Replica-aware, provenance-first extraction of trajectory event labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


CONFIG_SCHEMA = "trajectory-event-config.v1"
REPORT_SCHEMA = "trajectory-event-report.v1"
REQUIRED_COLUMNS = {"replica_id", "time_ns", "distance_nm"}


class TrajectoryError(RuntimeError):
    """Input or configuration failed a scientific integrity gate."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise TrajectoryError(f"unsupported config schema: {config.get('schema_version')!r}")
    normalized = dict(config)
    for field in ("dissociation_threshold_nm", "recapture_threshold_nm"):
        value = config.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise TrajectoryError(f"{field} must be finite")
        normalized[field] = float(value)
    if normalized["recapture_threshold_nm"] >= normalized["dissociation_threshold_nm"]:
        raise TrajectoryError("recapture threshold must be below dissociation threshold")
    for field in ("min_dwell_frames", "min_recapture_frames"):
        value = config.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise TrajectoryError(f"{field} must be a positive integer")
    return normalized


def read_rows(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
        raise TrajectoryError(f"CSV requires columns: {sorted(REQUIRED_COLUMNS)}")

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(reader, start=2):
        replica = (row.get("replica_id") or "").strip()
        if not replica:
            raise TrajectoryError(f"line {index}: empty replica_id")
        try:
            time_ns = float(row["time_ns"])
            distance_nm = float(row["distance_nm"])
        except (TypeError, ValueError) as exc:
            raise TrajectoryError(f"line {index}: non-numeric time or distance") from exc
        if not math.isfinite(time_ns) or not math.isfinite(distance_nm):
            raise TrajectoryError(f"line {index}: non-finite time or distance")
        if time_ns < 0 or distance_nm < 0:
            raise TrajectoryError(f"line {index}: negative time or distance")
        rows.append({
            "replica_id": replica,
            "time_ns": time_ns,
            "distance_nm": distance_nm,
        })
    if not rows:
        raise TrajectoryError("CSV contains no frames")
    return rows


def first_sustained_run(
    values: Iterable[float],
    predicate: Any,
    minimum: int,
) -> int | None:
    start: int | None = None
    length = 0
    for index, value in enumerate(values):
        if predicate(value):
            if start is None:
                start = index
            length += 1
            if length >= minimum:
                return start
        else:
            start = None
            length = 0
    return None


def extract_replica(replica_id: str, frames: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    if len(frames) < config["min_dwell_frames"]:
        raise TrajectoryError(f"{replica_id}: insufficient frames")
    # Preserve source order. Sorting here would hide a broken trajectory
    # concatenation or replica-lineage error instead of failing closed.
    frames = list(frames)
    times = [item["time_ns"] for item in frames]
    distances = [item["distance_nm"] for item in frames]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise TrajectoryError(f"{replica_id}: time must be strictly increasing")

    event_index = first_sustained_run(
        distances,
        lambda value: value >= config["dissociation_threshold_nm"],
        config["min_dwell_frames"],
    )
    exposure = times[-1] - times[0]
    base = {
        "replica_id": replica_id,
        "frame_count": len(frames),
        "start_time_ns": times[0],
        "end_time_ns": times[-1],
        "exposure_time_ns": exposure,
        "maximum_distance_nm": max(distances),
    }
    if event_index is None:
        return {
            **base,
            "event_observed": False,
            "event_time_ns": None,
            "event_class": "right_censored",
            "recaptured": False,
        }

    post_event = distances[event_index + config["min_dwell_frames"] :]
    recapture_offset = first_sustained_run(
        post_event,
        lambda value: value <= config["recapture_threshold_nm"],
        config["min_recapture_frames"],
    )
    recaptured = recapture_offset is not None
    recapture_time = None
    if recaptured:
        recapture_index = event_index + config["min_dwell_frames"] + int(recapture_offset)
        recapture_time = times[recapture_index]
    return {
        **base,
        "event_observed": True,
        "event_time_ns": times[event_index],
        "event_class": "observed_recaptured" if recaptured else "observed_no_recapture",
        "recaptured": recaptured,
        "recapture_time_ns": recapture_time,
    }


def build_report(rows: list[dict[str, Any]], config: dict[str, Any], input_sha256: str) -> dict[str, Any]:
    config = validate_config(config)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["replica_id"]].append(dict(row))
    replicas = [
        extract_replica(replica_id, frames, config)
        for replica_id, frames in sorted(grouped.items())
    ]
    counts = {
        label: sum(item["event_class"] == label for item in replicas)
        for label in ("observed_no_recapture", "observed_recaptured", "right_censored")
    }
    return {
        "schema_version": REPORT_SCHEMA,
        "manifest": {
            "input_sha256": input_sha256,
            "config_sha256": canonical_hash(config),
            "replica_count": len(replicas),
            "descriptive_event_labels_only": True,
            "kinetic_claim_authorized": False,
        },
        "counts": counts,
        "replicas": replicas,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    input_bytes = args.input.read_bytes()
    rows = read_rows(args.input)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = build_report(rows, config, sha256_bytes(input_bytes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
