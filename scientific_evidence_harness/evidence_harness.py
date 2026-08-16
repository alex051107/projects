#!/usr/bin/env python3
"""Fail-closed evidence validation for scientific agents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "scientific-evidence-card.v1"
REPORT_VERSION = "scientific-evidence-report.v1"
INFERENCE_ORDER = {
    "descriptive": 0,
    "structural": 1,
    "population": 2,
    "thermodynamic": 3,
    "kinetic": 4,
    "mechanistic": 5,
    "functional": 6,
}
MATURE_STATES = {"locally_reproduced", "science_validated"}


class EvidenceError(RuntimeError):
    """Raised when the input container is malformed."""


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def require_string(card: dict[str, Any], field: str, reasons: list[str]) -> str:
    value = card.get(field)
    if not isinstance(value, str) or not value.strip():
        reasons.append(f"missing_or_invalid:{field}")
        return ""
    return value.strip()


def validate_card(card: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []

    claim_id = require_string(card, "claim_id", reasons)
    source_id = require_string(card, "source_id", reasons)
    require_string(card, "source_type", reasons)
    require_string(card, "source_version", reasons)

    source_hash = card.get("source_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        reasons.append("invalid:source_sha256")

    if card.get("license_status") != "verified":
        reasons.append("unverified:license")
    if card.get("citation_readiness") != "ready":
        reasons.append("unready:citation")

    mapping = card.get("mapping")
    if not isinstance(mapping, dict):
        reasons.append("missing_or_invalid:mapping")
        mapping = {}
    for axis in ("source_axis", "target_axis", "mapping_version"):
        if not isinstance(mapping.get(axis), str) or not mapping.get(axis):
            reasons.append(f"missing_or_invalid:mapping.{axis}")

    coverage = mapping.get("coverage")
    if not isinstance(coverage, (int, float)) or isinstance(coverage, bool):
        reasons.append("missing_or_invalid:mapping.coverage")
    elif not 0 <= float(coverage) <= 1:
        reasons.append("out_of_range:mapping.coverage")
    elif float(coverage) < 0.95:
        reasons.append("insufficient:mapping.coverage")

    unresolved = mapping.get("unresolved_identifiers")
    if not isinstance(unresolved, list):
        reasons.append("missing_or_invalid:mapping.unresolved_identifiers")
    elif unresolved:
        warnings.append(f"unresolved_identifiers:{len(unresolved)}")

    measurement = card.get("measurement")
    if not isinstance(measurement, dict):
        reasons.append("missing_or_invalid:measurement")
        measurement = {}
    if not isinstance(measurement.get("observable"), str):
        reasons.append("missing_or_invalid:measurement.observable")
    if not isinstance(measurement.get("units"), str):
        reasons.append("missing_or_invalid:measurement.units")
    if measurement.get("time_semantics") not in {
        "not_applicable",
        "simulation_time",
        "experimental_exchange",
        "ensemble_without_time",
    }:
        reasons.append("missing_or_invalid:measurement.time_semantics")

    maturity = card.get("maturity_state")
    if maturity not in MATURE_STATES:
        reasons.append("insufficient:maturity_state")

    requested = card.get("requested_inference")
    ceiling = card.get("claim_ceiling")
    if requested not in INFERENCE_ORDER:
        reasons.append("missing_or_invalid:requested_inference")
    if ceiling not in INFERENCE_ORDER:
        reasons.append("missing_or_invalid:claim_ceiling")
    if requested in INFERENCE_ORDER and ceiling in INFERENCE_ORDER:
        if INFERENCE_ORDER[requested] > INFERENCE_ORDER[ceiling]:
            reasons.append(f"claim_exceeds_ceiling:{requested}>{ceiling}")

    blocked = card.get("blocked_claims", [])
    if not isinstance(blocked, list) or not all(isinstance(item, str) for item in blocked):
        reasons.append("missing_or_invalid:blocked_claims")
        blocked = []
    proposed = str(card.get("proposed_claim") or "")
    for phrase in blocked:
        if phrase.lower() in proposed.lower():
            reasons.append(f"explicitly_blocked_claim:{phrase}")

    hard_failures = [reason for reason in reasons if reason]
    if hard_failures:
        verdict = "rejected"
    elif warnings:
        verdict = "review_pending"
    else:
        verdict = "accepted"

    return {
        "claim_id": claim_id,
        "source_id": source_id,
        "verdict": verdict,
        "reasons": hard_failures,
        "warnings": warnings,
        "card_sha256": canonical_hash(card),
    }


def build_report(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceError(f"unsupported schema_version: {document.get('schema_version')!r}")
    cards = document.get("cards")
    if not isinstance(cards, list):
        raise EvidenceError("cards must be an array")
    if not all(isinstance(card, dict) for card in cards):
        raise EvidenceError("each card must be an object")

    results = [validate_card(card) for card in cards]
    counts = {key: sum(item["verdict"] == key for item in results) for key in (
        "accepted",
        "review_pending",
        "rejected",
    )}
    return {
        "schema_version": REPORT_VERSION,
        "input_sha256": canonical_hash(document),
        "counts": counts,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    document = json.loads(args.input.read_text(encoding="utf-8"))
    report = build_report(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

