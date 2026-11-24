"""Offline quality checks for a fictional library benchmark."""
import argparse
import json
from pathlib import Path
import re

QUESTION = re.compile(r"How many copies of (.+) are available\?")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def valid_answer(answer):
    if not isinstance(answer, dict) or set(answer) != {"status", "copies"}:
        return False
    status, copies = answer["status"], answer["copies"]
    return (status == "unknown" and copies is None) or (
        status == "known" and type(copies) is int and copies >= 0
    )


def validate_cases(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases must be a nonempty array")
    seen_ids, seen_titles = set(), {}
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"id", "split", "request", "catalog", "expected"}:
            raise ValueError("Invalid case fields")
        key, split, request, catalog, expected = (
            case[name] for name in ("id", "split", "request", "catalog", "expected")
        )
        if not isinstance(key, str) or not key.strip() or key in seen_ids:
            raise ValueError(f"Missing or duplicate case ID: {key}")
        seen_ids.add(key)
        if split not in ("development", "evaluation"):
            raise ValueError(f"{key}: invalid split")
        match = QUESTION.fullmatch(request) if isinstance(request, str) else None
        if not match or not match[1].strip():
            raise ValueError(f"{key}: invalid request")
        title = match[1]
        normalized_title = " ".join(title.casefold().split())
        if normalized_title in seen_titles and seen_titles[normalized_title] != split:
            raise ValueError(f"{key}: queried title overlaps development and evaluation")
        seen_titles[normalized_title] = split
        if not isinstance(catalog, dict) or any(
            not name.strip() or type(count) is not int or count < 0
            for name, count in catalog.items()
        ):
            raise ValueError(f"{key}: invalid catalog")
        derived = {"status": "known", "copies": catalog[title]} if title in catalog else {
            "status": "unknown", "copies": None
        }
        if not valid_answer(expected) or expected != derived:
            raise ValueError(f"{key}: expected answer contradicts the catalog or schema")


def evaluate(cases, predictions, split):
    validate_cases(cases)
    selected = [case for case in cases if case["split"] == split]
    if not selected:
        raise ValueError(f"No cases for split: {split}")
    if not isinstance(predictions, dict):
        raise ValueError("Predictions must be an object keyed by case ID")
    unexpected = sorted(set(predictions) - {case["id"] for case in selected})
    if unexpected:
        raise ValueError(f"Prediction IDs outside the requested split: {', '.join(unexpected)}")
    outcomes = {}
    for case in selected:
        key = case["id"]
        if key not in predictions:
            outcomes[key] = "missing_prediction"
        elif not valid_answer(predictions[key]):
            outcomes[key] = "invalid_prediction"
        elif predictions[key] != case["expected"]:
            outcomes[key] = "incorrect_answer"
        else:
            outcomes[key] = "correct"
    return {
        "split": split,
        "correct": sum(value == "correct" for value in outcomes.values()),
        "total": len(outcomes),
        "outcomes": outcomes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--split", choices=("development", "evaluation"), default="evaluation")
    args = parser.parse_args()
    try:
        report = evaluate(load(args.cases), load(args.predictions), args.split)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Input error: {error}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
