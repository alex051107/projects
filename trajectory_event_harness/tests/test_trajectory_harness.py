import copy
import json
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trajectory_harness import (  # noqa: E402
    TrajectoryError,
    build_report,
    extract_replica,
    read_rows,
    sha256_bytes,
    validate_config,
)


CONFIG = json.loads((ROOT / "examples" / "config.json").read_text())
CSV_PATH = ROOT / "examples" / "synthetic_trajectories.csv"


def frames(distances):
    return [
        {"replica_id": "r", "time_ns": float(index), "distance_nm": value}
        for index, value in enumerate(distances)
    ]


class TrajectoryHarnessTests(unittest.TestCase):
    def test_sustained_crossing_is_observed(self) -> None:
        result = extract_replica("r", frames([0.4, 1.2, 1.3, 1.4, 1.5]), CONFIG)
        self.assertTrue(result["event_observed"])
        self.assertEqual(1.0, result["event_time_ns"])
        self.assertEqual("observed_no_recapture", result["event_class"])

    def test_transient_crossing_is_censored(self) -> None:
        result = extract_replica("r", frames([0.4, 1.3, 0.6, 0.7, 0.8]), CONFIG)
        self.assertFalse(result["event_observed"])
        self.assertEqual("right_censored", result["event_class"])

    def test_recapture_is_preserved(self) -> None:
        result = extract_replica("r", frames([0.4, 1.2, 1.3, 1.4, 0.7, 0.6]), CONFIG)
        self.assertTrue(result["event_observed"])
        self.assertTrue(result["recaptured"])
        self.assertEqual("observed_recaptured", result["event_class"])
        self.assertEqual(4.0, result["recapture_time_ns"])

    def test_duplicate_time_fails_closed(self) -> None:
        bad = frames([0.4, 0.5, 0.6])
        bad[2]["time_ns"] = 1.0
        with self.assertRaises(TrajectoryError):
            extract_replica("r", bad, CONFIG)

    def test_out_of_order_time_fails_closed(self) -> None:
        bad = frames([0.4, 0.5, 0.6])
        bad[1]["time_ns"] = 2.0
        bad[2]["time_ns"] = 1.0
        with self.assertRaises(TrajectoryError):
            extract_replica("r", bad, CONFIG)

    def test_invalid_threshold_order_fails_closed(self) -> None:
        bad = copy.deepcopy(CONFIG)
        bad["recapture_threshold_nm"] = 1.3
        with self.assertRaises(TrajectoryError):
            validate_config(bad)

    def test_report_keeps_replica_and_censoring_counts(self) -> None:
        rows = read_rows(CSV_PATH)
        before = copy.deepcopy(rows)
        input_hash = sha256_bytes(CSV_PATH.read_bytes())
        report = build_report(rows, CONFIG, input_hash)
        self.assertEqual(before, rows)
        self.assertEqual(3, report["manifest"]["replica_count"])
        self.assertFalse(report["manifest"]["kinetic_claim_authorized"])
        self.assertEqual({
            "observed_no_recapture": 1,
            "observed_recaptured": 1,
            "right_censored": 1,
        }, report["counts"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
