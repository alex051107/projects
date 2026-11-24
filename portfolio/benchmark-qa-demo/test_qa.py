"""Integration tests from the command-line entry point."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


class QualityChecks(unittest.TestCase):
    def setUp(self):
        self.cases = json.loads((HERE / "cases.json").read_text())
        self.predictions = json.loads((HERE / "predictions.json").read_text())

    def run_cli(self, raw_predictions=None):
        with tempfile.TemporaryDirectory() as directory:
            cases = Path(directory) / "cases.json"
            predictions = Path(directory) / "predictions.json"
            cases.write_text(json.dumps(self.cases))
            predictions.write_text(
                json.dumps(self.predictions) if raw_predictions is None else raw_predictions
            )
            return subprocess.run(
                [sys.executable, str(HERE / "qa.py"), str(cases), str(predictions)],
                capture_output=True, text=True, check=False
            )

    def test_zero_is_not_unknown_and_missing_stays_in_denominator(self):
        del self.predictions["desk-05"]
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {
            "split":"evaluation","correct":1,"total":3,
            "outcomes":{"desk-03":"correct","desk-04":"incorrect_answer",
                        "desk-05":"missing_prediction"}
        })

    def test_wrong_split_prediction_rejected(self):
        self.predictions["desk-01"] = self.cases[0]["expected"]
        result = self.run_cli()
        self.assertEqual(result.returncode, 2)
        self.assertIn("IDs outside the requested split", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_normalized_title_overlap_rejected_before_scoring(self):
        self.cases[0]["request"] = "How many copies of cLOUd   mANUAl are available?"
        self.cases[0]["catalog"] = {"cLOUd   mANUAl":2}
        result = self.run_cli()
        self.assertEqual(result.returncode, 2)
        self.assertIn("queried title overlaps", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_expected_answer_drift_rejected(self):
        self.cases[2]["expected"] = {"status":"unknown","copies":None}
        result = self.run_cli()
        self.assertEqual(result.returncode, 2)
        self.assertIn("expected answer contradicts", result.stderr)

    def test_bool_is_not_a_copy_count(self):
        self.predictions["desk-05"]["copies"] = True
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["outcomes"]["desk-05"], "invalid_prediction")

    def test_duplicate_json_key_rejected(self):
        result = self.run_cli('{"desk-03":{},"desk-03":{}}')
        self.assertEqual(result.returncode, 2)
        self.assertIn("Duplicate JSON key", result.stderr)


if __name__ == "__main__":
    unittest.main()
