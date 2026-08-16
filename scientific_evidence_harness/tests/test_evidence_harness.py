import copy
import json
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evidence_harness import EvidenceError, build_report, validate_card  # noqa: E402


EXAMPLE = json.loads((ROOT / "examples" / "evidence_cards.json").read_text())


class EvidenceHarnessTests(unittest.TestCase):
    def test_accepts_reproduced_structural_claim(self) -> None:
        result = validate_card(copy.deepcopy(EXAMPLE["cards"][0]))
        self.assertEqual("accepted", result["verdict"])
        self.assertEqual([], result["reasons"])

    def test_rejects_claim_above_ceiling(self) -> None:
        result = validate_card(copy.deepcopy(EXAMPLE["cards"][1]))
        self.assertEqual("rejected", result["verdict"])
        self.assertIn("claim_exceeds_ceiling:kinetic>structural", result["reasons"])

    def test_mapping_gap_fails_closed(self) -> None:
        card = copy.deepcopy(EXAMPLE["cards"][0])
        card["mapping"]["coverage"] = 0.949
        result = validate_card(card)
        self.assertEqual("rejected", result["verdict"])
        self.assertIn("insufficient:mapping.coverage", result["reasons"])

    def test_unresolved_identifiers_require_review(self) -> None:
        card = copy.deepcopy(EXAMPLE["cards"][0])
        card["mapping"]["unresolved_identifiers"] = ["42A"]
        result = validate_card(card)
        self.assertEqual("review_pending", result["verdict"])

    def test_report_is_deterministic_and_input_is_not_mutated(self) -> None:
        document = copy.deepcopy(EXAMPLE)
        before = json.dumps(document, sort_keys=True)
        first = build_report(document)
        second = build_report(document)
        self.assertEqual(first, second)
        self.assertEqual(before, json.dumps(document, sort_keys=True))
        self.assertEqual({"accepted": 1, "review_pending": 0, "rejected": 1}, first["counts"])

    def test_unknown_schema_fails_closed(self) -> None:
        document = copy.deepcopy(EXAMPLE)
        document["schema_version"] = "future.v9"
        with self.assertRaises(EvidenceError):
            build_report(document)


if __name__ == "__main__":
    unittest.main(verbosity=2)

