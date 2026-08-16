from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from careplan_harness import (  # noqa: E402
    AuthorizationError,
    CarePlanHarness,
    FixtureDraftProvider,
    IdempotencyConflict,
    StateConflict,
)


ORDER = json.loads((ROOT / "examples" / "synthetic_order.json").read_text(encoding="utf-8"))


class CarePlanHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.harness = CarePlanHarness(Path(self.temporary.name) / "careplan.sqlite3")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _pending(self) -> dict:
        queued, reused = self.harness.submit(copy.deepcopy(ORDER), "demo-key")
        self.assertFalse(reused)
        self.assertEqual("DRAFT_QUEUED", queued["state"])
        pending = self.harness.draft(queued["plan_id"], FixtureDraftProvider())
        self.assertEqual("REVIEW_PENDING", pending["state"])
        return pending

    def test_valid_provider_stops_at_human_review(self) -> None:
        pending = self._pending()
        events = self.harness.events(pending["plan_id"])
        self.assertEqual("DRAFT_VALIDATED_REVIEW_REQUIRED", events[-1]["event_code"])
        self.assertNotIn("APPROVED", [event["to_state"] for event in events])

    def test_non_synthetic_input_is_hard_stopped_before_drafting(self) -> None:
        order = copy.deepcopy(ORDER)
        order["case_origin"] = "unknown"
        order["patient_token"] = "real-person-token"
        rejected, _ = self.harness.submit(order, "blocked")
        self.assertEqual("REJECTED", rejected["state"])
        reasons = self.harness.events(rejected["plan_id"])[0]["detail"]["hard_stops"]
        self.assertIn("ONLY_SYNTHETIC_CASES_ALLOWED", reasons)
        self.assertIn("INVALID_SYNTHETIC_PATIENT_TOKEN", reasons)

    def test_idempotent_replay_reuses_and_changed_reuse_is_rejected(self) -> None:
        first, _ = self.harness.submit(copy.deepcopy(ORDER), "same-key")
        second, reused = self.harness.submit(copy.deepcopy(ORDER), "same-key")
        self.assertTrue(reused)
        self.assertEqual(first["plan_id"], second["plan_id"])
        changed = copy.deepcopy(ORDER)
        changed["route"] = "changed-route"
        with self.assertRaises(IdempotencyConflict):
            self.harness.submit(changed, "same-key")

    def test_provider_approval_or_malformed_output_fails_closed(self) -> None:
        for index, mode in enumerate(("approval_field", "malformed", "invented_source")):
            queued, _ = self.harness.submit(copy.deepcopy(ORDER), f"invalid-{index}")
            failed = self.harness.draft(queued["plan_id"], FixtureDraftProvider(mode))
            self.assertEqual("FAILED", failed["state"])
            self.assertEqual(
                "DRAFT_SCHEMA_REJECTED",
                self.harness.events(failed["plan_id"])[-1]["event_code"],
            )

    def test_only_pharmacist_can_review_and_stale_review_is_rejected(self) -> None:
        pending = self._pending()
        with self.assertRaises(AuthorizationError):
            self.harness.review(
                pending["plan_id"],
                expected_state_version=pending["state_version"],
                reviewer_id="assistant:demo",
                decision="APPROVE",
            )
        with self.assertRaises(AuthorizationError):
            self.harness.review(
                pending["plan_id"],
                expected_state_version=pending["state_version"],
                reviewer_id="pharmacist:",
                decision="APPROVE",
            )
        approved = self.harness.review(
            pending["plan_id"],
            expected_state_version=pending["state_version"],
            reviewer_id="pharmacist:demo",
            decision="APPROVE",
        )
        self.assertEqual("APPROVED", approved["state"])
        with self.assertRaises(StateConflict):
            self.harness.review(
                pending["plan_id"],
                expected_state_version=pending["state_version"],
                reviewer_id="pharmacist:demo",
                decision="APPROVE",
            )

    def test_public_receipt_excludes_raw_order_and_patient_token(self) -> None:
        pending = self._pending()
        receipt = self.harness.receipt(pending["plan_id"])
        serialized = json.dumps(receipt)
        self.assertNotIn("patient_token", serialized)
        self.assertNotIn(ORDER["patient_token"], serialized)
        self.assertNotIn("order", receipt)
        self.assertTrue(receipt["human_decision_required"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
