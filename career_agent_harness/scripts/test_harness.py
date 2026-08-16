#!/usr/bin/env python3
"""Regression tests for the career agent harness."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().with_name("harness.py")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plan(urgent_items: list[dict[str, object]] | None = None) -> dict[str, object]:
    plan: dict[str, object] = {
        "planVersion": "test.v1",
        "timezone": "Asia/Shanghai",
        "weeks": [
            {
                "id": "W0",
                "start": "2026-07-20",
                "end": "2026-07-26",
                "phase": "test",
                "headline": "Test week",
                "artifact": "Public artifact",
                "market": "Context only",
                "practice": "Practice one current blocker",
                "gate": "Evidence is required",
                "gateTarget": "S3",
            }
        ],
    }
    if urgent_items is not None:
        plan["urgentItems"] = urgent_items
    return plan


class RefreshTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "private-state"
        self.run_refresh("--init", "--date", "2026-07-23")
        self.plan = Path(self.temp.name) / "test-plan.json"
        write_json(self.plan, test_plan())

    def tearDown(self) -> None:
        self.temp.cleanup()

    def run_refresh(self, *extra: str, plan: Path | None = None) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(SCRIPT), "--root", str(self.root)]
        command.extend(extra)
        if plan is not None:
            command.extend(["--plan", str(plan)])
        return subprocess.run(command, check=True, capture_output=True, text=True)

    def load_snapshot(self) -> dict[str, object]:
        return json.loads((self.root / "exports" / "site-snapshot.json").read_text(encoding="utf-8"))

    def test_init_is_non_destructive_and_export_is_sanitized(self) -> None:
        private_jobs = [
            {
                "id": "secret-job-id",
                "company": "SecretCo",
                "title": "Secret Role",
                "sourceUrl": "https://secret.example/job",
                "status": "shortlisted",
                "fit": "qualified",
                "eligibility": "verified",
                "acceptable": True,
            }
        ]
        write_json(self.root / "jobs.json", private_jobs)
        before = digest(self.root / "jobs.json")
        result = self.run_refresh("--init", "--date", "2026-07-23", plan=self.plan)
        self.assertIn("nothing overwritten", result.stdout.lower())
        self.assertEqual(before, digest(self.root / "jobs.json"))
        self.assertTrue((self.root / "daily" / "2026-07-23.md").exists())
        snapshot_text = (self.root / "exports" / "site-snapshot.json").read_text(encoding="utf-8")
        self.assertNotIn("SecretCo", snapshot_text)
        self.assertNotIn("Secret Role", snapshot_text)
        self.assertNotIn("secret-job-id", snapshot_text)
        self.assertNotIn("https://secret.example/job", snapshot_text)
        snapshot = json.loads(snapshot_text)
        self.assertEqual("ai-internship-site-snapshot.v1", snapshot["schemaVersion"])
        self.assertTrue(snapshot["privacy"]["sanitized"])
        self.assertEqual("device-local-records", snapshot["privacy"]["scope"])

    def test_priority_order_private_detail_and_no_state_mutation(self) -> None:
        config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        config["dailyCapacityMinutes"] = 600
        config["readinessEvidence"]["S0"] = {
            "status": "verified",
            "verifiedAt": "2026-07-22",
            "evidence": ["/private/evidence/resume-review.md"],
        }
        write_json(self.root / "config.json", config)
        write_json(
            self.root / "jobs.json",
            [
                {
                    "id": "overdue-job",
                    "company": "OldSecretCo",
                    "title": "Old Secret Role",
                    "status": "draft",
                    "fit": "needs-review",
                    "eligibility": "unknown",
                    "acceptable": False,
                    "nextAction": "Resolve private eligibility question",
                    "nextActionDue": "2026-07-22",
                },
                {
                    "id": "qualified-job",
                    "company": "QualifiedSecretCo",
                    "title": "Qualified Secret Role",
                    "status": "shortlisted",
                    "fit": "qualified",
                    "eligibility": "verified",
                    "acceptable": True,
                    "priority": 1,
                    "deadline": "2026-07-30",
                    "sourceUrl": "https://official.example/qualified-role",
                    "lastVerifiedAt": "2026-07-23",
                },
            ],
        )
        write_json(
            self.root / "contacts.json",
            [
                {
                    "id": "secret-contact",
                    "name": "Alice Private",
                    "company": "ContactSecretCo",
                    "status": "warm",
                    "nextAction": "Review promised note",
                    "nextActionDue": "2026-07-25",
                    "followUpCount": 0,
                }
            ],
        )
        write_json(
            self.root / "events.json",
            [
                {
                    "id": "secret-event",
                    "jobId": "qualified-job",
                    "type": "oa",
                    "title": "Secret OA",
                    "dueAt": "2026-07-24T12:00:00+08:00",
                    "status": "open",
                    "estimatedMinutes": 90,
                }
            ],
        )
        write_json(
            self.root / "activity.json",
            [{"date": "2026-07-21", "type": "research", "minutes": 30, "recordId": "secret-activity"}],
        )
        state_paths = [self.root / name for name in ("jobs.json", "contacts.json", "events.json", "activity.json")]
        before = {path.name: digest(path) for path in state_paths}
        self.run_refresh("--date", "2026-07-23", plan=self.plan)
        after = {path.name: digest(path) for path in state_paths}
        self.assertEqual(before, after)

        snapshot = self.load_snapshot()
        self.assertEqual(
            [
                "event",
                "overdue-next-action",
                "weekly-artifact",
                "market-application",
                "oa-practice",
                "connection",
            ],
            snapshot["today"]["categories"],
        )
        self.assertEqual("S0", snapshot["readiness"]["verifiedLevel"])
        self.assertEqual("S3", snapshot["readiness"]["gateTarget"])
        snapshot_text = json.dumps(snapshot, ensure_ascii=False)
        for secret in (
            "OldSecretCo",
            "QualifiedSecretCo",
            "Alice Private",
            "ContactSecretCo",
            "Secret OA",
            "secret-activity",
        ):
            self.assertNotIn(secret, snapshot_text)
        daily = (self.root / "daily" / "2026-07-23.md").read_text(encoding="utf-8")
        self.assertIn("QualifiedSecretCo", daily)
        self.assertIn("Alice Private", daily)
        self.assertIn("脚本没有投递、发送、预约或修改任何外部系统", daily)

    def test_date_and_gate_target_never_promote_readiness(self) -> None:
        config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        config["readinessEvidence"]["S0"] = {
            "status": "verified",
            "verifiedAt": "2026-07-20",
            "evidence": ["resume-v0-reviewed"],
        }
        config["readinessEvidence"]["S1"] = {
            "status": "verified",
            "verifiedAt": "2026-07-21",
            "evidence": [],
        }
        write_json(self.root / "config.json", config)
        self.run_refresh("--date", "2026-07-26", plan=self.plan)
        snapshot = self.load_snapshot()
        self.assertEqual("S0", snapshot["readiness"]["verifiedLevel"])
        self.assertEqual("S3", snapshot["readiness"]["gateTarget"])
        self.assertEqual(0, snapshot["readiness"]["verifiedEvidenceCounts"]["S1"])

    def test_stale_live_market_item_requires_reverification(self) -> None:
        stale_plan = Path(self.temp.name) / "stale-plan.json"
        write_json(
            stale_plan,
            test_plan(
                [
                    {
                        "id": "stale-role",
                        "label": "Previously live role",
                        "sourceUrl": "https://official.example/role",
                        "verifiedAt": "2026-07-20",
                        "anticipatedClose": "2026-08-01",
                        "status": "live",
                        "reverifyBy": "2026-07-22",
                    }
                ]
            ),
        )
        self.run_refresh("--date", "2026-07-23", plan=stale_plan)
        snapshot = self.load_snapshot()
        self.assertIn("market-reverify", snapshot["today"]["categories"])
        self.assertNotIn("verified-market-deadline", snapshot["today"]["categories"])
        snapshot_text = json.dumps(snapshot, ensure_ascii=False)
        self.assertNotIn("Previously live role", snapshot_text)
        self.assertNotIn("https://official.example/role", snapshot_text)
        daily = (self.root / "daily" / "2026-07-23.md").read_text(encoding="utf-8")
        self.assertIn("先核验，不直接申请", daily)

    def test_verify_now_is_urgent_verification_then_expires(self) -> None:
        verify_plan = Path(self.temp.name) / "verify-plan.json"
        item = {
            "id": "verify-role",
            "title": "Official role to recheck",
            "sourceUrl": "https://official.example/recheck",
            "verifiedAt": "2026-07-23",
            "anticipatedClose": "2026-07-24",
            "sourceTimezone": "America/Los_Angeles",
            "status": "verify_now",
            "requiresReverification": True,
            "reverifyBy": "2026-07-24",
            "staleAfter": "2026-07-24T23:59:59-07:00",
        }
        write_json(verify_plan, test_plan([item]))
        self.run_refresh("--date", "2026-07-23", plan=verify_plan)
        snapshot = self.load_snapshot()
        self.assertEqual("market-reverify", snapshot["today"]["categories"][0])
        self.assertNotIn("verified-market-deadline", snapshot["today"]["categories"])
        daily = (self.root / "daily" / "2026-07-23.md").read_text(encoding="utf-8")
        self.assertIn("先核验，不直接申请", daily)

        self.run_refresh("--date", "2026-07-26", plan=verify_plan)
        expired_snapshot = self.load_snapshot()
        self.assertNotIn("market-reverify", expired_snapshot["today"]["categories"])
        self.assertNotIn("verified-market-deadline", expired_snapshot["today"]["categories"])

    def test_noncontiguous_readiness_evidence_does_not_skip_levels(self) -> None:
        config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        config["readinessEvidence"]["S0"] = {
            "status": "unverified",
            "verifiedAt": None,
            "evidence": [],
        }
        config["readinessEvidence"]["S2"] = {
            "status": "verified",
            "verifiedAt": "2026-07-23",
            "evidence": ["premature-s2-claim"],
        }
        write_json(self.root / "config.json", config)
        self.run_refresh("--date", "2026-07-23", plan=self.plan)
        snapshot = self.load_snapshot()
        self.assertIsNone(snapshot["readiness"]["verifiedLevel"])
        self.assertEqual(1, snapshot["readiness"]["verifiedEvidenceCounts"]["S2"])

    def test_lower_priority_work_never_leapfrogs_capacity(self) -> None:
        config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        config["dailyCapacityMinutes"] = 90
        write_json(self.root / "config.json", config)
        self.run_refresh("--date", "2026-07-23", plan=self.plan)
        snapshot = self.load_snapshot()
        self.assertEqual(
            ["weekly-artifact", "market-monitoring"],
            snapshot["today"]["categories"],
        )
        self.assertNotIn("oa-practice", snapshot["today"]["categories"])
        self.assertNotIn("connection", snapshot["today"]["categories"])
        daily = (self.root / "daily" / "2026-07-23.md").read_text(encoding="utf-8")
        self.assertIn("P5 `oa-practice`", daily)
        self.assertIn("P6 `connection`", daily)

    def test_stale_qualified_job_returns_to_monitoring(self) -> None:
        write_json(
            self.root / "jobs.json",
            [
                {
                    "id": "stale-qualified",
                    "company": "Private stale company",
                    "title": "Private stale role",
                    "sourceUrl": "https://official.example/stale",
                    "status": "shortlisted",
                    "fit": "qualified",
                    "eligibility": "verified",
                    "acceptable": True,
                    "deadline": "2026-08-30",
                    "lastVerifiedAt": "2026-07-01",
                }
            ],
        )
        self.run_refresh("--date", "2026-07-23", plan=self.plan)
        snapshot = self.load_snapshot()
        self.assertIn("market-monitoring", snapshot["today"]["categories"])
        self.assertNotIn("market-application", snapshot["today"]["categories"])
        self.assertEqual(0, snapshot["metrics"]["qualifiedOpenJobs"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
