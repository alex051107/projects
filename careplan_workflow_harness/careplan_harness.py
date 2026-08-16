#!/usr/bin/env python3
"""Synthetic-only, reviewer-gated CarePlan workflow harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol


REQUIRED_ORDER_FIELDS = (
    "order_id",
    "medication_code",
    "dose_unit",
    "route",
    "declared_intent",
)
ALLOWED_SOURCE_FIELDS = frozenset(REQUIRED_ORDER_FIELDS[1:])
ALLOWED_DRAFT_FIELDS = frozenset(
    {"order_id", "plan_version", "summary", "reviewer_attention", "source_fields"}
)
TERMINAL_STATES = frozenset({"APPROVED", "REJECTED", "FAILED"})


class HarnessError(RuntimeError):
    """Base error for an expected workflow rejection."""


class IdempotencyConflict(HarnessError):
    pass


class StateConflict(HarnessError):
    pass


class AuthorizationError(HarnessError):
    pass


class DraftValidationError(HarnessError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_mapping(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def evaluate_order(order: Mapping[str, Any]) -> tuple[str, ...]:
    hard_stops: list[str] = []
    if order.get("case_origin") != "synthetic":
        hard_stops.append("ONLY_SYNTHETIC_CASES_ALLOWED")
    token = order.get("patient_token")
    if not isinstance(token, str) or not token.startswith("SYN-"):
        hard_stops.append("INVALID_SYNTHETIC_PATIENT_TOKEN")
    for field in REQUIRED_ORDER_FIELDS:
        value = order.get(field)
        if not isinstance(value, str) or not value.strip():
            hard_stops.append(f"MISSING_REQUIRED_FIELD:{field}")
    if order.get("synthetic_critical_flag") is True:
        hard_stops.append("SYNTHETIC_CRITICAL_HARD_STOP")
    return tuple(hard_stops)


def _validated_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise DraftValidationError(f"INVALID_DRAFT_FIELD:{field}")
    if len(value) != len(set(value)):
        raise DraftValidationError(f"DUPLICATE_DRAFT_VALUE:{field}")
    return value


def validate_draft(
    raw: Mapping[str, Any], order: Mapping[str, Any], plan_version: int
) -> dict[str, Any]:
    keys = set(raw)
    unexpected = keys - ALLOWED_DRAFT_FIELDS
    missing = ALLOWED_DRAFT_FIELDS - keys
    if unexpected:
        raise DraftValidationError("UNSUPPORTED_DRAFT_FIELD:" + ",".join(sorted(unexpected)))
    if missing:
        raise DraftValidationError("MISSING_DRAFT_FIELD:" + ",".join(sorted(missing)))
    if raw["order_id"] != order["order_id"]:
        raise DraftValidationError("ORDER_ID_MISMATCH")
    if raw["plan_version"] != plan_version:
        raise DraftValidationError("PLAN_VERSION_MISMATCH")
    if not isinstance(raw["summary"], str) or not raw["summary"].strip():
        raise DraftValidationError("INVALID_DRAFT_FIELD:summary")
    attention = _validated_string_list(raw["reviewer_attention"], "reviewer_attention")
    source_fields = _validated_string_list(raw["source_fields"], "source_fields")
    if not source_fields or not set(source_fields).issubset(ALLOWED_SOURCE_FIELDS):
        raise DraftValidationError("UNSUPPORTED_DRAFT_SOURCE_FIELD")
    return {
        "order_id": raw["order_id"],
        "plan_version": raw["plan_version"],
        "summary": raw["summary"].strip(),
        "reviewer_attention": attention,
        "source_fields": source_fields,
    }


class DraftProvider(Protocol):
    def draft(self, order: Mapping[str, Any], plan_version: int) -> Mapping[str, Any]:
        """Return a proposed draft. The provider has no state-transition authority."""


@dataclass
class FixtureDraftProvider:
    """Deterministic provider with explicit failure modes for regression tests."""

    mode: str = "valid"

    def draft(self, order: Mapping[str, Any], plan_version: int) -> Mapping[str, Any]:
        if self.mode == "malformed":
            return {"order_id": order["order_id"]}
        draft: dict[str, Any] = {
            "order_id": order["order_id"],
            "plan_version": plan_version,
            "summary": "Synthetic draft prepared for human review; no medical recommendation is made.",
            "reviewer_attention": ["VERIFY_SYNTHETIC_ORDER_CONTEXT"],
            "source_fields": ["medication_code", "dose_unit", "route", "declared_intent"],
        }
        if self.mode == "approval_field":
            draft["approval"] = "APPROVED"
        if self.mode == "invented_source":
            draft["source_fields"].append("invented_field")
        return draft


class CarePlanHarness:
    """SQLite-backed state machine with idempotent submit and human-owned review."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS plans (
                    plan_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    order_sha256 TEXT NOT NULL,
                    order_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    state_version INTEGER NOT NULL,
                    plan_version INTEGER NOT NULL,
                    draft_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    plan_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    from_state TEXT,
                    to_state TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    event_code TEXT NOT NULL,
                    detail_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (plan_id, sequence)
                );
                """
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "plan_id": row["plan_id"],
            "idempotency_key": row["idempotency_key"],
            "order_sha256": row["order_sha256"],
            "order": json.loads(row["order_json"]),
            "state": row["state"],
            "state_version": int(row["state_version"]),
            "plan_version": int(row["plan_version"]),
            "draft": json.loads(row["draft_json"]) if row["draft_json"] else None,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection,
        *,
        plan_id: str,
        from_state: str | None,
        to_state: str,
        actor: str,
        event_code: str,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        next_sequence = connection.execute(
            "SELECT COALESCE(MAX(sequence), -1) + 1 FROM audit_events WHERE plan_id = ?",
            (plan_id,),
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO audit_events(
                plan_id, sequence, from_state, to_state, actor,
                event_code, detail_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan_id,
                next_sequence,
                from_state,
                to_state,
                actor,
                event_code,
                canonical_json(dict(detail or {})),
                utc_now(),
            ),
        )

    def get(self, plan_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM plans WHERE plan_id = ?", (plan_id,)).fetchone()
        if row is None:
            raise KeyError(plan_id)
        return self._record(row)

    def submit(self, order: Mapping[str, Any], idempotency_key: str) -> tuple[dict[str, Any], bool]:
        if not idempotency_key.strip():
            raise ValueError("idempotency_key is required")
        frozen_order = dict(order)
        order_sha256 = sha256_mapping(frozen_order)
        hard_stops = evaluate_order(frozen_order)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM plans WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if existing is not None:
                record = self._record(existing)
                if record["order_sha256"] != order_sha256:
                    raise IdempotencyConflict("IDEMPOTENCY_KEY_REUSED_WITH_DIFFERENT_ORDER")
                return record, True

            now = utc_now()
            plan_id = f"plan_{uuid.uuid4().hex[:16]}"
            state = "REJECTED" if hard_stops else "DRAFT_QUEUED"
            connection.execute(
                """
                INSERT INTO plans(
                    plan_id, idempotency_key, order_sha256, order_json, state,
                    state_version, plan_version, draft_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 0, 1, NULL, ?, ?)
                """,
                (
                    plan_id,
                    idempotency_key,
                    order_sha256,
                    canonical_json(frozen_order),
                    state,
                    now,
                    now,
                ),
            )
            self._append_event(
                connection,
                plan_id=plan_id,
                from_state=None,
                to_state=state,
                actor="deterministic-intake",
                event_code="HARD_STOP_REJECTED" if hard_stops else "DRAFT_JOB_QUEUED",
                detail={"hard_stops": list(hard_stops)},
            )
            row = connection.execute("SELECT * FROM plans WHERE plan_id = ?", (plan_id,)).fetchone()
        return self._record(row), False

    def _transition(
        self,
        connection: sqlite3.Connection,
        record: Mapping[str, Any],
        *,
        expected_state: str,
        expected_version: int,
        next_state: str,
        actor: str,
        event_code: str,
        detail: Mapping[str, Any] | None = None,
        draft: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if record["state"] != expected_state:
            raise StateConflict(f"STATE_REQUIRED:{expected_state}:actual={record['state']}")
        if record["state_version"] != expected_version:
            raise StateConflict(
                f"STALE_STATE_VERSION:expected={expected_version}:actual={record['state_version']}"
            )
        next_version = expected_version + 1
        cursor = connection.execute(
            """
            UPDATE plans
            SET state = ?, state_version = ?, draft_json = COALESCE(?, draft_json), updated_at = ?
            WHERE plan_id = ? AND state_version = ?
            """,
            (
                next_state,
                next_version,
                canonical_json(draft) if draft is not None else None,
                utc_now(),
                record["plan_id"],
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            raise StateConflict("CONCURRENT_STATE_UPDATE")
        self._append_event(
            connection,
            plan_id=record["plan_id"],
            from_state=expected_state,
            to_state=next_state,
            actor=actor,
            event_code=event_code,
            detail=detail,
        )
        row = connection.execute(
            "SELECT * FROM plans WHERE plan_id = ?", (record["plan_id"],)
        ).fetchone()
        return self._record(row)

    def draft(self, plan_id: str, provider: DraftProvider) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM plans WHERE plan_id = ?", (plan_id,)).fetchone()
            if row is None:
                raise KeyError(plan_id)
            queued = self._record(row)
            drafting = self._transition(
                connection,
                queued,
                expected_state="DRAFT_QUEUED",
                expected_version=queued["state_version"],
                next_state="DRAFTING",
                actor="draft-worker",
                event_code="PROVIDER_CALL_STARTED",
            )
            try:
                raw = provider.draft(drafting["order"], drafting["plan_version"])
                if not isinstance(raw, Mapping):
                    raise DraftValidationError("DRAFT_ROOT_MUST_BE_OBJECT")
                validated = validate_draft(raw, drafting["order"], drafting["plan_version"])
            except DraftValidationError as exc:
                return self._transition(
                    connection,
                    drafting,
                    expected_state="DRAFTING",
                    expected_version=drafting["state_version"],
                    next_state="FAILED",
                    actor="draft-validator",
                    event_code="DRAFT_SCHEMA_REJECTED",
                    detail={"reason": str(exc)},
                )
            return self._transition(
                connection,
                drafting,
                expected_state="DRAFTING",
                expected_version=drafting["state_version"],
                next_state="REVIEW_PENDING",
                actor="draft-validator",
                event_code="DRAFT_VALIDATED_REVIEW_REQUIRED",
                draft=validated,
            )

    def review(
        self,
        plan_id: str,
        *,
        expected_state_version: int,
        reviewer_id: str,
        decision: str,
    ) -> dict[str, Any]:
        role, separator, identity = reviewer_id.partition(":")
        if role != "pharmacist" or separator != ":" or not identity.strip():
            raise AuthorizationError("PHARMACIST_ROLE_REQUIRED")
        normalized = decision.upper().strip()
        if normalized not in {"APPROVE", "REJECT"}:
            raise ValueError("decision must be APPROVE or REJECT")
        next_state = "APPROVED" if normalized == "APPROVE" else "REJECTED"
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM plans WHERE plan_id = ?", (plan_id,)).fetchone()
            if row is None:
                raise KeyError(plan_id)
            record = self._record(row)
            return self._transition(
                connection,
                record,
                expected_state="REVIEW_PENDING",
                expected_version=expected_state_version,
                next_state=next_state,
                actor=reviewer_id,
                event_code=f"PHARMACIST_{next_state}",
                detail={"decision": normalized},
            )

    def events(self, plan_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events WHERE plan_id = ? ORDER BY sequence", (plan_id,)
            ).fetchall()
        return [
            {
                "sequence": int(row["sequence"]),
                "from_state": row["from_state"],
                "to_state": row["to_state"],
                "actor": row["actor"],
                "event_code": row["event_code"],
                "detail": json.loads(row["detail_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def receipt(self, plan_id: str) -> dict[str, Any]:
        record = self.get(plan_id)
        return {
            "schema_version": "careplan-workflow-receipt.v1",
            "plan_id": record["plan_id"],
            "state": record["state"],
            "state_version": record["state_version"],
            "plan_version": record["plan_version"],
            "order_sha256": record["order_sha256"],
            "draft_schema_validated": record["draft"] is not None,
            "human_decision_required": record["state"] == "REVIEW_PENDING",
            "terminal": record["state"] in TERMINAL_STATES,
            "scope_boundary": (
                "Synthetic portfolio workflow only; no real patient data, medical advice, "
                "clinical validation, autonomous approval, or production-deployment claim."
            ),
            "audit_events": self.events(plan_id),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--idempotency-key", default="public-synthetic-demo-v1")
    parser.add_argument("--reviewer")
    parser.add_argument("--decision", choices=("APPROVE", "REJECT"))
    args = parser.parse_args()
    if bool(args.reviewer) != bool(args.decision):
        parser.error("--reviewer and --decision must be supplied together")

    order = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(order, dict):
        raise HarnessError("input must contain one JSON object")
    harness = CarePlanHarness(args.database)
    record, _ = harness.submit(order, args.idempotency_key)
    if record["state"] == "DRAFT_QUEUED":
        record = harness.draft(record["plan_id"], FixtureDraftProvider())
    if record["state"] == "REVIEW_PENDING" and args.reviewer and args.decision:
        record = harness.review(
            record["plan_id"],
            expected_state_version=record["state_version"],
            reviewer_id=args.reviewer,
            decision=args.decision,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(harness.receipt(record["plan_id"]), indent=2) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
