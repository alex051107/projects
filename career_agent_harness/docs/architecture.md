# Architecture

## Control loop

The harness separates four concerns that are often collapsed in a demo agent:

1. **Observation** — local JSON records describe verified market state,
   deadlines, contacts, activity, and evidence.
2. **Policy** — deterministic rules decide which records are actionable and
   how urgent work preempts planned work.
3. **Proposal** — the harness emits a bounded daily queue. It does not execute
   the queue.
4. **Disclosure** — a separate sanitizer exports only aggregate fields for a
   dashboard or portfolio surface.

This separation makes the system replayable: the same state, plan, date, and
timezone produce the same priority order.

## Core modules in `scripts/harness.py`

| Concern | Mechanism |
| --- | --- |
| Input validation | Top-level type checks, ISO date parsing, timezone normalization |
| Evidence | Contiguous S0–S3 readiness ladder with non-empty evidence |
| Freshness | `lastVerifiedAt`, `reverifyBy`, `staleAfter`, and deadline checks |
| Scheduling | Stable priorities, capacity budgeting, OA/interview preemption |
| Approval | `requiresConfirmation: true` on every proposed action |
| Privacy | Aggregate-only snapshot with an explicit omitted-field declaration |
| Durability | Atomic writes and non-destructive initialization |

## Agent integration

An LLM can sit upstream as a research or drafting component, but it should not
write outcomes or call external submission tools directly. Its proposed facts
enter the private state only after verification. The deterministic harness then
decides whether the fact is fresh and sufficient to create a human-reviewed
action.

That pattern is reusable beyond recruiting: replace the domain state and
policy while keeping evidence gates, freshness checks, approval boundaries,
append-only outcomes, and sanitized observability.

