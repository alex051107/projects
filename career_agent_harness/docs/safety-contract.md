# Safety contract

## Allowed

- Read local structured state.
- Validate evidence, freshness, eligibility, deadlines, and capacity.
- Generate a private proposal queue.
- Export sanitized aggregate metrics.
- Replay a prior date for testing.

## Forbidden

- Submit or withdraw an application.
- Send a connection request, email, direct message, or follow-up.
- Accept an interview time or offer.
- Infer work authorization, readiness, or application outcomes.
- Mark a proposed action as completed.
- Export company names, job titles, contact identities, source URLs, private
  notes, stable local IDs, or individual deadlines to the public snapshot.

## Invariants

- Every proposed external action requires human confirmation.
- A stale record fails closed.
- Readiness is evidence-based and contiguous.
- Planning never mutates source records.
- Zero applications and zero connections are valid outcomes.
- Public output is generated through an allowlist, not by redacting a private
  record after serialization.

