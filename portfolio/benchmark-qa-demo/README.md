# Benchmark QA Demo

An offline benchmark check for a fictional library help desk. Given a catalog snapshot, the system under test must report the available copies of a book, or say the catalog has no matching record. A recorded count of zero and a missing record are deliberately different answers.

**Scope:** All questions, catalog data, answer keys, and example predictions were written from scratch for this demo. No customer data, private evaluation artifacts, real model responses, or production metrics are included. The result below is a checker demonstration, not a model performance claim.

## Run locally

Python 3.9+; no dependencies, credentials, API calls, or network access.

```sh
python3 qa.py cases.json predictions.json
python3 -m unittest -v
```

The example returns **2 correct of 3** evaluation cases. The intentionally incorrect prediction reports zero copies of *Fern Calendar* even though that title has no catalog record.

## What the checker catches

- **Fixture errors:** Rejects malformed cases, duplicate IDs, inconsistent answer keys, and queried titles that recur across development and evaluation after case and whitespace normalization.
- **Prediction errors:** Rejects unexpected IDs. Reports missing, malformed, and wrong predictions separately; all expected cases remain in the denominator.
- **Observable results:** Emits a JSON report on a valid run, even if some answers are wrong. Invalid inputs produce a diagnostic and exit status 2 rather than publishing a misleading score.

```text
Synthetic cases → Validate fixtures → Select split
                                         ↓
Example predictions → Check prediction IDs → Score every selected case → JSON report
```

Editable Mermaid source: [flow.mmd](flow.mmd).

## Limits

This measures exact structured responses to one deterministic question template. Catalog lookups use exact title matching; case and whitespace normalization apply only to the cross-split title check. It does not evaluate paraphrases, production behavior, or broader leakage (such as shared catalog entries). The hand-written predictions make failures reproducible without calling a model.
