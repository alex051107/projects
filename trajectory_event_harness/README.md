# Trajectory Event Harness

A replica-aware extraction and validation layer for molecular-dynamics event
labels.

This project is distilled from an HSP90/LiGaMD trajectory-engineering workflow.
The public demo uses synthetic distance traces only. It shows how to turn noisy
frame-level coordinates into auditable first-passage labels without silently
counting transient crossings, recaptures, failed replicas, or right-censored
trajectories as equivalent observations.

## What the harness guarantees

- validates required columns, finite values, monotonic time, and replica
  identity;
- requires a sustained threshold crossing rather than one noisy frame;
- distinguishes observed events, post-event recapture, and right censoring;
- preserves replica-level exposure time;
- hashes the input and frozen config into a provenance manifest;
- emits descriptive labels only and explicitly refuses to claim a kinetic
  estimator from the synthetic demo.

```bash
python3 trajectory_harness.py \
  --input examples/synthetic_trajectories.csv \
  --config examples/config.json \
  --output /tmp/trajectory_events.json

python3 -m unittest discover -s tests -v
```

## Agent/harness framing

An upstream scientific agent can propose coordinate definitions or thresholds.
This deterministic layer freezes those choices, validates trajectory health,
extracts labels, and emits reviewable reasons. Downstream ML or survival
analysis should consume the manifest and censoring fields rather than raw
frames.

This is not a production `k_off` estimator. Any thermodynamic or kinetic claim
requires convergence, reweighting, uncertainty, and experimental calibration
beyond this public extraction demo.

