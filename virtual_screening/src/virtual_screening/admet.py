"""Simple ADMET filtering utilities."""
from __future__ import annotations

import pandas as pd


DEFAULT_THRESHOLDS = {
    "mol_wt": 500,
    "c_log_p": 5,
    "hbd": 5,
    "hba": 10,
    "tpsa": 140,
    "rot_bonds": 10,
    "vina_score": -6.0,
}


def filter_candidates(
    features: pd.DataFrame,
    rescoring: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
    top_k: int = 20,
) -> pd.DataFrame:
    """Apply Lipinski-like filters and rank ligands by consensus rescoring."""

    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    merged = features.merge(rescoring, on="ligand_id", how="inner")
    descriptor_filter = (merged["mol_wt"] <= thresholds["mol_wt"]) & (
        merged["c_log_p"] <= thresholds["c_log_p"]
    ) & (merged["hbd"] <= thresholds["hbd"]) & (merged["hba"] <= thresholds["hba"]) & (
        merged["tpsa"] <= thresholds["tpsa"]
    ) & (merged["rot_bonds"] <= thresholds["rot_bonds"])

    potency_filter = merged["vina_score"] <= thresholds["vina_score"]
    filtered = merged[descriptor_filter & potency_filter].copy()

    score_columns = [col for col in rescoring.columns if col.endswith("_score")]
    filtered["consensus_score"] = filtered[score_columns].mean(axis=1)

    return filtered.sort_values("consensus_score").head(top_k)
