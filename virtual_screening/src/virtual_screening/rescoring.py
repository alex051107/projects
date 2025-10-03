"""Ligand rescoring utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


@dataclass
class RescoreResult:
    model_name: str
    mae: float
    rmse: float
    r2: float


DESCRIPTOR_COLUMNS = [
    "mol_wt",
    "c_log_p",
    "tpsa",
    "hbd",
    "hba",
    "rot_bonds",
]


def _compute_descriptors(smiles: str) -> Dict[str, float]:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return {
        "mol_wt": Descriptors.MolWt(mol),
        "c_log_p": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "rot_bonds": Descriptors.NumRotatableBonds(mol),
    }



def build_feature_table(ligands: pd.DataFrame, docking: pd.DataFrame) -> pd.DataFrame:
    descriptor_rows = []
    for _, row in ligands.iterrows():
        descriptors = _compute_descriptors(row["smiles"])
        descriptors.update({
            "ligand_id": row["ligand_id"],
            "sdf_path": row["sdf_path"],
        })
        descriptor_rows.append(descriptors)

    feature_df = pd.DataFrame(descriptor_rows)
    merged = feature_df.merge(docking, on="ligand_id", how="inner")
    return merged



def _train_random_forest(X_train, y_train, random_state: int = 42) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=500,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model



def _train_xgboost(X_train, y_train, random_state: int = 42):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install xgboost to train XGBoost rescoring models.") from exc

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model



def train_models(
    features: pd.DataFrame,
    target_column: str = "vina_score",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Iterable[RescoreResult]]:
    X = features[DESCRIPTOR_COLUMNS]
    y = features[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = {
        "rf_score": _train_random_forest(X_train, y_train, random_state),
        "xgboost_qsar": _train_xgboost(X_train, y_train, random_state),
    }

    predictions = []
    metrics = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics.append(
            RescoreResult(
                model_name=name,
                mae=float(mean_absolute_error(y_test, y_pred)),
                rmse=float(np.sqrt(mean_squared_error(y_test, y_pred))),
                r2=float(r2_score(y_test, y_pred)),
            )
        )
        full_pred = model.predict(X)
        predictions.append(
            pd.DataFrame(
                {
                    "ligand_id": features["ligand_id"],
                    f"{name}_score": full_pred,
                }
            )
        )

    prediction_df = predictions[0]
    for df in predictions[1:]:
        prediction_df = prediction_df.merge(df, on="ligand_id")

    return prediction_df, metrics
