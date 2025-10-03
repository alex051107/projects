"""AutoDock Vina docking helpers."""
from __future__ import annotations

import pathlib
import re
import subprocess
from typing import Iterable, Sequence

import pandas as pd

from .prep import LigandRecord

SCORE_PATTERN = re.compile(r"^REMARK VINA RESULT:\s+([-0-9.]+)")


def _ensure_pdbqt(sdf_path: pathlib.Path, pdbqt_path: pathlib.Path | None) -> pathlib.Path:
    if pdbqt_path and pdbqt_path.exists():
        return pdbqt_path

    try:
        from meeko import MoleculePreparation
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install `meeko` and `rdkit` to convert SDF ligands to PDBQT.") from exc

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        raise ValueError(f"Failed to parse ligand SDF: {sdf_path}")
    mol = Chem.AddHs(mol)
    prep = MoleculePreparation()
    prepared = prep.prepare(mol)
    target = pdbqt_path or sdf_path.with_suffix(".pdbqt")
    with open(target, "w") as handle:
        handle.write(prepared.write_pdbqt_string())
    return target



def run_vina(
    receptor_pdbqt: pathlib.Path,
    ligands: Iterable[LigandRecord],
    center: Sequence[float],
    box_size: Sequence[float],
    output_dir: pathlib.Path,
    exhaustiveness: int = 12,
    num_modes: int = 10,
    seed: int = 42,
    vina_executable: str = "vina",
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for record in ligands:
        ligand_pdbqt = _ensure_pdbqt(record.sdf_path, record.pdbqt_path)
        out_path = output_dir / f"{record.ligand_id}_poses.pdbqt"
        log_path = output_dir / f"{record.ligand_id}.log"

        cmd = [
            vina_executable,
            "--receptor",
            str(receptor_pdbqt),
            "--ligand",
            str(ligand_pdbqt),
            "--center_x",
            str(center[0]),
            "--center_y",
            str(center[1]),
            "--center_z",
            str(center[2]),
            "--size_x",
            str(box_size[0]),
            "--size_y",
            str(box_size[1]),
            "--size_z",
            str(box_size[2]),
            "--exhaustiveness",
            str(exhaustiveness),
            "--num_modes",
            str(num_modes),
            "--seed",
            str(seed),
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ]

        subprocess.run(cmd, check=True)

        best_score = None
        for line in log_path.read_text().splitlines():
            match = SCORE_PATTERN.match(line)
            if match:
                best_score = float(match.group(1))
                break

        results.append(
            {
                "ligand_id": record.ligand_id,
                "vina_score": best_score,
                "pose_path": out_path,
                "log_path": log_path,
            }
        )

    return pd.DataFrame(results)
