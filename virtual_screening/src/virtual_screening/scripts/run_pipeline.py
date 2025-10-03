"""Run the end-to-end virtual screening workflow."""
from __future__ import annotations

import argparse
import pathlib

import pandas as pd
import yaml

from ..admet import filter_candidates
from ..docking import run_vina
from ..prep import LigandRecord, prepare_ligands, prepare_receptor
from ..reporting import build_report
from ..rescoring import build_feature_table, train_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())

    paths = config["paths"]
    prep_cfg = config.get("prep", {})
    ligand_cfg = config.get("ligands", {})
    docking_cfg = config.get("docking", {})
    admet_cfg = config.get("admet", {})

    receptor_path = pathlib.Path(paths["receptor"]).resolve()
    ligands_smiles = pathlib.Path(paths["ligands"]).resolve()
    workdir = pathlib.Path(paths.get("workdir", "virtual_screening/outputs")).resolve()
    receptor_dir = workdir / "receptor"
    ligand_dir = workdir / "ligands"
    docking_dir = workdir / "docking"
    report_dir = workdir / "reports"

    receptor_dir.mkdir(parents=True, exist_ok=True)
    ligand_dir.mkdir(parents=True, exist_ok=True)
    docking_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    prepared_receptor, receptor_pdbqt = prepare_receptor(
        receptor_path,
        receptor_dir,
        ph=prep_cfg.get("ph", 7.0),
        generate_pdbqt=prep_cfg.get("generate_pdbqt", False),
    )
    if receptor_pdbqt is not None:
        receptor_pdbqt_path = receptor_pdbqt
    else:
        receptor_pdbqt_config = paths.get("receptor_pdbqt")
        if not receptor_pdbqt_config:
            raise ValueError("Provide a receptor PDBQT file via prep.generate_pdbqt or paths.receptor_pdbqt")
        receptor_pdbqt_path = pathlib.Path(receptor_pdbqt_config).resolve()

    ligand_df = prepare_ligands(
        ligands_smiles,
        ligand_dir,
        num_conformers=ligand_cfg.get("num_conformers", 10),
        energy_minimize=ligand_cfg.get("energy_minimize", True),
        generate_pdbqt=ligand_cfg.get("generate_pdbqt", False),
    )
    ligand_records = [
        LigandRecord(
            ligand_id=row["ligand_id"],
            sdf_path=pathlib.Path(row["sdf_path"]),
            smiles=row["smiles"],
            pdbqt_path=pathlib.Path(row["pdbqt_path"]) if pd.notna(row.get("pdbqt_path")) else None,
        )
        for _, row in ligand_df.iterrows()
    ]

    docking_results = run_vina(
        receptor_pdbqt_path,
        ligand_records,
        center=docking_cfg["center"],
        box_size=docking_cfg["box_size"],
        output_dir=docking_dir,
        exhaustiveness=docking_cfg.get("exhaustiveness", 12),
        num_modes=docking_cfg.get("num_modes", 10),
        seed=docking_cfg.get("seed", 42),
        vina_executable=docking_cfg.get("vina_executable", "vina"),
    )

    feature_table = build_feature_table(ligand_df, docking_results)
    rescoring_predictions, rescoring_metrics = train_models(
        feature_table,
        target_column=docking_cfg.get("target_column", "vina_score"),
        test_size=docking_cfg.get("test_size", 0.2),
        random_state=docking_cfg.get("seed", 42),
    )

    filtered_hits = filter_candidates(
        feature_table,
        rescoring_predictions,
        thresholds=admet_cfg.get("thresholds"),
        top_k=admet_cfg.get("top_k", 20),
    )

    build_report(filtered_hits, rescoring_metrics, report_dir)

    print(f"Top hits saved to {report_dir / 'top20_hits.csv'}")


if __name__ == "__main__":
    main()
