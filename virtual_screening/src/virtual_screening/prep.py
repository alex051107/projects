"""Preparation utilities for receptors and ligands."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class LigandRecord:
    ligand_id: str
    sdf_path: pathlib.Path
    smiles: str
    pdbqt_path: pathlib.Path | None


def prepare_receptor(
    pdb_path: pathlib.Path,
    output_dir: pathlib.Path,
    ph: float = 7.0,
    generate_pdbqt: bool = False,
) -> tuple[pathlib.Path, pathlib.Path | None]:
    """Clean a receptor PDB and optionally create a PDBQT file."""

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = output_dir / f"{pdb_path.stem}_prepared.pdb"

    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Install `pdbfixer` and `openmm` to prepare receptors: pip install pdbfixer openmm"
        ) from exc

    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    with open(prepared_path, "w") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle)

    pdbqt_path: pathlib.Path | None = None
    if generate_pdbqt:
        try:
            from meeko import MoleculePreparation
            from rdkit import Chem
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Install `meeko` and `rdkit` to export receptor PDBQT files."
            ) from exc

        mol = Chem.MolFromPDBFile(str(prepared_path), removeHs=False)
        if mol is None:
            raise ValueError("RDKit failed to parse the prepared receptor.")
        prep = MoleculePreparation(receptor=True)
        prepared = prep.prepare(mol)
        pdbqt_path = prepared_path.with_suffix(".pdbqt")
        with open(pdbqt_path, "w") as handle:
            handle.write(prepared.write_pdbqt_string())

    return prepared_path, pdbqt_path


def prepare_ligands(
    smiles_file: pathlib.Path,
    output_dir: pathlib.Path,
    num_conformers: int = 10,
    energy_minimize: bool = True,
    generate_pdbqt: bool = False,
) -> pd.DataFrame:
    """Generate 3D ligand conformers from a SMILES library using RDKit."""

    from rdkit import Chem
    from rdkit.Chem import AllChem

    output_dir.mkdir(parents=True, exist_ok=True)
    supplier = Chem.SmilesMolSupplier(str(smiles_file), delimiter="\t", titleLine=False)

    if generate_pdbqt:
        try:
            from meeko import MoleculePreparation
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install `meeko` to export ligand PDBQT files.") from exc
        ligand_prep = MoleculePreparation()
    else:
        ligand_prep = None

    records: List[LigandRecord] = []
    for molecule in supplier:
        if molecule is None:
            continue
        ligand_id = molecule.GetProp("_Name") if molecule.HasProp("_Name") else Chem.MolToSmiles(molecule)
        smiles = Chem.MolToSmiles(molecule)
        mol = Chem.AddHs(molecule)
        params = AllChem.ETKDGv3()
        AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
        if energy_minimize:
            for conf_id in range(mol.GetNumConformers()):
                AllChem.UFFOptimizeMolecule(mol, confId=conf_id)
        sdf_path = output_dir / f"{ligand_id}.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        for conf_id in range(mol.GetNumConformers()):
            writer.write(mol, confId=conf_id)
        writer.close()

        pdbqt_path = None
        if generate_pdbqt and ligand_prep is not None:
            prepared = ligand_prep.prepare(mol)
            pdbqt_path = sdf_path.with_suffix(".pdbqt")
            with open(pdbqt_path, "w") as handle:
                handle.write(prepared.write_pdbqt_string())

        records.append(
            LigandRecord(ligand_id=ligand_id, sdf_path=sdf_path, smiles=smiles, pdbqt_path=pdbqt_path)
        )

    df = pd.DataFrame([record.__dict__ for record in records])
    return df
