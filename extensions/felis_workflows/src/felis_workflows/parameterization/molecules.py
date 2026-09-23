from pathlib import Path

from ..common import WorkflowError


def read_ligand(path, formal_charge=None):
    import numpy as np
    from rdkit import Chem
    molecules = list(Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True))
    if len(molecules) != 1 or molecules[0] is None:
        raise WorkflowError(f"Expected one valid SDF molecule: {path}")
    mol = molecules[0]
    if len(Chem.GetMolFrags(mol)) != 1:
        raise WorkflowError("Provide one connected ligand; salts/partners belong in separate inputs")
    if mol.GetNumConformers() != 1 or not mol.GetConformer().Is3D():
        raise WorkflowError("An explicit 3D bound pose is required")
    if any(a.GetNumImplicitHs() or a.GetNumExplicitHs() for a in mol.GetAtoms() if a.GetAtomicNum() != 1):
        raise WorkflowError("All hydrogens must be explicit atom records")
    if any(a.GetNumRadicalElectrons() or a.GetIsotope() for a in mol.GetAtoms()):
        raise WorkflowError("Radicals and isotope substitutions are not supported")
    if any(info.specified == Chem.StereoSpecified.Unspecified for info in Chem.FindPotentialStereo(mol)):
        raise WorkflowError("Resolve undefined atom/bond stereochemistry before parameterization")
    if formal_charge is not None and Chem.GetFormalCharge(mol) != formal_charge:
        raise WorkflowError("SDF formal charge does not match the campaign")
    xyz = np.asarray(mol.GetConformer().GetPositions())
    if not np.isfinite(xyz).all():
        raise WorkflowError("Nonfinite ligand coordinates")
    return mol, xyz
