#!/usr/bin/env python3
"""Generate a FELIS-ready Sage 2.3.0/AshGC ligand without changing the input pose."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from .topology import FF_NAME, FF_SHA256, MODEL, MODEL_SHA256, STEM, make_itp
from .molecules import read_ligand
from ..common import sha256, write as write_json

def parameterize(sdf, output_dir, formal_charge):
    # Charge inference is CPU-only and independent of the production FELIS env.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import numpy as np
    import openmm as mm
    from openmm import unit as ommu
    from rdkit import Chem
    from openff.toolkit import Molecule, ForceField
    from openff.units import unit
    from openff.toolkit.utils import RDKitToolkitWrapper, ToolkitRegistry
    from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
    from openff.toolkit.utils.toolkits import toolkit_registry_manager
    from .validation import validate

    sdf=Path(sdf).resolve();out=Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite parameterization directory: {out}")
    mol,xyz=read_ligand(sdf, formal_charge)
    ff_path=Path(__file__).resolve().parents[1]/"data"/FF_NAME
    if sha256(ff_path)!=FF_SHA256:raise ValueError("Bundled Sage force field checksum changed")
    charge_node=ET.parse(ff_path).getroot().find("NAGLCharges")
    if charge_node is None or charge_node.attrib.get("model_file")!=MODEL or charge_node.attrib.get("model_file_hash")!=MODEL_SHA256:
        raise ValueError("The bundled Sage force field does not declare the expected AshGC model")
    registry=ToolkitRegistry([RDKitToolkitWrapper(),NAGLToolkitWrapper()])
    with toolkit_registry_manager(registry):
        off=Molecule.from_rdkit(mol,hydrogens_are_explicit=True,allow_undefined_stereo=False)
        if [a.atomic_number for a in off.atoms] != [a.GetAtomicNum() for a in mol.GetAtoms()]:
            raise ValueError("RDKit/OpenFF atom order changed")
        original_bonds={tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))) for b in mol.GetBonds()}
        if original_bonds!={tuple(sorted((b.atom1_index,b.atom2_index))) for b in off.bonds}:
            raise ValueError("RDKit/OpenFF indexed bond graph changed")
        if not np.allclose(off.conformers[0].m_as(unit.angstrom),xyz,atol=1e-7,rtol=0):
            raise ValueError("RDKit/OpenFF coordinates changed")
        off.name="LIG"
        for i,a in enumerate(off.atoms):
            a.name=f"{Chem.GetPeriodicTable().GetElementSymbol(a.atomic_number)}{i+1}"
            a.metadata.update({"residue_name":"LIG","residue_number":1,"chain_id":"L"})
        off.partial_charges=None  # Never reuse charges from the GAFF2 SDF.
        ff=ForceField(str(ff_path),load_plugins=True)
        interchange=ff.create_interchange(off.to_topology())
        interchange.positions=xyz*unit.angstrom
        # No periodic box: native and exported systems both use NoCutoff for the
        # energy/force round trip. FELIS supplies the production PME protocol.
        native=interchange.to_openmm(combine_nonbonded_forces=True,add_constrained_forces=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix=out.name+".pending_",dir=out.parent))
    try:
        interchange.to_top(str(temp/"interchange.top"))
        (temp/f"{STEM}.itp").write_text(make_itp((temp/"interchange.top").read_text()))
        shutil.copyfile(sdf,temp/f"{STEM}.sdf")
        shutil.copyfile(ff_path,temp/FF_NAME)
        (temp/"native_system.xml").write_text(mm.XmlSerializer.serialize(native))
        nb=next(f for f in native.getForces() if isinstance(f,mm.NonbondedForce))
        charges=[nb.getParticleParameters(i)[0].value_in_unit(ommu.elementary_charge) for i in range(native.getNumParticles())]
        versions={}
        for package in ["openff-toolkit","openff-interchange","openff-nagl","openff-nagl-models","openff-forcefields","openmm","rdkit","torch","numpy"]:
            try:versions[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:versions[package]="unavailable"
        metadata={"formal_charge":formal_charge,"force_field":"OpenFF Sage 2.3.0","offxml":FF_NAME,"charge_model":MODEL,
                  "charge_model_sha256":MODEL_SHA256,"source_sdf":str(sdf),"source_sdf_sha256":sha256(sdf),
                  "canonical_isomeric_smiles":Chem.MolToSmiles(mol,isomericSmiles=True),
                  "atom_order":"identical to source SDF; no hydrogen addition or geometry optimization",
                  "net_charge_e":sum(charges),"partial_charges_e":charges,"versions":versions,
                  "hashes":{p.name:sha256(p) for p in temp.iterdir() if p.is_file()}}
        # Some Conda OpenFF distributions publish placeholder Python metadata.
        # Preserve the actual Conda package versions/builds as well.
        metadata['conda_packages']=[]
        for path in sorted((Path(__import__('sys').prefix)/'conda-meta').glob('*.json')):
            record=json.loads(path.read_text())
            metadata['conda_packages'].append({k:record.get(k) for k in ['name','version','build','channel']})
        write_json(temp/"parameters.json",metadata)
        result=validate(temp)
        write_json(temp/"validation.json",result)
        temp.rename(out)
        print(f"Sage/AshGC parameterization PASS: {out}")
        print(json.dumps(result,indent=2))
    except BaseException:
        shutil.rmtree(temp,ignore_errors=True)
        raise
