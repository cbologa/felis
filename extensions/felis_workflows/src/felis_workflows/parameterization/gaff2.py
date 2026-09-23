"""GAFF2/AM1-BCC with an independent Amber-to-GROMACS parameter check."""
from pathlib import Path
import importlib.metadata
import shutil
import tempfile

from ..common import WorkflowError, sha256, write
from . import acpype_io
from .molecules import read_ligand


def balance_native_charges(system, formal_charge):
    """Independently reproduce ACPYPE's documented integer-charge balancing.

    ACPYPE changes the largest-magnitude, tied partial charges in its export,
    but retains the pre-balance Amber prmtop. Do not compare different models
    or loosen the native/export tolerance to hide that difference.
    """
    import openmm as mm
    from openmm import unit as u
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    charges = [nb.getParticleParameters(i)[0].value_in_unit(u.elementary_charge)
               for i in range(nb.getNumParticles())]
    total = sum(charges)
    if abs(total - formal_charge) > .01:
        raise WorkflowError(f"Amber charge residual exceeds 0.01 e: {total} vs {formal_charge}")
    limit = max(charges) if abs(max(charges)) >= abs(min(charges)) else min(charges)
    indices = [i for i, q in enumerate(charges) if q == limit]
    delta = (formal_charge - total) / len(indices)
    balanced = [q + delta if i in indices else q for i, q in enumerate(charges)]
    for i in indices:
        _, sigma, epsilon = nb.getParticleParameters(i)
        nb.setParticleParameters(i, balanced[i], sigma, epsilon)
    for i in range(nb.getNumExceptions()):
        a, b, qprod, sigma, epsilon = nb.getExceptionParameters(i)
        old_product = charges[a] * charges[b]
        if old_product:
            qprod *= balanced[a] * balanced[b] / old_product
        nb.setExceptionParameters(i, a, b, qprod, sigma, epsilon)
    return {"method": "ACPYPE largest-absolute-charge balancing, independently reconstructed from Amber",
            "original_net_charge_e": total, "target_charge_e": formal_charge,
            "atom_indices_zero_based": indices, "per_atom_delta_e": delta}


def parameterize(sdf, output_dir, formal_charge):
    import numpy as np
    import openmm as mm
    from openmm import app
    from rdkit import Chem
    from .validation import validate
    sdf, out = Path(sdf).resolve(), Path(output_dir).resolve()
    if out.exists():
        raise WorkflowError(f"Refusing to overwrite {out}")
    mol, xyz = read_ligand(sdf, formal_charge)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=out.name + ".pending_", dir=out.parent) as directory:
        temp = Path(directory)
        build = temp / "build"
        build.mkdir()
        files = acpype_io.run_acpype(sdf, formal_charge, "gaff2", build)
        if "[ defaults ]" not in files["top"].read_text():
            raise WorkflowError("ACPYPE did not declare nonbonded defaults")
        acpype_io.assert_amber_defaults(files["top"])
        shutil.copyfile(sdf, temp / "ligand.sdf")
        # Indexed coordinates catch same-element/symmetric atom permutations
        # that a bond graph alone cannot identify. Preserve the source pose.
        gro = app.GromacsGroFile(str(files["gro"]))
        exported_xyz = gro.positions.value_in_unit(mm.unit.angstrom)
        if np.shape(exported_xyz) != xyz.shape or not np.allclose(exported_xyz, xyz, atol=.011, rtol=0):
            raise WorkflowError("ACPYPE changed atom order or coordinates; refusing an ambiguous SDF/ITP pairing")
        prmtops = list(files["acpdir"].glob("*.prmtop"))
        if len(prmtops) != 1:
            raise WorkflowError("ACPYPE -o all must produce one Amber prmtop for independent validation")
        native = app.AmberPrmtopFile(str(prmtops[0])).createSystem(
            nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False, removeCMMotion=False)
        balance = balance_native_charges(native, formal_charge)
        # ACPYPE's GMX writer rounds some bonded constants to five significant
        # digits. Export the Amber model through ParmEd at higher precision,
        # then compare it independently with OpenMM's Amber parser.
        import parmed
        from .topology import make_itp
        structure = parmed.load_file(str(prmtops[0]))
        nb = next(f for f in native.getForces() if isinstance(f, mm.NonbondedForce))
        for i, atom in enumerate(structure.atoms):
            atom.charge = nb.getParticleParameters(i)[0].value_in_unit(mm.unit.elementary_charge)
        structure.save(str(temp / "parmed.top"), format="gromacs")
        (temp / "ligand.itp").write_text(make_itp((temp / "parmed.top").read_text(),
            namespace="G2_", description="GAFF2/AM1-BCC; Amber to ParmEd export"))
        (temp / "native_system.xml").write_text(mm.XmlSerializer.serialize(native))
        shutil.copyfile(prmtops[0], temp / "native.prmtop")
        # Keep the chemistry outputs and logs needed to audit parameterization.
        for index, logfile in enumerate(sorted(build.rglob("*.log"))):
            shutil.copyfile(logfile, temp / f"acpype_{index}_{logfile.name}")
        shutil.rmtree(build)
        versions = {}
        for name in ["acpype", "parmed", "openmm", "rdkit", "numpy"]:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "unknown"
        write(temp / "parameters.json", {"force_field": "GAFF2", "charge_model": "AM1-BCC",
              "formal_charge": formal_charge, "versions": versions,
              "charge_balancing": balance,
              "source_sdf_sha256": sha256(sdf), "canonical_isomeric_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
              "atom_order": "source SDF; ACPYPE indexed coordinates checked", "hashes": {
                  p.name: sha256(p) for p in temp.iterdir() if p.is_file()}})
        write(temp / "validation.json", validate(temp))
        shutil.copytree(temp, out)
