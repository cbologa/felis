"""Independent OpenMM versus GROMACS fixture, including 1-4 interactions."""
from itertools import combinations, product
import math
import pytest
import numpy as np

from felis_workflows.common import WorkflowError, write
from felis_workflows.parameterization.molecules import read_ligand
from felis_workflows.parameterization.validation import validate, signatures, compare_signatures


@pytest.fixture
def ethane(tmp_path):
    import openmm as mm
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    assert AllChem.EmbedMolecule(mol, randomSeed=17) == 0
    with Chem.SDWriter(str(tmp_path / "ligand.sdf")) as w:
        w.write(mol)
    system, nb = mm.System(), mm.NonbondedForce()
    rows = ["[ atomtypes ]", "C 6 12.011 0.0 A .34 .2", "H 1 1.008 0.0 A .25 .1",
            "[ moleculetype ]", "LIG 3", "[ atoms ]"]
    for i, atom in enumerate(mol.GetAtoms()):
        carbon = atom.GetAtomicNum() == 6
        mass, charge, sigma, epsilon = (12.011, -.3, .34, .2) if carbon else (1.008, .1, .25, .1)
        system.addParticle(mass)
        nb.addParticle(charge, sigma, epsilon)
        rows.append(f'{i+1} {"C" if carbon else "H"} 1 LIG {atom.GetSymbol()}{i+1} {i+1} {charge} {mass}')
    bonds, force = [], mm.HarmonicBondForce()
    rows.append("[ bonds ]")
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        r = .154 if {a, b} == {0, 1} else .109
        force.addBond(a, b, r, 300000.)
        bonds.append((a, b))
        rows.append(f"{a+1} {b+1} 1 {r} 300000")
    system.addForce(force)
    nb.createExceptionsFromBonds(bonds, 5/6, .5)
    system.addForce(nb)
    rows.append("[ angles ]")
    force = mm.HarmonicAngleForce()
    for central in mol.GetAtoms():
        for a, c in combinations([n.GetIdx() for n in central.GetNeighbors()], 2):
            b = central.GetIdx()
            force.addAngle(a, b, c, math.radians(109.5), 200)
            rows.append(f"{a+1} {b+1} {c+1} 1 109.5 200")
    system.addForce(force)
    rows.append("[ dihedrals ]")
    force = mm.PeriodicTorsionForce()
    hydrogens = list(product([2, 3, 4], [5, 6, 7]))
    for a, d in hydrogens:
        force.addTorsion(a, 0, 1, d, 3, 0, .8)
        rows.append(f"{a+1} 1 2 {d+1} 9 0 .8 3")
    system.addForce(force)
    rows += ["[ pairs ]", *[f"{a+1} {d+1} 1" for a, d in hydrogens]]
    (tmp_path / "ligand.itp").write_text("\n".join(rows) + "\n")
    (tmp_path / "native_system.xml").write_text(mm.XmlSerializer.serialize(system))
    write(tmp_path / "parameters.json", {"formal_charge": 0, "hashes": {}})
    return tmp_path, system


def test_native_gromacs_parameters_energies_forces_constraints(ethane):
    directory, _ = ethane
    report = validate(directory)
    assert report["passed"] and report["particles"] == 8
    assert report["hydrogen_constraints"] == 6
    assert len(report["energy_force_checks"]) == 3


def test_native_export_charge_corruption_rejected(ethane):
    directory, _ = ethane
    itp = directory / "ligand.itp"
    itp.write_text(itp.read_text().replace("-0.3 12.011", "-0.2 12.011", 1))
    with pytest.raises(ValueError, match="Per-atom parameters"):
        validate(directory)


def test_native_export_torsion_corruption_rejected(ethane):
    directory, _ = ethane
    itp = directory / "ligand.itp"
    itp.write_text(itp.read_text().replace("9 0 .8 3", "9 180 .8 3", 1))
    with pytest.raises(ValueError, match="torsions mismatch"):
        validate(directory)


def test_assembly_permutation_and_serialization_tolerance(ethane):
    _, system = ethane
    expected = signatures(system)
    with pytest.raises(ValueError, match="Per-atom"):
        compare_signatures(expected, signatures(system, [2, 1, 0, 3, 4, 5, 6, 7]))
    system.setParticleMass(0, 12.01078)
    a = signatures(system)
    system.setParticleMass(0, 12.0108)
    compare_signatures(a, signatures(system), allow_mass_rounding=True)
    system.setParticleMass(0, 12.01)
    with pytest.raises(ValueError, match="mass"):
        compare_signatures(a, signatures(system), allow_mass_rounding=True)


def test_input_formal_charge_and_explicit_hydrogens(ethane, tmp_path):
    directory, _ = ethane
    with pytest.raises(WorkflowError, match="formal charge"):
        read_ligand(directory / "ligand.sdf", 1)
    from rdkit import Chem
    mol = Chem.RemoveHs(Chem.SDMolSupplier(str(directory / "ligand.sdf"), removeHs=False)[0])
    path = tmp_path / "implicit.sdf"
    with Chem.SDWriter(str(path)) as w:
        w.write(mol)
    with pytest.raises(WorkflowError, match="hydrogens"):
        read_ligand(path)


def test_amber_charge_balance_is_explicit_and_bounded():
    import openmm as mm
    from openmm import unit as u
    from felis_workflows.parameterization.gaff2 import balance_native_charges
    system, nb = mm.System(), mm.NonbondedForce()
    for q in [-.202, .2]:
        system.addParticle(12)
        nb.addParticle(q, .3, .2)
    nb.addException(0, 1, -.202*.2*5/6, .3, .1)
    system.addForce(nb)
    audit = balance_native_charges(system, 0)
    assert audit["atom_indices_zero_based"] == [0]
    assert audit["per_atom_delta_e"] == pytest.approx(.002)
    assert nb.getParticleParameters(0)[0].value_in_unit(u.elementary_charge) == pytest.approx(-.2)
    assert nb.getExceptionParameters(0)[2].value_in_unit(u.elementary_charge**2) == pytest.approx(-.2*.2*5/6)
    with pytest.raises(WorkflowError, match="residual"):
        balance_native_charges(system, 1)


def test_real_gaff2_output_regression(extension, repo, tmp_path, monkeypatch):
    from felis_workflows.parameterization import gaff2
    from felis_workflows.common import read
    source = extension / "tests/data/gaff2_ejm31"
    # Replay genuine ACPYPE outputs; no semiempirical executable is needed in CI.
    monkeypatch.setattr(gaff2.acpype_io, "run_acpype", lambda *args: {
        "acpdir": source, "itp": source / "LIG_GMX.itp", "top": source / "LIG_GMX.top",
        "gro": source / "LIG_GMX.gro"})
    destination = tmp_path / "parameters"
    gaff2.parameterize(repo / "examples/abfe/input/ejm_31.sdf", destination, 0)
    result = read(destination / "validation.json")
    assert result["passed"] and result["particles"] == 32
    assert max(x["energy_difference_kJ_mol"] for x in result["energy_force_checks"]) < .002
    assert result["counts"]["torsions"] == 91
