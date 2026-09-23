#!/usr/bin/env python3
"""Check native Sage against the FELIS-facing GROMACS topology in this environment."""
from __future__ import annotations
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
from .topology import STEM, ligand_top, rows
from .molecules import read_ligand
from ..common import sha256, verify_hashes, write as write_json

def load_export(directory, constraints=None):
    from openmm import app
    with tempfile.TemporaryDirectory(prefix="sage_validate_") as td:
        topfile = Path(td) / "ligand.top"
        topfile.write_text(ligand_top(Path(directory) / f"{STEM}.itp"))
        top = app.GromacsTopFile(str(topfile))
        system = top.createSystem(nonbondedMethod=app.NoCutoff, constraints=constraints,
                                  rigidWater=False, removeCMMotion=False)
    return top, system

def signatures(system, indices=None):
    """Index-normalized internal parameter multisets, including all exclusions."""
    import openmm as mm
    from openmm import unit as u
    if indices is None:
        indices = list(range(system.getNumParticles()))
    inverse = {int(i): j for j, i in enumerate(indices)}
    out = {"particles": [], "bonds": [], "angles": [], "torsions": [], "exceptions": []}
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    for i in indices:
        q, s, e = nb.getParticleParameters(int(i))
        out["particles"].append([q.value_in_unit(u.elementary_charge), s.value_in_unit(u.nanometer),
                                  e.value_in_unit(u.kilojoule_per_mole), system.getParticleMass(int(i)).value_in_unit(u.dalton)])
    def add(kind, ids, vals):
        if all(int(i) in inverse for i in ids):
            local = tuple(inverse[int(i)] for i in ids)
            out[kind].append([list(min(local, local[::-1])), list(vals)])
    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce):
            for n in range(f.getNumExceptions()):
                i,j,q,s,e=f.getExceptionParameters(n)
                q=q.value_in_unit(u.elementary_charge**2);e=e.value_in_unit(u.kilojoule_per_mole)
                # Sigma has no physical effect in a fully excluded pair.
                add("exceptions",(i,j),(q,s.value_in_unit(u.nanometer) if e else 0.,e))
        elif isinstance(f, mm.HarmonicBondForce):
            for n in range(f.getNumBonds()):
                i,j,r,k=f.getBondParameters(n)
                add("bonds",(i,j),(r.value_in_unit(u.nanometer),k.value_in_unit(u.kilojoule_per_mole/u.nanometer**2)))
        elif isinstance(f, mm.HarmonicAngleForce):
            for n in range(f.getNumAngles()):
                i,j,k,a,c=f.getAngleParameters(n)
                add("angles",(i,j,k),(a.value_in_unit(u.radian),c.value_in_unit(u.kilojoule_per_mole/u.radian**2)))
        elif isinstance(f, mm.PeriodicTorsionForce):
            for n in range(f.getNumTorsions()):
                i,j,k,l,p,a,c=f.getTorsionParameters(n)
                # ParmEd omits exactly zero-amplitude Amber torsions. They add
                # neither energy nor force; retain every nonzero term.
                if c.value_in_unit(u.kilojoule_per_mole) == 0:
                    continue
                add("torsions",(i,j,k,l),(p,a.value_in_unit(u.radian),c.value_in_unit(u.kilojoule_per_mole)))
        elif not isinstance(f, (mm.CMMotionRemover, mm.MonteCarloBarostat)):
            raise ValueError(f"Unexpected force type {type(f).__name__}")
    for name in out:
        if name != "particles":
            out[name].sort(key=lambda x:(x[0],x[1]))
    return out

def compare_signatures(a,b,*,allow_mass_rounding=False):
    reference=np.asarray(a["particles"],dtype=float)
    assembled=np.asarray(b["particles"],dtype=float)
    if reference.shape!=assembled.shape or reference.ndim!=2 or reference.shape[1]!=4:
        raise ValueError(f"Particle parameter shape mismatch: {reference.shape} vs {assembled.shape}")
    matches=np.isclose(reference,assembled,atol=2e-5,rtol=1e-7)
    matches &= np.isfinite(reference) & np.isfinite(assembled)
    if allow_mass_rounding:
        # The assembled topology writes masses to four decimal places. Accept
        # that exact serialization, rather than increasing every tolerance.
        # The native OpenFF/export comparison keeps the original strict check.
        rounded=np.round(reference[:,3],4)
        matches[:,3] |= (np.isfinite(rounded) & np.isfinite(assembled[:,3]) &
                         np.isclose(rounded,assembled[:,3],atol=1e-10,rtol=0))
    if not np.all(matches):
        labels=("charge (e)","sigma (nm)","epsilon (kJ/mol)","mass (Da)")
        bad=np.argwhere(~matches)
        details=[f"ligand atom {i+1} {labels[j]}: reference={reference[i,j]:.12g}, "
                 f"assembled={assembled[i,j]:.12g}, delta={assembled[i,j]-reference[i,j]:+.8g}"
                 for i,j in bad[:8]]
        raise ValueError(f"Per-atom parameters changed during export/build ({len(bad)} mismatches): "
                         + "; ".join(details))
    for key in ("bonds","angles","torsions","exceptions"):
        if len(a[key]) != len(b[key]):
            raise ValueError(f"{key} count changed: {len(a[key])} vs {len(b[key])}")
        for x,y in zip(a[key],b[key]):
            if x[0] != y[0] or not np.allclose(x[1],y[1],atol=2e-5,rtol=2e-7):
                raise ValueError(f"{key} mismatch: {x} vs {y}")

def evaluate(system, positions):
    import openmm as mm
    from openmm import unit as u
    itg=mm.VerletIntegrator(.001*u.picoseconds)
    ctx=mm.Context(system,itg,mm.Platform.getPlatformByName("Reference"))
    out=[]
    for xyz in positions:
        ctx.setPositions(xyz*u.nanometer)
        s=ctx.getState(energy=True,forces=True)
        out.append((s.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole),
                    s.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole/u.nanometer)))
    del ctx,itg
    return out

def validate(directory, check_hashes=True):
    import openmm as mm
    from openmm import app
    d=Path(directory)
    metadata=json.loads((d/"parameters.json").read_text())
    if check_hashes:
        verify_hashes(d,metadata["hashes"])
    mol,xyz=read_ligand(d/f"{STEM}.sdf", metadata["formal_charge"])
    data=rows((d/f"{STEM}.itp").read_text())
    graph={tuple(sorted((b.GetBeginAtomIdx()+1,b.GetEndAtomIdx()+1))) for b in mol.GetBonds()}
    if graph != {tuple(sorted(map(int,r[:2]))) for r in data.get("bonds",[])}:
        raise ValueError("SDF/ITP indexed bond graphs differ")
    top,export=load_export(d)
    if [a.element.atomic_number for a in top.topology.atoms()] != [a.GetAtomicNum() for a in mol.GetAtoms()]:
        raise ValueError("SDF/OpenMM atom order differs")
    native=mm.XmlSerializer.deserialize((d/"native_system.xml").read_text())
    if native.getNumConstraints() or export.getNumConstraints():
        raise ValueError("Validation requires unconstrained ligand systems")
    sa,sb=signatures(native),signatures(export)
    compare_signatures(sa,sb)
    if abs(sum(p[0] for p in sa["particles"])-metadata["formal_charge"])>1e-4:
        raise ValueError("Native ligand net charge disagrees with the declared formal charge")
    rng=np.random.default_rng(20260915)
    poses=[xyz/10,xyz/10+rng.normal(0,.001,xyz.shape),xyz/10+rng.normal(0,.002,xyz.shape)]
    errors=[]
    for a,b in zip(evaluate(native,poses),evaluate(export,poses)):
        de=abs(a[0]-b[0]);df=float(np.max(np.abs(a[1]-b[1])))
        if not np.isfinite([de,df]).all() or de>.002 or df>.02:
            raise ValueError(f"Native/export energy or force mismatch: {de} kJ/mol, {df} kJ/mol/nm")
        errors.append({"energy_difference_kJ_mol":de,"max_force_difference_kJ_mol_nm":df})
    _,constrained=load_export(d,app.HBonds)
    expected=sum(b.GetBeginAtom().GetAtomicNum()==1 or b.GetEndAtom().GetAtomicNum()==1 for b in mol.GetBonds())
    if constrained.getNumConstraints()!=expected:
        raise ValueError("FELIS HBonds constraint count differs from the SDF")
    return {"passed":True,"openmm_version":mm.__version__,"particles":mol.GetNumAtoms(),
            "hydrogen_constraints":expected,"counts":{k:len(v) for k,v in sa.items()},"energy_force_checks":errors}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("parameters",type=Path);p.add_argument("--report",type=Path)
    a=p.parse_args();result=validate(a.parameters)
    if a.report:write_json(a.report,result)
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
