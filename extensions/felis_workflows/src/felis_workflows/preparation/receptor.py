"""Explicit receptor chemistry; no target- or site-specific defaults."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import math
import re
import shutil
import subprocess
import sys

from ..common import WorkflowError, sha256, write
from ..planning import copy_topology
from . import caps

STANDARD = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL HID HIE HIP CYX ASH GLH LYN".split())


def residues(lines):
    result, seen, current = [], set(), None
    for line in lines:
        if not line.startswith("ATOM  "):
            continue
        key = line[21:22] + ":" + line[22:26].strip() + line[26:27].strip()
        if key != current:
            if key in seen:
                raise WorkflowError(f"Repeated/noncontiguous residue identifier: {key}")
            seen.add(key)
            result.append((key, []))
            current = key
        result[-1][1].append(line)
    return result


def clean_source(path, config):
    lines = Path(path).read_text().splitlines(keepends=True)
    if sum(line.startswith("MODEL") for line in lines) > 1:
        raise WorkflowError("Select one PDB model before preparation")
    allowed = set(config.get("remove_hetatm", []))
    actual = {line[17:20].strip() for line in lines if line.startswith("HETATM")}
    if actual - allowed:
        raise WorkflowError(f"Unspecified HETATM handling: {sorted(actual - allowed)}; use prepared GRO/TOP to retain cofactors/waters")
    out = []
    for line in lines:
        if line.startswith("TER"):
            if out and not out[-1].startswith("TER"):
                out.append("TER\n")
        elif line.startswith("ATOM  "):
            if line[16:17].strip():
                raise WorkflowError("Resolve PDB alternate locations before preparation")
            if line[17:20].strip() not in STANDARD:
                raise WorkflowError(f"Unsupported residue: {line[17:20]}")
            element = line[76:78].strip() if len(line) >= 78 else ""
            if element == "H" or (not element and line[12:16].strip().lstrip("0123456789").startswith("H")):
                continue
            if not line[21:22].strip():
                raise WorkflowError("Assign explicit chain identifiers")
            out.append(line)
    if not out:
        raise WorkflowError("No protein atoms")
    out.append("END\n")
    groups = residues(out)
    segments = caps.parse_segments(out)
    if len(segments) != config["expected_chains"]:
        raise WorkflowError("Protein segment count differs from expected_chains")
    return out, groups


def disulfide_pairs(groups, specification):
    sg = {}
    for key, atoms in groups:
        if atoms[0][17:20].strip() in {"CYS", "CYX"}:
            coords = [caps.xyz(line) for line in atoms if caps.atom_name(line) == "SG"]
            if len(coords) != 1:
                raise WorkflowError(f"Cysteine {key} must contain exactly one SG atom")
            sg[key] = coords[0]
    if specification["mode"] == "explicit":
        pairs = [tuple(p) for p in specification["pairs"]]
    elif specification["mode"] == "none":
        pairs = []
    else:
        import numpy as np
        names = list(sg)
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i+1:]
                 if np.linalg.norm(sg[a] - sg[b]) <= specification.get("cutoff_angstrom", 2.5)]
    flat = [key for pair in pairs for key in pair]
    if any(len(pair) != 2 for pair in pairs) or len(flat) != len(set(flat)):
        raise WorkflowError("Ambiguous disulfide pairing; supply explicit pairs")
    if any(key not in sg for key in flat) or len(pairs) != specification["expected"]:
        raise WorkflowError("Disulfide identities/count do not match the receptor specification")
    unpaired_cyx = [k for k, atoms in groups if atoms[0][17:20].strip() == "CYX" and k not in flat]
    if unpaired_cyx:
        raise WorkflowError(f"Unpaired CYX residues: {unpaired_cyx}")
    return pairs


def assign_chemistry(lines, groups, reduced, overrides, pairs):
    reduced_groups = residues(reduced)
    if len(groups) != len(reduced_groups):
        raise WorkflowError("Reduce changed residue count; inspect its output")
    states = {}
    for (key, atoms), (_, reduced_atoms) in zip(groups, reduced_groups):
        name, reduced_name = atoms[0][17:20].strip(), reduced_atoms[0][17:20].strip()
        equivalent = lambda x: "HIS" if x in {"HIS", "HID", "HIE", "HIP"} else "CYS" if x in {"CYX", "CYS"} else x
        if equivalent(name) != equivalent(reduced_name):
            raise WorkflowError("Reduce changed residue order/identity")
        if equivalent(name) == "HIS":
            names = {caps.atom_name(x) for x in reduced_atoms}
            states[key] = (name if name in {"HID", "HIE", "HIP"} else
                           "HIP" if {"HD1", "HE2"} <= names else "HID" if "HD1" in names else
                           "HIE" if "HE2" in names else None)
            if key in overrides:
                states[key] = overrides[key]
            if states[key] not in {"HID", "HIE", "HIP"}:
                raise WorkflowError(f"Explicit histidine state required for {key}")
    if set(overrides) - states.keys():
        raise WorkflowError("Histidine override refers to an unknown/non-histidine residue")
    states.update({key: "CYX" for pair in pairs for key in pair})
    output = []
    for line in lines:
        if line.startswith("ATOM  "):
            key = line[21:22] + ":" + line[22:26].strip() + line[26:27].strip()
            if key in states:
                line = line[:17] + f"{states[key]:>3s}" + line[20:]
        output.append(line)
    return output, states


def write_receptor(lines, policy, destination):
    import numpy as np
    output, mapping, serial, leap = [], {}, 1, 1
    for segment in caps.parse_segments(lines):
        groups = residues(segment)
        chain = segment[0][21:22]
        if policy == "ace_nme":
            groups[-1] = (groups[-1][0], [x for x in groups[-1][1] if caps.atom_name(x) != "OXT"])
            xyz = np.asarray([caps.xyz(x) for _, g in groups for x in g])
            ch3, carbon, oxygen, _ = caps.ace_geometry(groups[0][1], xyz)
            for name, pos, element in [("CH3", ch3, "C"), ("C", carbon, "C"), ("O", oxygen, "O")]:
                output.append(caps.pdb_new(serial, name, "ACE", chain, 1, pos, element)); serial += 1
            leap += 1
        for local, (key, atoms) in enumerate(groups, start=2 if policy == "ace_nme" else 1):
            mapping[key] = leap
            for atom in atoms:
                output.append(caps.rewrite_original(atom, serial, chain, local)); serial += 1
            leap += 1
        if policy == "ace_nme":
            nitrogen, carbon, _ = caps.nme_geometry(groups[-1][1], xyz)
            for name, pos, element in [("N", nitrogen, "N"), ("C", carbon, "C")]:
                output.append(caps.pdb_new(serial, name, "NME", chain, len(groups)+2, pos, element)); serial += 1
            leap += 1
        output.append("TER\n")
    output.append("END\n")
    Path(destination).write_text("".join(output))
    return mapping


def build(root, science):
    from openmm import app
    root = Path(root)
    output = root / "receptor"
    if (output / "validated.json").exists():
        from ..common import read, verify_hashes
        verify_hashes(output, read(output / "validated.json")["hashes"])
        return
    if output.exists():
        raise WorkflowError(f"Partial receptor build exists at {output}; inspect and move it aside before retrying")
    output.mkdir()
    config = science["campaign"]["receptor"]
    if "gro" in config:
        shutil.copyfile(root / config["gro"], output / "protein.gro")
        copy_topology(root / config["top"], output / "protein.top")
        audit = {"mode": "imported", "source": config, "chemistry": "as supplied; no atom or residue changes"}
    else:
        lines, groups = clean_source(root / config["pdb"], config)
        (output / "heavy.pdb").write_text("".join(lines))
        with (output / "reduce.log").open("w") as log:
            subprocess.run(["pdb4amber", "-i", "heavy.pdb", "-o", "reduced.pdb", "--dry", "--reduce", "--no-conect"],
                           cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
        pairs = disulfide_pairs(groups, config["disulfides"])
        named, states = assign_chemistry(lines, groups, (output / "reduced.pdb").read_text().splitlines(True),
                                         config.get("histidine_overrides", {}), pairs)
        mapping = write_receptor(named, config["caps"], output / "receptor.pdb")
        write(output / "residue_map.json", mapping)
        leap = ["source leaprc.protein.ff14SB", "receptor = loadpdb receptor.pdb"]
        leap += [f"bond receptor.{mapping[a]}.SG receptor.{mapping[b]}.SG" for a, b in pairs]
        leap += ["check receptor", "saveamberparm receptor receptor.prmtop receptor.inpcrd", "savepdb receptor receptor_final.pdb", "quit"]
        (output / "tleap.in").write_text("\n".join(leap) + "\n")
        with (output / "tleap.log").open("w") as log:
            subprocess.run(["tleap", "-f", "tleap.in"], cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
        text = (output / "tleap.log").read_text()
        if re.search(r"FATAL|does not have a type|Could not find.*parameter|Created a new atom named:|Errors\s*=\s*[1-9]", text, re.I):
            raise WorkflowError("LEaP chemistry error; inspect receptor/tleap.log")
        for line in text.splitlines():
            if "Added missing heavy atom:" in line and not (config["caps"] == "charged" and "OXT" in line):
                raise WorkflowError("LEaP invented a heavy atom; complete the source receptor first")
        import parmed
        structure = parmed.load_file(str(output / "receptor.prmtop"), str(output / "receptor.inpcrd"))
        counts = Counter(r.name for r in structure.residues)
        if config["caps"] == "ace_nme" and any(counts[name] != config["expected_chains"] for name in ["ACE", "NME"]):
            raise WorkflowError("Incorrect receptor cap count")
        structure.save(str(output / "protein.top"), format="gromacs")
        structure.save(str(output / "protein.gro"), format="gro")
        audit = {"mode": "built", "protein_forcefield": "ff14SB", "caps": config["caps"], "states": states,
                 "disulfides": pairs, "residue_map": mapping, "source": config}
    gro = app.GromacsGroFile(str(output / "protein.gro"))
    top = app.GromacsTopFile(str(output / "protein.top"))
    system = top.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)
    if system.getNumParticles() != len(gro.positions):
        raise WorkflowError("Receptor coordinate/topology atom count mismatch")
    write(output / "audit.json", {**audit, "particles": system.getNumParticles()})
    write(output / "validated.json", {"hashes": {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()}})
