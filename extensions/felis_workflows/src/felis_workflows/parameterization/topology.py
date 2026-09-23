"""Topology and provenance helpers; no FELIS/OpenFF imports at module import."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

STEM = "ligand"
FF_NAME = "openff_unconstrained-2.3.0.offxml"
FF_SHA256 = "65b926cf38ae4bc2f6f9cac907c3cadba154cc84988dd10ab2c3895aaf2dd995"
MODEL = "openff-gnn-am1bcc-1.0.0.pt"
MODEL_SHA256 = "7981e7f5b0b1e424c9e10a40d9e7606d96dcd3dd2b095cb4eeff6829f92238ee"
DEFAULTS = "1 2 yes 0.5 0.8333333333"
PREP_STAGES = ["makebox", "boresch_em", "boresch_npt", "boresch_post_process", "sysA_em", "sysB_em"]

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()

def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)

def sections(text):
    result = []
    current, lines = None, []
    for line in text.splitlines(keepends=True):
        m = re.match(r"\s*\[\s*([^]]+?)\s*\]", line)
        if m:
            if lines:
                result.append((current, "".join(lines)))
            current, lines = m[1].lower(), [line]
        else:
            lines.append(line)
    if lines:
        result.append((current, "".join(lines)))
    return result

def rows(text):
    result = {}
    for name, block in sections(text):
        if name is None:
            continue
        for line in block.splitlines()[1:]:
            data = line.split(";", 1)[0].strip()
            if data and not data.startswith("#"):
                result.setdefault(name, []).append(data.split())
    return result

def make_itp(top_text, namespace="S23_", description="OpenFF Sage 2.3.0 + AshGC"):
    """Extract a single molecule without discarding any unsupported physics."""
    if any(line.lstrip().startswith("#") for line in top_text.splitlines()):
        raise ValueError("Expected a monolithic Interchange TOP without preprocessor directives")
    data = rows(top_text)
    defs = data.get("defaults", [])
    if len(defs) != 1 or len(defs[0]) != 5:
        raise ValueError("Missing or ambiguous GROMACS defaults")
    nb, comb, gen, lj, qq = defs[0]
    if (nb, comb, gen.lower()) != ("1", "2", "yes") or abs(float(lj)-.5)>1e-8 or abs(float(qq)-5/6)>1e-6:
        raise ValueError(f"Incompatible nonbonded defaults: {defs}")
    if len(data.get("moleculetype", [])) != 1 or len(data.get("molecules", [])) != 1:
        raise ValueError("Exactly one molecule type is required")
    if data["molecules"][0] != [data["moleculetype"][0][0], "1"]:
        raise ValueError("Expected one copy of the ligand")
    if data["moleculetype"][0][1] != "3":
        raise ValueError("Expected nrexcl=3")
    allowed = {None, "defaults", "atomtypes", "moleculetype", "atoms", "pairs", "bonds", "angles", "dihedrals", "exclusions", "system", "molecules"}
    # Interchange emits an empty [settles] header even for non-water molecules.
    empty_headers = {s for s, _ in sections(top_text) if not data.get(s)}
    unexpected = {s for s, _ in sections(top_text)} - allowed - empty_headers
    if unexpected:
        raise ValueError(f"Unsupported sections (not silently dropped): {unexpected}")
    if not data.get("atomtypes") or not data.get("bonds"):
        raise ValueError("Atom types and bonds are required")
    if any(r[4] not in {"1", "4", "9"} for r in data.get("dihedrals", [])):
        raise ValueError("Unexpected Sage torsion function")
    # Namespace the types before FELIS/ByteMol applies its own namespace.
    mapping = {r[0]: namespace + r[0] for r in data["atomtypes"]}
    output = [f"; {description}; complete unrestrained ligand parameters\n"]
    for name, block in sections(top_text):
        if name in {None, "defaults", "system", "molecules"} or name in empty_headers:
            continue
        output.append(f"\n[ {name} ]\n")
        for r in rows(block).get(name, []):
            if name == "atomtypes":
                r[0] = mapping[r[0]]
                if len(r) == 8 and r[1] in mapping:
                    r[1] = mapping[r[1]]
            elif name == "atoms":
                r[1] = mapping[r[1]]
                r[3] = "LIG"
            elif name == "moleculetype":
                r[0] = "LIG"
            output.append(" ".join(r) + "\n")
    return "".join(output)

def ligand_top(itp):
    return f'[ defaults ]\n{DEFAULTS}\n\n#include "{Path(itp).resolve()}"\n\n[ system ]\nSage validation\n\n[ molecules ]\nLIG 1\n'
