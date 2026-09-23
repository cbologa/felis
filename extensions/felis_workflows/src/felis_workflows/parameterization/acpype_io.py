"""ACPYPE output helpers adapted from the user-provided ff archive."""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem


Block = Tuple[Optional[str], str]


COMMON_GAFF_ELEMENT_PREFIXES = (
    ("cl", "Cl"),
    ("br", "Br"),
    ("c", "C"),
    ("n", "N"),
    ("o", "O"),
    ("s", "S"),
    ("p", "P"),
    ("h", "H"),
    ("f", "F"),
    ("i", "I"),
)


MASS_TO_ELEMENT = [
    ("H", 0.5, 1.6),
    ("B", 10.0, 11.5),
    ("C", 11.5, 13.5),
    ("N", 13.5, 15.5),
    ("O", 15.5, 17.5),
    ("F", 18.0, 20.5),
    ("P", 30.0, 31.8),
    ("S", 31.8, 33.5),
    ("Cl", 34.0, 37.5),
    ("Br", 78.0, 81.5),
    ("I", 126.0, 128.5),
]


def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def run(cmd: Sequence[str], *, cwd: Optional[Path] = None) -> None:
    print("[run]", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), cwd=str(cwd) if cwd else None, check=True)


def run_acpype(sdf: Path, charge: int, atom_type: str, workdir: Path) -> Dict[str, Path]:
    base = "LIG"
    cmd = [
        "acpype",
        "-i", str(sdf.resolve()),
        "-b", base,
        "-c", "bcc",
        "-a", atom_type,
        "-n", str(charge),
        "-o", "all",
    ]
    run(cmd, cwd=workdir)

    acpdir = workdir / f"{base}.acpype"
    if not acpdir.is_dir():
        die(f"ACPYPE did not create expected directory: {acpdir}")

    files: Dict[str, Path] = {
        "acpdir": acpdir,
        "itp": acpdir / f"{base}_GMX.itp",
        "top": acpdir / f"{base}_GMX.top",
        "gro": acpdir / f"{base}_GMX.gro",
    }

    mol2_candidates = sorted(acpdir.glob("*.mol2"))
    files["mol2"] = mol2_candidates[0] if mol2_candidates else Path("")

    for key in ("itp", "top"):
        if not files[key].exists():
            die(f"ACPYPE did not produce {files[key]}")

    return files


def read_blocks(path: Path) -> List[Block]:
    """Split a GROMACS .top/.itp into ordered (section_header, raw_text) chunks."""
    chunks: List[Block] = []
    header: Optional[str] = None
    buf: List[str] = []

    with path.open() as handle:
        for line in handle:
            match = re.match(r"\s*\[\s*([^\]]+?)\s*\]", line)
            if match:
                if header is not None or buf:
                    chunks.append((header, "".join(buf)))
                header = match.group(1).strip().lower()
                buf = [line]
            else:
                buf.append(line)

    if header is not None or buf:
        chunks.append((header, "".join(buf)))

    return chunks


def section_text(chunks: Iterable[Block], section: str) -> str:
    section = section.lower()
    return "".join(text for header, text in chunks if header == section)


def assert_amber_defaults(top_path: Path) -> None:
    """Require AMBER/GAFF GROMACS defaults in ACPYPE's top file."""
    in_defaults = False

    with top_path.open() as handle:
        for raw in handle:
            sec_match = re.match(r"\s*\[\s*([^\]]+?)\s*\]", raw)
            if sec_match:
                in_defaults = sec_match.group(1).strip().lower() == "defaults"
                continue

            if not in_defaults:
                continue

            fields = raw.split(";", 1)[0].split()
            if not fields:
                continue
            if len(fields) < 5:
                continue

            try:
                nbfunc = fields[0]
                comb_rule = fields[1]
                gen_pairs = fields[2]
                fudge_lj = float(fields[3])
                fudge_qq = float(fields[4])
            except ValueError as exc:
                die(f"Could not parse [ defaults ] line in {top_path}: {raw.rstrip()} ({exc})")

            if nbfunc != "1":
                die(f"[ defaults ] nbfunc={nbfunc}; expected 1 for LJ+Coulomb GROMACS topology.")
            if comb_rule != "2":
                die(f"[ defaults ] comb-rule={comb_rule}; expected 2 (sigma/epsilon, Lorentz-Berthelot).")
            if gen_pairs.lower() not in {"yes", "no"}:
                die(f"[ defaults ] gen-pairs={gen_pairs}; expected yes/no.")
            if abs(fudge_lj - 0.5) > 1e-3 or abs(fudge_qq - (5.0 / 6.0)) > 1e-3:
                die(
                    "[ defaults ] fudgeLJ/fudgeQQ "
                    f"= {fudge_lj}/{fudge_qq}; expected 0.5 / 0.833333 for AMBER/GAFF."
                )

            print(
                "[check] AMBER/GAFF defaults OK: "
                f"nbfunc={nbfunc} comb-rule={comb_rule} gen-pairs={gen_pairs} "
                f"fudgeLJ={fudge_lj} fudgeQQ={fudge_qq}"
            )
            return

    die(f"No [ defaults ] data line found in {top_path}")


def build_itp(files: Dict[str, Path], out_itp: Path) -> None:
    """Build one Felis-facing ligand ITP containing [atomtypes] + molecule blocks."""
    itp_chunks = read_blocks(files["itp"])
    top_chunks = read_blocks(files["top"])

    # ACPYPE commonly keeps [ atomtypes ] in the ITP; older/variant outputs may
    # put it in the TOP. We always write it to the final ITP exactly once.
    atomtypes_txt = section_text(itp_chunks, "atomtypes")
    if not atomtypes_txt:
        atomtypes_txt = section_text(top_chunks, "atomtypes")

    if not atomtypes_txt:
        die(f"Could not find [ atomtypes ] in {files['itp']} or {files['top']}")

    drop_sections = {"defaults", "atomtypes", "system", "molecules"}
    molecule_txt = "".join(text for header, text in itp_chunks if header not in drop_sections)

    if "[ moleculetype ]" not in molecule_txt:
        die(f"No [ moleculetype ] block found in ACPYPE ITP: {files['itp']}")
    if "[ atoms ]" not in molecule_txt:
        die(f"No [ atoms ] block found in ACPYPE ITP: {files['itp']}")

    out_itp.parent.mkdir(parents=True, exist_ok=True)
    out_itp.write_text(
        "; Felis-ready GAFF/GAFF2 + AM1-BCC ligand topology assembled from ACPYPE\n"
        "; Generated by felis_gaff2_ligand.py\n\n"
        + atomtypes_txt.rstrip()
        + "\n\n"
        + molecule_txt.lstrip("\n")
    )

    print(f"[write] {out_itp}")
