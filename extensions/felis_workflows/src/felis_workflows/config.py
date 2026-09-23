"""Strict schemas: deployment configuration cannot override scientific choices."""
from __future__ import annotations
from copy import deepcopy
import math
import os
import re
from pathlib import Path

from .common import WorkflowError, identifier, keys, positive_int, read


def campaign_config(path):
    c = read(path)
    keys(c, {"schema_version", "name", "description", "receptor", "ligands", "calculations", "coupling"},
         {"schema_version", "name", "receptor", "ligands", "calculations"}, "campaign")
    if c["schema_version"] != 1:
        raise WorkflowError("Unsupported campaign schema")
    identifier(c["name"])
    r = c["receptor"]
    keys(r, {"pdb", "gro", "top", "include_dirs", "caps", "expected_chains", "remove_hetatm",
             "histidine_overrides", "disulfides", "description"}, context="receptor")
    if not (("pdb" in r and not ({"gro", "top"} & r.keys())) or ("pdb" not in r and {"gro", "top"} <= r.keys())):
        raise WorkflowError("Receptor requires either pdb or both gro and top")
    if "pdb" in r:
        if r.get("caps") not in {"ace_nme", "charged"}:
            raise WorkflowError("PDB preparation requires explicit caps: ace_nme or charged")
        positive_int(r.get("expected_chains"), "expected_chains")
        d = r.get("disulfides")
        keys(d, {"mode", "expected", "cutoff_angstrom", "pairs"}, {"mode", "expected"}, "disulfides")
        if d["mode"] not in {"detect", "explicit", "none"} or type(d["expected"]) is not int or d["expected"] < 0:
            raise WorkflowError("Invalid disulfide specification")
        if d["mode"] == "explicit" and len(d.get("pairs", [])) != d["expected"]:
            raise WorkflowError("Explicit disulfide pairs/count disagree")
        if d["mode"] == "none" and d["expected"]:
            raise WorkflowError("disulfides mode none requires expected: 0")
        if not 0 < d.get("cutoff_angstrom", 2.5) <= 3.0:
            raise WorkflowError("Disulfide cutoff must be in (0, 3] Angstrom")
    if not isinstance(c["ligands"], dict) or not c["ligands"]:
        raise WorkflowError("At least one ligand is required")
    for name, ligand in c["ligands"].items():
        identifier(name)
        keys(ligand, {"sdf", "formal_charge", "description"}, {"sdf", "formal_charge"}, f"ligand {name}")
        if type(ligand["formal_charge"]) is not int:
            raise WorkflowError("formal_charge must be an integer")
    ids = set()
    for calc in c["calculations"]:
        keys(calc, {"id", "target", "partners", "partner_site_max_distance_angstrom"},
             {"id", "target", "partners"}, "calculation")
        identifier(calc["id"])
        if calc["id"] in ids:
            raise WorkflowError("Duplicate calculation id")
        ids.add(calc["id"])
        partners = calc["partners"]
        if not isinstance(partners, list) or len(partners) != len(set(partners)):
            raise WorkflowError("partners must be a unique list")
        if calc["target"] in partners or set([calc["target"], *partners]) - c["ligands"].keys():
            raise WorkflowError("Unknown or duplicated target/partner")
        if partners and not 0 < calc.get("partner_site_max_distance_angstrom", 0) <= 20:
            raise WorkflowError("Partner calculations require an explicit site-distance threshold (0,20] Angstrom")
    if not ids:
        raise WorkflowError("No calculations specified")
    if c.get("coupling"):
        q = c["coupling"]
        keys(q, {"L", "P", "L_in_R", "P_in_R", "L_in_RP", "P_in_RL"},
             {"L", "P", "L_in_R", "P_in_R", "L_in_RP", "P_in_RL"}, "coupling")
        if q["L"] == q["P"] or len({q[k] for k in ("L_in_R", "P_in_R", "L_in_RP", "P_in_RL")}) != 4:
            raise WorkflowError("Coupling needs distinct ligands and four distinct calculations")
        byid = {v["id"]: v for v in c["calculations"]}
        for label, target, partners in [("L_in_R", q["L"], []), ("P_in_R", q["P"], []),
                                        ("L_in_RP", q["L"], [q["P"]]), ("P_in_RL", q["P"], [q["L"]])]:
            v = byid.get(q[label], {})
            if v.get("target") != target or v.get("partners") != partners:
                raise WorkflowError(f"Invalid thermodynamic endpoint for {label}")
    return c


def forcefield_config(path):
    f = read(path)
    keys(f, {"schema_version", "name", "ligand", "protein", "water", "ions"},
         {"schema_version", "name", "ligand", "protein", "water", "ions"}, "forcefield")
    keys(f["ligand"], {"backend", "version", "charge_model"}, {"backend", "version", "charge_model"}, "ligand forcefield")
    supported = {("gaff2", "2", "am1-bcc"), ("sage", "2.3.0", "ashgc")}
    if tuple(f["ligand"][k] for k in ("backend", "version", "charge_model")) not in supported:
        raise WorkflowError("Supported ligand models: GAFF2/AM1-BCC and Sage 2.3.0/AshGC")
    if f["schema_version"] != 1 or f["protein"] != "ff14SB" or f["water"] != "upstream-tip3p" or f["ions"] != "upstream":
        raise WorkflowError("This version preserves the pinned FELIS ff14SB/TIP3P/ion protocol")
    return f


def protocol_config(path):
    p = read(path)
    allowed = {"schema_version", "name", "purpose", "replicates", "seed", "temperature_K", "timestep_fs",
               "steps_per_iteration", "solvent_ns", "complex_ns", "equilibration_ns", "checkpoint_interval",
               "electrostatics", "vdw", "restraints", "groups_solvent", "groups_complex", "ionic_strength_M",
               "restraint_constants", "reuse_solvent"}
    keys(p, allowed, allowed, "protocol")
    if p["schema_version"] != 1 or p["purpose"] not in {"smoke", "validation", "production"}:
        raise WorkflowError("Invalid protocol version/purpose")
    # The upstream Boresch preparation and correction fix this temperature.
    # Do not expose an apparently configurable temperature that only changes one leg.
    if p["temperature_K"] != 298.15 or p["timestep_fs"] != 2 or p["steps_per_iteration"] != 2500:
        raise WorkflowError("Pinned FELIS requires 298.15 K, 2 fs and 2500 steps/ABFE iteration")
    for k in ["replicates", "seed", "checkpoint_interval", "groups_solvent", "groups_complex"]:
        positive_int(p[k], k)
    for k in ["solvent_ns", "complex_ns", "equilibration_ns"]:
        v = p[k] / 0.005
        if not math.isfinite(v) or v <= 0 or abs(v - round(v)) > 1e-7:
            raise WorkflowError(f"{k} must be a positive multiple of 0.005 ns")
    if p["equilibration_ns"] <= 0.5:
        raise WorkflowError("Boresch preparation discards 100 iterations; equilibration must exceed 0.5 ns")
    if type(p["reuse_solvent"]) is not bool or not math.isfinite(p["ionic_strength_M"]) or p["ionic_strength_M"] < 0:
        raise WorkflowError("Invalid solvent sharing or ionic strength")
    if len(p["restraint_constants"]) != 3 or any(not math.isfinite(v) or v <= 0 for v in p["restraint_constants"]):
        raise WorkflowError("Three positive restraint constants are required")
    return p


def site_config(path):
    s = read(path)
    keys(s, {"schema_version", "name", "backend", "repo", "bootstrap", "python", "resources", "slurm", "local", "mpi", "mps"},
         {"schema_version", "name", "backend", "repo", "python", "resources", "mpi", "mps"}, "site")
    if s["schema_version"] != 1 or s["backend"] not in {"local", "slurm"}:
        raise WorkflowError("Supported backends: local, slurm")
    identifier(s["name"])
    for key in ("repo", "bootstrap"):
        if s.get(key):
            s[key] = os.path.expandvars(os.path.expanduser(s[key]))
            if "$" in s[key]:
                raise WorkflowError(f"Unresolved environment variable in site.{key}: {s[key]}")
            s[key] = str((Path(path).resolve().parent / s[key]).resolve())
    keys(s["python"], {"simulation", "receptor", "gaff2", "sage"}, {"simulation", "receptor", "gaff2", "sage"}, "python runners")
    for command in s["python"].values():
        if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
            raise WorkflowError("Each Python runner must be a nonempty argv list ending in python")
    keys(s["mpi"], {"command", "ranks", "extra_args"}, {"command", "ranks", "extra_args"}, "MPI")
    positive_int(s["mpi"]["ranks"], "MPI ranks")
    if not isinstance(s["mpi"]["extra_args"], list) or type(s["mps"]) is not bool:
        raise WorkflowError("Invalid MPI/MPS settings")
    keys(s["resources"], {"prep", "array", "analysis"}, {"prep", "array", "analysis"}, "resources")
    for stage, r in s["resources"].items():
        keys(r, {"cpus", "memory", "walltime"}, {"cpus", "memory", "walltime"}, f"resources.{stage}")
        positive_int(r["cpus"], "cpus")
        if not all(isinstance(r[k], str) and r[k] for k in ("memory", "walltime")):
            raise WorkflowError("Memory and walltime must be nonempty strings")
    if s["backend"] == "slurm":
        t = s.get("slurm")
        keys(t, {"account", "partition", "prep_partition", "analysis_partition", "gpu_args", "array_concurrency", "prep_concurrency", "extra_args"},
             {"partition", "analysis_partition", "gpu_args", "array_concurrency", "prep_concurrency"}, "slurm")
        for k in ("partition", "analysis_partition", "prep_partition"):
            if k == "prep_partition" and k not in t:
                continue
            if not t[k] or "CHANGE_ME" in t[k]:
                raise WorkflowError(f"Set slurm.{k} to a partition you can use")
        for k in ("array_concurrency", "prep_concurrency"):
            positive_int(t[k], k)
        if len(t["gpu_args"]) != 1 or not re.fullmatch(r"(?:--gpus=(?:[A-Za-z0-9_-]+:)?1|--gres=gpu:(?:[A-Za-z0-9_-]+:)?1)", t["gpu_args"][0]):
            raise WorkflowError("gpu_args must explicitly request one GPU with --gpus= or --gres=")
        for arg in t.get("extra_args", []):
            if arg.split("=", 1)[0] not in {"--qos", "--constraint", "--exclude", "--reservation", "--requeue"}:
                raise WorkflowError(f"Unsupported Slurm extra argument: {arg}")
    else:
        keys(s.get("local"), {"gpus"}, {"gpus"}, "local")
        gpus = s["local"]["gpus"]
        if not isinstance(gpus, list) or not gpus or len(set(map(str, gpus))) != len(gpus):
            raise WorkflowError("local.gpus must list distinct GPU selectors")
    return deepcopy(s)
