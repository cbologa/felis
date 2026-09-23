from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import re
import shutil

from . import UPSTREAM_COMMIT, __version__
from .common import WorkflowError, digest, file_hashes, read, sha256, verify_hashes, write
from .config import campaign_config, forcefield_config, protocol_config
from .integrity import verify_upstream


def default_repo():
    return Path(os.environ.get("FELIS_REPO", Path(__file__).resolve().parents[4])).resolve()


def copy_topology(source, destination, include_dirs=()):
    """Freeze the complete include graph using relative, relocatable references."""
    destination = Path(destination)
    memo = {}
    def visit(src, dst):
        src, dst = Path(src).resolve(), Path(dst).resolve()
        if src in memo:
            return memo[src]
        if src.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise WorkflowError(f"Unmaterialized Git LFS input: {src}")
        memo[src] = dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for line in src.read_text().splitlines(keepends=True):
            match = re.match(r'\s*#\s*include\s*["<]([^">]+)[">]', line)
            if not match:
                if re.match(r"\s*#\s*include\b", line):
                    raise WorkflowError("Macro topology includes must be resolved before planning")
                lines.append(line)
                continue
            candidates = [src.parent / match[1], *(Path(p) / match[1] for p in include_dirs)]
            found = next((p for p in candidates if p.is_file()), None)
            if found is None:
                raise WorkflowError(f"Missing include {match[1]} from {src}")
            # A traversal index preserves include context without putting a
            # machine-specific source path into the scientific snapshot.
            key = f"{len(memo):04d}"
            child = visit(found, destination.parent / "includes" / f"{key}_{found.name}")
            lines.append(f'#include "{os.path.relpath(child, dst.parent)}"\n')
        dst.write_text("".join(lines))
        return dst
    return visit(source, destination)


def recipes(repo):
    path = Path(repo) / "felis/protocols/abfe/recipes.py"
    spec = importlib.util.spec_from_file_location("_pinned_felis_recipes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lambdas(repo, p):
    r = recipes(repo)
    if p["electrostatics"] not in r.electrostatic_lambda_recipes or p["vdw"] not in r.vdw_lambda_recipes or p["restraints"] not in r.restraint_lambda_recipes:
        raise WorkflowError("Unknown pinned FELIS lambda recipe")
    ve = r.get_ve_lambdas_with_res
    a = ve(p["electrostatics"], "e", False, 0.0, None) + ve(p["vdw"], "v", False, 0.0, None)[1:]
    b = (ve(p["vdw"], "v", True, 1.0, None) + ve(p["electrostatics"], "e", True, 1.0, None)[1:]
         + r.get_r_lambdas_dim3(p["restraints"], False, 1.0, 1.0, None)[1:])
    result = {}
    for leg, ladder, count in [("A", a, p["groups_solvent"]), ("B", b, p["groups_complex"])]:
        groups = r.split_replica_exchange_jobs(count, list(range(len(ladder))))
        edges = [edge for group in groups for edge in zip(group, group[1:])]
        if edges != list(zip(range(len(ladder) - 1), range(1, len(ladder)))):
            raise WorkflowError("Lambda partition must cover each adjacent transition exactly once")
        result[leg] = {"lambdas": ladder, "groups": groups}
    return result


def seed_for(base, *labels):
    return int(digest([base, *labels])[:15], 16) % 2147483646 + 1


def plan(campaign, forcefield, protocol, output, repo=None):
    repo = Path(repo or default_repo()).resolve()
    verify_upstream(repo)
    c, f, p = campaign_config(campaign), forcefield_config(forcefield), protocol_config(protocol)
    ladder = lambdas(repo, p)
    root = Path(output).resolve()
    if root.exists():
        raise WorkflowError(f"Use a new output directory: {root}")
    root.mkdir(parents=True)
    base = Path(campaign).resolve().parent
    sources = {}
    def snapshot(name, value):
        src = (base / value).resolve()
        if not src.is_file():
            raise WorkflowError(f"Missing input {src}; configure the campaign input paths")
        if src.read_bytes()[:80].startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise WorkflowError(f"Fetch this Git LFS object first: {src}")
        dst = root / "inputs" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        sources[str(dst.relative_to(root))] = str(src)
        return str(dst.relative_to(root))
    r = c["receptor"]
    if "pdb" in r:
        r["pdb"] = snapshot("receptor.pdb", r["pdb"])
    else:
        r["gro"] = snapshot("receptor.gro", r["gro"])
        source = (base / r["top"]).resolve()
        includes = [(base / v).resolve() for v in r.pop("include_dirs", [])]
        destination = root / "inputs/receptor.top"
        copy_topology(source, destination, includes)
        r["top"] = str(destination.relative_to(root))
        sources[r["top"]] = str(source)
    for name, lig in c["ligands"].items():
        lig["sdf"] = snapshot(f"ligands/{name}.sdf", lig["sdf"])
    calculations = []
    for replica in range(1, p["replicates"] + 1):
        owners = {}
        # Prefer a binary system as solvent owner when sharing is enabled.
        for calc in sorted(c["calculations"], key=lambda v: (len(v["partners"]), v["id"])):
            key = f'{calc["id"]}/r{replica}'
            owner = owners.setdefault(calc["target"], key) if p["reuse_solvent"] else key
            seeds = {leg: [seed_for(p["seed"], owner if leg == "A" else key, leg, i)
                           for i in range(len(ladder[leg]["groups"]))] for leg in "AB"}
            calculations.append({**calc, "key": key, "replica": replica, "solvent_owner": owner,
                                 "seeds": seeds, "prep_seed": seed_for(p["seed"], key, "prep")})
    value = {"schema_version": 1, "workflow_version": __version__, "upstream_commit": UPSTREAM_COMMIT,
             "campaign": c, "forcefield": f, "protocol": p, "ladders": ladder, "calculations": calculations,
             "input_hashes": file_hashes(root, ["inputs"])}
    write(root / "science.json", value)
    write(root / "science.lock.json", {"sha256": sha256(root / "science.json")})
    write(root / "input_origins.json", sources)
    return {"run": str(root), "calculations": len(calculations), "science_id": digest(value),
            "solvent_groups": len(ladder["A"]["groups"]), "complex_groups": len(ladder["B"]["groups"])}


def load_run(root, prepared=False):
    root = Path(root).resolve()
    if sha256(root / "science.json") != read(root / "science.lock.json")["sha256"]:
        raise WorkflowError("Scientific manifest changed; create a new plan")
    s = read(root / "science.json")
    if s["upstream_commit"] != UPSTREAM_COMMIT or s["workflow_version"] != __version__:
        raise WorkflowError("Workflow/upstream version mismatch")
    verify_hashes(root, s["input_hashes"])
    if prepared:
        ready = read(root / "prepared.json")
        if ready["science_id"] != digest(s):
            raise WorkflowError("Preparation belongs to a different science plan")
        verify_hashes(root, ready["hashes"])
    return s


def calculation(s, key):
    return next(v for v in s["calculations"] if v["key"] == key)


def workdir(root, calc):
    return Path(root) / "calculations" / calc["key"] / "work" / calc["target"]


def abfe_config(root, s, calc):
    root = Path(root).resolve()
    p = s["protocol"]
    return {"progro": str(root / "receptor/protein.gro"), "protop": str(root / "receptor/protein.top"),
            "sdffile": str(root / "parameters" / calc["target"] / "ligand.sdf"),
            "itpfile": str(root / "parameters" / calc["target"] / "ligand.itp"),
            "cofsdfs": [str(root / "parameters" / v / "ligand.sdf") for v in calc["partners"]],
            "cofitps": [str(root / "parameters" / v / "ligand.itp") for v in calc["partners"]],
            "outdir": str(workdir(root, calc).parent), "stages": ["all"],
            "elamrecipe": p["electrostatics"], "vlamrecipe": p["vdw"], "reslamrecipe": p["restraints"],
            "md_sol_nsnapshots": round(p["solvent_ns"] / .005), "md_pro_nsnapshots": round(p["complex_ns"] / .005),
            "md_eq_nsnapshots": round(p["equilibration_ns"] / .005), "md_checkpoint_interval": p["checkpoint_interval"],
            "pro_ionic_strength": p["ionic_strength_M"], "k_r_a_dih": p["restraint_constants"]}


def units(root, s, calc, leg):
    p = s["protocol"]
    result = []
    for i, group in enumerate(s["ladders"][leg]["groups"]):
        cfg = [f"prepare/sys{leg}_lam.json", f"prepare/sys{leg}_ab_ligatoms.json"]
        if leg == "B":
            cfg.append("prepare/sys_boresch_cfg.json")
        # Each calculation receives its own target-named SDF so FELIS's stem
        # convention agrees with the isolated calculation work directory.
        sdf = Path(root).resolve() / "calculations" / calc["key"] / "input" / f'{calc["target"]}.sdf'
        target = round(p["solvent_ns" if leg == "A" else "complex_ns"] / .005)
        stem = f"{leg.lower()}{i}"
        argv = ["-m", "felis.app.dyn.repex", "--cfg", *cfg, "--tkv",
                f"s:filename.stem:{stem}", f"s:filename.monomer:{sdf}",
                f"s:filename.sys:prepare/sys{leg}.top", f"s:filename.crd:prepare/sys{leg}_em.pdb",
                f"s:filename.atom_ids:prepare/sys{leg}_atom_ids.json", "s:dir.trj:trj",
                f'i:openmm.checkpoint_interval:{p["checkpoint_interval"]}', "i:integrator.npt:1",
                "i:integrator.nstep_per_snapshot:2500", f"i:integrator.nsnapshots:{target}",
                f'i:integrator.randomseed:{calc["seeds"][leg][i]}', "--rextkv", *[f"i:ab.ilam:{j}" for j in group]]
        result.append({"index": i, "stem": stem, "ilam": group, "iterations": target,
                       "checkpoint_interval": p["checkpoint_interval"], "argv": argv})
    return result
