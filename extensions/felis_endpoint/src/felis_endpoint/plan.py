"""Immutable, explicit physical-MD starts; no cluster or medoid selection."""
from __future__ import annotations

import math
from pathlib import Path
import shutil

from felis_workflows.artifacts import write_once
from felis_workflows.common import (WorkflowError, digest, file_hashes, identifier, keys,
                                    positive_int, read, sha256, verify_hashes, write)
from felis_workflows.config import site_config
from felis_workflows.planning import copy_topology, load_run, seed_for

WATERS = {"HOH", "SOL", "WAT", "TIP3", "TIP3P", "SPC", "SPCE"}
PROTEIN = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
           "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
           "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "ASH", "GLH"}


def finite_positive(value, label):
    # PyYAML loads unquoted scientific notation such as 2e-05 as a string.
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError as error:
            raise WorkflowError(f"{label} must be a positive finite number") from error
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise WorkflowError(f"{label} must be a positive finite number")
    return float(value)


def path_from_config(base, value, label):
    if not isinstance(value, str) or not value:
        raise WorkflowError(f"{label} must name a file")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def arithmetic(duration_ns, timestep_fs, interval_ps):
    steps = duration_ns * 1_000_000 / timestep_fs
    interval = interval_ps * 1_000 / timestep_fs
    if not steps.is_integer() or not interval.is_integer() or int(steps) % int(interval):
        raise WorkflowError("Duration and reporting interval must be exact multiples of the timestep")
    return int(steps), int(interval), int(steps / interval)


def compatible_atoms(pdb_path, top_path):
    """Read full topology and start geometry, checking order, elements and periodic solvent."""
    try:
        from openmm.app import GromacsTopFile, PDBFile
        pdb = PDBFile(str(pdb_path))
        box = pdb.topology.getPeriodicBoxVectors()
        if box is None:
            raise WorkflowError(f"Endpoint start has no periodic box: {pdb_path}")
        top = GromacsTopFile(str(top_path), periodicBoxVectors=box)
        pa, ta = list(pdb.topology.atoms()), list(top.topology.atoms())
        if len(pa) != len(ta) or not pa:
            raise WorkflowError("Endpoint topology and start atom counts differ")
        if not any(atom.residue.name.upper() in WATERS for atom in pa):
            raise WorkflowError("Endpoint start needs explicit solvating water")
        if not any(atom.residue.name.upper() in PROTEIN for atom in pa) or \
                not any(atom.residue.name.upper() not in WATERS | PROTEIN for atom in pa):
            raise WorkflowError("Endpoint start requires protein and separate ligand residues")
        for index, (a, b) in enumerate(zip(pa, ta)):
            water = a.residue.name.upper() in WATERS and b.residue.name.upper() in WATERS
            if ((a.element.symbol if a.element else None) != (b.element.symbol if b.element else None) or
                    (not water and (a.name != b.name or a.residue.name != b.residue.name)) or
                    (a.residue.name.upper() in WATERS) != (b.residue.name.upper() in WATERS)):
                raise WorkflowError(f"Endpoint topology/start atom {index} disagrees")
        return len(pa)
    except WorkflowError:
        raise
    except Exception as error:
        raise WorkflowError(f"Could not validate endpoint topology and PDB: {error}") from error


def frozen_start(config_dir, output, spec, source_run, science):
    keys(spec, {"id", "calculation", "topology", "include_dirs", "pdb", "trajectory", "trajectory_topology", "frame"},
         {"id", "calculation", "topology"}, "endpoint start")
    name = identifier(spec["id"])
    if spec["calculation"] not in {c["key"] for c in science["calculations"]}:
        raise WorkflowError(f"Unknown source calculation: {spec['calculation']}")
    from_pdb = "pdb" in spec
    if from_pdb == ("trajectory" in spec):
        raise WorkflowError("Specify exactly one PDB or trajectory/frame for each endpoint start")
    directory = output / "inputs" / name
    directory.mkdir(parents=True)
    top_src = path_from_config(config_dir, spec["topology"], "topology")
    if not top_src.is_file():
        raise WorkflowError(f"Missing endpoint topology: {top_src}")
    include_dirs = spec.get("include_dirs", [])
    if not isinstance(include_dirs, list) or any(not isinstance(v, str) for v in include_dirs):
        raise WorkflowError("include_dirs must be explicit directory paths")
    top_dst = copy_topology(top_src, directory / "system.top",
                            [path_from_config(config_dir, v, "topology include directory") for v in include_dirs])
    pdb_dst = directory / "start.pdb"
    if from_pdb:
        if set(spec) & {"frame", "trajectory_topology"}:
            raise WorkflowError("PDB starts cannot specify trajectory frame metadata")
        pdb_src = path_from_config(config_dir, spec["pdb"], "PDB start")
        if not pdb_src.is_file():
            raise WorkflowError(f"Missing endpoint PDB: {pdb_src}")
        original_hash = sha256(pdb_src)
        shutil.copyfile(pdb_src, pdb_dst)
        if sha256(pdb_src) != original_hash or sha256(pdb_dst) != original_hash:
            raise WorkflowError("PDB start changed while freezing endpoint inputs")
        source = {"kind": "pdb", "path": str(pdb_src), "source_sha256": original_hash}
    else:
        frame = spec.get("frame")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 or "trajectory_topology" not in spec:
            raise WorkflowError("Trajectory starts require an explicit nonnegative frame and topology PDB")
        trajectory = path_from_config(config_dir, spec["trajectory"], "source trajectory")
        if trajectory.suffix.lower() == ".nc":
            raise WorkflowError("Replica-exchange .nc needs thermodynamic-state mapping; extract an explicit PDB first")
        template = path_from_config(config_dir, spec["trajectory_topology"], "trajectory topology PDB")
        if not trajectory.is_file() or not template.is_file():
            raise WorkflowError("Source trajectory and topology PDB must exist")
        try:
            import mdtraj as md
            snapshot = md.load_frame(str(trajectory), frame, top=str(template))
            time_ps = float(snapshot.time[0]) if snapshot.time is not None else None
            snapshot.save_pdb(str(pdb_dst))
        except Exception as error:
            raise WorkflowError(f"Cannot freeze trajectory frame {frame}: {error}") from error
        source = {"kind": "trajectory_frame", "trajectory": str(trajectory), "frame": frame,
                  "time_ps": time_ps, "trajectory_topology": str(template),
                  "trajectory_topology_sha256": sha256(template)}
    atoms = compatible_atoms(pdb_dst, top_dst)
    frozen = file_hashes(output, [f"inputs/{name}"])
    return {"id": name, "calculation": spec["calculation"], "source_run": str(source_run),
            "source": source, "start_pdb": str(pdb_dst.relative_to(output)),
            "system_top": str(top_dst.relative_to(output)), "input_hashes": frozen,
            "topology_id": digest({k: v for k, v in frozen.items() if k != str(pdb_dst.relative_to(output))}),
            "start_sha256": frozen[str(pdb_dst.relative_to(output))], "atoms": atoms}


def plan(config, output, site_path):
    config, output = Path(config).resolve(), Path(output).resolve()
    value = read(config)
    keys(value, {"schema_version", "name", "source_run", "seed", "replicas", "duration_ns",
                 "report_interval_ps", "temperature_K", "timestep_fs", "pressure_bar", "starts"},
         {"schema_version", "name", "source_run", "seed", "replicas", "starts"}, "endpoint plan")
    if value["schema_version"] != 1:
        raise WorkflowError("Unsupported endpoint plan schema")
    name = identifier(value["name"])
    seed, replicas = positive_int(value["seed"], "endpoint seed"), positive_int(value["replicas"], "replicas")
    source_run = Path(value["source_run"]).expanduser()
    if not source_run.is_absolute():
        raise WorkflowError("source_run must be an absolute ABFE run path")
    source_run = source_run.resolve()
    science = load_run(source_run)
    duration = finite_positive(value.get("duration_ns", 50), "duration_ns")
    interval = finite_positive(value.get("report_interval_ps", 50), "report_interval_ps")
    temperature = finite_positive(value.get("temperature_K", 298.15), "temperature_K")
    timestep = finite_positive(value.get("timestep_fs", 2), "timestep_fs")
    pressure = finite_positive(value.get("pressure_bar", 1), "pressure_bar")
    steps, interval_steps, frames = arithmetic(duration, timestep, interval)
    starts = value["starts"]
    if not isinstance(starts, list) or not starts or any(not isinstance(s, dict) for s in starts) or \
            any(not isinstance(s.get("id"), str) for s in starts) or \
            len(starts) != len({s["id"] for s in starts}):
        raise WorkflowError("Endpoint starts must have distinct explicit IDs")
    site = site_config(site_path)
    if output.exists():
        raise WorkflowError(f"Use a new endpoint output directory: {output}")
    from .identity import runner_identity
    runtime = runner_identity(site_path, site)
    repo = Path(site["repo"]).resolve()
    if output.is_relative_to(repo):
        raise WorkflowError("Endpoint trajectories and checkpoints must be written outside the Git checkout")
    output.mkdir(parents=True)
    frozen = [frozen_start(config.parent, output, s, source_run, science) for s in starts]
    tasks = []
    for start in frozen:
        for replica in range(1, replicas + 1):
            labels = ("physical-endpoint", start["id"], start["start_sha256"], replica)
            tasks.append({"id": f'{start["id"]}/r{replica}', "start": start["id"], "replica": replica,
                          "seed": seed_for(seed, *labels, "initial-velocities"),
                          "integrator_seed": seed_for(seed, *labels, "langevin-integrator"),
                          "barostat_seed": seed_for(seed, *labels, "monte-carlo-barostat"),
                          "output": f'replicas/{start["id"]}/r{replica}'})
    stochastic_streams = [task[key] for task in tasks for key in ("seed", "integrator_seed", "barostat_seed")]
    if len(stochastic_streams) != len(set(stochastic_streams)):
        raise WorkflowError("Derived endpoint random streams collided; choose another endpoint seed")
    science_value = {"schema_version": 1, "name": name, "kind": "derived_physical_endpoint_md",
                     "source_run": str(source_run), "source_science_id": digest(science),
                     "protocol": {"ensemble": "NPT", "temperature_K": temperature, "pressure_bar": pressure,
                                  "timestep_fs": timestep, "duration_ns": duration,
                                  "report_interval_ps": interval, "steps": steps,
                                  "interval_steps": interval_steps, "expected_frames": frames,
                                  "alchemical": False, "boresch_restraints": False,
                                  "positional_restraints": False},
                     "endpoint_seed": seed, "replicas": replicas, "starts": frozen, "tasks": tasks,
                     "input_hashes": file_hashes(output, ["inputs"])}
    write_once(output / "plan.json", science_value)
    write_once(output / "plan.lock.json", {"sha256": sha256(output / "plan.json")})
    write_once(output / "runtime.json", runtime)
    write_once(output / "site.json", site)
    return {"endpoint_run": str(output), "starts": len(frozen), "replicas": replicas,
            "tasks": len(tasks), "expected_frames_per_task": frames}


def load(root):
    root = Path(root).resolve()
    if sha256(root / "plan.json") != read(root / "plan.lock.json")["sha256"]:
        raise WorkflowError("Endpoint plan changed; create a new endpoint run")
    value = read(root / "plan.json")
    if value.get("schema_version") != 1 or value.get("kind") != "derived_physical_endpoint_md":
        raise WorkflowError("Unsupported endpoint plan")
    science = load_run(value["source_run"])
    if digest(science) != value["source_science_id"]:
        raise WorkflowError("Source ABFE run identity changed")
    verify_hashes(root, value["input_hashes"])
    return value
