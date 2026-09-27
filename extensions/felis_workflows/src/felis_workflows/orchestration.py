from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess

from .backends import local, slurm
from .backends.common import graph, incomplete_graph, launch, snapshot
from .artifacts import (commit_manifest, global_dependencies, group_dependencies, group_state, paths,
                        validate_final, validate_global, validate_preparation)
from .common import WorkflowError, digest, file_hashes, lock, read
from .config import site_config
from .integrity import verify_upstream
from .planning import calculation, load_run, units, workdir
from .runtime import activate_source, fingerprint, source_environment, source_identity


def prepare(root, site_path):
    root = Path(root).resolve()
    site = site_config(site_path)
    activate_source(site)
    science = load_run(root)
    with lock(root / "execution.lock"):
        if (root / "prepared.json").exists():
            validate_global(root, science)
            return {"run": str(root), "prepared": True, "reused": True}
        attempt = snapshot(root, site, "prepare", science)
        launch(root, site, attempt / "site.json", "receptor", role="receptor", log=attempt / "receptor.log")
        backend = science["forcefield"]["ligand"]["backend"]
        for name in science["campaign"]["ligands"]:
            launch(root, site, attempt / "site.json", "parameterize", ["--ligand", name], role=backend,
                   log=attempt / f"{name}.log")
        outputs = [root / name for name in file_hashes(root, ["receptor", "parameters"])]
        commit_manifest(root / "prepared.json", root, science, "global_inputs", outputs,
                        global_dependencies(science), fingerprint(site), attempt.name)
        validate_global(root, science)
        return {"run": str(root), "prepared": True, "attempt": str(attempt)}


def execute(root, site_path, resume=False, dry_run=False):
    root = Path(root).resolve()
    site = site_config(site_path)
    activate_source(site)
    science = load_run(root, prepared=not dry_run)
    if max(len(g) for leg in science["ladders"].values() for g in leg["groups"]) * site["mpi"]["ranks"] > 48:
        raise WorkflowError("MPI ranks times states exceed 48; reduce site.mpi.ranks")
    if site["resources"]["prep"]["cpus"] < 4 or site["resources"]["array"]["cpus"] < site["mpi"]["ranks"]:
        raise WorkflowError("Allocate at least four CPUs for preparation and one CPU per array MPI rank")
    backend = slurm if site["backend"] == "slurm" else local
    with lock(root / "execution.lock"):
        # Rendering a new submission is read-only with respect to the scheduler.
        # Resume must first prove no trajectory writer is still running.
        if resume or not dry_run:
            if site["backend"] == "slurm":
                backend.ensure_idle(root, site, resume, cancel_pending=not dry_run)
            else:
                backend.ensure_idle(root, site, resume)
        attempt = snapshot(root, site, "preview" if dry_run else "resume" if resume else "submit", science)
        tasks = graph(science)
        if resume:
            load_run(root, prepared=True)
            probe = attempt / "probe.json"
            launch(root, site, attempt / "site.json", "probe", ["--output", str(probe)], log=attempt / "probe.log")
            tasks = incomplete_graph(science, read(probe))
        return backend.submit(root, site, attempt, tasks, dry_run)


def status(root):
    """Do not open NetCDF files while jobs may be writing to them."""
    root = Path(root).resolve()
    science = load_run(root)
    global_ready = (root / "prepared.json").exists()
    if global_ready:
        validate_global(root, science)
    output = []
    preparation = {}
    for calc in science["calculations"]:
        directory = root / "calculations" / calc["key"]
        if paths(root, calc).exists():
            validate_preparation(root, science, calc)
            preparation[calc["key"]] = "complete"
        else:
            preparation[calc["key"]] = "partial" if directory.exists() else "not_started"
    dependencies_cache = {}
    for calc in science["calculations"]:
        directory = root / "calculations" / calc["key"]
        prep = preparation[calc["key"]]
        groups = {}
        for leg in "AB":
            entries = []
            owner = calculation(science, calc["solvent_owner"]) if leg == "A" else calc
            shared = owner["key"] != calc["key"]
            owner_prep = preparation[owner["key"]]
            dependencies = None
            if owner_prep == "complete":
                key = (owner["key"], leg)
                if key not in dependencies_cache:
                    dependencies_cache[key] = group_dependencies(root, science, calc, leg)
                dependencies = dependencies_cache[key]
            for unit in units(root, science, calc, leg):
                if owner_prep == "complete":
                    state = group_state(root, science, calc, leg, unit, dependencies=dependencies)
                    group_status = "recorded_unverified" if state["complete"] else "incomplete"
                else:
                    if paths(root, owner, leg, unit).exists():
                        raise WorkflowError(f"Simulation record without validated system preparation: {paths(root, owner, leg, unit)}")
                    group_status = "not_started" if owner_prep == "not_started" else "preparing"
                entry = {"index": unit["index"], "state": group_status}
                if shared:
                    entry["owner"] = owner["key"]
                entries.append(entry)
            groups[leg] = {"owner": owner["key"], "shared": shared,
                           "completed_records": sum(v["state"] == "recorded_unverified" for v in entries),
                           "total": len(entries), "units": entries}
        analysis_dir = workdir(root, calc) / "analysis"
        partial_outputs = [directory / name for name in
                           ("completion_audit.json", "partner_occupancy.json", "result.json")]
        partial_outputs += [analysis_dir / name for name in
                            ("A_fe_table.tsv", "B_fe_table.tsv", "R_fe_table.tsv", "sys_abfe.tsv")]
        finalized = "not_started"
        if (directory / "finalized.json").exists():
            validate_final(root, science, calc)
            finalized = "complete"
        elif any(path.is_file() for path in partial_outputs):
            finalized = "partial"
        completed = sum(value["completed_records"] for value in groups.values())
        total = sum(value["total"] for value in groups.values())
        if finalized == "complete":
            state = "finalized"
        elif finalized == "partial":
            state = "finalizing"
        elif prep == "partial":
            state = "preparing"
        elif prep == "not_started":
            state = "not_started"
        elif completed == total:
            state = "ready_for_finalization"
        elif completed:
            state = "sampling"
        else:
            state = "prepared"
        output.append({"calculation": calc["key"], "state": state,
                       "system_preparation": prep, "groups": groups, "finalization": finalized})
    jobs = slurm.job_records(root)
    from .backends.common import prior_attempts
    attempts = []
    for directory, value in prior_attempts(root, include_all=True):
        graph_path = directory / "task_graph.json"
        selected = read(graph_path) if graph_path.exists() else None
        if selected and (selected.get("task_graph_id") != digest(selected.get("tasks")) or
                         selected.get("workflow_graph_id") != value.get("workflow_graph_id")):
            raise WorkflowError(f"Attempt task graph provenance changed: {graph_path}")
        attempts.append({"id": directory.name, "purpose": value["purpose"], "site": value["site"],
                         "science_id": value.get("science_id"), "site_id": value["site_id"],
                         "workflow_graph_id": value.get("workflow_graph_id"),
                         "task_graph_id": selected["task_graph_id"] if selected else None})
    return {"run": str(root), "inputs_prepared": global_ready,
            "global_inputs": "complete" if global_ready else "not_started", "calculations": output,
            "attempts": attempts, "submitted_jobs": jobs,
            "note": "Read-only manifest status; checkpointed trajectories are verified after writers stop during resume."}


def doctor(site_path):
    site = site_config(site_path)
    identity = source_identity(site)
    integrity = verify_upstream(site["repo"])
    commands = {site["mpi"]["command"], *(v[0] for v in site["python"].values())}
    if site["mps"]:
        commands.add("nvidia-cuda-mps-control")
    if site["backend"] == "slurm":
        commands.update({"sbatch", "squeue", "scancel", "sinfo"})
    # Resolve inside the user's bootstrap, including Conda/module initialization.
    import shlex
    script = "set -euo pipefail\n"
    if site.get("bootstrap"):
        script += "source " + shlex.quote(site["bootstrap"]) + "\n"
    for command in sorted(commands):
        script += "command -v " + shlex.quote(command) + "\n"
    env = source_environment(site)
    answer = subprocess.check_output(["bash", "-c", script], text=True, env=env)
    probe = ("import json, os; from felis_workflows.runtime import source_identity; "
             "print('FELIS_IDENTITY_JSON=' + json.dumps(source_identity({'repo': os.environ['FELIS_REPO']})))")
    runners = {}
    for role, command in site["python"].items():
        output = subprocess.check_output(["bash", "-c", script + "exec " + shlex.join([*command, "-c", probe])],
                                         text=True, env=env)
        records = [line.removeprefix("FELIS_IDENTITY_JSON=") for line in output.splitlines()
                   if line.startswith("FELIS_IDENTITY_JSON=")]
        if len(records) != 1:
            raise WorkflowError(f"Python runner {role} did not report source identity")
        try:
            runners[role] = json.loads(records[0])
        except json.JSONDecodeError as error:
            raise WorkflowError(f"Python runner {role} reported invalid source identity") from error
    return {"site": site["name"], "upstream": integrity, "commands": answer.splitlines(),
            "source": identity, "runners": runners,
            "note": "Configuration/commands checked. This does not allocate a GPU or qualify the scientific environments."}
