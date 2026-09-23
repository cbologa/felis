from __future__ import annotations
from pathlib import Path
import shutil
import subprocess

from .backends import local, slurm
from .backends.common import graph, incomplete_graph, launch, snapshot
from .common import WorkflowError, digest, file_hashes, lock, read, verify_hashes, write
from .config import site_config
from .integrity import verify_upstream
from .planning import load_run


def prepare(root, site_path):
    root = Path(root).resolve()
    site = site_config(site_path)
    verify_upstream(site["repo"])
    science = load_run(root)
    with lock(root / "execution.lock"):
        if (root / "prepared.json").exists():
            load_run(root, prepared=True)
            return {"run": str(root), "prepared": True, "reused": True}
        attempt = snapshot(root, site, "prepare")
        launch(root, site, attempt / "site.json", "receptor", role="receptor", log=attempt / "receptor.log")
        backend = science["forcefield"]["ligand"]["backend"]
        for name in science["campaign"]["ligands"]:
            launch(root, site, attempt / "site.json", "parameterize", ["--ligand", name], role=backend,
                   log=attempt / f"{name}.log")
        ready = {"science_id": digest(science), "hashes": file_hashes(root, ["receptor", "parameters"])}
        if not ready["hashes"]:
            raise WorkflowError("Preparation produced no files")
        write(root / "prepared.json", ready)
        return {"run": str(root), "prepared": True, "attempt": str(attempt)}


def execute(root, site_path, resume=False, dry_run=False):
    root = Path(root).resolve()
    site = site_config(site_path)
    verify_upstream(site["repo"])
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
        attempt = snapshot(root, site, "preview" if dry_run else "resume" if resume else "submit")
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
    output = []
    for calc in science["calculations"]:
        directory = root / "calculations" / calc["key"]
        state = "planned"
        if (directory / "prep.ok.json").exists():
            state = "system prepared"
        if (directory / "finalized.json").exists():
            verify_hashes(root, read(directory / "finalized.json")["hashes"])
            state = "finalized"
        output.append({"calculation": calc["key"], "state": state})
    jobs = slurm.job_records(root)
    return {"run": str(root), "inputs_prepared": (root / "prepared.json").exists(),
            "calculations": output, "submitted_jobs": jobs,
            "note": "File status only; use squeue/sacct for live Slurm state. Resume audits stopped trajectories."}


def doctor(site_path):
    site = site_config(site_path)
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
    answer = subprocess.check_output(["bash", "-c", script], text=True)
    return {"site": site["name"], "upstream": integrity, "commands": answer.splitlines(),
            "note": "Configuration/commands checked. This does not allocate a GPU or qualify the scientific environments."}
