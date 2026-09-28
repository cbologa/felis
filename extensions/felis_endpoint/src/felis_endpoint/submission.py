"""Scheduler-selected endpoint tasks, immutable scripts and exact attempt graphs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import os
from pathlib import Path
from queue import Queue
import shlex
import subprocess
import uuid

from felis_workflows.artifacts import write_once, write_script
from felis_workflows.common import WorkflowError, digest, lock, read, sha256, write
from felis_workflows.config import site_config

from .execution import task_state
from .identity import check_runner, environment
from .plan import load


def script(root, site, site_path, task):
    env = environment(site)
    lines = ["#!/bin/bash", "set -euo pipefail"]
    if site.get("bootstrap"):
        lines += ["set +u", f"source {shlex.quote(site['bootstrap'])}", "set -u"]
    for name in ("FELIS_REPO", "PYTHONPATH", "OMP_NUM_THREADS", "OPENMM_CPU_THREADS",
                 "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "JAX_PLATFORMS"):
        lines.append(f"export {name}={shlex.quote(env[name])}")
    lines.append(f"cd {shlex.quote(str(Path(root).resolve()))}")
    command = [*site["python"]["simulation"], "-m", "felis_endpoint.cli", "run-unit",
               "--run", str(Path(root).resolve()), "--site", str(Path(site_path).resolve()),
               "--task-id", task["id"]]
    lines.append("exec " + shlex.join(command))
    return "\n".join(lines) + "\n"


def slurm_command(root, site, attempt, task, path):
    resources, slurm = site["resources"]["array"], site["slurm"]
    cmd = ["sbatch", "--parsable", f"--job-name=endpoint_{task['id'].replace('/', '_')}",
           "--nodes=1", "--ntasks=1", f"--partition={slurm['partition']}",
           f"--cpus-per-task={resources['cpus']}", f"--mem={resources['memory']}",
           f"--time={resources['walltime']}", f"--chdir={root}",
           f"--output={attempt}/{task['id'].replace('/', '_')}_%j.out",
           f"--error={attempt}/{task['id'].replace('/', '_')}_%j.err"]
    if slurm.get("account"):
        cmd.append(f"--account={slurm['account']}")
    cmd.extend(slurm["gpu_args"])  # Exactly one GPU; site_config validates this.
    cmd.extend(slurm.get("extra_args", []))
    cmd.append(str(path))
    return cmd


def prior_jobs(root):
    jobs = []
    for path in sorted((Path(root) / "attempts").glob("*/jobs.json")):
        jobs.extend(read(path)["jobs"])
    return jobs


def previous_submissions(root):
    records = [read(path) for path in sorted((Path(root) / "attempts").glob("*/attempt.json"))]
    return [record for record in records if not record.get("dry_run")]


def ensure_idle(root, site, *, resume, dry_run):
    jobs = prior_jobs(root)
    previous = previous_submissions(root)
    if dry_run:
        return
    if previous and not resume:
        raise WorkflowError("Endpoint jobs were already submitted; use resume after they stop")
    if any(a["backend"] != site["backend"] or a["site"] != site["name"] for a in previous):
        raise WorkflowError("Endpoint continuation requires the original site and scheduler backend")
    if site["backend"] == "slurm":
        if not jobs:
            return
        output = subprocess.check_output(["squeue", "--me", "--noheader", "--format=%i|%T"], text=True)
        active = {line.split("|", 1)[0].split("_", 1)[0] for line in output.splitlines() if "|" in line}
        running = [job["job_id"] for job in jobs if job["job_id"] in active]
        if running:
            raise WorkflowError(f"Endpoint scheduler jobs still active: {running}; wait before resume")


def submit(root, site_path, *, resume=False, dry_run=False):
    """Serialize selection, attempt creation, and the entire submission transaction."""
    root = Path(root).resolve()
    with lock(root / "execution.lock"):
        return _submit_locked(root, site_path, resume=resume, dry_run=dry_run)


def _submit_locked(root, site_path, *, resume, dry_run):
    plan = load(root)
    site = site_config(site_path)
    runtime = check_runner(root, site_path, site)
    ensure_idle(root, site, resume=resume, dry_run=dry_run)
    selected = []
    for task in plan["tasks"]:
        state = task_state(root, plan, task, semantic=not dry_run, runtime=runtime)
        if state["state"] == "active" or state["state"] == "blocked":
            raise WorkflowError(f"Endpoint {task['id']} cannot be submitted: {state}")
        if state["state"] not in ("complete", "recorded_unverified"):
            selected.append(task)
    if not resume and not dry_run and prior_jobs(root):
        raise WorkflowError("Endpoint execution already submitted; use resume")
    attempt = root / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True)
    write_once(attempt / "site.json", site)
    graph = {"schema_version": 1, "plan_sha256": sha256(root / "plan.json"),
             "task_graph_id": digest(selected), "tasks": selected}
    write_once(attempt / "task_graph.json", graph)
    write_once(attempt / "attempt.json", {"schema_version": 1, "purpose": "resume" if resume else "submit",
               "dry_run": dry_run, "task_graph_id": graph["task_graph_id"],
               "runtime_stable": runtime["stable"], "source_diagnostic": runtime["diagnostic"],
               "site_id": digest(site), "site": site["name"], "backend": site["backend"]})
    preview = []
    for task in selected:
        script_path = attempt / "scripts" / f'{task["id"].replace("/", "_")}.sh'
        write_script(script_path, script(root, site, attempt / "site.json", task))
        argv = slurm_command(root, site, attempt, task, script_path) if site["backend"] == "slurm" else \
               ["bash", str(script_path)]
        preview.append({"task": task["id"], "argv": argv,
                        "script": str(script_path.relative_to(root)), "script_sha256": sha256(script_path)})
    submission_path = attempt / "submission_plan.json"
    write_once(submission_path, {"schema_version": 1, "dry_run": dry_run,
               "task_graph_id": graph["task_graph_id"], "commands": preview})
    write_once(attempt / "submission_plan.lock.json", {"sha256": sha256(submission_path)})
    jobs = []
    if not dry_run and site["backend"] == "slurm":
        for entry in preview:
            if sha256(submission_path) != read(attempt / "submission_plan.lock.json")["sha256"] or \
                    sha256(root / entry["script"]) != entry["script_sha256"]:
                raise WorkflowError("Endpoint submission command/script changed before sbatch")
            answer = subprocess.check_output(entry["argv"], text=True).strip().split(";", 1)[0]
            if not answer.isdigit():
                raise WorkflowError(f"Unrecognized sbatch reply: {answer}")
            jobs.append({"task": entry["task"], "job_id": answer,
                         "argv": entry["argv"], "script_sha256": entry["script_sha256"]})
            write(attempt / "jobs.json", {"jobs": jobs})
    elif not dry_run:
        devices = site["local"]["gpus"]
        available = Queue()
        for device in devices:
            available.put(device)
        def local_task(entry):
            device = available.get()
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(device)
            try:
                if sha256(submission_path) != read(attempt / "submission_plan.lock.json")["sha256"] or \
                        sha256(root / entry["script"]) != entry["script_sha256"]:
                    raise WorkflowError("Endpoint local submission command/script changed")
                with (attempt / f'{entry["task"].replace("/", "_")}.log').open("w") as output:
                    subprocess.run(entry["argv"], env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
            finally:
                available.put(device)
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [pool.submit(local_task, entry) for entry in preview]
            for future in futures:
                future.result()
    return {"attempt": str(attempt), "tasks": len(selected), "dry_run": dry_run, "jobs": jobs}
