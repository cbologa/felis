from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import os
import shlex
import subprocess
import uuid

from .. import artifacts
from ..artifacts import compatibility, write_once, write_script
from ..common import WorkflowError, digest, read, sha256
from ..runtime import fingerprint, source_environment


def snapshot(root, site, purpose, science=None):
    if science is None:
        from ..planning import load_run
        science = load_run(root)
    directory = Path(root) / "executions" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    write_once(directory / "site.json", site)
    runtime = fingerprint(site)
    write_once(directory / "attempt.json", {"schema_version": 1, "purpose": purpose,
              "site": site["name"], "backend": site["backend"], "root": str(Path(root).resolve()),
              "science_id": digest(science), "site_id": digest(site), "source_identity": runtime["source_identity"],
              "runtime_compatibility": compatibility(runtime),
              "workflow_graph_id": digest(artifacts.stage_graph(science))})
    return directory


def record_task_graph(attempt, tasks):
    """Commit the exact submitted/restart selection after the quiescent probe."""
    metadata = read(Path(attempt) / "attempt.json")
    value = {"schema_version": 1, "workflow_graph_id": metadata["workflow_graph_id"],
             "task_graph_id": digest(tasks), "tasks": tasks}
    write_once(Path(attempt) / "task_graph.json", value)
    return value


def worker_script(root, site, site_path, task, arguments=(), python_role="simulation", array=False):
    env = source_environment(site)
    lines = ["#!/bin/bash", "set -euo pipefail"]
    if site.get("bootstrap"):
        lines += ["set +u", f"source {shlex.quote(site['bootstrap'])}", "set -u"]
    for name in ["FELIS_REPO", "PYTHONPATH", "OMP_NUM_THREADS", "OPENMM_CPU_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "JAX_PLATFORMS"]:
        lines.append(f"export {name}={shlex.quote(env[name])}")
    lines += ['export TMPDIR="${SLURM_TMPDIR:-/tmp}"', 'test -d "$TMPDIR" && test -w "$TMPDIR"',
              f"cd {shlex.quote(str(Path(root).resolve()))}"]
    command = [*site["python"][python_role], "-m", "felis_workflows.worker", task,
               "--run", str(Path(root).resolve()), "--site", str(Path(site_path).resolve()), *map(str, arguments)]
    line = "exec " + shlex.join(command)
    if array:
        line += ' --index "${SLURM_ARRAY_TASK_ID:?Missing Slurm array index}"'
    lines.append(line)
    return "\n".join(lines) + "\n"


def launch(root, site, site_path, task, args=(), role="simulation", log=None, gpu=None):
    script = worker_script(root, site, site_path, task, args, role)
    attempt = Path(site_path).parent
    args = [str(value) for value in args]
    task_id = digest([task, args, role])[:16]
    script_path = attempt / "scripts" / f"{task_id}.sh"
    write_script(script_path, script)
    write_once(attempt / "commands" / f"{task_id}.json",
               {"task": task, "arguments": args, "role": role,
                "script": str(script_path.relative_to(root)), "script_sha256": sha256(script_path),
                "argv": ["bash", str(script_path)]})
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    elif task in {"receptor", "parameterize", "probe", "finalize"}:
        env["CUDA_VISIBLE_DEVICES"] = ""
    if log is None:
        subprocess.run(["bash", str(script_path)], env=env, check=True)
    else:
        with Path(log).open("a") as handle:
            subprocess.run(["bash", str(script_path)], env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def graph(science):
    """Aggregate canonical simulation-unit stages into scheduler arrays."""
    stages = artifacts.stage_graph(science)
    def task_id(stage):
        if stage["kind"] == "system_preparation":
            return "prep__" + stage["calculation"].replace("/", "__")
        if stage["kind"] == "simulation_group":
            return stage["leg"] + "__" + stage["calculation"].replace("/", "__")
        if stage["kind"] == "finalization":
            return "finalize__" + stage["calculation"].replace("/", "__")
        raise WorkflowError(f"No executable task for {stage['id']}")
    indexed = {stage["id"]: stage for stage in stages}
    tasks = {}
    for stage in stages:
        if stage["kind"] == "global_input":
            continue  # Global input preparation is the separate CPU prepare command.
        task_id_value = task_id(stage)
        if task_id_value not in tasks:
            kind = {"system_preparation": "prep", "simulation_group": "array",
                    "finalization": "finalize"}[stage["kind"]]
            task = {"id": task_id_value, "kind": kind, "calculation": stage["calculation"],
                    "dependencies": []}
            if kind == "array":
                task.update(leg=stage["leg"], indices=[])
            tasks[task_id_value] = task
        task = tasks[task_id_value]
        for name in stage["dependencies"]:
            dependency = indexed[name]
            if dependency["kind"] == "global_input":
                continue
            parent = task_id(dependency)
            if parent not in task["dependencies"]:
                task["dependencies"].append(parent)
        if stage["kind"] == "simulation_group":
            task["indices"].append(stage["index"])
    return list(tasks.values())


def incomplete_graph(science, status):
    pending = []
    for task in graph(science):
        entry = status[task["calculation"]]
        if task["kind"] == "array":
            task["indices"] = [i for i in task["indices"] if not entry[task["leg"]][i]]
            if not task["indices"]:
                continue
        elif entry[task["kind"]]:
            continue
        pending.append(task)
    ids = {t["id"] for t in pending}
    for task in pending:
        task["dependencies"] = [d for d in task["dependencies"] if d in ids]
    return pending


def prior_attempts(root, include_all=False):
    result = []
    for p in sorted((Path(root) / "executions").glob("*/attempt.json")):
        data = read(p)
        if include_all or data["purpose"] in {"submit", "resume"}:
            result.append((p.parent, data))
    return result
