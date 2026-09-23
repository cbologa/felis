from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import os
import shlex
import subprocess
import uuid

from ..common import WorkflowError, digest, read, write
from ..runtime import source_environment


def snapshot(root, site, purpose):
    directory = Path(root) / "executions" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    write(directory / "site.json", site)
    write(directory / "attempt.json", {"purpose": purpose, "site": site["name"], "backend": site["backend"],
                                        "root": str(Path(root).resolve()), "site_id": digest(site)})
    return directory


def worker_script(root, site, site_path, task, arguments=(), python_role="simulation", array=False):
    env = source_environment(site)
    lines = ["#!/bin/bash", "set -euo pipefail"]
    if site.get("bootstrap"):
        lines += ["set +u", f"source {shlex.quote(site['bootstrap'])}", "set -u"]
    for name in ["PYTHONPATH", "OMP_NUM_THREADS", "OPENMM_CPU_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "JAX_PLATFORMS"]:
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
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    elif task in {"receptor", "parameterize", "probe", "finalize"}:
        env["CUDA_VISIBLE_DEVICES"] = ""
    if log is None:
        subprocess.run(["bash", "-c", script], env=env, check=True)
    else:
        with Path(log).open("a") as handle:
            subprocess.run(["bash", "-c", script], env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def graph(science):
    tasks = []
    def id_for(kind, calc):
        return kind + "__" + calc["key"].replace("/", "__")
    for calc in science["calculations"]:
        tasks.append({"id": id_for("prep", calc), "kind": "prep", "calculation": calc["key"], "dependencies": []})
    for calc in science["calculations"]:
        for leg in "AB":
            if leg == "A" and calc["solvent_owner"] != calc["key"]:
                continue
            tasks.append({"id": id_for(leg, calc), "kind": "array", "leg": leg,
                          "calculation": calc["key"], "dependencies": [id_for("prep", calc)],
                          "indices": list(range(len(science["ladders"][leg]["groups"])) )})
    for calc in science["calculations"]:
        tasks.append({"id": id_for("finalize", calc), "kind": "finalize", "calculation": calc["key"],
                      "dependencies": ["A__" + calc["solvent_owner"].replace("/", "__"), id_for("B", calc)]})
    return tasks


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


def prior_attempts(root):
    result = []
    for p in sorted((Path(root) / "executions").glob("*/attempt.json")):
        data = read(p)
        if data["purpose"] in {"submit", "resume"}:
            result.append((p.parent, data))
    return result
