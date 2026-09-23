from __future__ import annotations
from pathlib import Path
import subprocess

from ..common import WorkflowError, read, write
from .common import prior_attempts, worker_script


def job_records(root):
    records = []
    for directory, _ in prior_attempts(root):
        path = directory / "jobs.json"
        if path.exists():
            records.extend(read(path)["jobs"])
    return records


def ensure_idle(root, site, resume=False, cancel_pending=True):
    attempts = prior_attempts(root)
    if not attempts:
        return
    if not resume:
        raise WorkflowError("Execution already submitted; use resume after previous jobs stop")
    if any(a["site"] != site["name"] or a["backend"] != "slurm" for _, a in attempts):
        raise WorkflowError("Checkpoint resume is supported on the original site/backend; portable campaigns can be planned at any site")
    text = subprocess.check_output(["squeue", "--me", "--noheader", "--format=%i|%T|%r"], text=True)
    rows = [line.strip().split("|", 2) for line in text.splitlines() if "|" in line]
    byid = {}
    for row in rows:
        byid.setdefault(row[0].split("_", 1)[0], []).append(row)
    cancellations = set()
    for job in job_records(root):
        for row in byid.get(job["job_id"], []):
            state, reason = row[1:]
            if state == "PENDING" and (job["kind"] == "finalize" or reason == "DependencyNeverSatisfied"):
                cancellations.add(job["job_id"])
            else:
                raise WorkflowError(f"Prior job {row[0]} is {state}; do not inspect/resume active trajectories")
    if cancel_pending:
        for jobid in sorted(cancellations):
            subprocess.run(["scancel", jobid], check=True)


def submission_command(root, site, attempt, task, script, dependencies):
    slurm = site["slurm"]
    stage = "array" if task["kind"] == "array" else "analysis" if task["kind"] == "finalize" else "prep"
    resources = site["resources"][stage]
    partition = slurm["analysis_partition"] if stage == "analysis" else slurm["partition"]
    cmd = ["sbatch", "--parsable", f"--job-name=felis_{task['id']}", "--nodes=1", "--ntasks=1",
           f"--partition={partition}", f"--cpus-per-task={resources['cpus']}", f"--mem={resources['memory']}",
           f"--time={resources['walltime']}", f"--chdir={root}",
           f"--output={attempt}/{task['id']}_%A_%a.out", f"--error={attempt}/{task['id']}_%A_%a.err"]
    if slurm.get("account"):
        cmd.append(f"--account={slurm['account']}")
    if stage != "analysis":
        cmd.extend(slurm["gpu_args"])
    if task["kind"] == "array":
        cmd.append("--array=" + ",".join(map(str, task["indices"])) + f"%{slurm['array_concurrency']}")
    if dependencies:
        cmd.append("--dependency=afterok:" + ":".join(dict.fromkeys(dependencies)))
    cmd.extend(slurm.get("extra_args", []))
    cmd.append(str(script))
    return cmd


def submit(root, site, attempt, tasks, dry_run=False):
    ids, jobs, prep_lanes = {}, [], [None] * site["slurm"]["prep_concurrency"]
    prep_index = 0
    previews = []
    for task in tasks:
        args = ["--calculation", task["calculation"]]
        action = task["kind"]
        if action == "array":
            action = "group"
            args += ["--leg", task["leg"]]
        script = attempt / f"{task['id']}.sh"
        script.write_text(worker_script(root, site, attempt / "site.json", action, args, array=task["kind"] == "array"))
        dependencies = [ids[d] for d in task["dependencies"]]
        lane = prep_index % len(prep_lanes)
        if task["kind"] == "prep":
            if prep_lanes[lane]:
                dependencies.append(prep_lanes[lane])
            prep_index += 1
        cmd = submission_command(root, site, attempt, task, script, dependencies)
        previews.append({"task": task, "argv": cmd})
        if dry_run:
            jobid = f"DRY_{len(ids)+1}"
        else:
            answer = subprocess.check_output(cmd, text=True).strip().split(";", 1)[0]
            if not answer.isdigit():
                raise WorkflowError(f"Unrecognized sbatch reply: {answer}")
            jobid = answer
            jobs.append({"task_id": task["id"], "kind": task["kind"], "job_id": jobid,
                         "calculation": task["calculation"], "leg": task.get("leg"), "indices": task.get("indices")})
            # Keep every successful submission visible if a later call fails.
            write(attempt / "jobs.json", {"jobs": jobs})
        ids[task["id"]] = jobid
        if task["kind"] == "prep":
            prep_lanes[lane] = jobid
    write(attempt / "submission_plan.json", {"tasks": previews, "dry_run": dry_run})
    return {"attempt": str(attempt), "jobs": jobs, "dry_run": dry_run, "task_count": len(tasks)}
