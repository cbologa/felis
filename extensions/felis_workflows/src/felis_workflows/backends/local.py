from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from ..common import WorkflowError, write
from .common import launch, prior_attempts


def ensure_idle(root, site, resume=False):
    previous = prior_attempts(root)
    if previous and not resume:
        raise WorkflowError("Execution exists; use resume")
    if any(v["backend"] != "local" or v["site"] != site["name"] for _, v in previous):
        raise WorkflowError("Resume on the original backend/site")
    # The orchestration layer holds the run-wide flock for the whole local run.


def submit(root, site, attempt, tasks, dry_run=False):
    write(attempt / "submission_plan.json", {"tasks": tasks, "dry_run": dry_run})
    if dry_run:
        return {"attempt": str(attempt), "dry_run": True, "task_count": len(tasks)}
    pool = Queue()
    for gpu in site["local"]["gpus"]:
        pool.put(gpu)
    complete = set()
    for task in tasks:
        if not set(task["dependencies"]) <= complete:
            raise WorkflowError("Unresolved local task dependency")
        args = ["--calculation", task["calculation"]]
        if task["kind"] == "array":
            def group(index):
                gpu = pool.get()
                try:
                    launch(root, site, attempt / "site.json", "group", args + ["--leg", task["leg"], "--index", str(index)],
                           log=attempt / f"{task['id']}_{index}.log", gpu=gpu)
                finally:
                    pool.put(gpu)
            with ThreadPoolExecutor(max_workers=len(site["local"]["gpus"])) as workers:
                list(workers.map(group, task["indices"]))
        else:
            launch(root, site, attempt / "site.json", task["kind"], args,
                   log=attempt / f"{task['id']}.log", gpu=site["local"]["gpus"][0] if task["kind"] == "prep" else None)
        complete.add(task["id"])
        write(attempt / "completed_tasks.json", {"tasks": sorted(complete)})
    return {"attempt": str(attempt), "completed_tasks": len(complete)}
