"""One independent, nonalchemical physical-MD unit with conservative restart."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path

from felis_workflows.artifacts import write_once
from felis_workflows.common import WorkflowError, digest, lock, read, sha256, write

from .identity import check
from .plan import compatible_atoms, load


def unit_dir(root, task):
    return Path(root) / task["output"]


def writer_active(path):
    if not path.exists():
        return False
    with path.open("rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        finally:
            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
            except OSError:
                pass
    return False


def task_state(root, plan, task, *, semantic=False, runtime=None, allow_locked=False):
    """Read-only status: never open a DCD if its writer could be active."""
    directory = unit_dir(root, task)
    if not allow_locked and writer_active(directory / "writer.lock"):
        return {"state": "active", "steps": None}
    marker = directory / "completed.json"
    trajectory, checkpoint, final = (directory / name for name in ("trajectory.dcd", "checkpoint.chk", "final_state.xml"))
    if marker.exists():
        value = read(marker)
        required = {"schema_version", "plan_sha256", "task", "seed", "integrator_seed", "barostat_seed",
                    "start_sha256", "topology_id",
                    "expected_steps", "report_interval_steps", "expected_frames", "steps", "frames",
                    "trajectory", "trajectory_bytes", "checkpoint_sha256", "final_state_sha256",
                    "runtime_stable", "producer_identity", "producer_attempt"}
        if set(value) != required:
            raise WorkflowError(f"Unsupported endpoint completion schema: {marker}")
        start = next(s for s in plan["starts"] if s["id"] == task["start"])
        expected = {"schema_version": 1, "plan_sha256": sha256(Path(root) / "plan.json"),
                    "task": task["id"], "seed": task["seed"],
                    "integrator_seed": task["integrator_seed"], "barostat_seed": task["barostat_seed"],
                    "start_sha256": start["start_sha256"],
                    "topology_id": start["topology_id"], "expected_steps": plan["protocol"]["steps"],
                    "report_interval_steps": plan["protocol"]["interval_steps"],
                    "expected_frames": plan["protocol"]["expected_frames"]}
        if any(value.get(k) != v for k, v in expected.items()) or value.get("steps") != expected["expected_steps"] or \
                value.get("frames") != expected["expected_frames"] or value.get("trajectory") != str(trajectory.relative_to(root)):
            raise WorkflowError(f"Endpoint completion identity mismatch: {marker}")
        if runtime is not None and value.get("runtime_stable") != runtime["stable"]:
            raise WorkflowError(f"Endpoint runtime differs from completed task: {marker}")
        if not trajectory.is_file() or trajectory.stat().st_size != value.get("trajectory_bytes"):
            raise WorkflowError(f"Endpoint trajectory missing or changed size: {trajectory}")
        if not checkpoint.is_file() or sha256(checkpoint) != value.get("checkpoint_sha256") or \
                not final.is_file() or sha256(final) != value.get("final_state_sha256"):
            raise WorkflowError(f"Endpoint checkpoint/final state missing or corrupt: {directory}")
        if semantic:
            if trajectory_frames(trajectory, Path(root) / start["start_pdb"], verify_last=True) != value["frames"]:
                raise WorkflowError(f"Endpoint DCD frame count differs from completion: {trajectory}")
        return {"state": "complete" if semantic else "recorded_unverified", "steps": value["steps"],
                "frames": value["frames"]}
    exists = [p for p in (trajectory, checkpoint, final, directory / "progress.json") if p.exists()]
    if not exists:
        return {"state": "not_started", "steps": 0}
    if final.exists() or not checkpoint.exists() or not trajectory.exists() or not (directory / "progress.json").exists():
        return {"state": "blocked", "reason": "Partial output lacks a matching trajectory/checkpoint/progress; inspect and repair before resume"}
    progress = read(directory / "progress.json")
    if progress.get("task") != task["id"] or progress.get("plan_sha256") != sha256(Path(root) / "plan.json") or \
            any(progress.get(key) != task[key] for key in ("seed", "integrator_seed", "barostat_seed")) or \
            isinstance(progress.get("steps"), bool) or not isinstance(progress.get("steps"), int) or \
            progress["steps"] <= 0 or progress["steps"] > plan["protocol"]["steps"] or \
            progress["steps"] % plan["protocol"]["interval_steps"] or \
            progress.get("frames") != progress["steps"] // plan["protocol"]["interval_steps"] or \
            sha256(checkpoint) != progress.get("checkpoint_sha256"):
        return {"state": "blocked", "reason": "Checkpoint progress or task identity mismatch"}
    if semantic:
        start = next(s for s in plan["starts"] if s["id"] == task["start"])
        if trajectory_frames(trajectory, Path(root) / start["start_pdb"], verify_last=True) != progress["frames"]:
            return {"state": "blocked", "reason": "DCD frames do not match the last atomic checkpoint"}
    return {"state": "checkpointed", "steps": progress["steps"], "frames": progress["frames"]}


def trajectory_frames(path, topology, *, verify_last=False):
    import mdtraj as md
    try:
        with md.open(str(path), "r") as handle:
            count = len(handle)
        if verify_last and count:
            md.load_frame(str(path), count - 1, top=str(topology))
        return count
    except Exception as error:
        raise WorkflowError(f"Could not inspect quiescent endpoint trajectory {path}: {error}") from error


def reject_restraints(system):
    """Only ordinary molecular mechanics and the NPT barostat are accepted."""
    allowed = {"NonbondedForce", "HarmonicBondForce", "HarmonicAngleForce",
               "PeriodicTorsionForce", "RBTorsionForce", "CMAPTorsionForce",
               "CMMotionRemover", "MonteCarloBarostat"}
    for force in system.getForces():
        if force.__class__.__name__ not in allowed or \
                (hasattr(force, "getNumGlobalParameters") and force.getNumGlobalParameters()):
            raise WorkflowError(f"Endpoint system contains unsupported restraint or alchemical force: {force.__class__.__name__}")


def build_simulation(root, start, protocol, task, platform_name="CUDA"):
    from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform, unit
    from openmm.app import GromacsTopFile, HBonds, PME, PDBFile, Simulation
    pdb_path = Path(root) / start["start_pdb"]
    top_path = Path(root) / start["system_top"]
    if compatible_atoms(pdb_path, top_path) != start["atoms"]:
        raise WorkflowError("Endpoint starting atom identity changed")
    pdb = PDBFile(str(pdb_path))
    top = GromacsTopFile(str(top_path), periodicBoxVectors=pdb.topology.getPeriodicBoxVectors())
    system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1 * unit.nanometer,
                              constraints=HBonds, rigidWater=True)
    barostat = MonteCarloBarostat(protocol["pressure_bar"] * unit.bar,
                                 protocol["temperature_K"] * unit.kelvin, 25)
    barostat.setRandomNumberSeed(task["barostat_seed"])
    system.addForce(barostat)
    reject_restraints(system)
    integrator = LangevinMiddleIntegrator(protocol["temperature_K"] * unit.kelvin,
                                          1 / unit.picosecond, protocol["timestep_fs"] * unit.femtosecond)
    integrator.setRandomNumberSeed(task["integrator_seed"])
    platform = Platform.getPlatformByName(platform_name)
    properties = {"DeviceIndex": "0"} if platform_name == "CUDA" else {}
    simulation = Simulation(top.topology, system, integrator, platform, properties)
    simulation.context.setPositions(pdb.positions)
    return simulation


def atomic_checkpoint(simulation, path):
    temporary = path.with_suffix(".tmp")
    try:
        simulation.saveCheckpoint(str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256(path)


def verify_attempt(root, task_id, site_path, site, runtime):
    """A generated worker must execute the exact frozen script and task graph."""
    attempt = Path(site_path).resolve().parent
    if attempt.parent != Path(root).resolve() / "attempts":
        raise WorkflowError("Endpoint worker site is not a frozen attempt site")
    saved_site = read(attempt / "site.json")
    if saved_site != site:
        raise WorkflowError("Endpoint worker site changed")
    graph = read(attempt / "task_graph.json")
    metadata = read(attempt / "attempt.json")
    if graph.get("plan_sha256") != sha256(Path(root) / "plan.json") or \
            graph.get("task_graph_id") != digest(graph.get("tasks")) or \
            metadata.get("task_graph_id") != graph["task_graph_id"] or \
            metadata.get("runtime_stable") != runtime["stable"] or \
            metadata.get("site_id") != digest(site) or \
            not any(task["id"] == task_id for task in graph["tasks"]):
        raise WorkflowError("Endpoint attempt task/runtime provenance changed")
    submission_path = attempt / "submission_plan.json"
    if sha256(submission_path) != read(attempt / "submission_plan.lock.json")["sha256"]:
        raise WorkflowError("Endpoint submitted argv provenance changed")
    commands = read(submission_path)["commands"]
    matches = [entry for entry in commands if entry["task"] == task_id]
    if len(matches) != 1 or sha256(Path(root) / matches[0]["script"]) != matches[0]["script_sha256"]:
        raise WorkflowError("Endpoint worker script differs from the submitted script")


def run_task(root, task_id, site, *, platform="CUDA", site_path=None):
    root = Path(root).resolve()
    plan = load(root)
    task = next((t for t in plan["tasks"] if t["id"] == task_id), None)
    if task is None:
        raise WorkflowError(f"Unknown endpoint task: {task_id}")
    runtime = check(root, site)
    if site_path is not None:
        verify_attempt(root, task_id, site_path, site, runtime)
    start = next(s for s in plan["starts"] if s["id"] == task["start"])
    directory = unit_dir(root, task)
    directory.mkdir(parents=True, exist_ok=True)
    with lock(directory / "writer.lock"):
        status = task_state(root, plan, task, semantic=True, runtime=runtime, allow_locked=True)
        if status["state"] == "complete":
            return status
        if status["state"] == "blocked":
            raise WorkflowError(f"Endpoint {task_id} cannot resume: {status['reason']}")
        trajectory, checkpoint = directory / "trajectory.dcd", directory / "checkpoint.chk"
        simulation = build_simulation(root, start, plan["protocol"], task, platform)
        if status["state"] == "checkpointed":
            try:
                simulation.loadCheckpoint(str(checkpoint))
            except Exception as error:
                raise WorkflowError(f"Cannot load endpoint checkpoint for {task_id}: {error}") from error
            if simulation.currentStep != status["steps"]:
                raise WorkflowError("Loaded checkpoint step differs from recorded progress")
        else:
            from openmm import unit
            simulation.context.setVelocitiesToTemperature(plan["protocol"]["temperature_K"] * unit.kelvin,
                                                          task["seed"])
        from openmm.app import DCDReporter
        reporter = DCDReporter(str(trajectory), plan["protocol"]["interval_steps"],
                               append=status["state"] == "checkpointed")
        simulation.reporters.append(reporter)
        interval = plan["protocol"]["interval_steps"]
        try:
            while simulation.currentStep < plan["protocol"]["steps"]:
                simulation.step(interval)
                reporter._out.flush()
                os.fsync(reporter._out.fileno())
                checkpoint_hash = atomic_checkpoint(simulation, checkpoint)
                write(directory / "progress.json", {"task": task_id, "plan_sha256": sha256(root / "plan.json"),
                      "seed": task["seed"], "integrator_seed": task["integrator_seed"],
                      "barostat_seed": task["barostat_seed"], "steps": simulation.currentStep,
                      "frames": simulation.currentStep // interval, "checkpoint_sha256": checkpoint_hash})
        finally:
            simulation.reporters.clear()
            if reporter._out is not None:
                reporter._out.close()
        if trajectory_frames(trajectory, root / start["start_pdb"], verify_last=True) != plan["protocol"]["expected_frames"]:
            raise WorkflowError("Endpoint trajectory does not contain the expected final frame count")
        final = directory / "final_state.xml"
        temporary = final.with_suffix(".tmp")
        simulation.saveState(str(temporary))
        os.replace(temporary, final)
        value = {"schema_version": 1, "plan_sha256": sha256(root / "plan.json"), "task": task_id,
                 "seed": task["seed"], "integrator_seed": task["integrator_seed"],
                 "barostat_seed": task["barostat_seed"], "start_sha256": start["start_sha256"],
                 "topology_id": start["topology_id"], "expected_steps": plan["protocol"]["steps"],
                 "report_interval_steps": interval, "expected_frames": plan["protocol"]["expected_frames"],
                 "steps": simulation.currentStep, "frames": plan["protocol"]["expected_frames"],
                 "trajectory": str(trajectory.relative_to(root)), "trajectory_bytes": trajectory.stat().st_size,
                 "checkpoint_sha256": sha256(checkpoint), "final_state_sha256": sha256(final),
                 "runtime_stable": runtime["stable"], "producer_identity": runtime["diagnostic"],
                 "producer_attempt": Path(site_path).resolve().parent.name if site_path else "direct"}
        write_once(directory / "completed.json", value)
        return task_state(root, plan, task, semantic=True, runtime=runtime, allow_locked=True)


def status(root):
    root = Path(root).resolve()
    from .identity import identity
    recorded = read(root / "runtime.json")
    current = identity(read(root / "site.json"))
    for key in ("endpoint_source_hashes",):
        if current["stable"][key] != recorded["stable"][key]:
            raise WorkflowError("Endpoint implementation differs from this run's source")
    if current["stable"]["abfe"]["source_hashes"] != recorded["stable"]["abfe"]["source_hashes"]:
        raise WorkflowError("ABFE source differs from this endpoint plan")
    plan = load(root)
    entries = {task["id"]: task_state(root, plan, task) for task in plan["tasks"]}
    return {"endpoint_run": str(root), "source_run": plan["source_run"], "plan_sha256": sha256(root / "plan.json"),
            "tasks": entries, "counts": {name: sum(v["state"] == name for v in entries.values())
                                        for name in {v["state"] for v in entries.values()}}}
