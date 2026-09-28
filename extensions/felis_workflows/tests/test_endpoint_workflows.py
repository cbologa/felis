"""Derived endpoint MD never modifies PR4 ABFE runtime or science."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from threading import Event

import pytest
import yaml

from felis_workflows.common import WorkflowError, read, sha256, write
from felis_workflows.planning import load_run
from felis_workflows.runtime import check_runtime, fingerprint
from felis_endpoint import execution, plan, submission


TOY_TOP = """[ defaults ]
1 2 yes 0.5 0.8333333
[ atomtypes ]
C C 12.011 0.0 A 0.34 0.2
OW OW 15.999 0.0 A 0.315 0.65
HW HW 1.008 0.0 A 0.1 0.0
[ moleculetype ]
LIG 3
[ atoms ]
1 C 1 LIG C1 1 0.0 12.011
[ moleculetype ]
PRO 3
[ atoms ]
1 C 1 ALA CA 1 0.0 12.011
[ moleculetype ]
SOL 2
[ atoms ]
1 OW 1 SOL OW 1 -0.834 15.999
2 HW 1 SOL HW1 1 0.417 1.008
3 HW 1 SOL HW2 1 0.417 1.008
[ bonds ]
1 2 1 0.09572 462750
1 3 1 0.09572 462750
[ angles ]
2 1 3 1 104.52 836.8
[ system ]
Toy
[ molecules ]
LIG 1
PRO 1
SOL 1
"""
TOY_PDB = """CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1
HETATM    1  C1  LIG A   1       5.000   5.000   5.000  1.00  0.00           C
ATOM      2  CA  ALA A   2       7.000   7.000   7.000  1.00  0.00           C
HETATM    3  OW  SOL A   3      10.000  10.000  10.000  1.00  0.00           O
HETATM    4  HW1 SOL A   3      10.957  10.000  10.000  1.00  0.00           H
HETATM    5  HW2 SOL A   3       9.760  10.927  10.000  1.00  0.00           H
END
"""


@pytest.fixture
def endpoint_inputs(cycle, site, tmp_path):
    root, science = cycle
    topology, pdb = tmp_path / "system.top", tmp_path / "start.pdb"
    topology.write_text(TOY_TOP)
    pdb.write_text(TOY_PDB)
    config = {"schema_version": 1, "name": "toy", "source_run": str(root),
              "seed": 1087, "replicas": 2, "duration_ns": 0.00002,
              "report_interval_ps": 0.002, "temperature_K": 298.15,
              "timestep_fs": 2, "pressure_bar": 1,
              "starts": [{"id": "start-one", "calculation": science["calculations"][0]["key"],
                          "topology": str(topology), "pdb": str(pdb)}]}
    path = tmp_path / "endpoint.yaml"
    write(path, config)
    return root, science, path, tmp_path / "endpoint-run", site, config, topology, pdb


def test_pr4_runtime_and_load_remain_compatible(cycle, site, repo):
    root, science = cycle
    before = fingerprint(read(site))
    write(root / "runtime.json", before)
    assert science["workflow_version"] == "0.2.0" and science["stage_model_version"] == 2
    assert load_run(root) == science
    from felis_endpoint import cli, networks, qualification  # Loading PR5 must not change the PR4 fingerprint.
    assert cli and networks and qualification
    assert check_runtime(root, read(site))["source_hashes"] == before["source_hashes"]
    assert load_run(root) == science
    assert all("felis_endpoint" not in name for name in before["source_hashes"])
    subprocess.run(["git", "diff", "--quiet", "d0bddaef0d9543f5ff831f4fe24ecc0e112c9133", "--",
                    "extensions/felis_workflows/src"], cwd=repo, check=True)


def test_plan_freezes_starts_seeds_and_independent_directories(endpoint_inputs):
    _, _, config, output, site, value, _, _ = endpoint_inputs
    result = plan.plan(config, output, site)
    frozen = plan.load(output)
    assert result["tasks"] == 2 and result["expected_frames_per_task"] == 10
    assert frozen["protocol"]["steps"] == 10 and frozen["protocol"]["interval_steps"] == 1
    assert frozen["protocol"]["alchemical"] is False
    assert frozen["protocol"]["boresch_restraints"] is False
    assert frozen["protocol"]["positional_restraints"] is False
    assert len({t["seed"] for t in frozen["tasks"]}) == len(frozen["tasks"])
    assert len({t["output"] for t in frozen["tasks"]}) == len(frozen["tasks"])
    value["starts"].append({**value["starts"][0], "id": "other-start"})
    write(config, value)
    second = plan.plan(config, output.parent / "other-run", site)
    other = plan.load(second["endpoint_run"])
    assert len({t["seed"] for t in other["tasks"]}) == 4
    assert len({t["output"] for t in other["tasks"]}) == 4
    assert frozen["starts"][0]["start_sha256"] == sha256(output / "inputs/start-one/start.pdb")


def test_default_duration_arithmetic_and_wrong_cwd_cli(endpoint_inputs, tmp_path, repo):
    assert plan.arithmetic(50, 2, 50) == (25_000_000, 25_000, 1_000)
    _, _, config, output, site, _, _, _ = endpoint_inputs
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(repo / "extensions/felis_endpoint/src"),
                                          str(repo / "extensions/felis_workflows/src"), str(repo)))
    outside = tmp_path / "unrelated-directory"
    outside.mkdir()
    result = subprocess.run([sys.executable, "-m", "felis_endpoint.cli", "plan", "--config", str(config),
                             "--output", str(output), "--site", str(site)], cwd=outside, env=env,
                            text=True, capture_output=True, check=True)
    assert '"tasks": 2' in result.stdout
    assert plan.load(output)["starts"][0]["source_run"] == str(endpoint_inputs[0])


@pytest.mark.parametrize("change,pattern", [
    ({"seed": True}, "positive integer"), ({"replicas": 0}, "positive integer"),
    ({"duration_ns": float("nan")}, "finite"), ({"duration_ns": float("inf")}, "finite"),
    ({"timestep_fs": 0}, "finite"), ({"report_interval_ps": 0.003}, "exact multiples"),
    ({"source_run": "relative/run"}, "absolute"),
])
def test_endpoint_config_rejects_invalid_science(endpoint_inputs, change, pattern):
    _, _, config, output, site, value, _, _ = endpoint_inputs
    value.update(change)
    config.write_text(yaml.safe_dump(value))
    with pytest.raises(WorkflowError, match=pattern):
        plan.plan(config, output, site)


def test_trajectory_frame_lineage_is_explicit_and_frozen(endpoint_inputs):
    import mdtraj as md
    _, _, config, output, site, value, _, pdb = endpoint_inputs
    first = md.load(str(pdb))
    trajectory = md.Trajectory(first.xyz.repeat(3, axis=0), first.topology,
                               time=[0, 2, 4], unitcell_lengths=first.unitcell_lengths.repeat(3, axis=0),
                               unitcell_angles=first.unitcell_angles.repeat(3, axis=0))
    trajectory.xyz[1, 0, 0] += 0.01
    frames = config.parent / "frames.dcd"
    trajectory.save_dcd(str(frames))
    value["starts"][0].pop("pdb")
    value["starts"][0].update({"trajectory": str(frames), "trajectory_topology": str(pdb), "frame": 1})
    write(config, value)
    plan.plan(config, output, site)
    start = plan.load(output)["starts"][0]
    assert start["source"]["kind"] == "trajectory_frame" and start["source"]["frame"] == 1
    assert start["source"]["trajectory"] == str(frames)
    assert start["source"]["trajectory_topology_sha256"] == sha256(pdb)
    assert start["start_sha256"] == sha256(output / start["start_pdb"])
    assert float(md.load(str(output / start["start_pdb"])).xyz[0, 0, 0]) > float(first.xyz[0, 0, 0])


def test_topology_mismatch_and_missing_frame_fail_closed(endpoint_inputs):
    _, _, config, output, site, value, _, pdb = endpoint_inputs
    pdb.write_text(TOY_PDB.replace(" C1  LIG", " C2  LIG"))
    with pytest.raises(WorkflowError, match="atom 0 disagrees"):
        plan.plan(config, output, site)
    value["starts"][0].pop("pdb")
    value["starts"][0].update({"trajectory": str(pdb), "trajectory_topology": str(pdb), "frame": -1})
    write(config, value)
    with pytest.raises(WorkflowError, match="nonnegative frame"):
        plan.plan(config, output.parent / "other", site)


def test_ligand_and_water_alone_are_not_a_solvated_complex(endpoint_inputs):
    _, _, config, output, site, _, top, pdb = endpoint_inputs
    pdb.write_text("\n".join(line for line in TOY_PDB.splitlines() if " ALA " not in line) + "\n")
    top.write_text(TOY_TOP.replace("[ moleculetype ]\nPRO 3\n[ atoms ]\n1 C 1 ALA CA 1 0.0 12.011\n", "")
                            .replace("PRO 1\n", ""))
    with pytest.raises(WorkflowError, match="protein and separate ligand"):
        plan.plan(config, output, site)


def test_replica_exchange_netcdf_requires_explicit_extracted_pdb(endpoint_inputs):
    _, _, config, output, site, value, _, pdb = endpoint_inputs
    value["starts"][0].pop("pdb")
    value["starts"][0].update({"trajectory": str(config.parent / "B-group.nc"),
                                "trajectory_topology": str(pdb), "frame": 0})
    write(config, value)
    with pytest.raises(WorkflowError, match="extract an explicit PDB first"):
        plan.plan(config, output, site)
    value["starts"][0]["source_group"] = {"leg": "B", "index": 0}
    write(config, value)
    with pytest.raises(WorkflowError, match="unknown keys .*source_group"):
        plan.plan(config, output.parent / "other", site)


def test_frozen_inputs_and_endpoint_runtime_cannot_change(endpoint_inputs):
    _, _, config, output, site, _, _, _ = endpoint_inputs
    plan.plan(config, output, site)
    frozen = plan.load(output)
    start = output / frozen["starts"][0]["start_pdb"]
    original = start.read_bytes()
    start.write_bytes(original + b"changed")
    with pytest.raises(WorkflowError, match="Missing or changed immutable"):
        plan.load(output)
    start.write_bytes(original)
    runtime = read(output / "runtime.json")
    runtime["stable"]["endpoint_version"] = "wrong"
    write(output / "runtime.json", runtime)
    with pytest.raises(WorkflowError, match="runtime source changed"):
        execution.run_task(output, frozen["tasks"][0]["id"], read(site), platform="Reference")


def test_no_custom_restraint_or_alchemical_force(endpoint_inputs):
    from openmm import CustomExternalForce, CustomBondForce, System
    from openmm.app import GromacsTopFile, PDBFile
    _, _, _, _, _, _, top, pdb = endpoint_inputs
    box = PDBFile(str(pdb)).topology.getPeriodicBoxVectors()
    system = GromacsTopFile(str(top), periodicBoxVectors=box).createSystem()
    execution.reject_restraints(system)
    system.addForce(CustomExternalForce("k*x*x"))
    with pytest.raises(WorkflowError, match="unsupported restraint"):
        execution.reject_restraints(system)
    other = System()
    other.addParticle(12)
    other.addForce(CustomBondForce("0.5*k*(r-r0)^2"))
    with pytest.raises(WorkflowError, match="unsupported restraint"):
        execution.reject_restraints(other)


def test_all_stochastic_streams_are_frozen_and_set_in_openmm(endpoint_inputs):
    from openmm import MonteCarloBarostat
    _, _, config, output, site, value, _, _ = endpoint_inputs
    value["starts"].append({**value["starts"][0], "id": "other-start"})
    write(config, value)
    plan.plan(config, output, site)
    frozen = plan.load(output)
    seeds = [task[key] for task in frozen["tasks"]
             for key in ("seed", "integrator_seed", "barostat_seed")]
    assert len(seeds) == 12 and len(set(seeds)) == 12
    plan.plan(config, output.parent / "same-science", site)
    assert plan.load(output.parent / "same-science")["tasks"] == frozen["tasks"]
    first = frozen["tasks"][0]
    simulation = execution.build_simulation(output, frozen["starts"][0], frozen["protocol"],
                                            first, platform_name="Reference")
    assert simulation.integrator.getRandomNumberSeed() == first["integrator_seed"]
    barostat = next(force for force in simulation.system.getForces()
                    if isinstance(force, MonteCarloBarostat))
    assert barostat.getRandomNumberSeed() == first["barostat_seed"]


def test_reference_execution_semantic_completion_and_corruption(endpoint_inputs, monkeypatch):
    _, _, config, output, site, _, _, _ = endpoint_inputs
    plan.plan(config, output, site)
    frozen = plan.load(output)
    task = frozen["tasks"][0]
    result = execution.run_task(output, task["id"], read(site), platform="Reference")
    assert result == {"state": "complete", "steps": 10, "frames": 10}
    marker = execution.unit_dir(output, task) / "completed.json"
    assert read(marker)["runtime_stable"] == read(output / "runtime.json")["stable"]
    assert read(marker)["integrator_seed"] == task["integrator_seed"]
    assert read(marker)["barostat_seed"] == task["barostat_seed"]
    monkeypatch.setattr(execution, "trajectory_frames", lambda *a, **kw: pytest.fail("status opened DCD"))
    assert execution.status(output)["tasks"][task["id"]]["state"] == "recorded_unverified"
    trajectory = marker.parent / "trajectory.dcd"
    original_trajectory = trajectory.read_bytes()
    trajectory.write_bytes(original_trajectory[:-1])
    with pytest.raises(WorkflowError, match="trajectory missing or changed size"):
        execution.status(output)
    trajectory.write_bytes(original_trajectory)
    checkpoint = marker.parent / "checkpoint.chk"
    original_checkpoint = checkpoint.read_bytes()
    checkpoint.write_bytes(b"corrupt")
    with pytest.raises(WorkflowError, match="checkpoint/final state missing or corrupt"):
        execution.status(output)
    checkpoint.write_bytes(original_checkpoint)
    value = read(marker)
    value["seed"] += 1
    write(marker, value)
    with pytest.raises(WorkflowError, match="completion identity mismatch"):
        execution.status(output)


def test_barostat_seed_mismatch_blocks_checkpoint_resume(endpoint_inputs, monkeypatch):
    _, _, config, output, site, _, _, _ = endpoint_inputs
    plan.plan(config, output, site)
    task = plan.load(output)["tasks"][0]
    original = execution.build_simulation
    def interrupted(*args, **kwargs):
        simulation = original(*args, **kwargs)
        step = simulation.step
        def once(count):
            if simulation.currentStep >= 1:
                raise RuntimeError("stop")
            return step(count)
        simulation.step = once
        return simulation
    with monkeypatch.context() as patch:
        patch.setattr(execution, "build_simulation", interrupted)
        with pytest.raises(RuntimeError, match="stop"):
            execution.run_task(output, task["id"], read(site), platform="Reference")
    progress_path = execution.unit_dir(output, task) / "progress.json"
    progress = read(progress_path)
    assert progress["integrator_seed"] == task["integrator_seed"]
    progress["barostat_seed"] += 1
    write(progress_path, progress)
    with pytest.raises(WorkflowError, match="cannot resume: Checkpoint progress"):
        execution.run_task(output, task["id"], read(site), platform="Reference")


def test_active_writer_status_does_not_inspect_trajectory(endpoint_inputs, monkeypatch):
    _, _, config, output, site, _, _, _ = endpoint_inputs
    plan.plan(config, output, site)
    task = plan.load(output)["tasks"][0]
    directory = execution.unit_dir(output, task)
    directory.mkdir(parents=True)
    lock_path = directory / "writer.lock"
    monkeypatch.setattr(execution, "trajectory_frames", lambda *a, **kw: pytest.fail("opened active DCD"))
    with lock_path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert execution.status(output)["tasks"][task["id"]]["state"] == "active"


def test_checkpoint_resume_and_no_silent_restart(endpoint_inputs, monkeypatch):
    _, _, config, output, site, _, _, _ = endpoint_inputs
    plan.plan(config, output, site)
    task = plan.load(output)["tasks"][0]
    original = execution.build_simulation
    def interrupted(*args, **kwargs):
        simulation = original(*args, **kwargs)
        step = simulation.step
        def once(count):
            if simulation.currentStep >= 1:
                raise RuntimeError("synthetic interruption")
            return step(count)
        simulation.step = once
        return simulation
    with monkeypatch.context() as patch:
        patch.setattr(execution, "build_simulation", interrupted)
        with pytest.raises(RuntimeError, match="synthetic interruption"):
            execution.run_task(output, task["id"], read(site), platform="Reference")
    assert execution.task_state(output, plan.load(output), task, semantic=True)["steps"] == 1
    assert execution.run_task(output, task["id"], read(site), platform="Reference")["state"] == "complete"
    assert execution.trajectory_frames(execution.unit_dir(output, task) / "trajectory.dcd",
                                       output / plan.load(output)["starts"][0]["start_pdb"]) == 10
    second = plan.load(output)["tasks"][1]
    other = execution.unit_dir(output, second)
    other.mkdir(parents=True)
    (other / "trajectory.dcd").write_bytes(b"partial")
    with pytest.raises(WorkflowError, match="cannot resume"):
        execution.run_task(output, second["id"], read(site), platform="Reference")


def test_corrupted_partial_checkpoint_blocks_resume(endpoint_inputs, monkeypatch):
    _, _, config, output, site, _, _, _ = endpoint_inputs
    plan.plan(config, output, site)
    task = plan.load(output)["tasks"][0]
    original = execution.build_simulation
    def interrupted(*args, **kwargs):
        simulation = original(*args, **kwargs)
        step = simulation.step
        def once(count):
            if simulation.currentStep >= 1:
                raise RuntimeError("stop")
            return step(count)
        simulation.step = once
        return simulation
    with monkeypatch.context() as patch:
        patch.setattr(execution, "build_simulation", interrupted)
        with pytest.raises(RuntimeError, match="stop"):
            execution.run_task(output, task["id"], read(site), platform="Reference")
    (execution.unit_dir(output, task) / "checkpoint.chk").write_bytes(b"corrupted")
    with pytest.raises(WorkflowError, match="cannot resume: Checkpoint progress"):
        execution.run_task(output, task["id"], read(site), platform="Reference")


def test_submission_scripts_and_scheduler_are_explicit_without_throttle(endpoint_inputs):
    _, _, config, output, site, _, _, _ = endpoint_inputs
    plan.plan(config, output, site)
    preview = submission.submit(output, site, dry_run=True)
    commands = read(Path(preview["attempt"]) / "submission_plan.json")["commands"]
    assert len(commands) == 2
    assert len({c["script"] for c in commands}) == 2
    for command in commands:
        assert not any(arg.startswith("--array") for arg in command["argv"])
        assert sum("--gres=gpu:" in arg or "--gpus=" in arg for arg in command["argv"]) == 1
        script_path = output / command["script"]
        text = script_path.read_text()
        assert "FELIS_REPO=" in text and "PYTHONPATH=" in text and f"cd {output}" in text
        assert "felis_endpoint.cli run-unit" in text
        assert sha256(script_path) == command["script_sha256"]
    task = plan.load(output)["tasks"][0]
    attempt = Path(preview["attempt"])
    execution.verify_attempt(output, task["id"], attempt / "site.json", read(attempt / "site.json"),
                             read(output / "runtime.json"))
    path = output / commands[0]["script"]
    path.write_text(path.read_text() + "# changed\n")
    with pytest.raises(WorkflowError, match="script differs"):
        execution.verify_attempt(output, task["id"], attempt / "site.json", read(attempt / "site.json"),
                                 read(output / "runtime.json"))


def test_concurrent_submit_and_resume_cannot_select_duplicate_tasks(endpoint_inputs, monkeypatch):
    _, _, config, output, site, _, _, _ = endpoint_inputs
    plan.plan(config, output, site)
    monkeypatch.setattr(submission, "check_runner", lambda *args: read(output / "runtime.json"))
    inside_first_submission, release_first_submission = Event(), Event()
    submitted = []

    def sbatch(argv, *, text):
        assert argv[0] == "sbatch"
        submitted.append(argv)
        if len(submitted) == 1:
            inside_first_submission.set()
            assert release_first_submission.wait(timeout=10)
        return f"{1000 + len(submitted)}\n"

    monkeypatch.setattr(submission.subprocess, "check_output", sbatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(submission.submit, output, site)
        try:
            assert inside_first_submission.wait(timeout=10)
            second = pool.submit(submission.submit, output, site, resume=True)
            with pytest.raises(WorkflowError, match="Another process owns .*execution.lock"):
                second.result(timeout=10)
        finally:
            release_first_submission.set()
        assert first.result(timeout=10)["tasks"] == 2
    assert len(submitted) == 2  # One submission for each planned task, not two attempts.
    attempts = list((output / "attempts").glob("*/attempt.json"))
    assert len(attempts) == 1 and not read(attempts[0])["dry_run"]
