from copy import deepcopy
from pathlib import Path
import subprocess
import pytest

from felis_workflows.backends.common import graph, incomplete_graph, worker_script
from felis_workflows.backends.slurm import submission_command, ensure_idle
from felis_workflows.common import WorkflowError, digest, read, write
from felis_workflows.config import campaign_config, forcefield_config, protocol_config, site_config
from felis_workflows.planning import abfe_config, copy_topology, load_run, units


def test_cycle_endpoints_sharing_and_native_tiling(cycle):
    root, science = cycle
    assert len(science["calculations"]) == 12
    assert len(science["ladders"]["A"]["lambdas"]) == 73
    assert len(science["ladders"]["B"]["lambdas"]) == 80
    seeds = []
    for calc in science["calculations"]:
        cfg = abfe_config(root, science, calc)
        assert len(cfg["cofsdfs"]) == len(calc["partners"])
        assert calc["target"] not in calc["partners"]
        for leg in "AB":
            if leg == "B" or calc["key"] == calc["solvent_owner"]:
                seeds.extend(calc["seeds"][leg])
            work_units = units(root, science, calc, leg)
            assert all(u["iterations"] == 2000 for u in work_units)
            edges = [(a, b) for u in work_units for a, b in zip(u["ilam"], u["ilam"][1:])]
            count = len(science["ladders"][leg]["lambdas"])
            assert edges == [(i, i+1) for i in range(count-1)]
    assert len(seeds) == len(set(seeds))
    tasks = graph(science)
    arrays = [t for t in tasks if t["kind"] == "array"]
    assert len(arrays) == 18  # two solvent + four complex legs, three replicates
    last = next(t for t in tasks if t["id"] == "finalize__L_in_RP__r1")
    assert last["dependencies"] == ["A__L_in_R__r1", "B__L_in_RP__r1"]


def test_science_identity_excludes_site(cycle, site, tmp_path):
    root, science = cycle
    before = digest(load_run(root))
    easley = site_config(site)
    hopper = deepcopy(easley)
    hopper.update(name="hopper")
    hopper["slurm"].update(partition="private", gpu_args=["--gres=gpu:1"])
    task = next(t for t in graph(science) if t["kind"] == "array")
    first = submission_command(root, easley, tmp_path, task, tmp_path / "job.sh", ["101"])
    second = submission_command(root, hopper, tmp_path, task, tmp_path / "job.sh", ["101"])
    assert first != second
    assert "--dependency=afterok:101" in first
    assert digest(load_run(root)) == before
    script = worker_script(root, easley, site, "group", ["--calculation", "L_in_R/r1", "--leg", "A"], array=True)
    assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0


def test_resume_only_missing_groups(cycle):
    _, science = cycle
    status = {c["key"]: {"prep": True, "finalize": True,
              **{leg: [True] * len(science["ladders"][leg]["groups"]) for leg in "AB"}}
              for c in science["calculations"]}
    status["L_in_R/r1"]["A"][2] = False
    status["L_in_R/r1"]["finalize"] = False
    status["L_in_RP/r1"]["finalize"] = False
    pending = incomplete_graph(science, status)
    assert len(pending) == 3
    assert pending[0]["indices"] == [2]
    assert pending[-1]["dependencies"] == ["A__L_in_R__r1"]


def test_modified_input_and_manifest_rejected(cycle):
    root, _ = cycle
    path = root / "inputs/receptor.pdb"
    original = path.read_bytes()
    path.write_text("changed")
    with pytest.raises(WorkflowError, match="changed immutable"):
        load_run(root)
    path.write_bytes(original)
    (root / "science.json").write_text("{}")
    with pytest.raises(WorkflowError, match="manifest changed"):
        load_run(root)


def test_topology_include_snapshot_relocatable(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "x.top").write_text('#include "molecule.itp"\n[ system ]\ntest\n')
    (source / "molecule.itp").write_text("[ moleculetype ]\nLIG 3\n")
    result = copy_topology(source / "x.top", tmp_path / "frozen/x.top")
    assert str(source) not in result.read_text()
    included = result.parent / result.read_text().split('"')[1]
    assert included.read_text() == (source / "molecule.itp").read_text()
    import shutil
    shutil.copytree(source, tmp_path / "other-machine")
    other = copy_topology(tmp_path / "other-machine/x.top", tmp_path / "frozen2/x.top")
    assert result.read_text() == other.read_text()
    (source / "molecule.itp").unlink()
    with pytest.raises(WorkflowError, match="Missing include"):
        copy_topology(source / "x.top", tmp_path / "another/x.top")


@pytest.mark.parametrize("gpu_args", [["--gpus=2"], ["--gpus=1", "--gres=gpu:1"], ["--gres=gpu:1,other:1"]])
def test_one_gpu_per_worker(site, gpu_args):
    s = read(site)
    s["slurm"]["gpu_args"] = gpu_args
    write(site, s)
    with pytest.raises(WorkflowError, match="one GPU"):
        site_config(site)


def test_science_keys_forbidden_in_site(site):
    s = read(site)
    s["solvent_ns"] = 5
    write(site, s)
    with pytest.raises(WorkflowError, match="unknown keys"):
        site_config(site)


def test_wrong_conditional_endpoint_rejected(extension, tmp_path):
    c = read(extension / "campaigns/coupling.yaml")
    c["calculations"][-1]["partners"] = []
    path = tmp_path / "bad.json"
    write(path, c)
    with pytest.raises(WorkflowError, match="endpoint"):
        campaign_config(path)


def test_profiles_load(extension, tmp_path):
    for path in (extension / "campaigns").glob("*.yaml"):
        campaign_config(path)
    for path in (extension / "forcefields").glob("*.yaml"):
        forcefield_config(path)
    for path in (extension / "protocols").glob("*.yaml"):
        protocol_config(path)
    p = read(extension / "protocols/validation.yaml")
    p["temperature_K"] = 300
    path = tmp_path / "bad.json"
    write(path, p)
    with pytest.raises(WorkflowError, match="298.15"):
        protocol_config(path)


def test_active_job_blocks_resume_before_trajectory_read(cycle, site, monkeypatch):
    from felis_workflows.backends.common import snapshot
    root, _ = cycle
    s = site_config(site)
    attempt = snapshot(root, s, "submit")
    write(attempt / "jobs.json", {"jobs": [{"job_id": "456", "kind": "array"}]})
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "456_2|RUNNING|None\n")
    with pytest.raises(WorkflowError, match="active trajectories"):
        ensure_idle(root, s, True)


def test_resume_preview_does_not_cancel_pending_jobs(cycle, site, monkeypatch):
    from felis_workflows.backends.common import snapshot
    root, _ = cycle
    s = site_config(site)
    attempt = snapshot(root, s, "submit")
    write(attempt / "jobs.json", {"jobs": [{"job_id": "456", "kind": "finalize"}]})
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "456|PENDING|DependencyNeverSatisfied\n")
    cancelled = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: cancelled.append(cmd))
    ensure_idle(root, s, True, cancel_pending=False)
    assert not cancelled
    ensure_idle(root, s, True)
    assert cancelled == [["scancel", "456"]]
