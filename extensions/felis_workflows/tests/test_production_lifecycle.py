"""Independent preproduction chains and conservative PR4 continuation."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import pytest

from felis_workflows import artifacts, orchestration, worker
from felis_workflows.backends import common, slurm
from felis_workflows.common import WorkflowError, digest, read, sha256, write
from felis_workflows.config import protocol_config, site_config
from felis_workflows.planning import abfe_config, load_run, seed_for, units, workdir
from test_artifacts_restart import (fake_iterations, global_ready, make_trajectory, runtime_value, system_ready)


@pytest.fixture
def three(cycle):
    root, original = cycle
    science = deepcopy(original)
    science["calculations"] = [c for c in science["calculations"] if c["id"] == "L_in_R"]
    for leg, count in (("A", 2), ("B", 1)):
        science["ladders"][leg]["groups"] = science["ladders"][leg]["groups"][:count]
        for calc in science["calculations"]:
            calc["seeds"][leg] = calc["seeds"][leg][:count]
    write(root / "science.json", science)
    write(root / "science.lock.json", {"sha256": sha256(root / "science.json")})
    assert load_run(root) == science
    return root, science


def test_production_science_and_independent_random_streams(cycle, extension):
    root, science = cycle
    protocol = protocol_config(extension / "protocols/production.yaml")
    assert science["workflow_version"] == "0.2.0"
    assert science["stage_model_version"] == 2
    assert {k: protocol[k] for k in ("replicates", "solvent_ns", "complex_ns", "equilibration_ns",
                                       "electrostatics", "vdw", "restraints", "reuse_solvent")} == {
        "replicates": 3, "solvent_ns": 10.0, "complex_ns": 10.0, "equilibration_ns": 10.0,
        "electrostatics": "e29", "vdw": "v45", "restraints": "r02", "reuse_solvent": True}
    assert [len(science["ladders"][leg]["lambdas"]) for leg in "AB"] == [73, 80]
    assert [len(science["ladders"][leg]["groups"]) for leg in "AB"] == [25, 30]
    assert {c["replica"] for c in science["calculations"]} == {1, 2, 3}
    seeds = set()
    for calc in science["calculations"]:
        cfg = abfe_config(root, science, calc)
        assert cfg["md_eq_nsnapshots"] == 2000  # 2 fs * 2500 = 5 ps per snapshot.
        assert cfg["md_sol_nsnapshots"] == cfg["md_pro_nsnapshots"] == 2000
        assert all(unit["iterations"] == 2000 for leg in "AB" for unit in units(root, science, calc, leg))
        assert calc["prep_seed"] == seed_for(protocol["seed"], calc["key"], "prep")
        streams = [calc["prep_seed"], *(seed_for(calc["prep_seed"], label) for label in (
            "s:filename.stem:sysB_boresch_em", "boresch_npt",
            "s:filename.stem:sysA_em", "s:filename.stem:sysB_boresch_filtered",
            "s:filename.stem:sysB_em"))]
        if calc["solvent_owner"] == calc["key"]:
            streams.extend(calc["seeds"]["A"])
        streams.extend(calc["seeds"]["B"])
        assert len(streams) == len(set(streams))
        assert not seeds.intersection(streams)
        seeds.update(streams)
    for name in ("smoke", "validation", "full-ladder-validation"):
        assert protocol_config(extension / "protocols" / f"{name}.yaml")["equilibration_ns"] == 3.0


def test_replica_graph_and_solvent_owner_never_cross(cycle):
    _, science = cycle
    nodes = artifacts.stage_index(science)
    tasks = {task["id"]: task for task in common.graph(science)}
    for calc in science["calculations"]:
        key = calc["key"]
        assert nodes[f"equil:{key}"]["dependencies"] == [f"prep:{key}"]
        prep = tasks["prep__" + key.replace("/", "__")]
        assert prep["dependencies"] == []
        for leg in "AB":
            if leg == "A" and calc["solvent_owner"] != key:
                owner = next(c for c in science["calculations"] if c["key"] == calc["solvent_owner"])
                assert owner["replica"] == calc["replica"]
                assert calc["seeds"]["A"] == owner["seeds"]["A"]
                assert f"A__{key.replace('/', '__')}" not in tasks
                continue
            array = tasks[f"{leg}__{key.replace('/', '__')}"]
            assert array["dependencies"] == [prep["id"]]
            assert nodes[f"group:{key}:{leg}:0"]["dependencies"] == [f"equil:{key}"]
        final = tasks["finalize__" + key.replace("/", "__")]
        assert final["dependencies"] == [
            "A__" + calc["solvent_owner"].replace("/", "__"), "B__" + key.replace("/", "__")]
        assert all(f"r{calc['replica']}" in dep for dep in final["dependencies"])


def test_three_replica_slurm_submission_has_no_throttle_or_prep_lanes(three, site):
    root, science = three
    profile = site_config(site)
    tasks = common.graph(science)
    assert len([t for t in tasks if t["kind"] == "prep"]) == 3
    attempt = root / "executions" / "synthetic"
    write(attempt / "attempt.json", {"workflow_graph_id": digest(artifacts.stage_graph(science))})
    slurm.submit(root, profile, attempt, tasks, dry_run=True)
    submitted = read(attempt / "submission_plan.json")["tasks"]
    ids = {entry["task"]["id"]: entry for entry in submitted}
    jobids = {entry["task"]["id"]: f"DRY_{number}"
              for number, entry in enumerate(submitted, 1)}
    assert all(not entry["dependencies"] for entry in submitted if entry["task"]["kind"] == "prep")
    for replica in (1, 2, 3):
        key = f"L_in_R__r{replica}"
        for leg in "AB":
            item = ids[f"{leg}__{key}"]
            assert item["dependencies"] == [jobids[f"prep__{key}"]]
            arrays = [arg for arg in item["argv"] if arg.startswith("--array=")]
            assert len(arrays) == 1 and "%" not in arrays[0]
        final = ids[f"finalize__{key}"]
        assert len(final["dependencies"]) == 2
        assert set(final["dependencies"]) == {jobids[f"{leg}__{key}"] for leg in "AB"}
    assert all("array_concurrency" not in read(path).get("slurm", {}) and
               "prep_concurrency" not in read(path).get("slurm", {})
               for path in (Path(__file__).parents[1] / "sites").glob("*.yaml"))


@pytest.mark.parametrize("obsolete", ("array_concurrency", "prep_concurrency"))
def test_obsolete_slurm_throttle_settings_are_rejected(site, obsolete):
    configured = read(site)
    configured["slurm"][obsolete] = 1
    write(site, configured)
    with pytest.raises(WorkflowError, match="unknown keys"):
        site_config(site)


def test_resume_after_preparation_commits_only_equilibration(three, runtime_value, site, monkeypatch):
    root, science = three
    calc = science["calculations"][0]
    global_ready(root, science, runtime_value)
    system_ready(root, science, runtime_value, calc, equilibrated=False)
    prep_manifest = artifacts.paths(root, calc)
    prep_hash = sha256(prep_manifest)
    assert orchestration.status(root)["calculations"][0]["state"] == "prepared"
    with pytest.raises(WorkflowError, match="Missing equilibration terminal artifact"):
        artifacts.group_dependencies(root, science, calc, "B", 0)
    work = workdir(root, calc)
    trj = work / "trj/sysB_boresch_em.pdb"
    trj.parent.mkdir(parents=True)
    trj.write_text("partial stage")
    assert orchestration.status(root)["calculations"][0]["state"] == "equilibrating"
    calls = []
    def execute_stages(config, stages, selected_site, seed, expected_work):
        calls.append((stages, seed, expected_work))
        assert stages == list(artifacts.EQUIL_STAGES)
        assert seed == calc["prep_seed"] and expected_work == work
        for stage in stages:
            (work / "progress" / f"{stage}.done").touch()
        for leg in "AB":
            (work / "prepare" / f"sys{leg}_em.pdb").write_text("start")
        (work / "prepare/sys_boresch_cfg.json").write_text("{}")
    monkeypatch.setattr(worker, "check_runtime", lambda *args: runtime_value)
    monkeypatch.setattr(worker, "allocation", lambda *args: nullcontext())
    monkeypatch.setattr(worker, "stages", execute_stages)
    worker.prepare_system(root, science, calc, read(site))
    assert len(calls) == 1 and sha256(prep_manifest) == prep_hash
    eq = artifacts.validate_equilibration(root, science, calc)
    assert eq["dependencies"]["system_preparation"] == prep_hash
    assert eq["dependencies"]["preparation_seed"] == calc["prep_seed"]
    assert eq["dependencies"]["equilibration_iterations"] == 2000
    assert artifacts.group_dependencies(root, science, calc, "B", 0) == {
        "equilibration": sha256(artifacts.paths(root, calc, "equil")),
        "group_seed": calc["seeds"]["B"][0]}
    assert orchestration.status(root)["calculations"][0]["state"] == "ready_for_sampling"
    worker.prepare_system(root, science, calc, read(site))
    assert len(calls) == 1


def test_fresh_preproduction_commits_assembly_before_equilibration(three, runtime_value, site, monkeypatch):
    root, science = three
    calc = science["calculations"][0]
    global_ready(root, science, runtime_value)
    write(root / "runtime.json", runtime_value)
    work = workdir(root, calc)
    calls = []
    monkeypatch.setattr(worker, "check_runtime", lambda *args: runtime_value)
    monkeypatch.setattr(worker, "allocation", lambda *args: nullcontext())
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/synthetic/gmx" if name == "gmx" else None)
    monkeypatch.setattr(worker.subprocess, "run", lambda argv, **kw: calls.append(("grompp", argv, kw)))
    import felis_workflows.validation as validation
    monkeypatch.setattr(validation, "validate_assembled", lambda *args: {"A": "validated", "B": "validated"})
    def execute_stages(config, selected, selected_site, seed, expected_work):
        assert seed == calc["prep_seed"] and expected_work == work
        (work / "prepare").mkdir(parents=True, exist_ok=True)
        (work / "progress").mkdir(exist_ok=True)
        if selected == ["makebox"]:
            assert not artifacts.paths(root, calc).exists()
            calls.append(("makebox",))
            for leg in "AB":
                for name in (f"sys{leg}.gro", f"sys{leg}.top", f"sys{leg}_atom_ids.json",
                             f"sys{leg}_ab_ligatoms.json", f"sys{leg}_posres.json"):
                    (work / "prepare" / name).write_text("assembled")
        else:
            assert selected == list(artifacts.EQUIL_STAGES)
            assert artifacts.validate_preparation(root, science, calc)
            assert not artifacts.paths(root, calc, "equil").exists()
            calls.append(("equilibration",))
            for leg in "AB":
                (work / "prepare" / f"sys{leg}_em.pdb").write_text("production start")
            (work / "prepare/sys_boresch_cfg.json").write_text("{}")
        for stage in selected:
            (work / "progress" / f"{stage}.done").touch()
    monkeypatch.setattr(worker, "stages", execute_stages)
    worker.prepare_system(root, science, calc, read(site))
    assert [entry[0] for entry in calls] == ["makebox", "grompp", "grompp", "equilibration"]
    assert all(entry[2]["check"] for entry in calls if entry[0] == "grompp")
    assert artifacts.validate_equilibration(root, science, calc)


def test_group_equilibration_lineage_missing_corrupt_or_wrong_replica(three, runtime_value, monkeypatch):
    root, science = three
    global_ready(root, science, runtime_value)
    first, second = science["calculations"][:2]
    system_ready(root, science, runtime_value, first)
    system_ready(root, science, runtime_value, second)
    unit = make_trajectory(root, science, first, "B", 0)
    import felis_workflows.validation as validation
    monkeypatch.setattr(validation, "iteration_status", fake_iterations(root, science, first, {("B", 0)}))
    artifacts.group_state(root, science, first, "B", unit, semantic=True,
                          producer="test", producer_runtime=runtime_value)
    marker = artifacts.paths(root, first, "B", unit)
    assert read(marker)["dependencies"] == {
        "equilibration": sha256(artifacts.paths(root, first, "equil")),
        "group_seed": first["seeds"]["B"][0]}
    monkeypatch.setattr(validation, "iteration_status", lambda *a, **kw: pytest.fail("status opened NetCDF"))
    saved = read(marker)
    wrong = deepcopy(saved)
    wrong["dependencies"]["equilibration"] = sha256(artifacts.paths(root, second, "equil"))
    write(marker, wrong)
    with pytest.raises(WorkflowError, match="dependency identity mismatch"):
        orchestration.status(root)
    write(marker, saved)
    equil_path = artifacts.paths(root, first, "equil")
    original = equil_path.read_bytes()
    equil_path.write_bytes(artifacts.paths(root, second, "equil").read_bytes())
    with pytest.raises(WorkflowError, match="stage/science/task identity mismatch"):
        orchestration.status(root)
    equil_path.write_bytes(original)
    (workdir(root, first) / "prepare/sysB_em.pdb").write_text("corrupt")
    with pytest.raises(WorkflowError, match="changed immutable"):
        orchestration.status(root)
    equil_path.unlink()
    with pytest.raises(WorkflowError, match="without validated equilibration"):
        orchestration.status(root)


def test_status_and_probe_validate_each_parent_once(three, runtime_value, site, monkeypatch):
    root, science = three
    global_ready(root, science, runtime_value)
    for calc in science["calculations"]:
        system_ready(root, science, runtime_value, calc)
    counts = {"preparation": 0, "equilibration": 0}
    original_prep, original_equil = artifacts.validate_preparation, artifacts.validate_equilibration

    def prep(*args, **kwargs):
        counts["preparation"] += 1
        return original_prep(*args, **kwargs)

    def equil(*args, **kwargs):
        counts["equilibration"] += 1
        return original_equil(*args, **kwargs)

    monkeypatch.setattr(artifacts, "validate_preparation", prep)
    monkeypatch.setattr(artifacts, "validate_equilibration", equil)
    state = orchestration.status(root)
    assert len(state["calculations"]) == 3
    assert all(c["groups"]["A"]["total"] == 2 and c["groups"]["B"]["total"] == 1
               for c in state["calculations"])
    assert counts == {"preparation": 3, "equilibration": 3}

    import felis_workflows.validation as validation
    monkeypatch.setattr(validation, "iteration_status", lambda *a, **kw: {"complete": False})
    monkeypatch.setattr(worker, "check_runtime", lambda *a: runtime_value)
    worker.probe(root, science, read(site), root / "probe.json")
    assert counts == {"preparation": 6, "equilibration": 6}


def test_simulate_group_reuses_verified_parent_and_exact_dependencies(three, runtime_value, site, monkeypatch):
    root, science = three
    calc = science["calculations"][0]
    global_ready(root, science, runtime_value)
    system_ready(root, science, runtime_value, calc)
    counts = {"preparation": 0, "equilibration": 0}
    original_prep, original_equil = artifacts.validate_preparation, artifacts.validate_equilibration

    def prep(*args, **kwargs):
        counts["preparation"] += 1
        return original_prep(*args, **kwargs)

    def equil(*args, **kwargs):
        counts["equilibration"] += 1
        return original_equil(*args, **kwargs)

    monkeypatch.setattr(artifacts, "validate_preparation", prep)
    monkeypatch.setattr(artifacts, "validate_equilibration", equil)
    monkeypatch.setattr(worker, "check_runtime", lambda *a: runtime_value)
    monkeypatch.setattr(worker, "allocation", lambda *a: nullcontext())
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **kw: None)
    seen = []

    def state(*args, **kwargs):
        seen.append(kwargs["dependencies"])
        return {"complete": len(seen) == 2}

    monkeypatch.setattr(worker, "group_state", state)
    worker.simulate_group(root, science, calc, read(site), "B", 0)
    assert counts == {"preparation": 1, "equilibration": 1}
    assert seen == [{"equilibration": sha256(artifacts.paths(root, calc, "equil")),
                     "group_seed": calc["seeds"]["B"][0]}] * 2


def test_final_dependencies_validate_owner_once_for_all_groups(three, runtime_value, monkeypatch):
    root, science = three
    calc = science["calculations"][0]
    global_ready(root, science, runtime_value)
    system_ready(root, science, runtime_value, calc)
    import felis_workflows.validation as validation
    completed = {("A", 0), ("A", 1), ("B", 0)}
    monkeypatch.setattr(validation, "iteration_status", fake_iterations(root, science, calc, completed))
    for leg, index in sorted(completed):
        item = make_trajectory(root, science, calc, leg, index)
        artifacts.group_state(root, science, calc, leg, item, semantic=True,
                              producer="test", producer_runtime=runtime_value)
    counts = {"preparation": 0, "equilibration": 0}
    original_prep, original_equil = artifacts.validate_preparation, artifacts.validate_equilibration

    def prep(*args, **kwargs):
        counts["preparation"] += 1
        return original_prep(*args, **kwargs)

    def equil(*args, **kwargs):
        counts["equilibration"] += 1
        return original_equil(*args, **kwargs)

    monkeypatch.setattr(artifacts, "validate_preparation", prep)
    monkeypatch.setattr(artifacts, "validate_equilibration", equil)
    expected = artifacts.final_dependencies(root, science, calc)
    assert len(expected) == 3
    assert counts == {"preparation": 1, "equilibration": 1}
    assert artifacts.final_dependencies(root, science, calc) == expected
    assert counts == {"preparation": 2, "equilibration": 2}


def test_shared_solvent_final_dependencies_validate_each_distinct_owner_once(cycle, runtime_value, monkeypatch):
    root, original = cycle
    science = deepcopy(original)
    science["calculations"] = [calc for calc in science["calculations"]
                               if calc["key"] in {"L_in_R/r1", "L_in_RP/r1"}]
    owner, consumer = science["calculations"]
    assert consumer["solvent_owner"] == owner["key"]
    for leg, count in (("A", 2), ("B", 1)):
        science["ladders"][leg]["groups"] = science["ladders"][leg]["groups"][:count]
        for calc in science["calculations"]:
            calc["seeds"][leg] = calc["seeds"][leg][:count]
    write(root / "science.json", science)
    write(root / "science.lock.json", {"sha256": sha256(root / "science.json")})
    global_ready(root, science, runtime_value)
    for calc in (owner, consumer):
        system_ready(root, science, runtime_value, calc)
    import felis_workflows.validation as validation

    def reporter(_root, _science, _calc, leg, item, factory=None):
        producer = owner if leg == "A" else consumer
        return fake_iterations(root, science, producer, {(leg, item["index"]) })(
            _root, _science, _calc, leg, item, factory)

    monkeypatch.setattr(validation, "iteration_status", reporter)
    for leg, producer, indices in (("A", owner, (0, 1)), ("B", consumer, (0,))):
        for index in indices:
            item = make_trajectory(root, science, producer, leg, index)
            artifacts.group_state(root, science, producer, leg, item, semantic=True,
                                  producer="test", producer_runtime=runtime_value)
    seen = []
    original_equil = artifacts.validate_equilibration

    def equil(_root, _science, calc, runtime=None):
        seen.append(calc["key"])
        return original_equil(_root, _science, calc, runtime)

    monkeypatch.setattr(artifacts, "validate_equilibration", equil)
    dependencies = artifacts.final_dependencies(root, science, consumer)
    assert len(dependencies) == 3
    assert sorted(seen) == sorted((owner["key"], consumer["key"]))


def test_three_replica_restart_selects_only_needed_stages(three, runtime_value, site, monkeypatch):
    root, science = three
    global_ready(root, science, runtime_value)
    first, second, third = science["calculations"]
    system_ready(root, science, runtime_value, first)
    system_ready(root, science, runtime_value, second, equilibrated=False)
    system_ready(root, science, runtime_value, third)
    import felis_workflows.validation as validation
    for leg, index in (("A", 0), ("B", 0)):
        unit = make_trajectory(root, science, third, leg, index)
        monkeypatch.setattr(validation, "iteration_status", fake_iterations(root, science, third, {(leg, index)}))
        artifacts.group_state(root, science, third, leg, unit, semantic=True,
                              producer="test", producer_runtime=runtime_value)
    original = fake_iterations(root, science, third, {("A", 0), ("B", 0)})
    def reporter(_root, _science, calc, leg, unit, factory=None):
        if calc["key"] != third["key"]:
            return {"complete": False, "reason": "missing trajectory", "stem": unit["stem"]}
        return original(_root, _science, calc, leg, unit, factory)
    monkeypatch.setattr(validation, "iteration_status", reporter)
    monkeypatch.setattr(worker, "check_runtime", lambda *args: runtime_value)
    monkeypatch.setattr(worker, "check_prepared", lambda *args: runtime_value)
    monkeypatch.setattr(worker, "check_equilibrated", lambda *args: runtime_value)
    destination = root / "probe.json"
    worker.probe(root, science, read(site), destination)
    state = read(destination)
    assert (state[first["key"]]["prep"], state[first["key"]]["equil"]) == (True, True)
    assert (state[second["key"]]["prep"], state[second["key"]]["equil"]) == (True, False)
    assert state[third["key"]]["A"] == [True, False]
    tasks = common.incomplete_graph(science, state)
    selected = {task["id"]: task for task in tasks}
    assert "prep__L_in_R__r1" not in selected and "prep__L_in_R__r3" not in selected
    assert selected["prep__L_in_R__r2"]["dependencies"] == []
    assert selected["A__L_in_R__r3"]["indices"] == [1]
    assert selected["finalize__L_in_R__r3"]["dependencies"] == ["A__L_in_R__r3"]
    assert selected["A__L_in_R__r2"]["dependencies"] == ["prep__L_in_R__r2"]
    assert selected["B__L_in_R__r2"]["dependencies"] == ["prep__L_in_R__r2"]
    assert selected["finalize__L_in_R__r1"]["dependencies"] == ["A__L_in_R__r1", "B__L_in_R__r1"]


def test_status_and_legacy_plans_fail_closed(three, runtime_value, monkeypatch):
    root, science = three
    global_ready(root, science, runtime_value)
    calc = science["calculations"][0]
    system_ready(root, science, runtime_value, calc, equilibrated=False)
    import felis_workflows.validation as validation
    monkeypatch.setattr(validation, "iteration_status", lambda *a, **kw: pytest.fail("status opened NetCDF"))
    assert orchestration.status(root)["calculations"][0]["equilibration"] == "not_started"
    nc = workdir(root, calc) / "trj/sysB_boresch_npt.nc"
    nc.parent.mkdir(parents=True)
    nc.write_bytes(b"active trajectory")
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path.suffix == ".nc":
            pytest.fail("status opened active NetCDF")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    partial = orchestration.status(root)["calculations"][0]
    assert partial["equilibration"] == "partial" and partial["state"] == "equilibrating"
    (root / "calculations" / calc["key"] / "prep.ok.json").write_text("legacy")
    with pytest.raises(WorkflowError, match="Legacy PR3 preproduction marker"):
        orchestration.status(root)
    monkeypatch.setattr(worker, "check_runtime", lambda *args: runtime_value)
    with pytest.raises(WorkflowError, match="Legacy PR3 preproduction marker"):
        worker.probe(root, science, {}, root / "probe.json")
    (root / "calculations" / calc["key"] / "prep.ok.json").unlink()
    old_version = deepcopy(science)
    old_version["workflow_version"] = "0.1.0"
    write(root / "science.json", old_version)
    write(root / "science.lock.json", {"sha256": sha256(root / "science.json")})
    with pytest.raises(WorkflowError, match="Workflow/upstream version mismatch"):
        load_run(root)
    old = deepcopy(science)
    del old["stage_model_version"]
    write(root / "science.json", old)
    write(root / "science.lock.json", {"sha256": sha256(root / "science.json")})
    with pytest.raises(WorkflowError, match="Pre-PR4 stage model.*plan a new run"):
        load_run(root)
