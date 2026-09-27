"""Terminal artifact contracts and conservative, scheduler-safe restart."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from felis_workflows import artifacts, orchestration, worker
from felis_workflows.backends import common as backend_common, slurm
from felis_workflows.common import WorkflowError, digest, read, sha256, write
from felis_workflows.config import site_config
from felis_workflows.planning import load_run, units, workdir


@pytest.fixture
def small_run(cycle):
    root, science = cycle
    science = deepcopy(science)
    science["calculations"] = science["calculations"][:1]
    science["ladders"]["A"]["groups"] = science["ladders"]["A"]["groups"][:2]
    science["ladders"]["B"]["groups"] = science["ladders"]["B"]["groups"][:1]
    for leg in "AB":
        science["calculations"][0]["seeds"][leg] = science["calculations"][0]["seeds"][leg][:len(science["ladders"][leg]["groups"])]
    write(root / "science.json", science)
    write(root / "science.lock.json", {"sha256": sha256(root / "science.json")})
    assert load_run(root) == science
    return root, science


@pytest.fixture
def runtime_value(repo):
    return {"versions": {"openmm": "test"}, "source_hashes": {"felis/__init__.py": "stable"},
            "source_identity": {"python_runtime": {"implementation": "cpython", "version": [3, 12, 14],
                                                   "cache_tag": "cpython-312"},
                                "repository": str(repo), "git_commit": "old",
                                "python_executable": "/old/bin/python",
                                "felis_source": str(repo / "felis/__init__.py"),
                                "felis_workflows_source": str(repo / "extensions/felis_workflows/src/felis_workflows/__init__.py"),
                                "bytemol_source": str(repo / "submodule/bytemol/bytemol/__init__.py")}}


def global_ready(root, science, runtime):
    outputs = []
    for name in ["receptor/protein.gro", "receptor/protein.top"] + [
        f"parameters/{ligand}/ligand.{ext}" for ligand in science["campaign"]["ligands"]
        for ext in ("sdf", "itp")]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        outputs.append(path)
    artifacts.commit_manifest(root / "prepared.json", root, science, "global_inputs", outputs,
                              artifacts.global_dependencies(science), runtime, "test:prepare")


def system_ready(root, science, runtime, calc=None):
    calc = calc or science["calculations"][0]
    directory, work = root / "calculations" / calc["key"], workdir(root, calc)
    work.joinpath("prepare").mkdir(parents=True)
    for stage in artifacts.PREP_STAGES:
        marker = work / "progress" / f"{stage}.done"
        marker.parent.mkdir(exist_ok=True)
        marker.touch()
    outputs = []
    for leg in "AB":
        for ext in ("gro", "top"):
            path = work / "prepare" / f"sys{leg}.{ext}"
            path.write_text(f"{leg}.{ext}")
            outputs.append(path)
        for suffix in ("em.pdb", "atom_ids.json", "lam.json", "ab_ligatoms.json"):
            path = work / "prepare" / f"sys{leg}_{suffix}"
            path.write_text(f"{leg}_{suffix}")
            outputs.append(path)
    boresch = work / "prepare/sys_boresch_cfg.json"
    boresch.write_text("{}")
    outputs.append(boresch)
    for name in ("abfecfg.json", "assembly_validation.json",
                 f'input/{calc["target"]}.sdf', f'input/{calc["target"]}.itp'):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        outputs.append(path)
    write(directory / "location.json", {"root": str(root), "science_id": digest(science)})
    write(root / "runtime.json", runtime)
    artifacts.commit_manifest(artifacts.paths(root, calc), root, science, "system_preparation", outputs,
                              artifacts.prep_dependencies(root, science, calc), runtime,
                              "test:prep", calculation=calc["key"], task=f"prep:{calc['key']}")
    return calc


def fake_iterations(root, science, calc, complete):
    def status(_root, _science, _calc, leg, unit, _factory=None):
        owner = workdir(root, calc)
        nc = owner / "trj" / f'{unit["stem"]}.nc'
        if not nc.exists():
            return {"complete": False, "reason": "missing trajectory", "stem": unit["stem"]}
        end = unit["iterations"] if (leg, unit["index"]) in complete else unit["iterations"] - 1
        return {"complete": end == unit["iterations"], "last_iteration": end,
                "last_checkpoint": end, "stored_target": unit["iterations"],
                "stem": unit["stem"], "trajectory": str(nc)}
    return status


def make_trajectory(root, science, calc, leg, index):
    unit = units(root, science, calc, leg)[index]
    nc = workdir(root, calc) / "trj" / f'{unit["stem"]}.nc'
    nc.parent.mkdir(parents=True, exist_ok=True)
    nc.write_bytes(b"checkpointed; never hash or inspect during status")
    nc.with_suffix(".create_done").touch()
    return unit


def test_stage_graph_and_write_once(small_run, tmp_path, runtime_value):
    root, science = small_run
    graph = artifacts.stage_graph(science)
    assert next(v for v in graph if v["id"].startswith("prep:"))["dependencies"] == [
        "global:receptor", *[f"global:ligand:{k}" for k in science["campaign"]["ligands"]]]
    assert next(v for v in graph if v["id"].startswith("finalize:"))["dependencies"] == [
        f"group:{science['calculations'][0]['key']}:A:0", f"group:{science['calculations'][0]['key']}:A:1",
        f"group:{science['calculations'][0]['key']}:B:0"]
    output = root / "test.txt"
    output.write_text("good")
    manifest = root / "artifact.json"
    args = (manifest, root, science, "global_inputs", [output], {}, runtime_value, "attempt")
    artifacts.commit_manifest(*args)
    assert artifacts.commit_manifest(*args)["producer"] == "attempt"
    with pytest.raises(WorkflowError, match="Conflicting committed"):
        artifacts.commit_manifest(*args[:-1], "other-attempt")
    output.write_text("changed")
    with pytest.raises(WorkflowError, match="changed immutable"):
        artifacts.validate_manifest(manifest, root, science, "global_inputs")


def test_manifest_rejects_escape_science_runtime_and_legacy(small_run, runtime_value):
    root, science = small_run
    output = root / "artifact.txt"
    output.write_text("stable")
    marker = root / "artifact.json"
    artifacts.commit_manifest(marker, root, science, "global_inputs", [output], {}, runtime_value, "attempt")
    relocated = deepcopy(runtime_value)
    relocated["source_identity"].update(repository="/another/checkout", git_commit="new",
                                          python_executable="/new/bin/python")
    assert artifacts.validate_manifest(marker, root, science, "global_inputs", runtime=relocated)
    changed = deepcopy(runtime_value)
    changed["source_hashes"]["felis/__init__.py"] = "different"
    with pytest.raises(WorkflowError, match="runtime compatibility"):
        artifacts.validate_manifest(marker, root, science, "global_inputs", runtime=changed)
    changed = deepcopy(runtime_value)
    changed["source_identity"]["python_runtime"]["version"] = [3, 13, 0]
    with pytest.raises(WorkflowError, match="runtime compatibility"):
        artifacts.validate_manifest(marker, root, science, "global_inputs", runtime=changed)
    with pytest.raises(WorkflowError, match="science/task identity"):
        artifacts.validate_manifest(marker, root, {**science, "other": True}, "global_inputs")
    altered = read(marker)
    altered["hashes"] = {"../escape": sha256(output)}
    write(marker, altered)
    with pytest.raises(WorkflowError, match="Invalid artifact path"):
        artifacts.validate_manifest(marker, root, science, "global_inputs")
    write(marker, {"science_id": digest(science), "hashes": {}})
    with pytest.raises(WorkflowError, match="Legacy"):
        artifacts.validate_manifest(marker, root, science, "global_inputs")


def test_global_and_partial_prep_integrity(small_run, runtime_value, site, monkeypatch):
    root, science = small_run
    calc = science["calculations"][0]
    directory = root / "calculations" / calc["key"]
    directory.mkdir(parents=True)
    report = orchestration.status(root)
    assert report["global_inputs"] == "not_started"
    assert report["calculations"][0]["system_preparation"] == "partial"
    global_ready(root, science, runtime_value)
    monkeypatch.setattr(orchestration, "activate_source", lambda _: None)
    assert orchestration.prepare(root, site)["reused"]
    assert orchestration.status(root)["global_inputs"] == "complete"
    system_ready(root, science, runtime_value)
    assert artifacts.validate_preparation(root, science, calc)
    assert orchestration.status(root)["calculations"][0]["system_preparation"] == "complete"
    (workdir(root, calc) / "prepare/sysB.top").write_text("corrupt")
    with pytest.raises(WorkflowError, match="changed immutable"):
        artifacts.validate_preparation(root, science, calc)
    with pytest.raises(WorkflowError, match="changed immutable"):
        orchestration.status(root)
    monkeypatch.setattr(worker, "check_runtime", lambda *_: runtime_value)
    with pytest.raises(WorkflowError, match="changed immutable"):
        worker.prepare_system(root, science, calc, read(site))


def test_partial_preparation_requires_original_runtime_identity(small_run, runtime_value, site):
    root, science = small_run
    global_ready(root, science, runtime_value)
    calc = science["calculations"][0]
    write(root / "calculations" / calc["key"] / "location.json",
          {"root": str(root), "science_id": digest(science)})
    with pytest.raises(WorkflowError, match="lacks its runtime identity"):
        worker.prepare_system(root, science, calc, read(site))


def test_group_records_restart_only_incomplete_and_detect_stale(small_run, runtime_value, monkeypatch):
    root, science = small_run
    global_ready(root, science, runtime_value)
    calc = system_ready(root, science, runtime_value)
    for leg, count in (("A", 2), ("B", 1)):
        for index in range(count):
            make_trajectory(root, science, calc, leg, index)
    import felis_workflows.validation as validation
    complete = {("A", 0), ("B", 0)}
    monkeypatch.setattr(validation, "iteration_status", fake_iterations(root, science, calc, complete))
    monkeypatch.setattr(worker, "check_prepared", lambda *args: runtime_value)
    probe_path = root / "probe.json"
    worker.probe(root, science, {"_attempt_id": "resume-1"}, probe_path)
    state = read(probe_path)
    assert state[calc["key"]]["A"] == [True, False]
    assert state[calc["key"]]["B"] == [True]
    from felis_workflows.backends.common import incomplete_graph
    pending = incomplete_graph(science, state)
    assert [(t["kind"], t.get("indices")) for t in pending] == [("array", [1]), ("finalize", None)]
    assert artifacts.paths(root, calc, "A", units(root, science, calc, "A")[0]).exists()
    saved = artifacts.paths(root, calc, "A", units(root, science, calc, "A")[0])
    changed = read(saved)
    changed["checkpoint"]["stored_target"] += 1
    write(saved, changed)
    with pytest.raises(WorkflowError, match="does not match its unit"):
        orchestration.status(root)
    changed["checkpoint"]["stored_target"] -= 1
    write(saved, changed)
    complete.remove(("A", 0))
    monkeypatch.setattr(validation, "iteration_status", fake_iterations(root, science, calc, complete))
    with pytest.raises(WorkflowError, match="invalid checkpoint"):
        worker.probe(root, science, {"_attempt_id": "resume-2"}, probe_path)


def test_producer_source_diagnostics_on_relocated_group_and_finalization(small_run, runtime_value, monkeypatch):
    root, science = small_run
    global_ready(root, science, runtime_value)
    calc = system_ready(root, science, runtime_value)
    current = deepcopy(runtime_value)
    current["source_identity"].update(repository="/relocated/felis", git_commit="new-revision",
                                      python_executable="/other/python", felis_source="/relocated/felis/felis/__init__.py")
    complete = {("A", 0), ("A", 1), ("B", 0)}
    for leg, index in complete:
        make_trajectory(root, science, calc, leg, index)
    import felis_workflows.validation as validation
    monkeypatch.setattr(validation, "iteration_status", fake_iterations(root, science, calc, complete))
    monkeypatch.setattr(worker, "check_runtime", lambda *args: current)
    worker.probe(root, science, {"_attempt_id": "relocated-attempt"}, root / "probe.json")
    group = read(artifacts.paths(root, calc, "A", units(root, science, calc, "A")[0]))
    assert group["source_diagnostics"] == current["source_identity"]
    assert group["runtime_compatibility"] == artifacts.compatibility(runtime_value)
    assert group["producer"] == "relocated-attempt"

    monkeypatch.setattr(validation, "partner_occupancy", lambda *args: {"passed": True})
    def stub(module_name, function_name, callback):
        module = ModuleType(module_name)
        setattr(module, function_name, callback)
        monkeypatch.setitem(sys.modules, module_name, module)
    prefix = "felis.protocols.abfe."
    stub(prefix + "main_fe_mbar", "calc_mbar", lambda *, stem, outdir, **kw:
         (Path(outdir) / f"{stem}_fe_table.tsv").write_text("data\n"))
    stub(prefix + "main_fe_restraints", "calc_restraints", lambda *, outdir, **kw:
         (Path(outdir) / "R_fe_table.tsv").write_text("data\n"))
    stub(prefix + "main_fe_summarize", "summarize_fe", lambda *, workdir, **kw:
         (Path(workdir) / "sys_abfe.tsv").write_text("ligand\tdG(kcal/mol)\nL\t-1.0\n"))
    worker.finalize(root, science, calc, {"_attempt_id": "relocated-attempt"})
    final = read(artifacts.paths(root, calc, "final"))
    assert final["source_diagnostics"] == current["source_identity"]
    assert final["runtime_compatibility"] == artifacts.compatibility(runtime_value)
    assert artifacts.validate_final(root, science, calc)


def test_recorded_group_requires_trajectory_without_opening_netcdf(small_run, runtime_value, monkeypatch):
    root, science = small_run
    global_ready(root, science, runtime_value)
    calc = system_ready(root, science, runtime_value)
    unit = make_trajectory(root, science, calc, "A", 0)
    import felis_workflows.validation as validation
    monkeypatch.setattr(validation, "iteration_status", fake_iterations(root, science, calc, {("A", 0)}))
    artifacts.group_state(root, science, calc, "A", unit, semantic=True,
                          producer="attempt", producer_runtime=runtime_value)
    trajectory = workdir(root, calc) / "trj" / f'{unit["stem"]}.nc'
    trajectory.unlink()
    monkeypatch.setattr(validation, "iteration_status", lambda *a, **kw: pytest.fail("NetCDF opened"))
    with pytest.raises(WorkflowError, match="Recorded checkpointed trajectory is missing"):
        artifacts.group_state(root, science, calc, "A", unit)
    with pytest.raises(WorkflowError, match="Recorded checkpointed trajectory is missing"):
        orchestration.status(root)


def test_new_group_requires_current_compatible_producer_runtime(small_run, runtime_value, monkeypatch):
    root, science = small_run
    global_ready(root, science, runtime_value)
    calc = system_ready(root, science, runtime_value)
    unit = make_trajectory(root, science, calc, "B", 0)
    import felis_workflows.validation as validation
    monkeypatch.setattr(validation, "iteration_status", fake_iterations(root, science, calc, {("B", 0)}))
    with pytest.raises(WorkflowError, match="current producer runtime"):
        artifacts.group_state(root, science, calc, "B", unit, semantic=True, producer="attempt")
    incompatible = deepcopy(runtime_value)
    incompatible["source_hashes"]["felis/__init__.py"] = "changed"
    with pytest.raises(WorkflowError, match="Producer runtime differs"):
        artifacts.group_state(root, science, calc, "B", unit, semantic=True,
                              producer="attempt", producer_runtime=incompatible)
    assert not artifacts.paths(root, calc, "B", unit).exists()


def test_orphaned_and_malformed_group_records_are_not_accepted(small_run, runtime_value, monkeypatch):
    root, science = small_run
    global_ready(root, science, runtime_value)
    calc = system_ready(root, science, runtime_value)
    unit = make_trajectory(root, science, calc, "A", 0)
    import felis_workflows.validation as validation
    monkeypatch.setattr(validation, "iteration_status", fake_iterations(root, science, calc, {("A", 0)}))
    marker = artifacts.paths(root, calc, "A", unit)
    artifacts.group_state(root, science, calc, "A", unit, semantic=True, producer="attempt",
                          producer_runtime=runtime_value)
    saved = read(marker)
    saved["checkpoint"] = "corrupt"
    write(marker, saved)
    with pytest.raises(WorkflowError, match="does not match its unit"):
        orchestration.status(root)
    write(marker, {**saved, "checkpoint": {"stored_target": unit["iterations"]}})
    with pytest.raises(WorkflowError, match="does not match its unit"):
        artifacts.group_state(root, science, calc, "A", unit)
    write(marker, {**saved, "checkpoint": {"trajectory": str((workdir(root, calc) / "trj" /
            f'{unit["stem"]}.nc').relative_to(root)), "stored_target": unit["iterations"],
            "checkpoint_interval": unit["checkpoint_interval"], "last_iteration": unit["iterations"],
            "last_checkpoint": unit["iterations"]}})
    artifacts.paths(root, calc).unlink()
    with pytest.raises(WorkflowError, match="without validated system preparation"):
        orchestration.status(root)
    with pytest.raises(WorkflowError, match="without validated system preparation"):
        worker.probe(root, science, {"_attempt_id": "attempt"}, root / "probe.json")


def test_final_manifest_and_status_never_read_trajectory(small_run, runtime_value, monkeypatch):
    root, science = small_run
    global_ready(root, science, runtime_value)
    calc = system_ready(root, science, runtime_value)
    complete = {("A", 0), ("A", 1), ("B", 0)}
    for leg, index in complete:
        make_trajectory(root, science, calc, leg, index)
    import felis_workflows.validation as validation
    monkeypatch.setattr(validation, "iteration_status", fake_iterations(root, science, calc, complete))
    for leg, index in sorted(complete):
        unit = units(root, science, calc, leg)[index]
        assert artifacts.group_state(root, science, calc, leg, unit, semantic=True, producer="attempt",
                                     producer_runtime=runtime_value)["complete"]
    result = root / "calculations" / calc["key"] / "result.json"
    write(result, {"science_id": digest(science)})
    outputs = [result]
    for name in ("completion_audit.json", "partner_occupancy.json"):
        path = result.parent / name
        path.write_text("{}")
        outputs.append(path)
    for name in ("A_fe_table.tsv", "B_fe_table.tsv", "R_fe_table.tsv", "sys_abfe.tsv"):
        path = workdir(root, calc) / "analysis" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test\n")
        outputs.append(path)
    artifacts.commit_manifest(artifacts.paths(root, calc, "final"), root, science, "finalization", outputs,
                              artifacts.final_dependencies(root, science, calc), runtime_value,
                              "attempt", calculation=calc["key"], task=f"finalize:{calc['key']}")
    assert artifacts.validate_final(root, science, calc)
    monkeypatch.setattr(worker, "check_prepared", lambda *a: runtime_value)
    worker.finalize(root, science, calc, {"_attempt_id": "reused"})
    monkeypatch.setattr(validation, "iteration_status", lambda *a, **kw: pytest.fail("status opened NetCDF"))
    before = {str(p) for p in root.rglob("*")}
    report = orchestration.status(root)
    assert report["calculations"][0]["finalization"] == "complete"
    assert {str(p) for p in root.rglob("*")} == before
    result.write_text("corrupt")
    with pytest.raises(WorkflowError, match="changed immutable"):
        artifacts.validate_final(root, science, calc)


def test_overall_status_tracks_sampling_and_finalization(small_run, runtime_value, monkeypatch):
    root, science = small_run
    global_ready(root, science, runtime_value)
    calc = system_ready(root, science, runtime_value)
    import felis_workflows.validation as validation
    complete = {("A", 0), ("A", 1), ("B", 0)}
    monkeypatch.setattr(validation, "iteration_status", fake_iterations(root, science, calc, complete))
    assert orchestration.status(root)["calculations"][0]["state"] == "prepared"
    for leg, index in sorted(complete):
        unit = make_trajectory(root, science, calc, leg, index)
        artifacts.group_state(root, science, calc, leg, unit, semantic=True,
                              producer="attempt", producer_runtime=runtime_value)
        report = orchestration.status(root)["calculations"][0]
        assert report["state"] == ("ready_for_finalization" if sum(
            group["completed_records"] for group in report["groups"].values()) == 3 else "sampling")
    write(root / "calculations" / calc["key"] / "result.json", {"incomplete": True})
    assert orchestration.status(root)["calculations"][0]["state"] == "finalizing"


@pytest.mark.parametrize("partial_name", (
    "completion_audit.json", "partner_occupancy.json",
    "analysis/A_fe_table.tsv", "analysis/B_fe_table.tsv",
    "analysis/R_fe_table.tsv", "analysis/sys_abfe.tsv",
))
def test_interrupted_finalizer_outputs_are_partial_without_netcdf_reads(
        small_run, runtime_value, monkeypatch, partial_name):
    root, science = small_run
    global_ready(root, science, runtime_value)
    calc = system_ready(root, science, runtime_value)
    import felis_workflows.validation as validation
    complete = {("A", 0), ("A", 1), ("B", 0)}
    monkeypatch.setattr(validation, "iteration_status", fake_iterations(root, science, calc, complete))
    for leg, index in sorted(complete):
        unit = make_trajectory(root, science, calc, leg, index)
        artifacts.group_state(root, science, calc, leg, unit, semantic=True,
                              producer="attempt", producer_runtime=runtime_value)
    monkeypatch.setattr(validation, "iteration_status", lambda *a, **kw: pytest.fail("status opened NetCDF"))
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path.suffix == ".nc":
            pytest.fail("status opened NetCDF")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    directory = root / "calculations" / calc["key"]
    analysis = workdir(root, calc) / "analysis"
    analysis.mkdir(parents=True)
    assert orchestration.status(root)["calculations"][0]["state"] == "ready_for_finalization"
    partial = analysis / partial_name.removeprefix("analysis/") if partial_name.startswith("analysis/") else directory / partial_name
    partial.write_text("unfinished finalization\n")
    assert not (directory / "result.json").exists()
    assert not (directory / "finalized.json").exists()
    before = {str(p) for p in root.rglob("*")}
    report = orchestration.status(root)["calculations"][0]
    assert report["finalization"] == "partial"
    assert report["state"] == "finalizing"
    assert {str(p) for p in root.rglob("*")} == before


def test_shared_solvent_status_reflects_owner_completion(cycle, runtime_value, monkeypatch):
    root, science = cycle
    science = deepcopy(science)
    owner = next(c for c in science["calculations"] if c["key"] == "L_in_R/r1")
    consumer = next(c for c in science["calculations"] if c["key"] == "L_in_RP/r1")
    assert consumer["solvent_owner"] == owner["key"]
    science["calculations"] = [owner, consumer]
    for leg, count in (("A", 2), ("B", 1)):
        science["ladders"][leg]["groups"] = science["ladders"][leg]["groups"][:count]
        for calc in science["calculations"]:
            calc["seeds"][leg] = calc["seeds"][leg][:count]
    write(root / "science.json", science)
    write(root / "science.lock.json", {"sha256": sha256(root / "science.json")})
    global_ready(root, science, runtime_value)
    system_ready(root, science, runtime_value, owner)
    unit = make_trajectory(root, science, owner, "A", 0)
    import felis_workflows.validation as validation
    monkeypatch.setattr(validation, "iteration_status", fake_iterations(root, science, owner, {("A", 0)}))
    artifacts.group_state(root, science, owner, "A", unit, semantic=True,
                          producer="owner-attempt", producer_runtime=runtime_value)
    status = {v["calculation"]: v for v in orchestration.status(root)["calculations"]}
    shared = status[consumer["key"]]["groups"]["A"]
    assert status[consumer["key"]]["state"] == "not_started"
    assert shared["owner"] == owner["key"] and shared["shared"] is True
    assert shared["completed_records"] == 1
    assert [v["state"] for v in shared["units"]] == ["recorded_unverified", "incomplete"]
    assert all(v["owner"] == owner["key"] for v in shared["units"])
    system_ready(root, science, runtime_value, consumer)
    status = {v["calculation"]: v for v in orchestration.status(root)["calculations"]}
    assert status[consumer["key"]]["state"] == "sampling"
    assert status[consumer["key"]]["groups"]["A"]["completed_records"] == 1


def test_canonical_stage_dependencies_drive_submission_and_finalization(small_run, runtime_value, monkeypatch):
    root, science = small_run
    initial = artifacts.stage_graph(science)
    final = next(v for v in initial if v["kind"] == "finalization")
    assert [t["id"] for t in backend_common.graph(science) if t["kind"] == "finalize"] == [
        "finalize__" + science["calculations"][0]["key"].replace("/", "__")]
    assert final["dependencies"] == [v["id"] for v in initial if v["kind"] == "simulation_group"]
    # Modify the central graph in isolation. Both submission and artifact
    # lineage must follow its new dependency list.
    altered = deepcopy(initial)
    next(v for v in altered if v["kind"] == "finalization")["dependencies"] = [final["dependencies"][-1]]
    monkeypatch.setattr(artifacts, "stage_graph", lambda _: altered)
    planned = backend_common.graph(science)
    assert next(v for v in planned if v["kind"] == "finalize")["dependencies"] == [
        "B__" + science["calculations"][0]["key"].replace("/", "__")]
    global_ready(root, science, runtime_value)
    calc = system_ready(root, science, runtime_value)
    unit = make_trajectory(root, science, calc, "B", 0)
    import felis_workflows.validation as validation
    monkeypatch.setattr(validation, "iteration_status", fake_iterations(root, science, calc, {("B", 0)}))
    artifacts.group_state(root, science, calc, "B", unit, semantic=True,
                          producer="attempt", producer_runtime=runtime_value)
    assert list(artifacts.final_dependencies(root, science, calc)) == [f'{calc["key"]}:B:0']


def test_resume_dry_run_uses_validated_groups_and_blocks_active_writer(small_run, runtime_value, site, monkeypatch):
    root, science = small_run
    global_ready(root, science, runtime_value)
    calc = system_ready(root, science, runtime_value)
    for leg, index in (("A", 0), ("A", 1), ("B", 0)):
        make_trajectory(root, science, calc, leg, index)
    import felis_workflows.validation as validation
    reads = []
    original = fake_iterations(root, science, calc, {("A", 0), ("B", 0)})
    def reporter(*args):
        reads.append((args[3], args[4]["index"]))
        return original(*args)
    monkeypatch.setattr(validation, "iteration_status", reporter)
    monkeypatch.setattr(worker, "check_prepared", lambda *a: runtime_value)
    monkeypatch.setattr(orchestration, "activate_source", lambda _: None)
    monkeypatch.setattr(backend_common, "fingerprint", lambda _: runtime_value)
    calls = []
    original_ensure_idle = slurm.ensure_idle
    monkeypatch.setattr(slurm, "ensure_idle", lambda *a, **kw: calls.append(kw))
    def local_probe(_root, _site, _snapshot, task, args=(), **kw):
        assert task == "probe"
        worker.probe(root, science, {"_attempt_id": "resume-test"}, Path(args[1]))
    monkeypatch.setattr(orchestration, "launch", local_probe)
    result = orchestration.execute(root, site, resume=True, dry_run=True)
    assert calls == [{"cancel_pending": False}]
    assert result["task_count"] == 2
    plan = read(Path(result["attempt"]) / "submission_plan.json")
    assert [(v["task"]["kind"], v["task"].get("indices")) for v in plan["tasks"]] == [
        ("array", [1]), ("finalize", None)]
    attempt = Path(result["attempt"])
    assert read(attempt / "attempt.json")["workflow_graph_id"] == digest(artifacts.stage_graph(science))
    selected = read(attempt / "task_graph.json")
    assert selected["task_graph_id"] == digest([v["task"] for v in plan["tasks"]])
    assert selected["task_graph_id"] != digest(backend_common.graph(science))
    assert orchestration.status(root)["attempts"][-1]["task_graph_id"] == selected["task_graph_id"]
    previous_reads = len(reads)
    monkeypatch.setattr(slurm, "ensure_idle", original_ensure_idle)
    # Exercise the real queue guard before any semantic probe can read NetCDF.
    active = root / "executions" / "active-writer"
    write(active / "attempt.json", {"purpose": "submit", "site": site_config(site)["name"],
                                    "backend": "slurm", "site_id": "synthetic"})
    write(active / "jobs.json", {"jobs": [{"job_id": "987", "kind": "array"}]})
    monkeypatch.setattr(slurm.subprocess, "check_output", lambda *a, **kw: "987_1|RUNNING|None\n")
    monkeypatch.setattr(orchestration, "launch", lambda *a, **kw: pytest.fail("quiescence check bypassed"))
    with pytest.raises(WorkflowError, match="active trajectories"):
        orchestration.execute(root, site, resume=True, dry_run=True)
    assert len(reads) == previous_reads


def test_attempt_submission_provenance_survives_later_failure(small_run, runtime_value, site, monkeypatch):
    root, science = small_run
    s = site_config(site)
    monkeypatch.setattr(backend_common, "fingerprint", lambda _: runtime_value)
    attempt = backend_common.snapshot(root, s, "submit", science)
    metadata = read(attempt / "attempt.json")
    assert metadata["science_id"] == digest(science)
    assert metadata["site_id"] == digest(s)
    assert metadata["workflow_graph_id"] == digest(artifacts.stage_graph(science))
    assert "task_graph_id" not in metadata
    calls = []
    def submitted(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(1, argv)
        return "123\n"
    monkeypatch.setattr(slurm.subprocess, "check_output", submitted)
    with pytest.raises(subprocess.CalledProcessError):
        slurm.submit(root, s, attempt, backend_common.graph(science))
    job = read(attempt / "jobs.json")["jobs"][0]
    assert job["job_id"] == "123" and job["argv"] == calls[0]
    assert job["script_sha256"] == sha256(attempt / f'{job["task_id"]}.sh')
    selected = read(attempt / "task_graph.json")
    assert selected["task_graph_id"] == digest(backend_common.graph(science))
    assert selected["workflow_graph_id"] == metadata["workflow_graph_id"]
    assert len(read(attempt / "submission_plan.json")["tasks"]) == 2
    selected["tasks"] = []
    write(attempt / "task_graph.json", selected)
    with pytest.raises(WorkflowError, match="Attempt task graph provenance changed"):
        orchestration.status(root)


def test_shipped_prep_memory_is_48g(extension):
    for path in (extension / "sites").glob("*.yaml"):
        site = read(path)
        assert site["resources"]["prep"]["memory"] == "48G"
        assert site["resources"]["array"]["memory"] == "64G"
