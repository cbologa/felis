"""Interrupted native creation and the ready-before-production contract."""
import ast
from contextlib import nullcontext
from copy import deepcopy
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from felis_workflows import artifacts, initialization, orchestration, repex, worker
from felis_workflows.backends import common
from felis_workflows.common import WorkflowError, read, sha256, write
from felis_workflows.config import site_config
from felis_workflows.planning import units, workdir
from felis_workflows.validation import iteration_status
from test_artifacts_restart import global_ready, runtime_value, system_ready


class Preempted(BaseException):
    """Interrupt without cleanup, like a killed job rather than an ordinary error."""


@pytest.fixture
def group(cycle, runtime_value):
    root, science = cycle
    calc = science["calculations"][0]
    global_ready(root, science, runtime_value)
    system_ready(root, science, runtime_value, calc)
    unit = units(root, science, calc, "B")[0]
    dependencies = artifacts.group_dependencies(root, science, calc, "B", unit["index"])
    return root, science, calc, "B", unit, dependencies, runtime_value


def begin(group):
    initialization.begin(*group, "synthetic-attempt")


def storage(group):
    return initialization.locations(*group[:5])[-1]


def inspect(group, reporter=None):
    root, science, calc, leg, unit, dependencies, runtime = group
    return artifacts.group_state(root, science, calc, leg, unit, semantic=True,
        producer="requeue", producer_runtime=runtime, reporter_factory=reporter,
        dependencies=dependencies, runtime=runtime)


def reporter_for(group, last=0, checkpoint=0, corrupt=False):
    unit = group[4]
    class Reporter:
        checkpoint_interval = unit["checkpoint_interval"]
        def __init__(self, *args, **kwargs): pass
        def read_last_iteration(self, last_checkpoint): return checkpoint if last_checkpoint else last
        def read_dict(self, key): return {"number_of_iterations": unit["iterations"]}
        def read_replica_thermodynamic_states(self, **kwargs): return list(range(len(unit["ilam"])))
        def read_sampler_states(self, **kwargs):
            return [SimpleNamespace(positions=[[float('nan') if corrupt else 0., 0., 0.]])
                    for _ in unit["ilam"]]
        def close(self): pass
    return Reporter


def comm_for(group, events):
    def barrier():
        assert initialization.state(*group) == "ready"
        events.append("ready barrier")
    return SimpleNamespace(mpi_rank=0, bcast=lambda value, **kw: value, barrier=barrier)


def gk_for(group):
    nc = storage(group)
    return SimpleNamespace(dir=SimpleNamespace(trj=str(nc.parent)),
        filename=SimpleNamespace(stem=nc.stem),
        integrator=SimpleNamespace(name="Langevin", dt_ps=.002, friction_1_ps=1,
            nstep_per_snapshot=2500, constraint_tol=1e-6, nsnapshots=group[4]["iterations"]))


@pytest.fixture
def native_factory(repo, group):
    # Execute the exact pinned function body with lightweight external objects.
    # This tests native ordering without requiring a GPU/system parameterization.
    source = repo / "felis/utils/omm/omm_tools.py"
    node = next(n for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "get_replica_exchange_sampler")
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    nc = storage(group)
    class Sampler:
        interrupt = True
        def __init__(self, **kwargs): pass
        def create(self, *args, storage, **kwargs):
            Path(storage).write_bytes(b"native initialization trajectory")
            Path(storage).with_name(Path(storage).stem + "_checkpoint.nc").write_bytes(b"native initial checkpoint")
            if self.interrupt:
                raise Preempted()
        def run(self):
            assert initialization.state(*group) == "ready"
    namespace = dict(Path=Path, os=os, _NP_VALUE=1, MPS_N_CONTEXT_LIMIT=48,
        mpi_rank=0, mpi_nproc=1, ReplicaExchangeSampler=Sampler,
        mcmc=SimpleNamespace(LangevinDynamicsMove=lambda *a, **kw: None),
        unit=SimpleNamespace(picoseconds=1), states=SimpleNamespace(SamplerState=lambda **kw: None),
        _get_storage_name=lambda _: str(nc), _get_new_state_reporter=lambda *a: str(nc),
        mpicomm=SimpleNamespace(barrier=lambda: None), logger=logging.getLogger(__name__))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), "exec"), namespace)
    return namespace["get_replica_exchange_sampler"], Sampler


def test_pinned_native_creation_reproduces_original_orphan_exception(group, native_factory):
    factory, _ = native_factory
    nc = storage(group)
    nc.parent.mkdir(parents=True)
    with pytest.raises(Preempted):
        factory([object()], [], [], gk_for(group))
    assert nc.is_file() and not nc.with_suffix(".create_done").exists()
    with pytest.raises(WorkflowError, match="Trajectory without FELIS creation marker"):
        iteration_status(*group[:5], reporter_factory=object)
    with pytest.raises(WorkflowError, match="creation marker"):
        inspect(group, object)
    with pytest.raises(WorkflowError, match="Unowned initialization"):
        begin(group)


def test_pinned_native_hdf_fallback_can_remove_marker_from_existing_checkpoint(
        group, native_factory, monkeypatch):
    factory, sampler_type = native_factory
    nc = storage(group)
    nc.parent.mkdir(parents=True)
    nc.write_bytes(b"previously valid production trajectory")
    checkpoint = nc.with_name(nc.stem + "_checkpoint.nc")
    checkpoint.write_bytes(b"previously valid production checkpoint")
    nc.with_suffix(".create_done").write_text("create_done")
    def fail_restore(path): raise RuntimeError("NetCDF: HDF error")
    monkeypatch.setattr(sampler_type, "from_storage", staticmethod(fail_restore), raising=False)
    original_unlink = Path.unlink
    def interrupted_unlink(path, *args, **kwargs):
        if path == nc: raise Preempted()
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", interrupted_unlink)
    with pytest.raises(Preempted): factory([object()], [], [], gk_for(group))
    assert not nc.with_suffix(".create_done").exists()
    assert nc.read_bytes() == b"previously valid production trajectory"
    assert checkpoint.read_bytes() == b"previously valid production checkpoint"
    with pytest.raises(WorkflowError, match="creation marker"):
        inspect(group, object)


def test_native_interrupted_creation_requeues_with_preserved_files(group, native_factory):
    begin(group)
    factory, sampler_type = native_factory
    nc = storage(group)
    nc.parent.mkdir(parents=True)
    events = []
    guarded = repex.guarded_factory(factory, lambda *a: pytest.fail("unexpected restore"),
        comm_for(group, events), *group)
    with pytest.raises(Preempted):
        guarded([object()], [], [], gk_for(group))
    original = nc.read_bytes()
    status = inspect(group, object)
    assert not status["complete"] and "initialization" in status["reason"]
    assert events == [] and nc.read_bytes() == original
    # Quiescent status/probe do not open these uncommitted NetCDF files.
    before = {p: p.read_bytes() for p in nc.parent.glob("*.nc")}
    begin(group)
    assert not nc.exists()
    archives = list((initialization.locations(*group[:5])[1].parent / nc.stem).glob("*/quarantine.json"))
    assert len(archives) == 1
    for path, content in before.items():
        assert (archives[0].parent / path.name).read_bytes() == content
    sampler_type.interrupt = False
    sampler = guarded([object()], [], [], gk_for(group))
    assert events == ["ready barrier"]
    sampler.run()
    assert not inspect(group, reporter_for(group))["complete"]
    assert not artifacts.paths(group[0], group[2], group[3], group[4]).exists()


def test_marker_written_but_ready_interrupted_is_restored_not_recreated(group, monkeypatch):
    begin(group)
    nc = storage(group)
    nc.parent.mkdir(parents=True)
    def factory(*args):
        nc.write_bytes(b"completed initialization")
        nc.with_suffix(".create_done").write_text("create_done")
        return object()
    original_commit = initialization.commit_ready
    monkeypatch.setattr(initialization, "commit_ready", lambda *a: (_ for _ in ()).throw(Preempted()))
    first = repex.guarded_factory(factory, None, SimpleNamespace(mpi_rank=0,
        bcast=lambda value, **kw: value, barrier=lambda: pytest.fail("production gate passed")), *group)
    with pytest.raises(Preempted):
        first([object()], [], [], gk_for(group))
    assert initialization.state(*group) == "created"
    assert not inspect(group, reporter_for(group))["complete"]
    before = nc.read_bytes()
    begin(group)
    monkeypatch.setattr(initialization, "commit_ready", original_commit)
    events = []
    restore = lambda path: events.append(("restore", path)) or object()
    resumed = repex.guarded_factory(lambda *a: pytest.fail("recreated"), restore, comm_for(group, events), *group)
    resumed([object()], [], [], gk_for(group))
    assert events == [("restore", str(nc)), "ready barrier"] and nc.read_bytes() == before


def ready_files(group):
    begin(group)
    nc = storage(group)
    nc.parent.mkdir(parents=True, exist_ok=True)
    nc.write_bytes(b"valid checkpointed trajectory")
    checkpoint = nc.with_name(nc.stem + "_checkpoint.nc")
    checkpoint.write_bytes(b"valid checkpoint")
    nc.with_suffix(".create_done").write_text("create_done")
    initialization.commit_ready(*group)
    return nc, checkpoint


def test_checkpoint_after_preemption_preserved_and_completion_still_validated(group):
    nc, checkpoint = ready_files(group)
    original = (nc.read_bytes(), checkpoint.read_bytes())
    status = inspect(group, reporter_for(group, last=1961, checkpoint=1950))
    assert not status["complete"] and status["last_checkpoint"] == 1950
    begin(group)
    events = []
    sampler = object()
    guarded = repex.guarded_factory(lambda *a: pytest.fail("valid trajectory recreated"),
        lambda path: sampler, comm_for(group, events), *group)
    assert guarded([], [], [], gk_for(group)) is sampler
    assert (nc.read_bytes(), checkpoint.read_bytes()) == original
    assert inspect(group, reporter_for(group, last=2000, checkpoint=2000))["complete"]
    saved = read(artifacts.paths(group[0], group[2], group[3], group[4]))
    assert saved["checkpoint"]["trajectory"] == str(nc.relative_to(group[0]))
    assert saved["hashes"] == {str(nc.with_suffix(".create_done").relative_to(group[0])):
                               sha256(nc.with_suffix(".create_done"))}


@pytest.mark.parametrize("damage", ["missing marker", "changed marker", "invalid coordinates", "missing trajectory"])
def test_ready_group_corruption_is_not_reclassified_as_initialization(group, damage):
    nc, checkpoint = ready_files(group)
    if damage == "missing marker": nc.with_suffix(".create_done").unlink()
    elif damage == "changed marker": nc.with_suffix(".create_done").write_text("changed")
    elif damage == "missing trajectory": nc.unlink()
    with pytest.raises(WorkflowError):
        inspect(group, reporter_for(group, last=100, checkpoint=100, corrupt=damage == "invalid coordinates"))
    assert checkpoint.read_bytes() == b"valid checkpoint"


def test_transient_native_restore_error_does_not_delete_valid_lineage(group):
    nc, checkpoint = ready_files(group)
    original = (nc.read_bytes(), checkpoint.read_bytes(), nc.with_suffix(".create_done").read_bytes())
    def restore(path): raise RuntimeError("NetCDF: HDF error")
    guarded = repex.guarded_factory(lambda *a: pytest.fail("native destructive fallback used"),
        restore, comm_for(group, []), *group)
    with pytest.raises(RuntimeError, match="NetCDF"):
        guarded([], [], [], gk_for(group))
    assert (nc.read_bytes(), checkpoint.read_bytes(), nc.with_suffix(".create_done").read_bytes()) == original


@pytest.mark.parametrize("field", ["science_id", "calculation", "task", "unit", "dependencies", "runtime_compatibility"])
def test_initialization_identity_cannot_cross_groups_or_runtimes(group, field):
    begin(group)
    started = initialization.locations(*group[:5])[1]
    saved = read(started)
    saved[field] = "wrong"
    write(started, saved)
    with pytest.raises(WorkflowError, match="initialization identity mismatch"):
        inspect(group, object)
    with pytest.raises(WorkflowError, match="initialization identity mismatch"):
        begin(group)


def test_ready_record_requires_original_initialization_intent(group):
    ready_files(group)
    initialization.locations(*group[:5])[1].unlink()
    with pytest.raises(WorkflowError, match="ready record without intent"):
        inspect(group, object)


def test_missing_ready_record_after_sampling_is_not_an_initialization_retry(group):
    nc, checkpoint = ready_files(group)
    initialization.locations(*group[:5])[2].unlink()
    with pytest.raises(WorkflowError, match="sampled before its initialization ready record"):
        inspect(group, reporter_for(group, last=100, checkpoint=100))
    assert nc.read_bytes() == b"valid checkpointed trajectory"
    assert checkpoint.read_bytes() == b"valid checkpoint"


def test_legacy_marked_checkpoint_is_restored_without_fabricating_intent(group):
    nc = storage(group)
    nc.parent.mkdir(parents=True)
    nc.write_bytes(b"existing PR4 or PR5 trajectory")
    nc.with_suffix(".create_done").write_text("create_done")
    assert not inspect(group, reporter_for(group, last=100, checkpoint=100))["complete"]
    begin(group)
    comm = SimpleNamespace(mpi_rank=0, bcast=lambda mode, **kw: mode, barrier=lambda: None)
    restored = object()
    guarded = repex.guarded_factory(lambda *a: pytest.fail("legacy trajectory recreated"),
        lambda path: restored, comm, *group)
    assert guarded([], [], [], gk_for(group)) is restored
    assert initialization.state(*group) is None
    assert not initialization.locations(*group[:5])[1].exists()


def test_other_rank_waits_for_ready_before_production(group):
    ready_files(group)
    events = []
    comm = comm_for(group, events)
    comm.mpi_rank = 1
    comm.bcast = lambda mode, **kw: "restore"
    restored = object()
    guarded = repex.guarded_factory(lambda *a: pytest.fail("nonroot recreated trajectory"),
        lambda path: restored, comm, *group)
    assert guarded([], [], [], gk_for(group)) is restored
    assert events == ["ready barrier"]


def test_new_child_cannot_initialize_without_group_intent(group):
    guarded = repex.guarded_factory(lambda *a: pytest.fail("unowned initialization"), None,
        SimpleNamespace(mpi_rank=0), *group)
    with pytest.raises(WorkflowError, match="requires a recorded group intent"):
        guarded([], [], [], gk_for(group))


def test_concurrent_ranks_cannot_start_production_before_ready_commit(group, monkeypatch):
    begin(group)
    nc = storage(group)
    nc.parent.mkdir(parents=True)
    broadcast_gate, production_gate = threading.Barrier(2, timeout=5), threading.Barrier(2, timeout=5)
    commit_entered, allow_commit, nonroot_waiting = threading.Event(), threading.Event(), threading.Event()
    shared, returned, errors = {}, [], []
    original_commit = initialization.commit_ready
    def delayed_commit(*args):
        commit_entered.set()
        assert allow_commit.wait(5)
        original_commit(*args)
    monkeypatch.setattr(initialization, "commit_ready", delayed_commit)
    def rank_worker(rank):
        def bcast(mode, **kw):
            if rank == 0: shared["mode"] = mode
            broadcast_gate.wait()
            return shared["mode"]
        def barrier():
            if rank == 1: nonroot_waiting.set()
            production_gate.wait()
        comm = SimpleNamespace(mpi_rank=rank, bcast=bcast, barrier=barrier)
        def factory(*args):
            if rank == 0:
                nc.write_bytes(b"iteration zero")
                nc.with_suffix(".create_done").write_text("create_done")
            return object()
        try:
            guarded = repex.guarded_factory(factory, None, comm, *group)
            guarded([], [], [], gk_for(group))
            assert initialization.state(*group) == "ready"
            returned.append(rank)
        except BaseException as error:
            errors.append(error)
    threads = [threading.Thread(target=rank_worker, args=(rank,)) for rank in (0, 1)]
    try:
        for thread in threads: thread.start()
        assert commit_entered.wait(5) and nonroot_waiting.wait(5)
        assert returned == [] and not initialization.locations(*group[:5])[2].exists()
    finally:
        allow_commit.set()
        for thread in threads: thread.join(6)
    assert not errors and all(not thread.is_alive() for thread in threads)
    assert sorted(returned) == [0, 1]


def test_repeated_preemption_during_quarantine_preserves_both_files(group, monkeypatch):
    begin(group)
    nc = storage(group)
    nc.parent.mkdir(parents=True)
    checkpoint = nc.with_name(nc.stem + "_checkpoint.nc")
    nc.write_bytes(b"partial trajectory")
    checkpoint.write_bytes(b"partial checkpoint")
    original_rename = initialization.os.rename
    calls = []
    def interrupted_rename(source, destination):
        calls.append(source)
        if len(calls) == 2: raise Preempted()
        original_rename(source, destination)
    monkeypatch.setattr(initialization.os, "rename", interrupted_rename)
    with pytest.raises(Preempted): begin(group)
    monkeypatch.setattr(initialization.os, "rename", original_rename)
    assert not inspect(group, object)["complete"]
    begin(group)
    archive = initialization.locations(*group[:5])[1].parent / nc.stem
    assert sorted(p.read_bytes() for p in archive.glob("*/*.nc")) == sorted([b"partial trajectory", b"partial checkpoint"])
    assert not nc.exists() and not checkpoint.exists()


def test_worker_requeue_creates_intent_before_subprocess_and_recovers(group, site, monkeypatch):
    root, science, calc, leg, unit, dependencies, runtime = group
    monkeypatch.setattr(worker, "check_runtime", lambda *a: runtime)
    monkeypatch.setattr(worker, "allocation", lambda *a: nullcontext())
    import felis_workflows.validation as validation
    reporter = reporter_for(group)
    original_iterations = validation.iteration_status
    monkeypatch.setattr(validation, "iteration_status", lambda *a: original_iterations(*a[:5], reporter_factory=reporter))
    nc = storage(group)
    calls = []
    def launch(argv, **kwargs):
        assert initialization.state(*group) == "pending"
        assert "felis_workflows.repex" in argv and kwargs["cwd"] == nc.parent.parent
        assert not nc.exists()
        calls.append(argv)
        nc.write_bytes(b"initialized by guarded child")
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, argv)
        nc.with_suffix(".create_done").write_text("create_done")
        initialization.commit_ready(*group)
        reporter.read_last_iteration = lambda self, last_checkpoint: unit["iterations"]
    monkeypatch.setattr(worker.subprocess, "run", launch)
    with pytest.raises(subprocess.CalledProcessError):
        worker.simulate_group(root, science, calc, read(site), leg, unit["index"])
    worker.simulate_group(root, science, calc, read(site), leg, unit["index"])
    assert len(calls) == 2 and inspect(group, reporter)["complete"]


def test_group_launcher_preserves_native_science_arguments_and_restores_adapter(
        group, native_factory, monkeypatch):
    pytest.importorskip("openmmtools")
    from felis.app.dyn import repex as native
    from felis.protocols.dyn import main_replica_exchange as protocol
    begin(group)
    nc = storage(group)
    nc.parent.mkdir(parents=True)
    factory, sampler_type = native_factory
    sampler_type.interrupt = False
    monkeypatch.setattr(protocol, "get_replica_exchange_sampler", factory)
    monkeypatch.setattr(repex, "activate_source", lambda *a: None)
    monkeypatch.setenv("FELIS_REPO", group[-1]["source_identity"]["repository"])
    calls = []
    def run_protocol(gk_list):
        sampler = protocol.get_replica_exchange_sampler([object()], [], [], gk_for(group))
        sampler.run()
    def run_native(argv, selected_protocol):
        calls.append(argv)
        selected_protocol([])
    monkeypatch.setattr(protocol, "mainfunc", run_protocol)
    monkeypatch.setattr(native, "mainfunc", run_native)
    root, _, calc, leg, unit, _, _ = group
    repex.main(["--run", str(root), "--calculation", calc["key"], "--leg", leg,
                "--index", str(unit["index"])])
    assert calls == [unit["argv"][2:]]
    assert protocol.get_replica_exchange_sampler is factory
    assert initialization.state(*group) == "ready"


def test_shared_solvent_uses_owner_intent_and_rejects_wrong_replica(cycle, runtime_value):
    root, science = cycle
    owner = next(c for c in science["calculations"] if c["key"] == "L_in_R/r1")
    consumer = next(c for c in science["calculations"] if c["key"] == "L_in_RP/r1")
    other = next(c for c in science["calculations"] if c["key"] == "L_in_R/r2")
    unit = units(root, science, owner, "A")[0]
    dependencies = {"equilibration": "owner equilibration", "group_seed": owner["seeds"]["A"][0]}
    group = (root, science, owner, "A", unit, dependencies, runtime_value)
    begin(group)
    consumer_unit = units(root, science, consumer, "A")[0]
    assert consumer_unit != unit and consumer["seeds"]["A"] == owner["seeds"]["A"]
    assert initialization.state(root, science, consumer, "A", consumer_unit, dependencies, runtime_value) == "pending"
    destination = initialization.locations(root, science, other, "A", units(root, science, other, "A")[0])[1]
    write(destination, read(initialization.locations(*group[:5])[1]))
    with pytest.raises(WorkflowError, match="initialization identity mismatch"):
        initialization.state(root, science, other, "A", units(root, science, other, "A")[0], dependencies, runtime_value)


@pytest.fixture
def shared_solvent_run(cycle, runtime_value, monkeypatch):
    root, original = cycle
    science = deepcopy(original)
    science["calculations"] = [c for c in science["calculations"]
                               if c["key"] in {"L_in_R/r1", "L_in_RP/r1"}]
    owner, consumer = science["calculations"]
    assert science["protocol"]["reuse_solvent"] is True
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
    monkeypatch.setattr(worker, "check_runtime", lambda *a: runtime_value)
    return root, science, owner, consumer, runtime_value


@pytest.mark.parametrize("wrong", ("seed", "ilam", "iterations", "checkpoint_interval"))
def test_shared_solvent_canonicalization_preserves_other_unit_identity(shared_solvent_run, wrong):
    root, science, owner, consumer, runtime = shared_solvent_run
    owner_unit = units(root, science, owner, "A")[0]
    consumer_unit = deepcopy(units(root, science, consumer, "A")[0])
    dependencies = artifacts.group_dependencies(root, science, owner, "A", 0)
    begin((root, science, owner, "A", owner_unit, dependencies, runtime))
    if wrong == "seed":
        consumer_unit["argv"] = [arg + "0" if arg.startswith("i:integrator.randomseed:") else arg
                                 for arg in consumer_unit["argv"]]
    elif wrong == "ilam":
        consumer_unit["ilam"] = consumer_unit["ilam"][1:]
    else:
        consumer_unit[wrong] += 1
    with pytest.raises(WorkflowError, match="initialization identity mismatch"):
        initialization.state(root, science, consumer, "A", consumer_unit, dependencies, runtime)


def test_shared_solvent_consumer_status_and_probe_accept_pending_owner_intent(shared_solvent_run):
    root, science, owner, consumer, runtime = shared_solvent_run
    unit = units(root, science, owner, "A")[0]
    dependencies = artifacts.group_dependencies(root, science, owner, "A", 0)
    group = (root, science, owner, "A", unit, dependencies, runtime)
    begin(group)
    nc = storage(group)
    nc.parent.mkdir(parents=True)
    nc.write_bytes(b"interrupted owner initialization; not readable NetCDF")
    started = initialization.locations(*group[:5])[1]
    before = (sha256(started), sha256(nc))
    status = {v["calculation"]: v for v in orchestration.status(root)["calculations"]}
    assert status[consumer["key"]]["groups"]["A"]["owner"] == owner["key"]
    assert status[consumer["key"]]["groups"]["A"]["units"][0]["state"] == "incomplete"
    worker.probe(root, science, {"_attempt_id": "consumer-probe"}, root / "probe.json")
    assert read(root / "probe.json")[consumer["key"]]["A"] == [False, False]
    assert (sha256(started), sha256(nc)) == before
    assert not (started.parent / unit["stem"]).exists()  # Probe must not quarantine.


def test_shared_solvent_consumer_status_probe_and_finalize_accept_owner_initialization(
        shared_solvent_run, monkeypatch):
    root, science, owner, consumer, runtime = shared_solvent_run
    for calc, legs in ((owner, "AB"), (consumer, "B")):
        for leg in legs:
            for unit in units(root, science, calc, leg):
                group = (root, science, calc, leg, unit,
                         artifacts.group_dependencies(root, science, calc, leg, unit["index"]), runtime)
                ready_files(group)
    owner_unit = units(root, science, owner, "A")[0]
    consumer_unit = units(root, science, consumer, "A")[0]
    assert owner_unit != consumer_unit
    started = initialization.locations(root, science, owner, "A", owner_unit)[1]
    assert read(started)["unit"] == owner_unit
    started_hash = sha256(started)
    import felis_workflows.validation as validation

    def completed(_root, _science, calc, leg, unit, factory=None):
        reporter = reporter_for((_root, _science, calc, leg, unit),
                                last=unit["iterations"], checkpoint=unit["iterations"])
        return iteration_status(_root, _science, calc, leg, unit, reporter)

    monkeypatch.setattr(validation, "iteration_status", completed)
    orchestration.status(root)  # Ready owner intent without a terminal group record.
    worker.probe(root, science, {"_attempt_id": "consumer-probe"}, root / "probe.json")
    probe = read(root / "probe.json")[consumer["key"]]
    assert probe["A"] == [True, True] and probe["B"] == [True]
    status = {v["calculation"]: v for v in orchestration.status(root)["calculations"]}
    assert status[consumer["key"]]["state"] == "ready_for_finalization"
    dependencies = artifacts.final_dependencies(root, science, consumer)
    assert dependencies == {
        f"{producer['key']}:{leg}:{unit['index']}": sha256(artifacts.paths(root, producer, leg, unit))
        for producer, leg in ((owner, "A"), (consumer, "B"))
        for unit in units(root, science, producer, leg)
    }
    monkeypatch.setattr(validation, "partner_occupancy", lambda *a: {"passed": True})
    mbar_sources = {}

    def mbar(*, stem, nc_list, outdir, **kwargs):
        producer = owner if stem == "A" else consumer
        assert nc_list == [str(workdir(root, producer) / "trj" / f"{u['stem']}.nc")
                           for u in units(root, science, producer, stem)]
        mbar_sources[stem] = nc_list
        (Path(outdir) / f"{stem}_fe_table.tsv").write_text("data\n")

    callbacks = {
        "main_fe_mbar": ("calc_mbar", mbar),
        "main_fe_restraints": ("calc_restraints", lambda *, outdir, **kw:
                              (Path(outdir) / "R_fe_table.tsv").write_text("data\n")),
        "main_fe_summarize": ("summarize_fe", lambda *, workdir, **kw:
                             (Path(workdir) / "sys_abfe.tsv").write_text("ligand\tdG(kcal/mol)\nL\t-1.0\n")),
    }
    for name, (function, callback) in callbacks.items():
        module = ModuleType("felis.protocols.abfe." + name)
        setattr(module, function, callback)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    worker.finalize(root, science, consumer, {"_attempt_id": "consumer-finalize"})
    assert set(mbar_sources) == {"A", "B"}
    assert artifacts.validate_final(root, science, consumer)["dependencies"] == dependencies
    worker.probe(root, science, {"_attempt_id": "after-finalize"}, root / "probe.json")
    assert read(root / "probe.json")[consumer["key"]]["finalize"] is True
    status = {v["calculation"]: v for v in orchestration.status(root)["calculations"]}
    assert status[consumer["key"]]["state"] == "finalized"
    assert sha256(started) == started_hash  # Consumer reads must never rewrite owner intent.


@pytest.mark.parametrize("wrong", ("group", "seed", "runtime"))
@pytest.mark.parametrize("operation", ("status", "probe", "dependencies", "finalize"))
def test_shared_solvent_consumers_reject_wrong_owner_identity(shared_solvent_run, wrong, operation):
    root, science, owner, consumer, runtime = shared_solvent_run
    unit = units(root, science, owner, "A")[0]
    group = (root, science, owner, "A", unit,
             artifacts.group_dependencies(root, science, owner, "A", 0), runtime)
    ready_files(group)
    started = initialization.locations(*group[:5])[1]
    record = read(started)
    if wrong == "group":
        record["task"] = f"group:{owner['key']}:A:1"
    elif wrong == "seed":
        record["unit"]["argv"] = [arg + "0" if arg.startswith("i:integrator.randomseed:") else arg
                                    for arg in record["unit"]["argv"]]
    else:
        record["runtime_compatibility"]["versions"]["openmm"] = "different"
    write(started, record)
    before = (sha256(started), sha256(storage(group)))
    actions = {
        "status": lambda: orchestration.status(root),
        "probe": lambda: worker.probe(root, science, {}, root / "probe.json"),
        "dependencies": lambda: artifacts.final_dependencies(root, science, consumer),
        "finalize": lambda: worker.finalize(root, science, consumer, {}),
    }
    with pytest.raises(WorkflowError, match="initialization identity mismatch"):
        actions[operation]()
    assert (sha256(started), sha256(storage(group))) == before
    assert not artifacts.paths(root, consumer, "final").exists()


def test_easley_submit_and_resume_render_current_policy(group, site, monkeypatch):
    root, science, calc, leg, unit, dependencies, runtime = group
    configured = read(site)
    configured["slurm"].update(extra_args=["--requeue", "--exclude=easley061,easley052"], prep_partition="l40s")
    configured["resources"]["prep"]["walltime"] = "12:00:00"
    configured["resources"]["array"]["walltime"] = "06:00:00"
    write(site, configured)
    monkeypatch.setattr(orchestration, "activate_source", lambda *a: None)
    monkeypatch.setattr(common, "fingerprint", lambda *a: runtime)
    monkeypatch.setattr(worker, "check_runtime", lambda *a: runtime)
    import felis_workflows.validation as validation
    original_iterations = validation.iteration_status
    monkeypatch.setattr(validation, "iteration_status", lambda *a: original_iterations(*a[:5], reporter_factory=object))
    first = orchestration.execute(root, site, dry_run=True)
    begin(group)
    nc = storage(group)
    nc.parent.mkdir(parents=True)
    nc.write_bytes(b"interrupted initialization; never open as NetCDF")
    before = nc.read_bytes()
    def probe(_root, _site, _snapshot, task, args=(), **kw):
        assert task == "probe"
        worker.probe(root, science, {"_attempt_id": "resume"}, Path(args[1]))
    monkeypatch.setattr(orchestration, "launch", probe)
    resumed = orchestration.execute(root, site, resume=True, dry_run=True)
    assert nc.read_bytes() == before  # The probe must not quarantine or alter files.
    for result in (first, resumed):
        attempt = Path(result["attempt"])
        assert read(attempt / "site.json") == site_config(site)
        for entry in read(attempt / "submission_plan.json")["tasks"]:
            argv = entry["argv"]
            assert "--requeue" in argv and "--exclude=easley061,easley052" in argv
            if entry["task"]["kind"] == "array":
                assert "--partition=scavenger" in argv and "--gpus=1" in argv
                assert "--cpus-per-task=8" in argv and "--mem=64G" in argv and "--time=06:00:00" in argv
                assert "%" not in next(arg for arg in argv if arg.startswith("--array="))
            elif entry["task"]["kind"] == "prep":
                assert "--partition=l40s" in argv and "--mem=48G" in argv and "--time=12:00:00" in argv
            else:
                assert "--partition=general" in argv and "--mem=64G" in argv and "--cpus-per-task=4" in argv
                assert "--gpus=1" not in argv


def test_real_netcdf_interrupted_initialization_then_checkpoint_resume(cycle, runtime_value):
    omt = pytest.importorskip("openmmtools")
    import openmm as mm
    from openmm import unit as u
    root, science = cycle
    science = deepcopy(science)
    science["calculations"] = science["calculations"][:1]
    science["protocol"].update(complex_ns=.01, checkpoint_interval=1)
    science["ladders"]["B"]["groups"] = [[0, 1]]
    calc = science["calculations"][0]
    calc["seeds"]["B"] = calc["seeds"]["B"][:1]
    global_ready(root, science, runtime_value)
    system_ready(root, science, runtime_value, calc)
    unit = units(root, science, calc, "B")[0]
    group = (root, science, calc, "B", unit,
             artifacts.group_dependencies(root, science, calc, "B", 0), runtime_value)
    nc = storage(group)
    nc.parent.mkdir(parents=True)
    system = mm.System()
    system.addParticle(12)
    force = mm.CustomExternalForce("10*(x*x+y*y+z*z)")
    force.addParticle(0, [])
    system.addForce(force)
    thermo = [omt.states.ThermodynamicState(system, temperature=t*u.kelvin) for t in (298.15, 300.)]
    sample = omt.states.SamplerState(positions=[[.01, .01, .01]]*u.nanometer)
    previous = omt.cache.global_context_cache.platform
    omt.cache.global_context_cache.platform = mm.Platform.getPlatformByName("Reference")
    interrupt = [True]
    def factory(*args):
        sampler = omt.multistate.ReplicaExchangeSampler(
            mcmc_moves=omt.mcmc.LangevinDynamicsMove(n_steps=1),
            number_of_iterations=2, online_analysis_interval=None)
        reporter = omt.multistate.MultiStateReporter(str(nc), checkpoint_interval=1)
        sampler.create(thermo, sample, storage=reporter)
        if interrupt[0]:
            reporter.close()
            raise Preempted()
        nc.with_suffix(".create_done").write_text("create_done")
        return sampler
    guarded = repex.guarded_factory(factory, omt.multistate.ReplicaExchangeSampler.from_storage,
                                   comm_for(group, []), *group)
    reporters = []
    try:
        begin(group)
        with pytest.raises(Preempted): guarded(thermo, [], [], gk_for(group))
        original = {p.name: sha256(p) for p in nc.parent.glob("*.nc")}
        assert not inspect(group)["complete"]
        begin(group)
        archived = initialization.locations(*group[:5])[1].parent / nc.stem
        assert {p.name: sha256(p) for p in archived.glob("*/*.nc")} == original
        interrupt[0] = False
        sampler = guarded(thermo, [], [], gk_for(group))
        reporters.append(sampler._reporter)
        assert initialization.state(*group) == "ready"
        sampler.run(n_iterations=1)
        sampler._reporter.close()
        partial = inspect(group)
        assert partial["last_iteration"] == partial["last_checkpoint"] == 1
        assert not partial["complete"]
        begin(group)
        restored = guarded(thermo, [], [], gk_for(group))
        reporters.append(restored._reporter)
        assert restored.iteration == 1  # The committed checkpoint was recovered.
        restored.run()
        restored._reporter.close()
        assert inspect(group)["complete"]
    finally:
        for reporter in reporters: reporter.close()
        omt.cache.global_context_cache.empty()
        omt.cache.global_context_cache.platform = previous
