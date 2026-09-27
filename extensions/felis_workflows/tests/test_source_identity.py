"""Source selection and FELIS calculation-relative path regressions."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import types

import pytest

from felis_workflows.backends.common import worker_script
from felis_workflows.common import WorkflowError, read, write
from felis_workflows.orchestration import doctor
from felis_workflows.planning import default_repo, workdir
from felis_workflows import runtime, worker


def test_configured_source_and_path_safe_environment(repo, site, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", f":/opt/extra::relative:{repo}:")
    monkeypatch.setenv("FELIS_REPO", str(repo))
    env = runtime.source_environment(read(site))
    assert env["FELIS_REPO"] == str(repo)
    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(repo / "extensions/felis_workflows/src"), str(repo), "/opt/extra"]
    identity = runtime.source_identity(read(site))
    assert identity["repository"] == str(repo)
    assert identity["felis_source"] == str(repo / "felis/__init__.py")
    assert identity["bytemol_source"] == str((repo / "submodule/bytemol/bytemol/__init__.py").resolve())
    assert identity["felis_workflows_source"] == str(repo / "extensions/felis_workflows/src/felis_workflows/__init__.py")
    assert identity["python_executable"] == sys.executable
    assert identity["python_runtime"]["version"] == list(sys.version_info[:3])
    assert len(identity["git_commit"]) == 40


@pytest.mark.parametrize("package", ["felis", "felis_workflows", "bytemol"])
def test_preimported_other_checkout_fails(repo, site, tmp_path, monkeypatch, package):
    other = tmp_path / "other-checkout" / package / "__init__.py"
    other.parent.mkdir(parents=True)
    other.touch()
    fake = types.ModuleType(package)
    fake.__file__ = str(other)
    monkeypatch.setitem(sys.modules, package, fake)
    with pytest.raises(WorkflowError, match=f"{package} imported from"):
        runtime.activate_source(read(site))


def test_preimported_other_bytemol_builder_fails(repo, site, tmp_path, monkeypatch):
    name = "bytemol.toolkit.system_builder"
    fake = types.ModuleType(name)
    fake.__file__ = str(tmp_path / "site-packages/bytemol/toolkit/system_builder/__init__.py")
    monkeypatch.setitem(sys.modules, name, fake)
    with pytest.raises(WorkflowError, match="bytemol.toolkit.system_builder imported from"):
        runtime.select_source(read(site))


def test_integrity_failure_precedes_new_core_import(repo, tmp_path):
    """A fresh interpreter proves verification does not execute FELIS init code."""
    script = """
import sys
from felis_workflows import runtime
from felis_workflows.common import WorkflowError
assert 'felis' not in sys.modules and 'bytemol' not in sys.modules
def reject(repo):
    assert 'felis' not in sys.modules and 'bytemol' not in sys.modules
    raise WorkflowError('integrity gate')
runtime.verify_upstream = reject
try:
    runtime.activate_source({'repo': sys.argv[1]})
except WorkflowError as error:
    assert str(error) == 'integrity gate'
else:
    raise AssertionError('integrity gate was bypassed')
assert 'felis' not in sys.modules and 'bytemol' not in sys.modules
print('integrity-before-core-import')
"""
    output = subprocess.check_output([sys.executable, "-c", script, str(repo)], cwd=tmp_path,
                                     env=runtime.source_environment({"repo": str(repo)}), text=True)
    assert output.strip() == "integrity-before-core-import"


def test_repository_discovery_ignores_unrelated_cwd(repo, tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv("FELIS_REPO", raising=False)
    assert default_repo() == repo
    monkeypatch.setenv("FELIS_REPO", str(repo))
    env = runtime.source_environment({"repo": str(repo)})
    output = subprocess.check_output(
        [sys.executable, "-m", "felis_workflows.cli", "verify-upstream"],
        cwd=elsewhere, env=env, text=True)
    assert json.loads(output)["verified_paths"] == 4035
    second = tmp_path / "another-directory"
    second.mkdir()
    monkeypatch.chdir(second)
    assert default_repo() == repo
    assert runtime.select_source({"repo": str(repo)}) == repo
    monkeypatch.setenv("FELIS_REPO", "relative-checkout")
    with pytest.raises(WorkflowError, match="identity must be absolute"):
        default_repo()
    with pytest.raises(WorkflowError, match="identity must be absolute"):
        runtime.select_source({"repo": str(repo)})
    monkeypatch.delenv("FELIS_REPO")
    with pytest.raises(WorkflowError, match="identity must be absolute"):
        runtime.select_source({"repo": "relative-checkout"})
    monkeypatch.setenv("FELIS_REPO", str(elsewhere))
    with pytest.raises(WorkflowError, match="Not a FELIS source checkout"):
        default_repo()


def test_non_source_install_and_conflicting_repo_fail(repo, site, tmp_path, monkeypatch):
    monkeypatch.delenv("FELIS_REPO", raising=False)
    monkeypatch.setattr(runtime, "__file__", str(tmp_path / "site-packages/felis_workflows/runtime.py"))
    with pytest.raises(WorkflowError, match="does not identify a source checkout"):
        default_repo()
    alternate = tmp_path / "alternate"
    for relative in ("felis/__init__.py", "bytemol/__init__.py",
                     "extensions/felis_workflows/src/felis_workflows/__init__.py"):
        path = alternate / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setenv("FELIS_REPO", str(alternate))
    with pytest.raises(WorkflowError, match="site.repo selects"):
        runtime.select_source(read(site))


def test_worker_script_establishes_checkout_from_other_cwd(repo, site, tmp_path, monkeypatch):
    run = tmp_path / "run with spaces"
    run.mkdir()
    monkeypatch.setenv("PYTHONPATH", ":")
    script = worker_script(run, read(site), site, "group", ["--calculation", "example/r1"], array=True)
    assert "export FELIS_REPO=" in script
    assert "export PYTHONPATH=" in script
    assert "--index \"${SLURM_ARRAY_TASK_ID:?Missing Slurm array index}\"" in script
    assert "cd " + shlex.quote(str(run)) in script
    probe = ("import json, os, felis, felis_workflows, bytemol; "
             "print(json.dumps([os.getcwd(), os.environ['FELIS_REPO'], felis.__file__, "
             "felis_workflows.__file__, bytemol.__file__, os.environ['PYTHONPATH'].split(os.pathsep)[:2]]))")
    lines = script.splitlines()
    lines[-1] = "exec " + shlex.join([sys.executable, "-c", probe])
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    result = subprocess.run(["bash", "-c", "\n".join(lines)], cwd=unrelated,
                            env={**os.environ, "PYTHONPATH": ":"}, capture_output=True, text=True, check=True)
    cwd, selected, felis_file, workflows_file, bytemol_file, prefixes = json.loads(result.stdout)
    assert (cwd, selected) == (str(run), str(repo))
    assert felis_file == str(repo / "felis/__init__.py")
    assert workflows_file == str(repo / "extensions/felis_workflows/src/felis_workflows/__init__.py")
    assert bytemol_file == str(repo / "bytemol/__init__.py")
    assert prefixes == [str(repo / "extensions/felis_workflows/src"), str(repo)]


def test_stage_runs_at_calculation_work_directory(repo, site, tmp_path, monkeypatch):
    import felis.protocols.abfe.config_types as config_types
    work = tmp_path / "run/calculations/example/r1/work/ligand"
    (work / "prepare").mkdir(parents=True)
    (work / "prepare/sysB.top").write_text("topology\n")
    cfg = types.SimpleNamespace(tmpdir=str(work.parent), sdffile=str(tmp_path / "ligand.sdf"),
                                stages=None, check=lambda: None)
    monkeypatch.setattr(config_types.ABFEInputConfig, "from_file", lambda _: cfg)
    monkeypatch.setattr(runtime, "install_adapter", lambda *args: None)
    import felis.protocols.abfe.main_abfe4 as main_abfe4
    observed = []
    def run(_):
        observed.append((Path.cwd(), Path("prepare/sysB.top").read_text()))
    monkeypatch.setattr(main_abfe4, "mainfunc", run)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    runtime.stages(tmp_path / "abfecfg.json", ["boresch_em"], read(site), 1, work)
    assert observed == [(work, "topology\n")]
    assert Path.cwd() == unrelated
    with pytest.raises(WorkflowError, match="work directory mismatch"):
        runtime.stages(tmp_path / "abfecfg.json", ["boresch_em"], read(site), 1, unrelated)


def test_real_cpu_felis_relative_system_from_wrong_cwd(repo, extension, tmp_path):
    """The FELIS system loader used by stages needs the calculation CWD."""
    source = extension / "tests/data/gaff2_ejm31"
    work = tmp_path / "calculation/work/ligand"
    prepare = work / "prepare"
    prepare.mkdir(parents=True)
    for original, frozen in (("LIG_GMX.top", "sysB.top"), ("LIG_GMX.itp", "LIG_GMX.itp"),
                             ("LIG_GMX.gro", "sysB.gro")):
        shutil.copyfile(source / original, prepare / frozen)
    gro = prepare / "sysB.gro"
    lines = gro.read_text().splitlines()
    lines[-1] = "   5.00000     5.00000     5.00000"  # Small CPU PME box.
    gro.write_text("\n".join(lines) + "\n")
    script = """
import os
from felis_workflows.runtime import activate_source
activate_source({'repo': os.environ['FELIS_REPO']})
from felis.configs import GlobalKeys
from felis.utils.omm.omm_system import get_simulation
config = GlobalKeys()
for value in ('s:filename.sys:prepare/sysB.top', 's:filename.crd:prepare/sysB.gro',
              's:openmm.platform:CPU', 'i:integrator.minimize:0',
              'i:integrator.nstep_per_snapshot:1', 'i:integrator.nsnapshots:1'):
    config.update_by_tkv(value)
config.check()
simulation = get_simulation(config)
simulation.step(1)
print('FELIS_CPU_STEP_OK', simulation.currentStep)
"""
    env = runtime.source_environment({"repo": str(repo)})
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    wrong = subprocess.run([sys.executable, "-c", script], cwd=unrelated, env=env,
                           capture_output=True, text=True)
    assert wrong.returncode != 0 and "_get_gro" in wrong.stderr
    right = subprocess.run([sys.executable, "-c", script], cwd=work, env=env,
                           capture_output=True, text=True, check=True)
    assert right.stdout.strip() == "FELIS_CPU_STEP_OK 1"


def test_group_subprocess_uses_workdir_and_source_environment(cycle, site, monkeypatch):
    root, science = cycle
    calc = science["calculations"][0]
    work = workdir(root, calc)
    (work / "trj").mkdir(parents=True)
    monkeypatch.setattr(worker, "check_prepared", lambda *args: None)
    monkeypatch.setattr(worker, "allocation", lambda *args: nullcontext())
    import felis_workflows.validation as validation
    state = iter([{"complete": False}, {"complete": True}])
    monkeypatch.setattr(validation, "iteration_status", lambda *args: next(state))
    calls = []
    monkeypatch.setattr(worker.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
    worker.simulate_group(root, science, calc, read(site), "A", 0)
    assert calls[0][1]["cwd"] == work
    assert calls[0][1]["env"]["FELIS_REPO"] == read(site)["repo"]
    assert calls[0][1]["env"]["PYTHONPATH"].split(os.pathsep)[:2] == [
        str(Path(read(site)["repo"]) / "extensions/felis_workflows/src"), read(site)["repo"]]


def test_runtime_identity_is_checked_but_revision_is_diagnostic(repo, site, tmp_path, monkeypatch):
    path = tmp_path / "runtime.json"
    source = {"repository": str(repo), "felis_source": str(repo / "felis/__init__.py"),
              "bytemol_source": str(repo / "submodule/bytemol/bytemol/__init__.py"),
              "felis_workflows_source": str(repo / "extensions/felis_workflows/src/felis_workflows/__init__.py"),
              "python_executable": sys.executable, "git_commit": "a" * 40,
              "python_runtime": {"implementation": "cpython", "version": [3, 12, 14], "cache_tag": "cpython-312"}}
    baseline = {"versions": {"openmm": "test"}, "source_hashes": {"felis/a.py": "hash"},
                "source_identity": source}
    monkeypatch.setattr(runtime, "fingerprint", lambda _: baseline)
    runtime.check_runtime(tmp_path, read(site))
    assert read(path)["source_identity"]["git_commit"] == "a" * 40
    relocated = {**baseline, "source_identity": {**source, "git_commit": "b" * 40,
                "repository": "/other/felis", "felis_source": "/other/felis/felis/__init__.py",
                "felis_workflows_source": "/other/felis/extensions/felis_workflows/src/felis_workflows/__init__.py",
                "bytemol_source": "/other/felis/submodule/bytemol/bytemol/__init__.py",
                "python_executable": "/other/python"}}
    monkeypatch.setattr(runtime, "fingerprint", lambda _: relocated)
    runtime.check_runtime(tmp_path, read(site))
    assert read(path)["source_identity"] == source  # Keep the original attribution.
    changed_hash = {**relocated, "source_hashes": {"felis/a.py": "changed"}}
    changed_packages = {**relocated, "versions": {"openmm": "other"}}
    changed_python = {**relocated, "source_identity": {**relocated["source_identity"],
                      "python_runtime": {**source["python_runtime"], "version": [3, 13, 0]}}}
    for wrong in (changed_hash, changed_packages, changed_python):
        monkeypatch.setattr(runtime, "fingerprint", lambda _, wrong=wrong: wrong)
        with pytest.raises(WorkflowError, match="Scientific runtime differs"):
            runtime.check_runtime(tmp_path, read(site))
    write(path, {"versions": {}, "source_hashes": {}})
    with pytest.raises(WorkflowError, match="lacks source identity"):
        runtime.check_runtime(tmp_path, read(site))


def test_doctor_reports_cli_and_runner_identity(repo, extension, tmp_path, monkeypatch):
    site = read(extension / "sites/local-gpu.yaml")
    site["repo"] = str(repo)
    site["mps"] = False
    site["mpi"]["command"] = "python"
    site["python"] = {role: [sys.executable] for role in site["python"]}
    path = tmp_path / "local.json"
    write(path, site)
    report = doctor(path)
    assert report["source"]["repository"] == str(repo)
    assert report["source"]["felis_source"] == str(repo / "felis/__init__.py")
    assert report["source"]["bytemol_source"] == str(repo / "submodule/bytemol/bytemol/__init__.py")
    assert report["source"]["felis_workflows_source"] == str(repo / "extensions/felis_workflows/src/felis_workflows/__init__.py")
    assert report["source"]["python_executable"] == sys.executable
    assert set(report["runners"]) == {"simulation", "receptor", "gaff2", "sage"}
    assert all(identity["repository"] == str(repo) for identity in report["runners"].values())
    assert all(identity["bytemol_source"] == report["source"]["bytemol_source"]
               for identity in report["runners"].values())
    stale = types.ModuleType("felis_workflows")
    stale.__file__ = str(tmp_path / "stale/felis_workflows/__init__.py")
    monkeypatch.setitem(sys.modules, "felis_workflows", stale)
    with pytest.raises(WorkflowError, match="felis_workflows imported from"):
        doctor(path)
