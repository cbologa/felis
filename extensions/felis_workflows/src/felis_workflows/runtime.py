"""Deployment helpers and a process-local adapter for the pinned FELIS launcher."""
from __future__ import annotations
from contextlib import contextmanager
import ctypes
import importlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .common import WorkflowError, digest, read, sha256, write
from .integrity import verify_upstream
from .planning import seed_for


def source_environment(site):
    env = os.environ.copy()
    repo = repository_root(site["repo"])
    extension = repo / "extensions/felis_workflows/src"
    prefixes = [str(extension), str(repo)]
    # Relative and empty inherited entries depend on the worker's CWD.
    inherited = [part for part in env.get("PYTHONPATH", "").split(os.pathsep)
                 if part and Path(part).is_absolute() and part not in prefixes]
    env["PYTHONPATH"] = os.pathsep.join([*prefixes, *inherited])
    env["FELIS_REPO"] = str(repo)
    for name in ["OMP_NUM_THREADS", "OPENMM_CPU_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        env[name] = "1"
    env["JAX_PLATFORMS"] = "cpu"
    return env


def repository_root(value):
    if not value:
        raise WorkflowError("Select a FELIS checkout with --repo, site.repo or FELIS_REPO")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise WorkflowError(f"FELIS repository identity must be absolute: {value}")
    repo = path.resolve()
    expected = [repo / "felis/__init__.py", repo / "bytemol/__init__.py",
                repo / "extensions/felis_workflows/src/felis_workflows/__init__.py"]
    if not all(path.is_file() for path in expected):
        raise WorkflowError(f"Not a FELIS source checkout: {repo}")
    return repo


def source_checkout():
    """Use an editable source installation only when its layout identifies a checkout."""
    package = Path(__file__).resolve()
    candidate = package.parents[4]
    try:
        repo = repository_root(candidate)
    except WorkflowError as error:
        raise WorkflowError("Workflow installation does not identify a source checkout; specify --repo or FELIS_REPO") from error
    if package != repo / "extensions/felis_workflows/src/felis_workflows/runtime.py":
        raise WorkflowError("Workflow installation does not identify a source checkout; specify --repo or FELIS_REPO")
    return repo


def _check_imports(repo):
    roots = {"felis": (repo / "felis").resolve(),
             "bytemol": (repo / "submodule/bytemol/bytemol").resolve(),
             "felis_workflows": (repo / "extensions/felis_workflows/src/felis_workflows").resolve()}
    for name, module in tuple(sys.modules.items()):
        package = name.partition(".")[0]
        if package not in roots or module is None:
            continue
        filename = getattr(module, "__file__", None)
        path = Path(filename).resolve() if filename else None
        expected = roots[package]
        namespace = [Path(p).resolve() for p in getattr(module, "__path__", ())] if path is None else []
        if ((path is None and (name == package or not namespace or
                              any(not p.is_relative_to(expected) for p in namespace))) or
                (path is not None and name == package and path != expected / "__init__.py") or
                (path is not None and name != package and not path.is_relative_to(expected))):
            raise WorkflowError(f"{name} imported from {path}; configured checkout is {repo}")


def select_source(site):
    """Select a checkout and reject conflicting imports without loading FELIS."""
    repo = repository_root(site["repo"])
    selected = os.environ.get("FELIS_REPO")
    if selected and repository_root(selected) != repo:
        raise WorkflowError(f"FELIS_REPO selects {selected}, but site.repo selects {repo}")
    _check_imports(repo)
    return repo


def activate_source(site):
    repo = select_source(site)
    verify_upstream(repo)  # No new FELIS or bytemol code is imported before this succeeds.
    env = source_environment(site)
    os.environ.update(env)
    sys.path[:] = [*env["PYTHONPATH"].split(os.pathsep),
                   *(p for p in sys.path if p and str(Path(p).resolve()) not in
                     {str(repo), str(repo / "extensions/felis_workflows/src")})]
    importlib.invalidate_caches()
    importlib.import_module("felis")
    importlib.import_module("bytemol")
    importlib.import_module("felis_workflows")
    _check_imports(repo)
    return repo


def source_identity(site):
    repo = activate_source(site)
    try:
        revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=False)
        commit = revision.stdout.strip() if revision.returncode == 0 else None
    except OSError:
        commit = None
    return {"repository": str(repo), "git_commit": commit,
            "python_executable": str(Path(sys.executable).absolute()),
            "felis_source": str(Path(sys.modules["felis"].__file__).resolve()),
            "bytemol_source": str(Path(sys.modules["bytemol"].__file__).resolve()),
            "python_runtime": {"implementation": sys.implementation.name,
                               "version": list(sys.version_info[:3]), "cache_tag": sys.implementation.cache_tag},
            "felis_workflows_source": str(Path(sys.modules["felis_workflows"].__file__).resolve())}


def fingerprint(site):
    source = source_identity(site)
    repo = Path(source["repository"])
    versions = {}
    for package in ["openmm", "openmmtools", "numpy", "pymbar", "mdtraj", "mpi4py", "parmed", "rdkit"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    hashes = {}
    for directory in ["felis", "submodule/bytemol/bytemol", "extensions/felis_workflows/src"]:
        for path in sorted((repo / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".xml", ".offxml", ".mdp"}:
                hashes[str(path.relative_to(repo))] = sha256(path)
    from felis.configs import GlobalKeys
    config = GlobalKeys()
    if config.integrator.dt_ps != .002 or config.integrator.targetT_K != 298.15:
        raise WorkflowError("Pinned FELIS integrator defaults changed")
    return {"versions": versions, "source_hashes": hashes,
            "source_identity": source}


def check_runtime(root, site):
    path = Path(root) / "runtime.json"
    current = fingerprint(site)
    if path.exists():
        recorded = read(path)
        # Absolute origins and Git revision diagnose selection at each invocation.
        # Continuation depends on executable content, package versions and ABI,
        # not on the mount point or the spelling of an equivalent Python runner.
        if not isinstance(recorded.get("source_identity"), dict) or "python_runtime" not in recorded["source_identity"]:
            raise WorkflowError("Runtime snapshot lacks source identity; cannot safely continue this run")
        def compatible(value):
            return {**value, "source_identity": {"python_runtime": value["source_identity"]["python_runtime"]}}
        if compatible(recorded) != compatible(current):
            raise WorkflowError("Scientific runtime differs from preparation; use a compatible environment or a new run")
    else:
        write(path, current)
    return current


def mpi_command(site, python_args, ranks=None):
    ranks = ranks or site["mpi"]["ranks"]
    return [site["mpi"]["command"], *site["mpi"]["extra_args"], "-np", str(ranks), sys.executable, *python_args]


def allocated_uuid():
    """Resolve CUDA ordinal 0 through the driver, respecting scheduler visibility.

    nvidia-smi indexes physical devices and must not be used to reinterpret a
    Slurm/cgroup-remapped CUDA ordinal.
    """
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise WorkflowError("Each worker must be assigned exactly one full GPU")
    if os.environ["CUDA_VISIBLE_DEVICES"].startswith("MIG-"):
        raise WorkflowError("MIG execution is not validated; request one full GPU")
    driver = ctypes.CDLL("libcuda.so.1")
    def checked(code):
        if code:
            raise WorkflowError(f"CUDA driver call failed ({code})")
    checked(driver.cuInit(0))
    count, device = ctypes.c_int(), ctypes.c_int()
    checked(driver.cuDeviceGetCount(ctypes.byref(count)))
    if count.value != 1:
        raise WorkflowError(f"Expected one CUDA-visible device, found {count.value}")
    checked(driver.cuDeviceGet(ctypes.byref(device), 0))
    value = (ctypes.c_ubyte * 16)()
    checked(driver.cuDeviceGetUuid(ctypes.byref(value), device))
    raw = bytes(value).hex()
    return "GPU-" + "-".join([raw[:8], raw[8:12], raw[12:16], raw[16:20], raw[20:]])


@contextmanager
def allocation(site):
    original = {k: os.environ.get(k) for k in ["CUDA_VISIBLE_DEVICES", "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY", "NP_VALUE"]}
    directory = None
    started = False
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = allocated_uuid()
        os.environ["NP_VALUE"] = str(site["mpi"]["ranks"])
        if site["mps"]:
            directory = tempfile.mkdtemp(prefix="felis_mps_", dir="/tmp")
            for name, sub in [("CUDA_MPS_PIPE_DIRECTORY", "pipe"), ("CUDA_MPS_LOG_DIRECTORY", "log")]:
                value = str(Path(directory) / sub)
                Path(value).mkdir()
                os.environ[name] = value
            subprocess.run(["nvidia-cuda-mps-control", "-d"], check=True)
            started = True
        yield
    finally:
        if started:
            try:
                subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True,
                               check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=60)
            except subprocess.TimeoutExpired:
                # The control client can hang after all GPU stages have returned.
                # subprocess.run kills that client on timeout; let the worker
                # finish its checks and release the Slurm allocation.
                print("CUDA MPS shutdown exceeded 60 seconds; continuing cleanup", file=sys.stderr,
                      flush=True)
        if directory:
            shutil.rmtree(directory)
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def install_adapter(site, prep_seed):
    import felis.utils
    import felis.utils.cuda_tools
    from felis.protocols.abfe.job_context import JobContext
    original = JobContext._run_one_subprocess
    if getattr(original, "_workflow_adapter", False):
        return
    def subprocess_in_allocation(self, args, working_dir, extra_envs=None):
        if args == "echo quit | nvidia-cuda-mps-control" or args == ["nvidia-cuda-mps-control", "-d"]:
            return 0
        args = list(args) if not isinstance(args, str) else args
        if isinstance(args, list) and args and args[0] == "python3":
            args[0] = sys.executable
            if "--tkv" in args and not any("integrator.randomseed:" in x for x in args):
                label = next((v for v in args if v.startswith("s:filename.stem:")), "prep")
                idx = args.index("--rextkv") if "--rextkv" in args else len(args)
                args.insert(idx, f"i:integrator.randomseed:{seed_for(prep_seed, label)}")
        return original(self, args, working_dir, extra_envs)
    subprocess_in_allocation._workflow_adapter = True
    def mpi_in_allocation(cls, common_cmd, gpu_id, np):
        command = list(common_cmd)
        if command[0] == "python3":
            command = command[1:]
        if "--tkv" in command and not any("integrator.randomseed:" in x for x in command):
            idx = command.index("--rextkv") if "--rextkv" in command else len(command)
            command.insert(idx, f"i:integrator.randomseed:{seed_for(prep_seed, 'boresch_npt')}")
        # Upstream preparation requires four ranks irrespective of array ranks.
        return ["env", f"NP_VALUE={np}", *mpi_command(site, command, np)]
    JobContext._run_one_subprocess = subprocess_in_allocation
    JobContext.build_cuda_mps_mpirun_command = classmethod(mpi_in_allocation)
    felis.utils.get_visible_cuda_devices = lambda: ["0"] if os.environ.get("CUDA_VISIBLE_DEVICES") else []
    felis.utils.cuda_tools.get_visible_cuda_devices = felis.utils.get_visible_cuda_devices


def stages(config_path, stages_to_run, site, seed, expected_work):
    activate_source(site)
    install_adapter(site, seed)
    from felis.protocols.abfe.config_types import ABFEInputConfig
    from felis.protocols.abfe.main_abfe4 import mainfunc
    cfg = ABFEInputConfig.from_file(str(config_path))
    cfg.stages = stages_to_run
    cfg.check()
    work = (Path(cfg.tmpdir) / Path(cfg.sdffile).stem).resolve()
    if work != Path(expected_work).resolve():
        raise WorkflowError(f"FELIS calculation work directory mismatch: {work} != {expected_work}")
    work.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(work)
        mainfunc(cfg)
    finally:
        os.chdir(previous)
