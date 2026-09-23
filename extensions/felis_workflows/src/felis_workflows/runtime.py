"""Deployment helpers and a process-local adapter for the pinned FELIS launcher."""
from __future__ import annotations
from contextlib import contextmanager
import ctypes
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .common import WorkflowError, digest, read, sha256, write
from .planning import seed_for


def source_environment(site):
    env = os.environ.copy()
    repo = Path(site["repo"])
    extension = repo / "extensions/felis_workflows/src"
    env["PYTHONPATH"] = os.pathsep.join([str(extension), str(repo), env.get("PYTHONPATH", "")])
    for name in ["OMP_NUM_THREADS", "OPENMM_CPU_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        env[name] = "1"
    env["JAX_PLATFORMS"] = "cpu"
    return env


def activate_source(site):
    repo = Path(site["repo"]).resolve()
    for path in [str(repo), str(repo / "extensions/felis_workflows/src")]:
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    os.environ.update(source_environment(site))
    import felis
    if not Path(felis.__file__).resolve().is_relative_to(repo):
        raise WorkflowError("FELIS imported from the wrong checkout")
    return repo


def fingerprint(site):
    repo = activate_source(site)
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
    return {"versions": versions, "source_hashes": hashes}


def check_runtime(root, site):
    path = Path(root) / "runtime.json"
    current = fingerprint(site)
    if path.exists():
        if read(path) != current:
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


def stages(config_path, stages_to_run, site, seed):
    activate_source(site)
    install_adapter(site, seed)
    from felis.protocols.abfe.config_types import ABFEInputConfig
    from felis.protocols.abfe.main_abfe4 import mainfunc
    cfg = ABFEInputConfig.from_file(str(config_path))
    cfg.stages = stages_to_run
    cfg.check()
    mainfunc(cfg)
