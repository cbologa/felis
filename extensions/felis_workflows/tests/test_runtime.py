import os
from pathlib import Path
import subprocess

from felis_workflows import runtime


def test_mps_shutdown_timeout_releases_allocation(monkeypatch, tmp_path, capsys):
    for name in ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY", "NP_VALUE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runtime, "allocated_uuid", lambda: "GPU-test")
    original_mkdtemp = runtime.tempfile.mkdtemp
    monkeypatch.setattr(runtime.tempfile, "mkdtemp", lambda **kwargs: original_mkdtemp(
        prefix=kwargs["prefix"], dir=tmp_path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command == ["nvidia-cuda-mps-control"]:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runtime.subprocess, "run", run)
    with runtime.allocation({"mps": True, "mpi": {"ranks": 4}}):
        directory = Path(os.environ["CUDA_MPS_PIPE_DIRECTORY"]).parent
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-test"
        assert directory.exists()

    assert calls[-1][1]["timeout"] == 60
    assert not directory.exists()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert "CUDA_MPS_PIPE_DIRECTORY" not in os.environ
    assert "shutdown exceeded 60 seconds" in capsys.readouterr().err
