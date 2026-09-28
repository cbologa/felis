"""Separate endpoint identity; the PR4 ABFE source fingerprint stays unchanged."""
from __future__ import annotations

import os
import json
from pathlib import Path
import shlex
import subprocess
import sys

from felis_workflows.artifacts import compatibility
from felis_workflows.common import WorkflowError, read, sha256, write
from felis_workflows.runtime import activate_source, fingerprint, source_environment

from . import __version__


def activate(site):
    repo = activate_source(site)  # Integrity precedes newly imported FELIS code.
    expected = (repo / "extensions/felis_endpoint/src/felis_endpoint").resolve()
    for name, module in tuple(sys.modules.items()):
        if name != "felis_endpoint" and not name.startswith("felis_endpoint."):
            continue
        file = getattr(module, "__file__", None)
        if file is None or not Path(file).resolve().is_relative_to(expected):
            raise WorkflowError(f"{name} is not from the configured endpoint checkout: {repo}")
    if Path(__file__).resolve() != expected / "identity.py":
        raise WorkflowError(f"Endpoint workflow is not from the configured checkout: {repo}")
    return repo


def environment(site):
    repo = activate(site)
    env = source_environment(site)
    endpoint_src = str(repo / "extensions/felis_endpoint/src")
    env["PYTHONPATH"] = os.pathsep.join([endpoint_src, *(p for p in env["PYTHONPATH"].split(os.pathsep)
                                                     if p and p != endpoint_src)])
    return env


def identity(site):
    repo = activate(site)
    base = fingerprint(site)
    endpoint_root = repo / "extensions/felis_endpoint/src/felis_endpoint"
    hashes = {str(path.relative_to(repo)): sha256(path)
              for path in sorted(endpoint_root.rglob("*.py"))}
    if not hashes:
        raise WorkflowError("Endpoint implementation is missing")
    return {"schema_version": 1,
            "stable": {"abfe": compatibility(base), "endpoint_version": __version__,
                       "endpoint_source_hashes": hashes},
            "diagnostic": {**base["source_identity"],
                           "endpoint_source": str((endpoint_root / "__init__.py").resolve())}}


def check(root, site, *, create=False):
    current = identity(site)
    path = Path(root) / "runtime.json"
    if not path.exists():
        if not create:
            raise WorkflowError("Endpoint runtime is missing; replan or restore its provenance")
        write(path, current)
    elif read(path).get("stable") != current["stable"]:
        raise WorkflowError("Endpoint scientific/runtime source changed; use a compatible checkout or a new plan")
    return current


def runner_identity(site_path, site):
    """Fingerprint the configured simulation interpreter, not the CLI interpreter."""
    repo = activate(site)
    command = [*site["python"]["simulation"], "-m", "felis_endpoint.cli", "runtime-probe",
               "--site", str(Path(site_path).resolve())]
    script = "set -euo pipefail\n"
    if site.get("bootstrap"):
        script += "set +u\nsource " + shlex.quote(site["bootstrap"]) + "\nset -u\n"
    script += "exec " + shlex.join(command)
    try:
        result = subprocess.run(["bash", "-c", script], cwd=repo, env=environment(site), text=True,
                                capture_output=True, check=True)
        records = [line.removeprefix("FELIS_ENDPOINT_IDENTITY_JSON=") for line in result.stdout.splitlines()
                   if line.startswith("FELIS_ENDPOINT_IDENTITY_JSON=")]
        if len(records) != 1:
            raise WorkflowError("Configured simulation runner did not report exactly one endpoint identity")
        return json.loads(records[0])
    except (subprocess.CalledProcessError, ValueError) as error:
        raise WorkflowError(f"Configured endpoint runner could not establish source identity: {error}") from error


def check_runner(root, site_path, site):
    current = runner_identity(site_path, site)
    if read(Path(root) / "runtime.json").get("stable") != current["stable"]:
        raise WorkflowError("Endpoint simulation runner differs from the planned source/runtime")
    return current
