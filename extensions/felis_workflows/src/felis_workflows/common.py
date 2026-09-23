from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import yaml


class WorkflowError(ValueError):
    pass


def read(path):
    with Path(path).open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise WorkflowError(f"Expected a mapping: {path}")
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise WorkflowError(f"Invalid identifier: {value!r}")
    return value


def keys(value, allowed, required=(), context="configuration"):
    if not isinstance(value, dict):
        raise WorkflowError(f"{context} must be a mapping")
    unknown, missing = set(value) - set(allowed), set(required) - set(value)
    if unknown or missing:
        raise WorkflowError(f"{context}: unknown keys {sorted(unknown)}; missing keys {sorted(missing)}")


def positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkflowError(f"{label} must be a positive integer")
    return value


def run(argv, *, cwd=None, env=None, capture=False):
    return subprocess.run([str(x) for x in argv], cwd=cwd, env=env, check=True,
                          text=True, stdout=subprocess.PIPE if capture else None).stdout


def verify_hashes(root, hashes):
    root = Path(root).resolve()
    for relative, expected in hashes.items():
        path = root / relative
        if not path.resolve().is_relative_to(root):
            raise WorkflowError(f"Input escapes run directory: {relative}")
        if not path.is_file() or sha256(path) != expected:
            raise WorkflowError(f"Missing or changed immutable file: {path}")


def file_hashes(root, directories):
    return {str(p.relative_to(root)): sha256(p) for name in directories
            for p in sorted((root / name).rglob("*")) if p.is_file()}


@contextlib.contextmanager
def lock(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise WorkflowError(f"Another process owns {path}") from error
        yield
