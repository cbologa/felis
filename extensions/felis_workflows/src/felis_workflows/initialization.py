"""Durable group initialization intent, separate from simulation completion.

Only files created after a recorded intent may be quarantined for a fresh
initialization. A ready record is committed before any rank starts production.
Legacy marked trajectories retain their existing validation/restart contract.
"""
from pathlib import Path
import os
import uuid

from .artifacts import compatibility, write_once
from .common import WorkflowError, digest, read, sha256
from .planning import calculation, units, workdir


def locations(root, science, calc, leg, unit):
    owner = calculation(science, calc["solvent_owner"]) if leg == "A" else calc
    directory = Path(root) / "calculations" / owner["key"] / "initialization"
    nc = workdir(root, owner) / "trj" / f'{unit["stem"]}.nc'
    return owner, directory / f'{unit["stem"]}.started.json', directory / f'{unit["stem"]}.ready.json', nc


def identity(root, science, owner, leg, unit, dependencies, runtime):
    if leg == "A":
        # Shared solvent runs use the owner's SDF, not the consumer's local
        # copy. Keep every other unit/argv field (including the seed) strict.
        owner_unit = units(root, science, owner, leg)[unit["index"]]
        monomer = next(arg for arg in owner_unit["argv"] if arg.startswith("s:filename.monomer:"))
        unit = {**unit, "argv": [monomer if arg.startswith("s:filename.monomer:") else arg
                                for arg in unit["argv"]]}
    return {"schema_version": 1, "root": str(Path(root).resolve()),
            "science_id": digest(science), "calculation": owner["key"],
            "task": f"group:{owner['key']}:{leg}:{unit['index']}",
            "unit": unit, "dependencies": dependencies,
            "runtime_compatibility": compatibility(runtime)}


def state(root, science, calc, leg, unit, dependencies, runtime):
    owner, started, ready, nc = locations(root, science, calc, leg, unit)
    marker = nc.with_suffix(".create_done")
    if not started.exists():
        if ready.exists():
            raise WorkflowError(f"Initialization ready record without intent: {ready}")
        return None
    saved = read(started)
    expected = identity(root, science, owner, leg, unit, dependencies, runtime)
    producer = saved.get("producer")
    if not isinstance(producer, str) or not producer or saved != {**expected, "producer": producer}:
        raise WorkflowError(f"Group initialization identity mismatch: {started}")
    if ready.exists():
        saved_ready = read(ready)
        if not marker.is_file() or saved_ready != {
                "schema_version": 1, "started_sha256": sha256(started), "marker_sha256": sha256(marker)}:
            raise WorkflowError(f"Missing or changed initialized group marker: {ready}")
        if not nc.is_file():
            raise WorkflowError(f"Initialized group trajectory is missing: {nc}")
        return "ready"
    return "created" if marker.is_file() else "pending"


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def begin(root, science, calc, leg, unit, dependencies, runtime, producer):
    """Called under the group lock, after semantic validation and before launch."""
    owner, started, _, nc = locations(root, science, calc, leg, unit)
    marker = nc.with_suffix(".create_done")
    checkpoint = nc.with_name(nc.stem + "_checkpoint.nc")
    current = state(root, science, calc, leg, unit, dependencies, runtime)
    if current in {"ready", "created"} or marker.is_file():
        return  # Resume; never reinitialize a committed creation marker.
    if current is None:
        if any(p.exists() or p.is_symlink() for p in (nc, checkpoint, marker)):
            raise WorkflowError(f"Unowned initialization artifacts; refusing to recreate: {nc}")
        write_once(started, {**identity(root, science, owner, leg, unit, dependencies, runtime),
                             "producer": producer})
    # An interrupted create has no ready record and no creation marker. Preserve
    # its files, even if unreadable, before the native factory sees a clean path.
    files = [p for p in (nc, checkpoint) if p.exists() or p.is_symlink()]
    if any(p.is_symlink() or not p.is_file() for p in files):
        raise WorkflowError(f"Nonregular initialization artifact: {nc}")
    if files:
        archive = started.parent / unit["stem"] / uuid.uuid4().hex
        archive.mkdir(parents=True)
        write_once(archive / "quarantine.json", {
            "started_sha256": sha256(started), "reason": "interrupted initialization before production",
            "files": [str(p.relative_to(root)) for p in files]})
        for path in files:
            os.rename(path, archive / path.name)
            fsync_directory(archive)
            fsync_directory(path.parent)


def commit_ready(root, science, calc, leg, unit, dependencies, runtime):
    current = state(root, science, calc, leg, unit, dependencies, runtime)
    if current is None:
        return  # Existing PR4/PR5 trajectory; no fabricated initialization intent.
    if current not in {"created", "ready"}:
        raise WorkflowError("Sampler returned without its FELIS creation marker")
    _, started, ready, nc = locations(root, science, calc, leg, unit)
    if not nc.is_file():
        raise WorkflowError(f"Sampler returned without its trajectory: {nc}")
    write_once(ready, {"schema_version": 1, "started_sha256": sha256(started),
                       "marker_sha256": sha256(nc.with_suffix(".create_done"))})
