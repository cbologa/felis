"""Versioned, write-once terminal artifacts and the workflow stage graph.

Trajectory files are checkpointed, not immutable outputs. Their terminal records
are only committed after ``iteration_status`` validates a quiescent reporter.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .common import WorkflowError, digest, read, sha256

SCHEMA = 1
PREP_STAGES = ("makebox",)
EQUIL_STAGES = ("boresch_em", "boresch_npt", "boresch_post_process", "sysA_em", "sysB_em")
KINDS = {"global_inputs", "system_preparation", "equilibration", "simulation_group", "finalization"}


def compatibility(runtime):
    """Strip diagnostic mount points, revision and interpreter pathname."""
    try:
        return {"versions": runtime["versions"], "source_hashes": runtime["source_hashes"],
                "python_runtime": runtime["source_identity"]["python_runtime"]}
    except (KeyError, TypeError) as error:
        raise WorkflowError("Runtime snapshot lacks stable compatibility identity") from error


def relative_file(root, name):
    root = Path(root).resolve()
    if not isinstance(name, str) or not name or Path(name).is_absolute() or \
            any(part in {"", ".", ".."} for part in Path(name).parts):
        raise WorkflowError(f"Invalid artifact path: {name!r}")
    path = root / name
    if not path.resolve().is_relative_to(root):
        raise WorkflowError(f"Artifact escapes run directory: {name}")
    return path


def output_hashes(root, paths):
    root = Path(root).resolve()
    result = {}
    for path in paths:
        path = Path(path)
        name = str(path.relative_to(root))
        checked = relative_file(root, name)
        if not checked.is_file():
            raise WorkflowError(f"Missing immutable output: {checked}")
        result[name] = sha256(checked)
    if not result:
        raise WorkflowError("Terminal artifact has no immutable outputs")
    return dict(sorted(result.items()))


def write_once(path, value):
    """Publish with a hard link so another writer cannot replace a terminal record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text() != content:
                raise WorkflowError(f"Conflicting committed artifact: {path}")
            return False
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return True
    finally:
        os.unlink(temporary)


def write_script(path, content):
    """Persist exactly the worker script that is executed or submitted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.read_text() != content:
            raise WorkflowError(f"Conflicting generated worker script: {path}")
        return
    with os.fdopen(fd, "w") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def commit_manifest(path, root, science, kind, outputs, dependencies, runtime,
                    producer, *, calculation=None, task=None, checkpoint=None, diagnostics=None):
    if kind not in KINDS:
        raise WorkflowError(f"Unknown artifact kind: {kind}")
    value = {"schema_version": SCHEMA, "kind": kind, "science_id": digest(science),
             "calculation": calculation, "task": task, "producer": producer,
             "dependencies": dependencies, "hashes": output_hashes(root, outputs),
             "runtime_compatibility": compatibility(runtime),
             "source_diagnostics": diagnostics if diagnostics is not None else runtime["source_identity"],
             "checkpoint": checkpoint}
    write_once(path, value)
    return validate_manifest(path, root, science, kind, dependencies, runtime,
                             calculation=calculation, task=task)


def validate_manifest(path, root, science, kind, dependencies=None, runtime=None,
                      *, calculation=None, task=None):
    path = Path(path)
    if not path.is_file():
        raise WorkflowError(f"Missing {kind} terminal artifact: {path}")
    value = read(path)
    required = {"schema_version", "kind", "science_id", "calculation", "task", "producer",
                "dependencies", "hashes", "runtime_compatibility", "source_diagnostics", "checkpoint"}
    if set(value) != required or value.get("schema_version") != SCHEMA:
        raise WorkflowError(f"Legacy or unsupported terminal artifact {path}; safe continuation requires a validated PR3 manifest")
    if value["kind"] != kind or value["science_id"] != digest(science) or \
            value["calculation"] != calculation or value["task"] != task:
        raise WorkflowError(f"Artifact stage/science/task identity mismatch: {path}")
    if dependencies is not None and value["dependencies"] != dependencies:
        raise WorkflowError(f"Artifact dependency identity mismatch: {path}")
    if runtime is not None and value["runtime_compatibility"] != compatibility(runtime):
        raise WorkflowError(f"Artifact runtime compatibility mismatch: {path}")
    hashes = value["hashes"]
    if not isinstance(hashes, dict) or not hashes:
        raise WorkflowError(f"Artifact has no immutable outputs: {path}")
    for name, expected in hashes.items():
        output = relative_file(root, name)
        if not isinstance(expected, str) or not output.is_file() or sha256(output) != expected:
            raise WorkflowError(f"Missing or changed immutable artifact: {output}")
    return value


def manifest_id(path):
    return sha256(path)


def global_dependencies(science):
    return {"frozen_inputs": digest(science["input_hashes"])}


def validate_global(root, science):
    path = paths(root)
    value = validate_manifest(path, root, science, "global_inputs", global_dependencies(science))
    required = {"receptor/protein.gro", "receptor/protein.top"}
    for ligand in science["campaign"]["ligands"]:
        required.update({f"parameters/{ligand}/ligand.sdf", f"parameters/{ligand}/ligand.itp"})
    if not required <= value["hashes"].keys():
        raise WorkflowError("Global preparation manifest lacks required receptor/ligand outputs")
    return value


def stage_index(science):
    """The single dependency graph used by artifact lineage and submissions."""
    return {stage["id"]: stage for stage in stage_graph(science)}


def prep_dependencies(root, science, calc):
    stages = stage_index(science)
    expected = [name for name, node in stages.items() if node["kind"] == "global_input"]
    if stages[f"prep:{calc['key']}"]["dependencies"] != expected:
        raise WorkflowError(f"Invalid global preparation dependencies: {calc['key']}")
    validate_global(root, science)
    return {"global_inputs": manifest_id(paths(root)), "preparation_seed": calc["prep_seed"]}


def equil_dependencies(root, science, calc):
    stage = stage_index(science)[f"equil:{calc['key']}"]
    if stage["dependencies"] != [f"prep:{calc['key']}"]:
        raise WorkflowError(f"Invalid equilibration dependencies: {calc['key']}")
    validate_preparation(root, science, calc)
    return {"system_preparation": manifest_id(paths(root, calc)),
            "preparation_seed": calc["prep_seed"],
            "equilibration_iterations": round(science["protocol"]["equilibration_ns"] / .005)}


def validate_preparation(root, science, calc, runtime=None):
    from .planning import workdir
    legacy = Path(root) / "calculations" / calc["key"] / "prep.ok.json"
    if legacy.exists():
        raise WorkflowError(f"Legacy PR3 preproduction marker {legacy}; plan a new PR4 run")
    path = paths(root, calc)
    value = validate_manifest(path, root, science, "system_preparation", prep_dependencies(root, science, calc),
                              runtime if runtime is not None else runtime_snapshot(root),
                              calculation=calc["key"], task=f"prep:{calc['key']}")
    work = workdir(root, calc)
    required = {str((work / "prepare" / f"sys{leg}.{ext}").relative_to(root))
                for leg in "AB" for ext in ("gro", "top")}
    required.update(str((work / "prepare" / f"sys{leg}_{suffix}").relative_to(root))
                    for leg in "AB" for suffix in ("atom_ids.json", "ab_ligatoms.json", "posres.json"))
    directory = Path(root) / "calculations" / calc["key"]
    required.update(str((directory / name).relative_to(root)) for name in
                    ("abfecfg.json", "assembly_validation.json",
                     f'input/{calc["target"]}.sdf', f'input/{calc["target"]}.itp'))
    if not required <= value["hashes"].keys():
        raise WorkflowError(f"Prepared system lacks required outputs: {calc['key']}")
    for stage in PREP_STAGES:
        if not (work / "progress" / f"{stage}.done").is_file():
            raise WorkflowError(f"Prepared system lacks FELIS stage {stage}: {calc['key']}")
    anchor = read(directory / "location.json")
    if anchor != {"root": str(Path(root).resolve()), "science_id": digest(science)}:
        raise WorkflowError("Initialized FELIS system moved or changed scientific settings")
    return value


def validate_equilibration(root, science, calc, runtime=None):
    from .planning import workdir
    value = validate_manifest(paths(root, calc, "equil"), root, science, "equilibration",
                              equil_dependencies(root, science, calc),
                              runtime if runtime is not None else runtime_snapshot(root),
                              calculation=calc["key"], task=f"equil:{calc['key']}")
    work = workdir(root, calc)
    required = {str((work / "prepare" / f"sys{leg}_{suffix}").relative_to(root))
                for leg in "AB" for suffix in ("em.pdb", "lam.json")}
    required.add(str((work / "prepare/sys_boresch_cfg.json").relative_to(root)))
    if not required <= value["hashes"].keys():
        raise WorkflowError(f"Equilibration lacks required production starts/configuration: {calc['key']}")
    for stage in EQUIL_STAGES:
        if not (work / "progress" / f"{stage}.done").is_file():
            raise WorkflowError(f"Equilibration lacks FELIS stage {stage}: {calc['key']}")
    return value


class ValidatedEquilibrationParents:
    """Cache verified terminal parent identities for one operation only."""

    def __init__(self, root, science, runtime=None):
        self.root, self.science, self.runtime = root, science, runtime
        self._identities = {}
        self._stages = stage_index(science)

    def identity(self, calc, runtime=None):
        if runtime is not None:
            if self.runtime is not None and compatibility(self.runtime) != compatibility(runtime):
                raise WorkflowError("Conflicting runtime identities for validated equilibration parents")
            self.runtime = runtime
        key = calc["key"]
        if key not in self._identities:
            if self.runtime is None:
                self.runtime = runtime_snapshot(self.root)
            validate_equilibration(self.root, self.science, calc, self.runtime)
            self._identities[key] = manifest_id(paths(self.root, calc, "equil"))
        return self._identities[key]

    def dependencies(self, calc, leg, index):
        from .planning import calculation
        owner = calculation(self.science, calc["solvent_owner"]) if leg == "A" else calc
        stage = self._stages[f"group:{owner['key']}:{leg}:{index}"]
        if stage["dependencies"] != [f"equil:{owner['key']}"]:
            raise WorkflowError(f"Invalid group equilibration dependencies: {stage['id']}")
        return {"equilibration": self.identity(owner), "group_seed": owner["seeds"][leg][index]}


def group_dependencies(root, science, calc, leg, index):
    return ValidatedEquilibrationParents(root, science).dependencies(calc, leg, index)


def runtime_snapshot(root):
    path = Path(root) / "runtime.json"
    if not path.is_file():
        raise WorkflowError("Simulation runtime snapshot is missing")
    return read(path)


def group_state(root, science, calc, leg, unit, *, semantic=False, producer=None,
                producer_runtime=None, reporter_factory=None, dependencies=None, runtime=None):
    """Only the semantic branch may inspect a checkpointed NetCDF trajectory."""
    from .planning import calculation, workdir
    from .validation import iteration_status
    owner = calculation(science, calc["solvent_owner"]) if leg == "A" else calc
    path = paths(root, owner, leg, unit)
    runtime = runtime if runtime is not None else runtime_snapshot(root)
    dependencies = dependencies if dependencies is not None else group_dependencies(root, science, calc, leg, unit["index"])
    saved = None
    if path.exists():
        saved = validate_manifest(path, root, science, "simulation_group", dependencies, runtime,
                                  calculation=owner["key"], task=f"group:{owner['key']}:{leg}:{unit['index']}")
        trajectory = str((workdir(root, owner) / "trj" / f'{unit["stem"]}.nc').relative_to(root))
        checkpoint = saved["checkpoint"]
        if not isinstance(checkpoint, dict) or checkpoint.get("trajectory") != trajectory or \
                checkpoint.get("stored_target") != unit["iterations"] or \
                checkpoint.get("checkpoint_interval") != unit["checkpoint_interval"] or \
                any(type(checkpoint.get(name)) is not int or checkpoint[name] < 0
                    for name in ("last_iteration", "last_checkpoint")) or \
                checkpoint["last_checkpoint"] > checkpoint["last_iteration"] or \
                checkpoint["last_iteration"] != unit["iterations"]:
            raise WorkflowError(f"Simulation record does not match its unit: {path}")
        # stat only: status may run while another worker is writing this file.
        if not relative_file(root, trajectory).is_file():
            raise WorkflowError(f"Recorded checkpointed trajectory is missing: {trajectory}")
        marker = str(Path(trajectory).with_suffix(".create_done"))
        if set(saved["hashes"]) != {marker}:
            raise WorkflowError(f"Simulation record lacks its FELIS creation marker: {path}")
    from .initialization import state as initialization_state
    initialization = initialization_state(root, science, calc, leg, unit, dependencies, runtime)
    if not semantic:
        return {"complete": saved is not None, "verified": False,
                "reason": "recorded; trajectory inspection deferred" if saved else "no terminal record"}
    if saved is None and initialization == "pending":
        return {"complete": False, "verified": True, "stem": unit["stem"],
                "reason": "interrupted group initialization; production has not started"}
    status = iteration_status(root, science, calc, leg, unit, reporter_factory)
    if initialization == "created" and status["last_iteration"] != 0:
        raise WorkflowError("Group sampled before its initialization ready record was committed")
    if saved and (not status["complete"] or
                  status["last_iteration"] != saved["checkpoint"]["last_iteration"] or
                  status["last_checkpoint"] < saved["checkpoint"]["last_checkpoint"]):
        raise WorkflowError(f"Recorded simulation completion has invalid checkpoint: {path}")
    if status["complete"] and saved is None:
        if producer is None or producer_runtime is None:
            raise WorkflowError("Semantic completion needs an attempt and its current producer runtime")
        if compatibility(runtime) != compatibility(producer_runtime):
            raise WorkflowError("Producer runtime differs from the recorded simulation runtime")
        marker = workdir(root, owner) / "trj" / f'{unit["stem"]}.create_done'
        checkpoint = {"trajectory": str(Path(status["trajectory"]).relative_to(root)),
                      "last_iteration": status["last_iteration"], "last_checkpoint": status["last_checkpoint"],
                      "stored_target": status["stored_target"], "checkpoint_interval": unit["checkpoint_interval"]}
        commit_manifest(path, root, science, "simulation_group", [marker], dependencies, producer_runtime,
                        producer, calculation=owner["key"],
                        task=f"group:{owner['key']}:{leg}:{unit['index']}", checkpoint=checkpoint)
    return {**status, "verified": True}


def final_dependencies(root, science, calc, *, parents=None):
    from .planning import calculation, units
    records = {}
    stages = stage_index(science)
    parents = parents if parents is not None else ValidatedEquilibrationParents(root, science)
    for name in stages[f"finalize:{calc['key']}"]["dependencies"]:
        stage = stages[name]
        leg, index = stage["leg"], stage["index"]
        owner = calculation(science, stage["calculation"])
        unit = units(root, science, calc, leg)[index]
        group_state(root, science, calc, leg, unit,
                    dependencies=parents.dependencies(calc, leg, index), runtime=parents.runtime)
        path = paths(root, owner, leg, unit)
        if not path.exists():
            raise WorkflowError(f"Finalization needs a recorded simulation group: {path}")
        records[f"{owner['key']}:{leg}:{index}"] = manifest_id(path)
    return records


def validate_final(root, science, calc, *, parents=None):
    value = validate_manifest(paths(root, calc, "final"), root, science, "finalization",
                              final_dependencies(root, science, calc, parents=parents), runtime_snapshot(root),
                              calculation=calc["key"], task=f"finalize:{calc['key']}")
    from .planning import workdir
    directory, analysis = Path(root) / "calculations" / calc["key"], workdir(root, calc) / "analysis"
    required = {str((directory / name).relative_to(root)) for name in
                ("result.json", "completion_audit.json", "partner_occupancy.json")}
    required.update(str((analysis / name).relative_to(root)) for name in
                    ("A_fe_table.tsv", "B_fe_table.tsv", "R_fe_table.tsv", "sys_abfe.tsv"))
    if not required <= value["hashes"].keys():
        raise WorkflowError(f"Finalization manifest lacks required outputs: {calc['key']}")
    return value


def paths(root, calc=None, leg=None, unit=None):
    root = Path(root)
    if calc is None:
        return root / "prepared.json"
    directory = root / "calculations" / calc["key"]
    if unit is not None:
        return directory / "completion" / f'{unit["stem"]}.json'
    return directory / ("assembly.ok.json" if leg is None else
                        "equil.ok.json" if leg == "equil" else "finalized.json")


def stage_graph(science):
    """Canonical artifact graph; submission arrays are derived from these nodes."""
    stages = [{"id": "global:receptor", "kind": "global_input", "dependencies": ["science"]}]
    stages += [{"id": f"global:ligand:{name}", "kind": "global_input", "dependencies": ["science"]}
               for name in science["campaign"]["ligands"]]
    global_ids = [s["id"] for s in stages]
    for calc in science["calculations"]:
        key = calc["key"]
        stages.append({"id": f"prep:{key}", "kind": "system_preparation", "calculation": key,
                       "dependencies": global_ids})
        stages.append({"id": f"equil:{key}", "kind": "equilibration", "calculation": key,
                       "dependencies": [f"prep:{key}"]})
        for leg in "AB":
            if leg == "A" and calc["solvent_owner"] != key:
                continue
            for index, _ in enumerate(science["ladders"][leg]["groups"]):
                stages.append({"id": f"group:{key}:{leg}:{index}", "kind": "simulation_group",
                               "calculation": key, "leg": leg, "index": index,
                               "dependencies": [f"equil:{key}"]})
    for calc in science["calculations"]:
        deps = [f"group:{calc['solvent_owner']}:A:{i}"
                for i in range(len(science["ladders"]["A"]["groups"]))]
        deps += [f"group:{calc['key']}:B:{i}"
                 for i in range(len(science["ladders"]["B"]["groups"]))]
        stages.append({"id": f"finalize:{calc['key']}", "kind": "finalization",
                       "calculation": calc["key"], "dependencies": deps})
    return stages
