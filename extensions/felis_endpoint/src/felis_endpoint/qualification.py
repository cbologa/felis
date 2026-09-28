"""Evidence inventory for ABFE profiles; convergence remains a scientific judgment."""
from __future__ import annotations

import csv
from pathlib import Path

from felis_workflows.artifacts import compatibility
from felis_workflows.common import WorkflowError, digest, read, write
from felis_workflows.orchestration import status as abfe_status
from felis_workflows.planning import load_run, workdir

from .identity import activate


NATIVE = ("A_fe_table.tsv", "B_fe_table.tsv", "R_fe_table.tsv", "sys_abfe.tsv")
CONVERGENCE = ("A_converge_table.tsv", "B_converge_table.tsv")


def table(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise WorkflowError(f"Empty native FELIS analysis table: {path}")
    return {"columns": list(rows[0]), "rows": rows}


def qualify(root, site, output=None):
    activate(site)
    root = Path(root).resolve()
    science = load_run(root)
    errors = []
    runtime_path = root / "runtime.json"
    if runtime_path.is_file():
        from felis_workflows.runtime import fingerprint
        try:
            if compatibility(read(runtime_path)) != compatibility(fingerprint(site)):
                errors.append("Current ABFE runner differs from the recorded scientific runtime")
        except WorkflowError as error:
            errors.append(str(error))
    else:
        errors.append("ABFE runtime snapshot is missing")
    try:
        snapshot = abfe_status(root)
        state = {entry["calculation"]: entry for entry in snapshot["calculations"]}
    except WorkflowError as error:
        errors.append(str(error))
        state = {}
        snapshot = {"global_inputs": "unverified"}
    results, missing = {}, []
    for calc in science["calculations"]:
        analysis = workdir(root, calc) / "analysis"
        native = {}
        for name in (*NATIVE, *CONVERGENCE):
            path = analysis / name
            if not path.is_file():
                native[name] = {"state": "missing"}
                continue
            try:
                native[name] = {"state": "present", **table(path)}
            except (WorkflowError, UnicodeError, csv.Error) as error:
                native[name] = {"state": "invalid", "reason": str(error)}
        entry = state.get(calc["key"], {})
        final = entry.get("finalization") == "complete"
        evidence = [name for name, value in native.items() if value["state"] != "present"]
        if not final:
            evidence.append("finalized.json")
            missing.append(calc["key"])
        qualified = final and not evidence
        results[calc["key"]] = {"calculation": calc["id"], "replica": calc["replica"],
                                "state": entry.get("state", "unverified"),
                                "assembly": entry.get("system_preparation"),
                                "equilibration": entry.get("equilibration"),
                                "groups": {leg: {"recorded": entry.get("groups", {}).get(leg, {}).get("completed_records", 0),
                                                 "expected": len(science["ladders"][leg]["groups"])}
                                           for leg in "AB"},
                                "terminal_finalization": final, "terminal_qualified": qualified,
                                "missing_evidence": evidence,
                                "native_analysis": native}
    calculations_by_replica = {}
    for calc in science["calculations"]:
        calculations_by_replica.setdefault(calc["replica"], []).append(calc["key"])
    completed_replicas = sorted(replica for replica, keys in calculations_by_replica.items()
                                if all(results[key]["terminal_qualified"] for key in keys))
    report = {"schema_version": 1, "source_run": str(root), "science_id": digest(science),
              "profile": science["protocol"]["name"], "purpose": science["protocol"]["purpose"],
              "expected_calculations": len(science["calculations"]),
              "global_inputs": snapshot["global_inputs"],
              "expected_replicas": science["protocol"]["replicates"],
              "planned_replicas": sorted({c["replica"] for c in science["calculations"]}),
              "completed_replicas": completed_replicas,
              "sampling_ns": {"solvent": science["protocol"]["solvent_ns"],
                              "complex": science["protocol"]["complex_ns"]},
              "alchemical_states": {leg: len(science["ladders"][leg]["lambdas"]) for leg in "AB"},
              "missing_calculations": missing, "artifact_errors": errors, "calculations": results,
              "scientific_convergence": "not automatically classified; review A/B tables, overlap, restraints and receptor state"}
    if output is not None:
        if Path(output).resolve().is_relative_to(Path(site["repo"]).resolve()):
            raise WorkflowError("Generated qualification reports must be outside the Git checkout")
        write(output, report)
    return report
