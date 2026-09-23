"""Paired-replicate coupling estimates; no independence assumption for shared legs."""
from __future__ import annotations
import csv
import math
from pathlib import Path
import statistics

from .common import WorkflowError, digest, keys, read, verify_hashes, write
from .planning import load_run


def estimates(values):
    n = len(values)
    return {"n": n, "mean_kcal_mol": statistics.mean(values) if n else None,
            "standard_error_kcal_mol": statistics.stdev(values) / math.sqrt(n) if n > 1 else None}


def coupling(science, records):
    definition = science["campaign"].get("coupling")
    if not definition:
        return None
    rows, excluded = [], []
    for replicate in range(1, science["protocol"]["replicates"] + 1):
        names = ["L_in_R", "P_in_R", "L_in_RP", "P_in_RL"]
        entries = [records.get(f"{definition[name]}/r{replicate}") for name in names]
        if any(r is None for r in entries):
            excluded.append({"replica": replicate, "reason": "incomplete cycle"})
            continue
        if any(not r["occupancy_passed"] for r in entries):
            excluded.append({"replica": replicate, "reason": "partner occupancy check failed"})
            continue
        if any(r["formal_charge"] and not r.get("correction") for r in entries):
            excluded.append({"replica": replicate, "reason": "charged target requires documented external finite-size correction"})
            continue
        a, b, c, d = [r["dG_kcal_mol"] for r in entries]
        rows.append({"replica": replicate, "via_L_kcal_mol": c - a, "via_P_kcal_mol": d - b,
                     "closure_kcal_mol": a + d - b - c})
    return {"definition": "coupling = G(L|RP)-G(L|R) = G(P|RL)-G(P|R); negative favors co-binding",
            "replicas": rows, "excluded": excluded,
            "via_L": estimates([v["via_L_kcal_mol"] for v in rows]),
            "via_P": estimates([v["via_P_kcal_mol"] for v in rows]),
            "closure": estimates([v["closure_kcal_mol"] for v in rows]),
            "uncertainty": "SE across matched independent replicate differences; shared solvent covariance retained. No automatic route averaging. External correction uncertainty is not included."}


def analyze(root, corrections=None):
    root = Path(root).resolve()
    science = load_run(root, prepared=True)
    records, missing = {}, []
    adjustments = read(corrections) if corrections else {}
    known = {c["key"] for c in science["calculations"]}
    if set(adjustments) - known:
        raise WorkflowError("Correction names must match calculation/replica keys")
    for name, value in adjustments.items():
        keys(value, {"value_kcal_mol", "provenance"}, {"value_kcal_mol", "provenance"}, f"correction {name}")
        if not isinstance(value["provenance"], str) or not value["provenance"].strip() or not math.isfinite(value["value_kcal_mol"]):
            raise WorkflowError("Corrections require a finite additive value and nonempty provenance")
    for calc in science["calculations"]:
        directory = root / "calculations" / calc["key"]
        marker = directory / "finalized.json"
        if not marker.exists():
            missing.append(calc["key"])
            continue
        ready = read(marker)
        if ready["science_id"] != digest(science):
            raise WorkflowError("Analysis belongs to a different scientific plan")
        verify_hashes(root, ready["hashes"])
        result = read(directory / "result.json")
        if result["science_id"] != digest(science) or (result["calculation"], result["replica"]) != (calc["id"], calc["replica"]):
            raise WorkflowError("Result identity mismatch")
        result["correction"] = adjustments.get(calc["key"])
        result["dG_kcal_mol"] = result["raw_dG_kcal_mol"] + (result["correction"]["value_kcal_mol"] if result["correction"] else 0)
        result["interpretation"] = "raw, charge correction pending" if result["formal_charge"] and not result["correction"] else "binding estimate"
        if not result["occupancy_passed"]:
            result["interpretation"] = "raw, partner occupancy failed"
        records[calc["key"]] = result
    report = {"science_id": digest(science), "purpose": science["protocol"]["purpose"],
              "forcefield": science["forcefield"], "missing": missing, "results": records,
              "coupling": coupling(science, records),
              "qualification": "Numerical estimates require convergence, overlap, restraints and receptor-state review; smoke/validation runs are not production predictions."}
    destination = root / "analysis"
    write(destination / "report.json", report)
    with (destination / "abfe.tsv").open("w") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["calculation/replica", "raw_dG_kcal_mol", "dG_kcal_mol", "interpretation"])
        for key, row in records.items():
            writer.writerow([key, row["raw_dG_kcal_mol"], row["dG_kcal_mol"], row["interpretation"]])
    return {"report": str(destination / "report.json"), "completed": len(records), "missing": missing, "coupling": report["coupling"]}
