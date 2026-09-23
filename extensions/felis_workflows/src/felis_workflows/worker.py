"""One execution unit. Invoked in the appropriate site-selected Python environment."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .common import WorkflowError, digest, file_hashes, lock, read, sha256, verify_hashes, write
from .planning import abfe_config, calculation, load_run, units, workdir
from .runtime import allocation, check_runtime, mpi_command, stages

PREP_STAGES = ["makebox", "boresch_em", "boresch_npt", "boresch_post_process", "sysA_em", "sysB_em"]


def calcdir(root, calc):
    return Path(root) / "calculations" / calc["key"]


def check_prepared(root, science, calc, site):
    ready = read(calcdir(root, calc) / "prep.ok.json")
    if ready["science_id"] != digest(science):
        raise WorkflowError("Prepared system belongs to a different plan")
    if ready["root"] != str(Path(root).resolve()):
        raise WorkflowError("An initialized FELIS system moved; use its original absolute mount path or prepare a new run")
    verify_hashes(root, ready["hashes"])
    check_runtime(root, site)
    return ready


def prepare_system(root, science, calc, site):
    from .validation import validate_assembled
    directory = calcdir(root, calc)
    directory.mkdir(parents=True, exist_ok=True)
    with lock(directory / "prep.lock"):
        if (directory / "prep.ok.json").exists():
            check_prepared(root, science, calc, site)
            return
        check_runtime(root, site)
        anchor = directory / "location.json"
        current = {"root": str(root), "science_id": digest(science)}
        if anchor.exists() and read(anchor) != current:
            raise WorkflowError("A partial FELIS preparation moved or changed scientific settings")
        write(anchor, current)
        input_dir = directory / "input"
        input_dir.mkdir(exist_ok=True)
        # FELIS identifies the work directory by the target SDF stem.
        for suffix in ["sdf", "itp"]:
            shutil.copyfile(root / "parameters" / calc["target"] / f"ligand.{suffix}",
                            input_dir / f'{calc["target"]}.{suffix}')
        cfg = abfe_config(root, science, calc)
        cfg["sdffile"] = str(input_dir / f'{calc["target"]}.sdf')
        cfg["itpfile"] = str(input_dir / f'{calc["target"]}.itp')
        config = directory / "abfecfg.json"
        write(config, cfg)
        work = workdir(root, calc)
        with allocation(site):
            stages(config, ["makebox"], site, calc["prep_seed"])
            write(directory / "assembly_validation.json", validate_assembled(root, science, calc))
            gmx = shutil.which("gmx")
            if not gmx:
                raise WorkflowError("gmx is required for full-system topology validation")
            for leg in "AB":
                with (directory / f"grompp_{leg}.log").open("w") as log:
                    subprocess.run([gmx, "grompp", "-f", str(Path(__file__).parent / "data/grompp_check.mdp"),
                                    "-c", str(work / f"prepare/sys{leg}.gro"), "-p", str(work / f"prepare/sys{leg}.top"),
                                    "-o", str(directory / f"grompp_{leg}.tpr"), "-po", str(directory / f"grompp_{leg}.mdp")],
                                   cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)
            stages(config, PREP_STAGES[1:], site, calc["prep_seed"])
        for stage in PREP_STAGES:
            if not (work / "progress" / f"{stage}.done").exists():
                raise WorkflowError(f"Incomplete preparation: {stage}")
        for leg in "AB":
            write(work / f"prepare/sys{leg}_lam.json", {"ab": {"lam_list": science["ladders"][leg]["lambdas"]}})
        files = [*input_dir.rglob("*"), *list((work / "prepare").rglob("*")), config,
                 directory / "assembly_validation.json"]
        write(directory / "prep.ok.json", {"science_id": digest(science), "root": str(root), "hashes": {
            str(p.relative_to(root)): sha256(p) for p in files if p.is_file()}})


def simulate_group(root, science, calc, site, leg, index):
    from .validation import iteration_status
    if leg == "A" and calc["solvent_owner"] != calc["key"]:
        raise WorkflowError("A shared solvent calculation must be run through its owner")
    check_prepared(root, science, calc, site)
    item = units(root, science, calc, leg)[index]
    work = workdir(root, calc)
    if site["mpi"]["ranks"] * len(item["ilam"]) > 48:
        raise WorkflowError("MPI ranks times states exceed the pinned FELIS context limit; reduce site MPI ranks")
    with lock(work / "trj" / f'{item["stem"]}.lock'):
        status = iteration_status(root, science, calc, leg, item)
        if status["complete"]:
            return
        with allocation(site):
            subprocess.run(mpi_command(site, item["argv"]), cwd=work, check=True)
        status = iteration_status(root, science, calc, leg, item)
        if not status["complete"]:
            raise WorkflowError(f"Simulation stopped before its iteration target: {status}")
        write(calcdir(root, calc) / "completion" / f'{item["stem"]}.json', status)


def finalize(root, science, calc, site):
    from .validation import iteration_status, partner_occupancy
    check_prepared(root, science, calc, site)
    directory, work = calcdir(root, calc), workdir(root, calc)
    with lock(directory / "analysis.lock"):
        if (directory / "finalized.json").exists():
            verify_hashes(root, read(directory / "finalized.json")["hashes"])
            return
        owner = calculation(science, calc["solvent_owner"])
        check_prepared(root, science, owner, site)
        reports = [iteration_status(root, science, calc, leg, item)
                   for leg in "AB" for item in units(root, science, calc, leg)]
        write(directory / "completion_audit.json", {"groups": reports})
        if not all(v["complete"] for v in reports):
            raise WorkflowError("Finalization requires every group to reach its iteration target")
        occupancy = partner_occupancy(root, science, calc)
        write(directory / "partner_occupancy.json", occupancy)
        # Calculate raw ABFE even when occupancy fails, but mark it ineligible
        # for conditional-binding/coupling interpretation.
        from felis.protocols.abfe.main_fe_mbar import calc_mbar
        from felis.protocols.abfe.main_fe_restraints import calc_restraints
        from felis.protocols.abfe.main_fe_summarize import summarize_fe
        analysis = work / "analysis"
        analysis.mkdir(exist_ok=True)
        p = science["protocol"]
        for leg, source in [("A", workdir(root, owner)), ("B", work)]:
            paths = [str(source / "trj" / f'{u["stem"]}.nc') for u in units(root, science, calc, leg)]
            calc_mbar(stem=leg, nc_list=paths, checkpoint_interval=p["checkpoint_interval"], outdir=str(analysis))
        calc_restraints(boresch_cfg_json=str(work / "prepare/sys_boresch_cfg.json"), outdir=str(analysis))
        summarize_fe(workdir=str(analysis), ligand=calc["target"],
                     lam_sol_cfg=str(work / "prepare/sysA_lam.json"), lam_pro_cfg=str(work / "prepare/sysB_lam.json"))
        with (analysis / "sys_abfe.tsv").open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        if len(rows) != 1:
            raise WorkflowError("Expected exactly one ABFE result")
        dg = float(rows[0]["dG(kcal/mol)"])
        import math
        if not math.isfinite(dg):
            raise WorkflowError("Nonfinite ABFE result")
        record = {"calculation": calc["id"], "replica": calc["replica"], "target": calc["target"],
                  "partners": calc["partners"], "raw_dG_kcal_mol": dg, "solvent_owner": calc["solvent_owner"],
                  "occupancy_passed": occupancy["passed"], "science_id": digest(science),
                  "formal_charge": science["campaign"]["ligands"][calc["target"]]["formal_charge"]}
        write(directory / "result.json", record)
        files = [analysis / name for name in ["A_fe_table.tsv", "B_fe_table.tsv", "R_fe_table.tsv", "sys_abfe.tsv"]]
        files += [directory / "result.json", directory / "completion_audit.json", directory / "partner_occupancy.json"]
        write(directory / "finalized.json", {"science_id": digest(science), "hashes": {
            str(path.relative_to(root)): sha256(path) for path in files}})


def probe(root, science, site, destination):
    from .validation import iteration_status
    output = {}
    for calc in science["calculations"]:
        directory = calcdir(root, calc)
        done = (directory / "prep.ok.json").exists()
        if done:
            check_prepared(root, science, calc, site)
        entry = {"prep": done, "A": [], "B": [], "finalize": False}
        for leg in "AB":
            for item in units(root, science, calc, leg):
                entry[leg].append(iteration_status(root, science, calc, leg, item)["complete"] if done else False)
        if (directory / "finalized.json").exists():
            verify_hashes(root, read(directory / "finalized.json")["hashes"])
            entry["finalize"] = True
        output[calc["key"]] = entry
    write(destination, output)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["receptor", "parameterize", "prep", "group", "finalize", "probe"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--calculation")
    parser.add_argument("--ligand")
    parser.add_argument("--leg", choices=["A", "B"])
    parser.add_argument("--index", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.run.resolve()
    science = load_run(root, prepared=args.task not in {"receptor", "parameterize"})
    site = read(args.site)
    from .integrity import verify_upstream
    verify_upstream(site["repo"])
    if args.task == "receptor":
        from .preparation.receptor import build
        build(root, science)
    elif args.task == "parameterize":
        name = args.ligand
        ligand = science["campaign"]["ligands"][name]
        output = root / "parameters" / name
        if output.exists():
            metadata = read(output / "parameters.json")
            if metadata["source_sdf_sha256"] != sha256(root / ligand["sdf"]):
                raise WorkflowError("Existing ligand parameters have a different source molecule")
            from .parameterization.validation import validate
            validate(output)
        else:
            if science["forcefield"]["ligand"]["backend"] == "sage":
                from .parameterization.sage import parameterize
            else:
                from .parameterization.gaff2 import parameterize
            parameterize(root / ligand["sdf"], output, ligand["formal_charge"])
    elif args.task == "probe":
        probe(root, science, site, args.output)
    else:
        calc = calculation(science, args.calculation)
        if args.task == "prep":
            prepare_system(root, science, calc, site)
        elif args.task == "group":
            simulate_group(root, science, calc, site, args.leg, args.index)
        else:
            finalize(root, science, calc, site)


if __name__ == "__main__":
    main()
