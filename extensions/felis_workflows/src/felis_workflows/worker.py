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

from .artifacts import (EQUIL_STAGES, PREP_STAGES, ValidatedEquilibrationParents, commit_manifest, equil_dependencies,
                        final_dependencies, group_state, paths, prep_dependencies, validate_equilibration,
                        validate_final, validate_preparation, write_once)
from .common import WorkflowError, digest, lock, read, sha256, write
from .planning import abfe_config, calculation, load_run, units, workdir
from .runtime import activate_source, allocation, check_runtime, mpi_command, source_environment, stages

def calcdir(root, calc):
    return Path(root) / "calculations" / calc["key"]


def check_prepared(root, science, calc, site):
    current_runtime = check_runtime(root, site)
    validate_preparation(root, science, calc, current_runtime)
    return current_runtime


def check_equilibrated(root, science, calc, site, *, parents=None):
    current_runtime = check_runtime(root, site)
    if parents is None:
        validate_equilibration(root, science, calc, current_runtime)
    else:
        parents.identity(calc, current_runtime)
    return current_runtime


def prepare_system(root, science, calc, site):
    from .validation import validate_assembled
    directory = calcdir(root, calc)
    directory.mkdir(parents=True, exist_ok=True)
    with lock(directory / "prep.lock"):
        if (directory / "prep.ok.json").exists():
            raise WorkflowError("Legacy PR3 preproduction marker; plan a new PR4 run")
        if paths(root, calc, "equil").exists():
            check_equilibrated(root, science, calc, site)
            return
        anchor = directory / "location.json"
        if anchor.exists() and not (root / "runtime.json").is_file():
            raise WorkflowError("Partial FELIS preparation lacks its runtime identity; cannot safely continue")
        runtime = check_runtime(root, site)
        current = {"root": str(root), "science_id": digest(science)}
        if anchor.exists() and read(anchor) != current:
            raise WorkflowError("A partial FELIS preparation moved or changed scientific settings")
        write_once(anchor, current)
        input_dir = directory / "input"
        config = directory / "abfecfg.json"
        work = workdir(root, calc)
        with allocation(site):
            if not paths(root, calc).exists():
                input_dir.mkdir(exist_ok=True)
                # FELIS identifies the work directory by the target SDF stem.
                for suffix in ("sdf", "itp"):
                    source = root / "parameters" / calc["target"] / f"ligand.{suffix}"
                    destination = input_dir / f'{calc["target"]}.{suffix}'
                    if destination.exists():
                        if sha256(destination) != sha256(source):
                            raise WorkflowError(f"Conflicting partial preparation input: {destination}")
                    else:
                        shutil.copyfile(source, destination)
                cfg = abfe_config(root, science, calc)
                cfg["sdffile"] = str(input_dir / f'{calc["target"]}.sdf')
                cfg["itpfile"] = str(input_dir / f'{calc["target"]}.itp')
                write_once(config, cfg)
                stages(config, list(PREP_STAGES), site, calc["prep_seed"], work)
                if not (work / "progress/makebox.done").is_file():
                    raise WorkflowError("Incomplete system assembly: makebox")
                write_once(directory / "assembly_validation.json", validate_assembled(root, science, calc))
                gmx = shutil.which("gmx")
                if not gmx:
                    raise WorkflowError("gmx is required for full-system topology validation")
                for leg in "AB":
                    with (directory / f"grompp_{leg}.log").open("w") as log:
                        subprocess.run([gmx, "grompp", "-f", str(Path(__file__).parent / "data/grompp_check.mdp"),
                                        "-c", str(work / f"prepare/sys{leg}.gro"), "-p", str(work / f"prepare/sys{leg}.top"),
                                        "-o", str(directory / f"grompp_{leg}.tpr"), "-po", str(directory / f"grompp_{leg}.mdp")],
                                       cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)
                files = [config, directory / "assembly_validation.json",
                         *(input_dir / f'{calc["target"]}.{suffix}' for suffix in ("sdf", "itp")),
                         *(work / "prepare" / f"sys{leg}.{ext}" for leg in "AB" for ext in ("gro", "top")),
                         *(work / "prepare" / f"sys{leg}_{suffix}" for leg in "AB"
                           for suffix in ("atom_ids.json", "ab_ligatoms.json", "posres.json"))]
                commit_manifest(paths(root, calc), root, science, "system_preparation", files,
                                prep_dependencies(root, science, calc), runtime,
                                site.get("_attempt_id", "direct"), calculation=calc["key"], task=f"prep:{calc['key']}")
            validate_preparation(root, science, calc, runtime)
            stages(config, list(EQUIL_STAGES), site, calc["prep_seed"], work)
            for stage in EQUIL_STAGES:
                if not (work / "progress" / f"{stage}.done").is_file():
                    raise WorkflowError(f"Incomplete equilibration: {stage}")
            for leg in "AB":
                write_once(work / f"prepare/sys{leg}_lam.json", {"ab": {"lam_list": science["ladders"][leg]["lambdas"]}})
            files = [*(work / "prepare" / f"sys{leg}_{suffix}" for leg in "AB"
                       for suffix in ("em.pdb", "lam.json")), work / "prepare/sys_boresch_cfg.json"]
            commit_manifest(paths(root, calc, "equil"), root, science, "equilibration", files,
                            equil_dependencies(root, science, calc), runtime,
                            site.get("_attempt_id", "direct"), calculation=calc["key"], task=f"equil:{calc['key']}")
            validate_equilibration(root, science, calc, runtime)


def simulate_group(root, science, calc, site, leg, index):
    if leg == "A" and calc["solvent_owner"] != calc["key"]:
        raise WorkflowError("A shared solvent calculation must be run through its owner")
    parents = ValidatedEquilibrationParents(root, science)
    producer_runtime = check_equilibrated(root, science, calc, site, parents=parents)
    dependencies = parents.dependencies(calc, leg, index)
    item = units(root, science, calc, leg)[index]
    work = workdir(root, calc)
    if site["mpi"]["ranks"] * len(item["ilam"]) > 48:
        raise WorkflowError("MPI ranks times states exceed the pinned FELIS context limit; reduce site MPI ranks")
    with lock(work / "trj" / f'{item["stem"]}.lock'):
        status = group_state(root, science, calc, leg, item, semantic=True,
                             producer=site.get("_attempt_id", "direct"), producer_runtime=producer_runtime,
                             dependencies=dependencies, runtime=producer_runtime)
        if status["complete"]:
            return
        from .initialization import begin
        begin(root, science, calc, leg, item, dependencies, producer_runtime,
              site.get("_attempt_id", "direct"))
        with allocation(site):
            command = ["-m", "felis_workflows.repex", "--run", str(root),
                       "--calculation", calc["key"], "--leg", leg, "--index", str(index)]
            subprocess.run(mpi_command(site, command), cwd=work,
                           env=source_environment(site), check=True)
        status = group_state(root, science, calc, leg, item, semantic=True,
                             producer=site.get("_attempt_id", "direct"), producer_runtime=producer_runtime,
                             dependencies=dependencies, runtime=producer_runtime)
        if not status["complete"]:
            raise WorkflowError(f"Simulation stopped before its iteration target: {status}")


def finalize(root, science, calc, site):
    from .validation import partner_occupancy
    parents = ValidatedEquilibrationParents(root, science)
    producer_runtime = check_equilibrated(root, science, calc, site, parents=parents)
    directory, work = calcdir(root, calc), workdir(root, calc)
    with lock(directory / "analysis.lock"):
        owner = calculation(science, calc["solvent_owner"])
        parents.identity(owner)
        reports = [group_state(root, science, calc, leg, item, semantic=True,
                               producer=site.get("_attempt_id", "direct"), producer_runtime=producer_runtime,
                               dependencies=parents.dependencies(calc, leg, item["index"]), runtime=producer_runtime)
                   for leg in "AB" for item in units(root, science, calc, leg)]
        if not all(v["complete"] for v in reports):
            raise WorkflowError("Finalization requires every group to reach its iteration target")
        if (directory / "finalized.json").exists():
            validate_final(root, science, calc, parents=parents)
            return
        write(directory / "completion_audit.json", {"groups": reports})
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
            trajectories = [str(source / "trj" / f'{u["stem"]}.nc') for u in units(root, science, calc, leg)]
            calc_mbar(stem=leg, nc_list=trajectories, checkpoint_interval=p["checkpoint_interval"], outdir=str(analysis))
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
        commit_manifest(paths(root, calc, "final"), root, science, "finalization", files,
                        final_dependencies(root, science, calc, parents=parents), producer_runtime,
                        site.get("_attempt_id", "direct"), calculation=calc["key"],
                        task=f"finalize:{calc['key']}")


def probe(root, science, site, destination):
    output = {}
    producer_runtime = check_runtime(root, site)
    parents = ValidatedEquilibrationParents(root, science, producer_runtime)
    for calc in science["calculations"]:
        directory = calcdir(root, calc)
        if (directory / "prep.ok.json").exists():
            raise WorkflowError(f"Legacy PR3 preproduction marker in {directory}; plan a new PR4 run")
        prepared = paths(root, calc).exists()
        equilibrated = paths(root, calc, "equil").exists()
        if equilibrated and not prepared:
            raise WorkflowError(f"Equilibration without system preparation: {calc['key']}")
        if equilibrated:
            parents.identity(calc)
        elif prepared:
            validate_preparation(root, science, calc, producer_runtime)
        entry = {"prep": prepared, "equil": equilibrated, "A": [], "B": [], "finalize": False}
        for leg in "AB":
            owner = calculation(science, calc["solvent_owner"]) if leg == "A" else calc
            owner_equilibrated = paths(root, owner, "equil").exists()
            for item in units(root, science, calc, leg):
                if not owner_equilibrated and paths(root, owner, leg, item).exists():
                    raise WorkflowError(f"Simulation record without validated equilibration: {paths(root, owner, leg, item)}")
                entry[leg].append(group_state(root, science, calc, leg, item, semantic=True,
                                              producer=site.get("_attempt_id", "direct"),
                                              producer_runtime=producer_runtime,
                                              dependencies=parents.dependencies(calc, leg, item["index"]),
                                              runtime=producer_runtime)["complete"] if owner_equilibrated else False)
        if (directory / "finalized.json").exists():
            validate_final(root, science, calc, parents=parents)
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
    site = read(args.site)
    if args.site.parent.parent.name == "executions":
        site["_attempt_id"] = args.site.parent.name
    activate_source(site)
    science = load_run(root, prepared=args.task not in {"receptor", "parameterize"})
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
