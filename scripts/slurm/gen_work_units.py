#!/usr/bin/env python3
# Generate per-group repex work units for a Felis ABFE run, for execution as
# Slurm array tasks. This REUSES Felis's own lambda construction and
# split_replica_exchange_jobs so the ladder tiling/overlap stays identical to
# what the single-node orchestrator produces -- which is what the unmodified
# mbar stage (stage_post_analysis.py) expects to read back.
#
# Output: a JSON manifest with one entry per group, for legs sysA and sysB.
# Each entry carries the exact `felis.app.dyn.repex` argv (minus mpirun/MPS,
# which the array task wrapper adds) plus the stem (a{idx}/b{idx}) and the
# ilam index list for that group.
#
# Usage:
#   python gen_work_units.py --abfecfg examples/abfe/abfecfg.yaml \
#       --na 12 --nb 14 --workdir <abs path to tyk2_example/ejm_31> \
#       --sdf-abs <abs path to ejm_31.sdf> --out work_units.json
#
# --na / --nb are the NUMBER OF GROUPS (== array sizes == the n_cuda_devices
# value recorded in the .done files). Pick them for preemption granularity;
# they need NOT equal any physical GPU count.

import argparse
import json
import os
from pathlib import Path

from felis.protocols.abfe.config_types import ABFEInputConfig
from felis.protocols.abfe.recipes import (
    get_ve_lambdas_with_res,
    get_r_lambdas_dim3,
    split_replica_exchange_jobs,
)


def build_sysA_lams(acfg):
    lam_e = get_ve_lambdas_with_res(acfg.elamrecipe, "e", ascend=False, reslam=0.0,
                                    suppl_lams=acfg.supplementary_elec_lambda_list or [])
    lam_v = get_ve_lambdas_with_res(acfg.vlamrecipe, "v", ascend=False, reslam=0.0,
                                    suppl_lams=acfg.supplementary_vdw_lambda_list or [])
    assert lam_e[-1] == lam_v[0]
    return (lam_e + lam_v[1:]).copy()


def build_sysB_lams(acfg):
    lam_v = get_ve_lambdas_with_res(acfg.vlamrecipe, "v", ascend=True, reslam=1.0,
                                    suppl_lams=acfg.supplementary_vdw_lambda_list or [])
    lam_e = get_ve_lambdas_with_res(acfg.elamrecipe, "e", ascend=True, reslam=1.0,
                                    suppl_lams=acfg.supplementary_elec_lambda_list or [])
    lam_r = get_r_lambdas_dim3(acfg.reslamrecipe, ascend=False, vlam=1.0, elam=1.0,
                               suppl_lams=acfg.supplementary_restraint_lambda_list or [])
    assert lam_v[-1] == lam_e[0]
    assert lam_e[-1] == lam_r[0]
    return (lam_v + lam_e[1:] + lam_r[1:]).copy()


def make_repex_argv(leg, idx, igroup, acfg, sdf_abs):
    # Mirror stage_sysA.py / stage_sysB.py EXACTLY (minus mpirun/MPS wrapper).
    if leg == "A":
        cfg = ["prepare/sysA_lam.json", "prepare/sysA_ab_ligatoms.json"]
        sys_top = "prepare/sysA.top"
        crd = "prepare/sysA_em.pdb"
        atom_ids = "prepare/sysA_atom_ids.json"
        nsnap = acfg.md_sol_nsnapshots
        extra_cfg = []
    else:
        # sysB also passes the boresch cfg json as a third --cfg entry
        from felis.protocols.boresch.main_boresch_restraints import BORESCH_CFG_JSON
        cfg = ["prepare/sysB_lam.json", "prepare/sysB_ab_ligatoms.json", f"prepare/{BORESCH_CFG_JSON}"]
        sys_top = "prepare/sysB.top"
        crd = "prepare/sysB_em.pdb"
        atom_ids = "prepare/sysB_atom_ids.json"
        nsnap = acfg.md_pro_nsnapshots
        extra_cfg = []

    stem = f"{'a' if leg == 'A' else 'b'}{idx}"
    argv = [
        "python3", "-m", "felis.app.dyn.repex",
        "--cfg", *cfg,
        "--tkv",
        f"s:filename.stem:{stem}",
        f"s:filename.monomer:{sdf_abs}",
        f"s:filename.sys:{sys_top}",
        f"s:filename.crd:{crd}",
        f"s:filename.atom_ids:{atom_ids}",
        "s:dir.trj:trj",
        f"i:openmm.checkpoint_interval:{acfg.md_checkpoint_interval}",
        "i:integrator.npt:1",
        "i:integrator.nstep_per_snapshot:2500",
        f"i:integrator.nsnapshots:{nsnap}",
        "--rextkv",
        *[f"i:ab.ilam:{vv}" for vv in igroup],
    ]
    return stem, argv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abfecfg", required=True)
    ap.add_argument("--na", type=int, required=True, help="number of sysA groups (array size A)")
    ap.add_argument("--nb", type=int, required=True, help="number of sysB groups (array size B)")
    ap.add_argument("--workdir", required=True, help="abs path to <outdir>/<ligand> (has prepare/ trj/ progress/)")
    ap.add_argument("--sdf-abs", required=True)
    ap.add_argument("--out", default="work_units.json")
    args = ap.parse_args()

    acfg = ABFEInputConfig.from_file(args.abfecfg)

    sysA = build_sysA_lams(acfg)
    sysB = build_sysB_lams(acfg)
    groupsA = split_replica_exchange_jobs(args.na, list(range(len(sysA))))
    groupsB = split_replica_exchange_jobs(args.nb, list(range(len(sysB))))

    # split_replica_exchange_jobs may emit FEWER groups than requested if
    # njobs is small; the .done count must match the ACTUAL number emitted.
    na_actual = len(groupsA)
    nb_actual = len(groupsB)

    manifest = {
        "workdir": args.workdir,
        "na": na_actual,
        "nb": nb_actual,
        "sysA_lam": sysA,
        "sysB_lam": sysB,
        "units": {"A": [], "B": []},
    }
    for idx, g in enumerate(groupsA):
        stem, argv = make_repex_argv("A", idx, g, acfg, args.sdf_abs)
        manifest["units"]["A"].append({"idx": idx, "stem": stem, "ilam": g, "argv": argv})
    for idx, g in enumerate(groupsB):
        stem, argv = make_repex_argv("B", idx, g, acfg, args.sdf_abs)
        manifest["units"]["B"].append({"idx": idx, "stem": stem, "ilam": g, "argv": argv})

    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)

    # Also write the sysX_lam.json that the repex commands reference, exactly
    # as the stages do (dump_config writes {"ab": {"lam_list": [...]}}).
    prep = Path(args.workdir) / "prepare"
    prep.mkdir(parents=True, exist_ok=True)
    with open(prep / "sysA_lam.json", "w") as f:
        json.dump({"ab": {"lam_list": sysA}}, f, indent=4)
    with open(prep / "sysB_lam.json", "w") as f:
        json.dump({"ab": {"lam_list": sysB}}, f, indent=4)

    print(f"sysA: {len(sysA)} windows -> {na_actual} groups "
          f"(largest {max(len(g) for g in groupsA)} windows)")
    print(f"sysB: {len(sysB)} windows -> {nb_actual} groups "
          f"(largest {max(len(g) for g in groupsB)} windows)")
    print(f"Wrote {args.out}")
    print(f"IMPORTANT: array A must be 0-{na_actual-1}, array B must be 0-{nb_actual-1}")
    print(f"           .done files must record n_cuda_devices = {na_actual} (A) / {nb_actual} (B)")


if __name__ == "__main__":
    main()
