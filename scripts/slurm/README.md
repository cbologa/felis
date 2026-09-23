# Felis ABFE — Slurm array backend

Run Felis absolute binding free energy (ABFE) calculations on a Slurm cluster
by submitting each replica-exchange **lambda-window group as its own array
task** on one GPU, instead of Felis's built-in single-node model (which expects
all GPUs in one box and launches groups as local subprocesses).

This is purpose-built for **preemptible / scavenger** partitions: small,
independently restartable tasks, with automatic requeue. It leaves Felis's
science untouched — the lambda partitioning, the `repex` sampling, and the
`mbar` + `summarize` analysis are all the unmodified upstream code. Only the
*execution backend* changes.

> **Two-phase model.** This backend only parallelizes the alchemical legs
> (the `sysA`/`sysB` repex groups). The preparatory stages — `makebox`,
> `boresch_em`, `boresch_npt`, `boresch_post_process`, `sysA_em`, `sysB_em` —
> are **not** parallelized here and must be run first, once, via the normal
> Felis orchestrator (they write `prepare/` and the `progress/*.done` flags the
> array tasks depend on). See **Phase 1** below.

## Why this works (and the one rule that must hold)

Felis already runs lambda-window **groups** independently — separate process,
separate GPU, separate `.nc` file, with replica exchange only *within* a group.
The `mbar` stage (`felis/protocols/abfe/stage_post_analysis.py`) then stitches
the per-group `.nc` files into one ladder by summing adjacent-state free-energy
differences in index order.

The groups are produced by `split_replica_exchange_jobs(...)`, which makes them
**overlap by one boundary state** so the ladder tiles contiguously. The mbar
stage relies on this exactly, and additionally **requires the file count to
equal the recorded `n_cuda_devices`** (it reads that back from the stage's
`.done` file and checks `a0.nc … a{N-1}.nc` are all present).

**The rule:** the work units MUST be generated with Felis's own
`split_replica_exchange_jobs` (this is what `gen_work_units.py` does). Do not
hand-roll a disjoint split — the overlap is load-bearing for a correct dG.

## Files

| file | role |
|------|------|
| `gen_work_units.py` | Rebuilds the sysA/sysB lambda ladders the same way the stages do, partitions them with `split_replica_exchange_jobs`, and writes `work_units.json` (one `repex` command per group) plus the `sysA_lam.json`/`sysB_lam.json` the commands reference. Validates that integer config fields (e.g. `md_checkpoint_interval`) are set, so a `None` can't silently become the string `"None"` and crash each rank. |
| `felis_array.sbatch` | One array task = one group, on one GPU via MPS. `--requeue`-safe; resumes via Felis's own `.create_done` / `from_storage` logic. Submitted once per leg (A, B). |
| `felis_finalize.sbatch` | Runs `afterok` on both arrays: verifies all `.nc` exist, writes the `.done` files with the correct group counts, runs `mbar` + `summarize` (unmodified), prints the dG and the tyk2 experimental reference. |
| `submit.sh` | Driver: generates the manifest, reads back the *actual* group counts, submits both arrays + finalize with the right dependency chain. |

## Phase 1 — prepare the system (run once, before the arrays)

The array backend picks up at the alchemical legs. First build the system and
run equilibration via the normal app, on one GPU:

```bash
cd examples/abfe
# copy the example inputs from the benchmark dataset (what run.sh does):
tyk2=../../pl_bfe_dataset/Schrodinger/jacs/tyk2
cp $tyk2/protein_ff14sb/protein_w_cofactors.gro \
   $tyk2/protein_ff14sb/protein_w_cofactors.top \
   $tyk2/ligands/ejm_31.sdf \
   $tyk2/ligands_itps_joint-25/ejm_31.itp  input/

# run the prep stages only (needs a GPU; ~30 min on an L40S):
python -m felis.app.abfe --abfecfg abfecfg.yaml \
    --stages makebox boresch_em boresch_npt boresch_post_process sysA_em sysB_em
```

When this finishes you should have all six `progress/*.done` flags and a
populated `prepare/` (`sysA.top`, `sysA_em.pdb`, `sysB.top`, `sys_boresch_cfg.json`,
the atom-id JSONs, …). Only then run Phase 2.

## Phase 2 — quick start (the alchemical legs)

Set `FELIS_REPO` (and `CONDA_SH` if your conda isn't at `$HOME/miniconda3`) and
run the driver:

```bash
cd scripts/slurm
FELIS_REPO=/path/to/felis ./submit.sh
```

Everything is environment-overridable. For a fast, non-preemptible end-to-end
smoke test on a short partition:

```bash
FELIS_REPO=/path/to/felis NA=6 NB=8 PARTITION=debug TIME=01:00:00 NP_VALUE=4 ./submit.sh
```

## Recommended de-risking order

Do NOT jump straight to a multi-day scavenger production run. Validate the
chain cheaply first, in this order:

1. **Inspect the manifest tiling** — confirm the `ilam` index lists tile
   contiguously and **overlap by one** at each boundary
   (e.g. `[0,1,2,3,4] [4,5,6,7,8] [8,…]`: each group's last index equals the
   next group's first). If they look disjoint or gapped, STOP — that means the
   partition is wrong and dG would be biased:
   ```bash
   python gen_work_units.py --abfecfg examples/abfe/abfecfg.yaml \
       --na 6 --nb 8 \
       --workdir "$PWD/examples/abfe/tyk2_example/ejm_31" \
       --sdf-abs "$PWD/examples/abfe/input/ejm_31.sdf" \
       --out work_units.json
   python -c "import json;m=json.load(open('work_units.json'));[print(u['stem'],u['ilam']) for u in m['units']['A']]"
   ```

2. **Validation run on a non-preemptible partition** (`debug`, `l40s`, or
   `general` with a GPU) with the small recipe (`e05`+`v18`, the shipped
   `abfecfg.yaml`) — prove a dG drops out end to end. For tyk2/ejm_31 this
   lands around -5.6 kcal/mol vs experimental -9.54; the ~4 kcal/mol gap is
   expected at this coarse sampling and is not an error — it validates the
   *pipeline*, not the accuracy.

3. **Kill-and-resume test** — `scancel` one array task mid-run; confirm
   `--requeue` brings it back and Felis resumes that group from its
   `.create_done` / intact `.nc` rather than recomputing from scratch. Only
   after this passes should you trust scavenger.

4. **Production** — switch to `--partition=scavenger`, the full `e29`+`v45`
   recipe with longer sampling (`md_*nsnapshots: 2000`), larger `NA`/`NB`
   (e.g. 25/30), `TIME=08:00:00`, and 3 replicas into separate `outdir`s.

## Knobs (env vars, all overridable on the `submit.sh` line)

| var | default | meaning |
|-----|---------|---------|
| `FELIS_REPO` | `$HOME/felis` | repo root on the cluster (override this) |
| `ABFECFG` | `…/examples/abfe/abfecfg.yaml` | the ABFE config |
| `WORKDIR` | `…/tyk2_example/ejm_31` | run dir (has `prepare/ trj/ progress/`) |
| `NA` / `NB` | `6` / `8` | requested group counts = array sizes |
| `PARTITION` | `scavenger` | array (compute) partition |
| `TIME` | `04:00:00` | per-task wall (must exceed one group's runtime) |
| `NP_VALUE` | `2` | MPI ranks per group over MPS |
| `CPUS` / `MEM` | `8` / `64G` | per-task CPUs and memory |
| `FIN_PARTITION` | `general` | finalize (CPU-only) partition |

### Sizing `NA`/`NB`, `NP_VALUE`, and `TIME`

The MPS context assertion is `NP_VALUE * windows_in_group <= 48`. With small
groups (3–5 windows) this is satisfied even at `NP_VALUE=4`. More groups → more
concurrency and shorter (more restartable) tasks, but more *overlap waste*
(each shared boundary state is simulated in two groups). ~3–5 windows/group is a
reasonable balance.

`TIME` must exceed one group's runtime. A complex (`sysB`) group of ~4 windows
at 2000 snapshots takes several hours on an L40S — so for production use small
groups *and* a generous wall (`TIME=08:00:00`). Measure your own per-iteration
rate from a validation log (`grep "Iteration took" slurm_logs/*.out`) and
multiply by `nsnapshots * windows_in_group` to size it.

`gen_work_units.py` may emit **fewer** groups than requested for short ladders;
`submit.sh` reads the actual counts back, so array ranges and the recorded
`n_cuda_devices` always match.

## Cluster / environment notes

These came up bringing the backend up on a real cluster; all are baked into the
scripts but worth knowing if you adapt them:

- **GPU request flag.** The scripts use `--gpus=1`. Some schedulers want
  `--gres=gpu:1` instead — change the `#SBATCH` line in `felis_array.sbatch`
  and the `ARRAY_OPTS` line in `submit.sh` together if a submission is rejected
  for GPU resources.
- **OpenMPI ≥ 5** enforces slot accounting and refuses to place `NP_VALUE`
  ranks on one node (Felis intentionally oversubscribes one GPU via MPS). The
  scripts export `PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe` to allow
  it (harmless on OpenMPI 4).
- **Per-task memory.** The default per-CPU memory on many partitions is tiny
  (a few GB); `NP_VALUE` ranks each loading OpenMM + the system will OOM under
  it. The scripts request `--mem=64G`; keep it explicit.
- **conda + `set -u`.** Some conda deactivate hooks (e.g. gromacs) reference
  unset vars and abort under `set -u`; activation is wrapped in `set +u` / `set -u`.
- **System vs conda python in MPI ranks.** Bare `python3` inside an `mpirun`
  rank can resolve to the system interpreter (no `felis`). The array script
  invokes `$CONDA_PREFIX/bin/python3` explicitly and forwards `PATH` with `-x`.
- **Relative `outdir`.** `abfecfg.yaml`'s `outdir` is relative; the finalize job
  runs the app from the config's directory (not the ligand workdir) so it
  resolves to the same path the prep run used, rather than doubling.

## Caveats

- Group **overlap states are simulated redundantly** — the price of the tiling
  design. Finer granularity trades GPU-hours for restartability.
- If you set `tmpdir` in the config, point both `tmpdir` and `outdir` at
  **shared** storage (not node-local scratch that vanishes on preemption). The
  default minimal config leaves `tmpdir` unset and operates in place.
- These scripts wrap Felis from outside; they are not part of the importable
  `felis` package.
