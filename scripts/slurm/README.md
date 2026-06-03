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
hand-roll a disjoint split — the overlap is load-bearing for a correct ΔG.

## Files

| file | role |
|------|------|
| `gen_work_units.py` | Rebuilds the sysA/sysB lambda ladders the same way the stages do, partitions them with `split_replica_exchange_jobs`, and writes `work_units.json` (one `repex` command per group) plus the `sysA_lam.json`/`sysB_lam.json` the commands reference. |
| `felis_array.sbatch` | One array task = one group, on one GPU via MPS. `--requeue`-safe; resumes via Felis's own `.create_done` / `from_storage` logic. Submitted once per leg (A, B). |
| `felis_finalize.sbatch` | Runs `afterok` on both arrays: verifies all `.nc` exist, writes the `.done` files with the correct group counts, runs `mbar` + `summarize` (unmodified), prints the ΔG and the tyk2 experimental reference. |
| `submit.sh` | Driver: generates the manifest, reads back the *actual* group counts, submits both arrays + finalize with the right dependency chain. |

## Quick start

Edit the CONFIG block at the top of `submit.sh` (especially `FELIS_REPO` and
`CONDA_SH` for your cluster), then:

```bash
cd scripts/slurm
./submit.sh
```

Everything is environment-overridable. For a fast, non-preemptible end-to-end
smoke test on a short partition:

```bash
NA=4 NB=4 PARTITION=debug TIME=01:00:00 NP_VALUE=4 ./submit.sh
```

## Recommended de-risking order

Do NOT jump straight to a multi-day scavenger production run. Validate the
chain cheaply first, in this order:

1. **Stage-parse check** — proves `--stages mbar summarize` parses (it will
   fail fast for lack of `.nc`, which is fine):
   ```bash
   python -m felis.app.abfe --abfecfg examples/abfe/abfecfg.yaml --stages mbar summarize 2>&1 | head -30
   ```

2. **Inspect the manifest tiling** — confirm the `ilam` index lists tile
   contiguously and **overlap by one** at each boundary
   (e.g. `[0,1,2] [2,3,4,5] [5,6,…]`). If they look disjoint or gapped, STOP —
   that means the partition is wrong and ΔG would be biased:
   ```bash
   python gen_work_units.py --abfecfg examples/abfe/abfecfg.yaml \
       --na 6 --nb 8 \
       --workdir "$PWD/examples/abfe/tyk2_example/ejm_31" \
       --sdf-abs "$PWD/examples/abfe/input/ejm_31.sdf" \
       --out work_units.json
   python -c "import json;m=json.load(open('work_units.json'));[print(u['stem'],u['ilam']) for u in m['units']['A']]"
   ```

3. **Validation run on `debug`** with a small recipe (e.g. `e05`+`v18`) — prove
   a ΔG drops out and lands near tyk2's experimental ejm_31 value.

4. **Kill-and-resume test** — `scancel` one array task mid-run; confirm
   `--requeue` brings it back and Felis resumes that group rather than
   recomputing it. Only after this passes should you trust scavenger.

5. **Production** — switch to `--partition=scavenger`, the full `e29`+`v45`
   recipe, and larger `NA`/`NB`.

## Knobs (env vars, all overridable on the `submit.sh` line)

| var | default | meaning |
|-----|---------|---------|
| `FELIS_REPO` | `$HOME/felis` | repo root on the cluster |
| `ABFECFG` | `…/examples/abfe/abfecfg.yaml` | the ABFE config |
| `WORKDIR` | `…/tyk2_example/ejm_31` | run dir (has `prepare/ trj/ progress/`) |
| `NA` / `NB` | `6` / `8` | requested group counts = array sizes |
| `PARTITION` | `scavenger` | array (compute) partition |
| `TIME` | `04:00:00` | per-task wall (must exceed one group's runtime) |
| `NP_VALUE` | `2` | MPI ranks per group over MPS |
| `FIN_PARTITION` | `general` | finalize (CPU-only) partition |

### Sizing `NA`/`NB` and `NP_VALUE`

The MPS context assertion is `NP_VALUE * windows_in_group <= 48`. With small
groups (3–5 windows) this is satisfied even at `NP_VALUE=4`. More groups → more
concurrency and shorter (more restartable) tasks, but more *overlap waste*
(each shared boundary state is simulated in two groups). ~3–5 windows/group is a
reasonable balance.

`gen_work_units.py` may emit **fewer** groups than requested for short ladders;
`submit.sh` reads the actual counts back, so array ranges and the recorded
`n_cuda_devices` always match.

## Caveats

- Group **overlap states are simulated redundantly** — the price of the tiling
  design. Finer granularity trades GPU-hours for restartability.
- If you set `tmpdir` in the config, point both `tmpdir` and `outdir` at
  **shared** storage (not node-local scratch that vanishes on preemption). The
  default minimal config leaves `tmpdir` unset and operates in place.
- These scripts wrap Felis from outside; they are not part of the importable
  `felis` package.
