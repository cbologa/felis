#!/bin/bash -l
# Finalize job: runs AFTER both arrays complete (afterok). Writes the .done
# files the unmodified mbar stage needs, then runs mbar + summarize via Felis's
# own stage code, then compares the resulting dG to the experimental value.
#
# Submit with a dependency on BOTH array job IDs:
#   AID=$(sbatch --parsable --array=0-$((NA-1)) --export=ALL,LEG=A,MANIFEST=$PWD/work_units.json felis_array.sbatch)
#   BID=$(sbatch --parsable --array=0-$((NB-1)) --export=ALL,LEG=B,MANIFEST=$PWD/work_units.json felis_array.sbatch)
#   sbatch --dependency=afterok:${AID}:${BID} --export=ALL,MANIFEST=$PWD/work_units.json,ABFECFG=$PWD/examples/abfe/abfecfg.yaml felis_finalize.sbatch

#SBATCH --job-name=felis_finalize
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

: "${MANIFEST:?set MANIFEST=/abs/path/work_units.json}"
: "${ABFECFG:?set ABFECFG=/abs/path/abfecfg.yaml}"
: "${FELIS_ENV:=felis}"
: "${CONDA_SH:=$HOME/miniconda3/etc/profile.d/conda.sh}"
# Repo root, used for the (optional) experimental-reference comparison.
# Defaults to two levels up from this script (scripts/slurm/ -> repo root).
: "${FELIS_REPO:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

export PATH="$HOME/app:$HOME/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

set +u
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
source "$CONDA_SH"
conda activate "$FELIS_ENV"
set -u

# Same OpenMPI-5 / threading env as the array tasks: the felis app imports the
# MPI stack (mpiplus/mpi4py) at startup even for the CPU-only mbar+summarize.
export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe
export OMP_NUM_THREADS=1
export OPENMM_CPU_THREADS=1
export RDMAV_FORK_SAFE=1

WORKDIR=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['workdir'])" "$MANIFEST")
NA=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['na'])" "$MANIFEST")
NB=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['nb'])" "$MANIFEST")
cd "$WORKDIR"

echo "[finalize] workdir=$WORKDIR  na=$NA  nb=$NB"

# --- 1. Verify every group's .nc exists before analysis ------------------
missing=0
for i in $(seq 0 $((NA-1))); do [ -f "trj/a${i}.nc" ] || { echo "MISSING trj/a${i}.nc"; missing=1; }; done
for i in $(seq 0 $((NB-1))); do [ -f "trj/b${i}.nc" ] || { echo "MISSING trj/b${i}.nc"; missing=1; }; done
[ "$missing" -eq 0 ] || { echo "[finalize] aborting: missing .nc files"; exit 1; }

# --- 2. Write the .done files the mbar stage reads back ------------------
# stage_post_analysis.py does: load_config(open("sysA.done"))["n_cuda_devices"]
# and requires exactly a0..a{NA-1}.nc. We record the ACTUAL group counts.
mkdir -p progress
python3 - "$NA" "$NB" <<'PY'
import json, sys
na, nb = int(sys.argv[1]), int(sys.argv[2])
# Match felis dump_config (compact JSON). Keys must match what mbar reads.
with open("progress/sysA.done", "w") as f:
    json.dump({"n_cuda_devices": na}, f)
with open("progress/sysB.done", "w") as f:
    json.dump({"n_cuda_devices": nb}, f)
# sysA_em was already done in the original run; recreate its flag if absent so
# the orchestrator (if ever re-run) doesn't redo it. Harmless if present.
import os
for flag in ("makebox", "boresch_em", "boresch_npt", "boresch_post_process", "sysA_em", "sysB_em"):
    p = f"progress/{flag}.done"
    if not os.path.exists(p):
        open(p, "w").close()
print("wrote progress/sysA.done, sysB.done (+ stage flags)")
PY

# --- 3. Run mbar + summarize via Felis's OWN stage code ------------------
# We invoke the abfe app with stages restricted to mbar+summarize. Because the
# .done flags for all earlier stages exist, planned_stages filters them out and
# only mbar+summarize run -- consuming the per-group .nc files we produced.
LIG=$(python3 -c "import json,sys;from pathlib import Path;c=__import__('yaml').safe_load(open(sys.argv[1]));print(Path(c['sdffile']).stem)" "$ABFECFG")
echo "[finalize] ligand=$LIG ; running mbar + summarize"

# CRITICAL: abfecfg.yaml's outdir is RELATIVE (e.g. "tyk2_example"), so the app
# must run from the same directory the prep run used -- the dir CONTAINING the
# abfecfg -- otherwise outdir resolves under WORKDIR and the path doubles to
# .../tyk2_example/ejm_31/tyk2_example/ejm_31/, where no .nc files exist.
# We verified files / wrote .done above using WORKDIR-relative paths; now cd to
# the config's directory for the app call so outdir resolves as in prep.
ABFECFG_DIR=$(cd "$(dirname "$ABFECFG")" && pwd)
cd "$ABFECFG_DIR"
echo "[finalize] running app from $ABFECFG_DIR (so relative outdir resolves)"
"$CONDA_PREFIX/bin/python3" -m felis.app.abfe --abfecfg "$ABFECFG" --stages mbar summarize

# return to WORKDIR for the result reporting below
cd "$WORKDIR"

# --- 4. Report dG and compare to experiment ------------------------------
echo "[finalize] analysis outputs:"
ls -1 analysis/ 2>/dev/null || true
echo
echo "[finalize] summarized free energy:"
cat analysis/*summary* 2>/dev/null || cat analysis/*fe_table* 2>/dev/null || true

# tyk2 ejm_31 experimental reference, if the benchmark tree is present:
EXP="$FELIS_REPO/pl_bfe_dataset/Schrodinger/jacs/tyk2/exp.txt"
if [ -f "$EXP" ]; then
    echo
    echo "[finalize] experimental reference (tyk2/exp.txt), grep ejm_31:"
    grep -i "ejm_31" "$EXP" || true
fi
echo "[finalize] done"
