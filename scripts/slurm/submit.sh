#!/bin/bash
# One-shot driver for the Felis ABFE Slurm pipeline.
#   1. generate work-unit manifest (reuses Felis's own partitioning)
#   2. submit sysA array  (one repex group per task, one GPU each)
#   3. submit sysB array
#   4. submit finalize (afterok on BOTH arrays) -> mbar + summarize + dG
#
# Array ranges are read back from the manifest's ACTUAL group counts, so they
# always match what split_replica_exchange_jobs emitted (which can be fewer
# than requested for small ladders). Those same counts are what the finalize
# job records in the .done files for the mbar stage.
#
# Usage (edit the CONFIG block, then just run):
#   ./submit.sh
#
# Override anything inline, e.g. a quick non-preemptible validation run:
#   NA=4 NB=4 PARTITION=debug TIME=01:00:00 NP_VALUE=4 ./submit.sh

set -euo pipefail

# ============================ CONFIG ====================================
# Absolute paths -- Slurm tasks start in a fresh shell with no inherited CWD.
FELIS_REPO="${FELIS_REPO:-$HOME/felis}"                  # repo root on easley
ABFECFG="${ABFECFG:-$FELIS_REPO/examples/abfe/abfecfg.yaml}"
SDF_ABS="${SDF_ABS:-$FELIS_REPO/examples/abfe/input/ejm_31.sdf}"
WORKDIR="${WORKDIR:-$FELIS_REPO/examples/abfe/tyk2_example/ejm_31}"
MANIFEST="${MANIFEST:-$PWD/work_units.json}"

# Group counts (= array sizes = recorded n_cuda_devices). Pick for preemption
# granularity: more groups -> shorter, more-restartable tasks, more overlap
# waste. ~3-5 windows/group is a good balance. For the e05+v18 validation
# recipe (sysA=22, sysB=29 windows) NA=6/NB=8 gives ~4-window groups.
NA="${NA:-6}"
NB="${NB:-8}"

# Slurm knobs for the ARRAY (compute) jobs.
PARTITION="${PARTITION:-scavenger}"     # scavenger for production; debug to test
TIME="${TIME:-04:00:00}"                # per-task wall; must exceed one group
NP_VALUE="${NP_VALUE:-2}"               # MPI ranks/group over MPS (assert: np*group_windows<=48)
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
# GPU request flag: easley wanted --gpus=1. If your cluster rejects it, change
# the ARRAY_OPTS line below (and felis_array.sbatch's #SBATCH) to --gres=gpu:1.

# Finalize job partition (CPU-only, short).
FIN_PARTITION="${FIN_PARTITION:-general}"

# Conda
FELIS_ENV="${FELIS_ENV:-felis}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARRAY_SBATCH="${ARRAY_SBATCH:-$SCRIPT_DIR/felis_array.sbatch}"
FINALIZE_SBATCH="${FINALIZE_SBATCH:-$SCRIPT_DIR/felis_finalize.sbatch}"
GEN="${GEN:-$SCRIPT_DIR/gen_work_units.py}"
# ========================================================================

mkdir -p slurm_logs

echo "==> activating conda env '$FELIS_ENV' for manifest generation"
set +u
source "$CONDA_SH"
conda activate "$FELIS_ENV"
set -u

echo "==> generating work-unit manifest"
python3 "$GEN" \
    --abfecfg "$ABFECFG" \
    --na "$NA" --nb "$NB" \
    --workdir "$WORKDIR" \
    --sdf-abs "$SDF_ABS" \
    --out "$MANIFEST"

# Read back the ACTUAL group counts the generator emitted.
NA_ACT=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['na'])" "$MANIFEST")
NB_ACT=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['nb'])" "$MANIFEST")
echo "==> actual groups: sysA=$NA_ACT  sysB=$NB_ACT"

if [ "$NA_ACT" -lt 1 ] || [ "$NB_ACT" -lt 1 ]; then
    echo "ERROR: zero groups emitted -- check recipe/window counts" >&2
    exit 1
fi

COMMON_EXPORT="ALL,MANIFEST=$MANIFEST,NP_VALUE=$NP_VALUE,FELIS_ENV=$FELIS_ENV,CONDA_SH=$CONDA_SH"
ARRAY_OPTS=(--partition="$PARTITION" --time="$TIME" --gpus=1
            --cpus-per-task="$CPUS" --mem="$MEM")

echo "==> submitting sysA array (0-$((NA_ACT-1)))"
AID=$(sbatch --parsable "${ARRAY_OPTS[@]}" \
        --array=0-$((NA_ACT-1)) \
        --export="$COMMON_EXPORT,LEG=A" \
        "$ARRAY_SBATCH")
echo "    sysA array job id: $AID"

echo "==> submitting sysB array (0-$((NB_ACT-1)))"
BID=$(sbatch --parsable "${ARRAY_OPTS[@]}" \
        --array=0-$((NB_ACT-1)) \
        --export="$COMMON_EXPORT,LEG=B" \
        "$ARRAY_SBATCH")
echo "    sysB array job id: $BID"

# afterok on the WHOLE arrays: dependency on the array job ids waits for every
# task in both arrays to succeed. If any task fails, finalize stays pending and
# can be cancelled; fix the failed task (resubmit just that index) and finalize
# will release once all succeed.
echo "==> submitting finalize (afterok:$AID:$BID)"
FID=$(sbatch --parsable \
        --partition="$FIN_PARTITION" \
        --dependency=afterok:"$AID":"$BID" \
        --export="ALL,MANIFEST=$MANIFEST,ABFECFG=$ABFECFG,FELIS_ENV=$FELIS_ENV,CONDA_SH=$CONDA_SH,FELIS_REPO=$FELIS_REPO" \
        "$FINALIZE_SBATCH")
echo "    finalize job id: $FID"

cat <<EOF

==> submitted.
    sysA array : $AID  (0-$((NA_ACT-1)))
    sysB array : $BID  (0-$((NB_ACT-1)))
    finalize   : $FID  (afterok on both)

  watch:    squeue -u \$USER
  A logs:   slurm_logs/felis_abfe_${AID}_*.out
  finalize: slurm_logs/felis_finalize_${FID}.out
  result:   $WORKDIR/analysis/   (dG tables after finalize)

  If a single array task is preempted and NOT auto-requeued, resubmit just it:
    sbatch ${ARRAY_OPTS[*]} --array=<idx> --export="$COMMON_EXPORT,LEG=A" $ARRAY_SBATCH
EOF
