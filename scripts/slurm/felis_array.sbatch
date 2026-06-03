#!/bin/bash -l
# Slurm array job: one Felis repex GROUP per array task, on ONE GPU via MPS.
# Preemption-tolerant (scavenger): --requeue + Felis's own .create_done /
# from_storage resume means a killed task picks up its group where it left off.
#
# Submit TWICE -- once per leg -- with the right --array range and LEG env:
#   NA=$(jq -r .na work_units.json); NB=$(jq -r .nb work_units.json)
#   sbatch --array=0-$((NA-1)) --export=ALL,LEG=A,MANIFEST=$PWD/work_units.json felis_array.sbatch
#   sbatch --array=0-$((NB-1)) --export=ALL,LEG=B,MANIFEST=$PWD/work_units.json felis_array.sbatch
#
# NOTE the deliberate choices below:
#   --gpus=1              one GPU per task; MPS pins to it as local device 0
#                         (easley wanted --gpus; some clusters need --gres=gpu:1)
#   --requeue             scavenger preemption -> automatic requeue
#   NP_VALUE              MPI ranks sharing the one GPU via MPS (np). For
#                         scavenger keep this modest (2-4). The MPS_N_CONTEXT
#                         assert is np * windows_in_THIS_group <= 48; with small
#                         groups that's easily satisfied even at np=4.

#SBATCH --job-name=felis_abfe
#SBATCH --partition=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
# GPU request: easley's scheduler wanted --gpus=1 (not --gres=gpu:1). If a
# submission is rejected for GPU resources on your cluster, swap this for
# `#SBATCH --gres=gpu:1` (and the matching line in submit.sh's ARRAY_OPTS).
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

: "${LEG:?set LEG=A or LEG=B}"
: "${MANIFEST:?set MANIFEST=/abs/path/work_units.json}"
: "${NP_VALUE:=2}"          # MPI ranks per group over MPS; override at submit
: "${FELIS_ENV:=felis}"     # conda env name
: "${CONDA_SH:=$HOME/miniconda3/etc/profile.d/conda.sh}"

mkdir -p slurm_logs

# --- environment ---------------------------------------------------------
# Batch shells can start with a narrower PATH than interactive shells; widen it
# so nvidia-cuda-mps-control, mpirun, etc. are findable. ($HOME/app, $HOME/bin
# are easley-specific user dirs -- harmless to keep elsewhere, they're just
# skipped if absent.)
export PATH="$HOME/app:$HOME/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

# Activate conda. set +u guards against conda hooks (gromacs deactivate, CUDA
# nvcc flags) that reference unset vars and would trip `set -u`.
set +u
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
source "$CONDA_SH"
conda activate "$FELIS_ENV"
set -u

# OpenMPI 5+ enforces slot accounting and refuses to place N ranks on a node
# with fewer "slots" -- but Felis deliberately packs np ranks onto ONE GPU via
# MPS (intentional GPU oversubscription). This tells PRRTE (OMPI 5's runtime)
# to allow it, equivalent to `mpirun --map-by :OVERSUBSCRIBE`.
# REQUIRED on OpenMPI >= 5; harmless on 4.x.
export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe
# Pin per-rank threading so N MPI ranks sharing one GPU don't each spawn many
# CPU threads and thrash. From the upstream run.sh.
export OMP_NUM_THREADS=1
export OPENMM_CPU_THREADS=1
export RDMAV_FORK_SAFE=1

# --- locate this task's work unit ---------------------------------------
WORKDIR=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['workdir'])" "$MANIFEST")
cd "$WORKDIR"

IDX="${SLURM_ARRAY_TASK_ID}"

# Extract the exact repex argv for (LEG, IDX) from the manifest as a NUL-safe
# array, so paths/values with spaces survive.
mapfile -t ARGV < <(python3 - "$MANIFEST" "$LEG" "$IDX" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); leg = sys.argv[2]; idx = int(sys.argv[3])
unit = next(u for u in m["units"][leg] if u["idx"] == idx)
print("\n".join(unit["argv"]))
PY
)

STEM="$(python3 -c "import json,sys;m=json.load(open(sys.argv[1]));leg=sys.argv[2];idx=int(sys.argv[3]);print(next(u for u in m['units'][leg] if u['idx']==idx)['stem'])" "$MANIFEST" "$LEG" "$IDX")"

echo "[felis_array] LEG=$LEG IDX=$IDX STEM=$STEM NP_VALUE=$NP_VALUE host=$(hostname)"
echo "[felis_array] argv: ${ARGV[*]}"

# --- per-node MPS on the single allocated GPU ----------------------------
# Slurm exposes exactly one GPU; CUDA sees it as local index 0. MPS pipe/log
# dirs are made unique per job to avoid any cross-job clash on shared /tmp.
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mpslog_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cleanup() {
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" 2>/dev/null || true
}
trap cleanup EXIT
cleanup            # clear any stale state from a previous (preempted) attempt
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

# Start MPS daemon WITHOUT a device filter, then pin the run to the GPU.
unset CUDA_VISIBLE_DEVICES
nvidia-cuda-mps-control -d 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=0

# --- run the group's replica-exchange via mpirun -------------------------
# Felis resume: if trj/<stem>.create_done exists and the .nc is intact,
# ReplicaExchangeSampler.from_storage resumes; on a corrupt-NetCDF (hard kill)
# it deletes and recreates that group. Either way the task is idempotent.
#
# IMPORTANT: the manifest's argv starts with a bare "python3", which inside an
# mpirun rank can resolve to the SYSTEM python (/usr/bin/python3) rather than
# the conda env's -- the ranks don't reliably inherit the activated PATH. So we
# substitute the explicit conda python ($CONDA_PREFIX/bin/python3) for the
# leading element, and forward PATH + CUDA env to every rank with -x.
PYBIN="$CONDA_PREFIX/bin/python3"
mpirun --allow-run-as-root -np "$NP_VALUE" \
    -x NP_VALUE="$NP_VALUE" -x PATH -x CUDA_VISIBLE_DEVICES \
    -x CUDA_MPS_PIPE_DIRECTORY -x CUDA_MPS_LOG_DIRECTORY \
    "$PYBIN" "${ARGV[@]:1}"

echo "[felis_array] LEG=$LEG IDX=$IDX STEM=$STEM completed"
