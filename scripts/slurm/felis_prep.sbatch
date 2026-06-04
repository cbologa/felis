#!/bin/bash -l
# Phase 1 (prep) for the Felis ABFE Slurm pipeline: builds the system and runs
# equilibration via the normal app on ONE GPU, writing prepare/ and the
# progress/*.done flags that the Phase-2 alchemical-leg arrays depend on.
#
# Run this ONCE per replica/run, before submitting the arrays. Non-preemptible
# partition recommended (debug/l40s/general-with-GPU) since it's a single
# ~30-min job and restarting it is cheap but pointless to risk.
#
# Submit:
#   sbatch --partition=debug --gpus=1 \
#       --export=ALL,ABFECFG=/abs/path/abfecfg.yaml,RUNDIR=/abs/path/rundir \
#       felis_prep.sbatch
#
# ABFECFG : absolute path to the run's abfecfg.yaml
# RUNDIR  : the directory CONTAINING abfecfg.yaml and input/ (the app is run
#           from here so the config's relative progro/sdffile/etc. resolve).
#           Defaults to the directory of ABFECFG.

#SBATCH --job-name=felis_prep
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

: "${ABFECFG:?set ABFECFG=/abs/path/abfecfg.yaml}"
: "${FELIS_ENV:=felis}"
: "${CONDA_SH:=$HOME/miniconda3/etc/profile.d/conda.sh}"
# Directory to run the app from (where the config's relative paths resolve).
: "${RUNDIR:=$(cd "$(dirname "$ABFECFG")" && pwd)}"

mkdir -p slurm_logs

# --- environment (same fixes as the array job) --------------------------
export PATH="$HOME/app:$HOME/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

set +u
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
source "$CONDA_SH"
conda activate "$FELIS_ENV"
set -u

export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe
export OMP_NUM_THREADS=1
export OPENMM_CPU_THREADS=1
export RDMAV_FORK_SAFE=1
export CUDA_VISIBLE_DEVICES=0

# Keep Python's mkdtemp temp files (gromacs scratch, tpr/itp writes) on the
# same filesystem as the run dir, so gmx rename()s don't cross filesystems.
# NOTE: this does NOT control Felis's own tmpdir -- that MUST be handled by
# leaving `tmpdir` UNSET in abfecfg.yaml, so Felis sets tmpdir = outdir (see
# the README gotcha: a *set* tmpdir is discarded for a home mkdtemp, which
# breaks gmx genion when outdir is on a different filesystem than home).
export TMPDIR="${TMPDIR_BASE:-$RUNDIR/.tmp}"
mkdir -p "$TMPDIR"

cd "$RUNDIR"

echo "[felis_prep] host=$(hostname) rundir=$RUNDIR"
echo "[felis_prep] abfecfg=$ABFECFG"
echo "[felis_prep] TMPDIR=$TMPDIR"

# Guard: refuse to run if abfecfg sets a tmpdir (the inverted-logic bug makes a
# set tmpdir actively harmful -- it gets replaced by an unwritable home path).
if grep -qE '^[[:space:]]*tmpdir:' "$ABFECFG"; then
    echo "[felis_prep] ERROR: '$ABFECFG' sets 'tmpdir'. Remove that line --" >&2
    echo "  Felis discards a user-set tmpdir for a home mkdtemp, which breaks"  >&2
    echo "  gmx genion when outdir is on scratch. Leave tmpdir unset so Felis"  >&2
    echo "  sets tmpdir = outdir (your scratch path)."                          >&2
    exit 1
fi

# --- run the prep stages (build + equilibrate, no alchemical legs) -------
python -m felis.app.abfe --abfecfg "$ABFECFG" \
    --stages makebox boresch_em boresch_npt boresch_post_process sysA_em sysB_em

echo "[felis_prep] done. progress flags:"
OUTDIR=$(python3 -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['outdir'])" "$ABFECFG")
LIG=$(python3 -c "import sys,yaml;from pathlib import Path;print(Path(yaml.safe_load(open(sys.argv[1]))['sdffile']).stem)" "$ABFECFG")
ls -1 "$OUTDIR/$LIG/progress/" 2>/dev/null || true
