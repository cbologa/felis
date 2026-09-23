# Installing Felis on a Slurm cluster

Step-by-step install of this Felis fork into a conda environment on an HPC
cluster, validated on an Auburn "easley"–class system (Slurm, conda from a
user-local Miniconda, NVIDIA L40S / H100 nodes, driver supporting CUDA 12.8).

It is written so the failure modes that actually bit during bring-up can't bite
you. The two that cost the most time, called out where they occur:

1. **Do not `conda remove jax`.** Current `openmmtools` (0.26) depends on jax;
   removing jax silently cascades-removes openmmtools, pymbar, mdtraj, and
   pdbfixer. (Older recipes removed jax — that was for an environment where
   openmmtools didn't depend on it. Don't carry it forward.)
2. **Install every conda dependency in ONE transaction, and verify the stack
   imports BEFORE running any `pip`.** If openmmtools is missing when `pip
   install .` runs, pip falls back to PyPI for the whole dependency tree, pulls
   incompatible numpy/pandas, undoes the setuptools pin, and then hard-fails
   because openmmtools isn't on PyPI at all.

Guiding principle: **do package installs on the login node** (it has internet;
compute nodes usually don't) and **verify the OpenMM CUDA platform on a GPU
node** (login nodes have no GPU). Don't run MD on the login node.

---

## 0. Prerequisites — what the cluster provides

On the login node:

```bash
which conda || module avail 2>&1 | grep -iE "conda|miniconda|anaconda"
df -h "$HOME" | tail -1            # need a few GB for the env + repo
```

Two cases for conda:

- **You already have a user Miniconda** (initialized in `~/.bashrc`, so `conda`
  is a shell function). Use it; its `conda.sh` is
  `$HOME/miniconda3/etc/profile.d/conda.sh`. This is the simplest, most stable
  case — the path won't move.
- **Only a `miniconda3`/`anaconda` module exists.** `module load` it. Note that
  a module-provided conda may need `source "$(conda info --base)/etc/profile.d/conda.sh"`
  to be activatable inside a non-login batch shell.

Record the `conda.sh` path — the Slurm scripts read it as `CONDA_SH` (default
`$HOME/miniconda3/etc/profile.d/conda.sh`).

Confirm conda activates in the harshest shell a Slurm task will see (stripped
environment, no `.bashrc`):

```bash
env -i bash --norc -c '
  source '"$HOME"'/miniconda3/etc/profile.d/conda.sh
  conda activate base && echo "ACTIVATE_OK: $(which python)"
'
```

If `ACTIVATE_OK` prints a python under your conda, the batch scripts will
activate the env fine as written.

---

## 1. Clone the fork (recursively)

```bash
cd "$HOME"                         # or any shared path the compute nodes mount
git clone --recursive https://github.com/cbologa/felis.git
cd felis

# confirm the bytemol force-field XMLs AND the slurm scripts are present
ls submodule/bytemol/bytemol/toolkit/protein/residue_reference/   # two .xml files
ls scripts/slurm/                                                 # gen/submit/sbatch/README
```

The `--recursive` matters: `bytemol` is pulled in as a submodule. The two XMLs
under `residue_reference/` (`residue_template_lib.xml`,
`ion_solvent_template_lib.xml`) are the fork's fix for an upstream packaging gap
— without them, `from bytemol.toolkit.protein import parse_pdb` fails at import
and nothing downstream runs. Confirm both are present now.

> **git-lfs:** if the benchmark dataset (`pl_bfe_dataset/`) is LFS-tracked,
> install git-lfs (step 2 does this via conda) then `git lfs pull`. If a needed
> input file is a tiny "pointer" file rather than real content, that's an
> unpulled LFS object.

---

## 2. Create the env and install ALL conda dependencies in ONE transaction

```bash
conda create -n felis python=3.11 -y
conda activate felis

# Single transaction so the solver picks mutually compatible versions
# (notably numpy<2, which openmmtools requires). Match cuda-version to the
# cluster driver's max CUDA (12.8 here -> 12.6 runtime is safely under it;
# conda ships the CUDA runtime, so no system CUDA module is needed).
conda install -n felis -c conda-forge \
    openmm openmmtools gromacs git-lfs cuda-version=12.6 \
    "numpy~=1.26" "pandas~=2.0" mdanalysis scipy matplotlib -y

git lfs install
```

**Do NOT run `conda remove jax`.** jax arrives as an `openmmtools` dependency;
removing it takes openmmtools (and pymbar, mdtraj, pdbfixer, mpiplus) with it.
The jax that comes in is CPU-only and harmless — it does not touch GPU MD.

Pinning `numpy~=1.26` and `pandas~=2.0` here (matching felis's `pyproject.toml`)
means the env already satisfies felis's requirements, which is what makes the
later `pip install --no-deps` safe.

---

## 3. Verify the conda stack BEFORE any pip — this is the gate

```bash
python -c "import openmmtools; print('openmmtools', openmmtools.__version__)"  # ~0.26.0
python -c "import openmm; print('openmm', openmm.__version__)"                 # ~8.4
python -c "import numpy; print('numpy', numpy.__version__)"                    # MUST be 1.26.x
python -c "import pandas; print('pandas', pandas.__version__)"                 # MUST be 2.x
python -c "import pymbar; print('pymbar', pymbar.__version__)"                 # came with openmmtools
```

**Do not proceed if openmmtools fails to import or numpy is 2.x.** If
openmmtools is missing, the conda install didn't take — stop and fix it (do not
let pip try to supply it; it can't). If numpy is 2.x, force it down:
`conda install -n felis -c conda-forge "numpy~=1.26" -y`.

pymbar prints two harmless banners on import (a `timeseries` statistical-
inefficiency note and a `PyMBAR will use 64-bit JAX` notice). Ignore both; they
appear every time pymbar is imported, including during the `mbar` analysis.

---

## 4. Pin setuptools (ProLIF needs `pkg_resources`)

```bash
pip install "setuptools<81"
python -c "import setuptools; print('setuptools', setuptools.__version__)"     # < 81
```

ProLIF imports `pkg_resources`, which setuptools removed in 81+. This pip
downgrade shadows conda's setuptools; that's intended and harmless. (You'll see
a `pkg_resources is deprecated` warning later — that's the expected state, not
an error.)

---

## 5. ProLIF from source + the bundled patch, with `--no-deps`

Cloned outside the felis tree so it doesn't pollute the repo. `--no-deps` so pip
can't drag PyPI numpy/pandas back over the conda versions.

```bash
cd "$HOME"
git clone https://github.com/chemosim-lab/ProLIF.git
cd ProLIF
git checkout v2.0.3
git apply "$HOME/felis/submodule/prolif.patch"     # the patch ships in the fork
pip install . --no-deps
cd "$HOME/felis"
```

If `git apply` reports the patch doesn't apply, confirm you're on the v2.0.3
commit (`git log --oneline -1`); `git apply --check ../felis/submodule/prolif.patch`
explains why without modifying anything.

---

## 6. rdkit — the dependency `--no-deps` skips

ProLIF needs rdkit, but `--no-deps` won't have installed it. Add it via conda:

```bash
conda install -n felis -c conda-forge rdkit -y
python -c "import prolif; print('ProLIF', prolif.__version__)"   # expect 2.0.3
```

If the ProLIF import surfaces another `ModuleNotFoundError`, install that one
package the same way (conda-forge) and retry — that's the expected `--no-deps`
trade-off: pip won't pull pure dependencies, so genuinely-needed ones are added
by hand. rdkit is the main one.

---

## 7. Install Felis itself, with `--no-deps`

```bash
pip install . --no-deps
```

`--no-deps` installs only felis and trusts the env that step 3 verified. Letting
pip resolve felis's pins instead reaches PyPI (wrong numpy/pandas, no
openmmtools) — the exact failure this whole recipe is structured to avoid.

---

## 8. Full verification (login node)

```bash
python -c "import felis; print('felis OK')"
python -c "from bytemol.toolkit.protein import parse_pdb; print('bytemol parse_pdb OK')"
```

The second line is the meaningful one: it's the import that fails on a clean
clone without the reconstructed `residue_reference/*.xml`, and it loads both
template files. `OK` here confirms the fork's fix works on a fresh machine.

> **Durable XML note.** `pip install .` copies bytemol's `.py` files but not its
> `.xml` data files (the upstream packaging doesn't declare them as package
> data). So a plain install can still raise `FileNotFoundError` for
> `residue_template_lib.xml` at runtime even though the files are in the repo.
> Two fixes:
> - **Quick:** copy them into the installed package:
>   ```bash
>   SRC=submodule/bytemol/bytemol/toolkit/protein/residue_reference
>   DST=$(python -c "import bytemol,os;print(os.path.dirname(bytemol.__file__))")/toolkit/protein/residue_reference
>   mkdir -p "$DST" && cp "$SRC"/*.xml "$DST"/
>   ```
> - **Durable:** use an editable install (`pip install -e . --no-deps`) so
>   bytemol imports from the repo tree where the XMLs live, and `git pull`
>   updates take effect without reinstalling.

---

## 9. Verify the OpenMM CUDA platform — on a GPU node

Login nodes have no GPU, so confirm CUDA on a short interactive allocation
(`debug`: short wall, non-preemptible):

```bash
srun --partition=debug --gpus=1 --time=00:15:00 --pty bash
# on the node:
conda activate felis
nvidia-smi                              # note the driver's max CUDA version
python -c "from openmm import Platform; print([Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())])"
# want 'CUDA' in the list, e.g. ['Reference','CPU','CUDA','OpenCL']
exit
```

If `CUDA` is absent, the conda CUDA runtime is newer than the node driver
supports — lower `cuda-version` in step 2 (e.g. to match a `cuda/12.x` the
cluster advertises). The driver's `CUDA Version` in `nvidia-smi` is the
**maximum** runtime it supports; any conda `cuda-version` at or below it works.

This step also confirms the env activates cleanly on a compute node, which is
what the Slurm jobs rely on.

---

## 10. Prepare the example config and inputs

The `examples/abfe` example populates its `input/` from the benchmark dataset
(this is what `examples/abfe/run.sh` does). Copy the four files for tyk2/ejm_31:

```bash
cd examples/abfe
tyk2=../../pl_bfe_dataset/Schrodinger/jacs/tyk2
mkdir -p input
cp "$tyk2/protein_ff14sb/protein_w_cofactors.gro" \
   "$tyk2/protein_ff14sb/protein_w_cofactors.top" \
   "$tyk2/ligands/ejm_31.sdf" \
   "$tyk2/ligands_itps_joint-25/ejm_31.itp"  input/
ls input/        # expect all four: .gro .top .sdf .itp
```

The shipped `abfecfg.yaml` is the fast **validation** config (coarse recipe,
~2 ns/window, single replica). Confirm it parses with non-`None` integer fields
(a `None` snapshot or checkpoint count serializes as the string `"None"` and
crashes the samplers):

```bash
cd "$HOME/felis"
python -c "
from felis.protocols.abfe.config_types import ABFEInputConfig
c = ABFEInputConfig.from_file('examples/abfe/abfecfg.yaml')
print('recipes:', c.elamrecipe, c.vlamrecipe, c.reslamrecipe)
print('snapshots:', c.md_sol_nsnapshots, c.md_pro_nsnapshots, c.md_nsnapshots)
print('checkpoint:', c.md_checkpoint_interval)
"
```

Expect real integers (e.g. `400 / 400 / 400`, checkpoint `50`) — not `None`.

---

## 11. Run it

The calculation is two phases — system prep (once, via the normal app) then the
alchemical legs (via the Slurm array backend). See **scripts/slurm/README.md**
for the prep command, the array submission, and the de-risking sequence
(inspect the lambda tiling, validate on a non-preemptible partition, test
kill-and-resume, then go to scavenger for production).

---

## Appendix — environment gotchas baked into the Slurm scripts

These surfaced during bring-up and are already handled in the `scripts/slurm`
files; listed here so they're not a mystery if you adapt them or hit them
running prep by hand.

- **conda + `set -u`.** Some conda deactivate hooks (gromacs) reference unset
  variables and abort under `set -u`. Wrap activation in `set +u` / `set -u`
  (and export empty `NVCC_PREPEND_FLAGS`/`NVCC_APPEND_FLAGS`).
- **No MPI library.** `mpi4py` is only Python bindings; it needs a real
  `libmpi.so`. If the env was built with `nompi` openmm/gromacs, install one:
  `conda install -n felis -c conda-forge openmpi mpi4py -y`.
- **OpenMPI ≥ 5 slot accounting.** OMPI 5 refuses to place N ranks on one node
  when Felis intentionally packs N ranks onto one GPU via MPS. Export
  `PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe` (harmless on OMPI 4).
- **Per-task memory.** The default per-CPU memory on many partitions is only a
  few GB; N MPI ranks each loading OpenMM + JAX will OOM (SIGKILL) under it.
  Request `--mem=64G --cpus-per-task=8` explicitly.
- **System vs conda python in MPI ranks.** A bare `python3` inside an `mpirun`
  rank can resolve to `/usr/bin/python3` (no felis). Invoke
  `$CONDA_PREFIX/bin/python3` explicitly and forward PATH with `mpirun -x PATH`.
- **GPU request flag.** easley wanted `--gpus=1`; some schedulers want
  `--gres=gpu:1`. Swap if a submission is rejected for GPU resources.

## Versions this was validated against

openmm 8.4, openmmtools 0.26.0, pymbar 4.2.0, prolif 2.0.3, rdkit (conda-forge),
numpy 1.26.4, pandas 2.3.x, gromacs 2026.0, openmpi 5.x, python 3.11, CUDA
runtime 12.6 under a 12.8-capable driver.
