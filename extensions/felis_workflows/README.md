# Portable FELIS workflows

This extension separates molecular inputs, force fields, sampling protocols and
execution sites. It calls the pinned FELIS engine without editing its files.
The upstream revision is `4d2556bfe09753ff63ddf549c3838a517f173b12`.

| Location | Responsibility |
| --- | --- |
| `campaigns/` | Receptor chemistry, bound ligand poses, binary/conditional calculations |
| `forcefields/` | GAFF2/AM1-BCC or OpenFF Sage 2.3.0/AshGC |
| `protocols/` | Lambda recipes, sampling lengths, replicates and solvent sharing |
| `sites/` | Local/Slurm execution, environments, partitions, GPUs, memory, MPI and MPS |
| `environments/` | Separate simulation, receptor/GAFF2 and Sage environments |
| `src/felis_workflows/` | Preparation, planning, launchers, validation and analysis |
| `tests/` | Scientific invariants, scheduler behavior and CPU integration checks |
| `legacy/` | Original supplied source archives, retained for provenance |

Changing site settings does not change a planned campaign's scientific identity.
Changing a force field, pose, receptor, lambda recipe or sampling length requires
a **new run directory**. Runs belong on persistent shared storage, outside the
source checkout. A finished plan can move before system initialization; native
FELIS preparation/checkpoints subsequently require the same absolute mount path.

## Install and configure a site

Use Python 3.11 for the scientific environments. An existing working FELIS
environment can be used. For a fresh installation, the YAML files are starting
specifications; follow the upstream README for its dependencies, including the
ProLIF patch. The simulation environment needs GROMACS, CUDA OpenMM, OpenMMTools,
OpenMPI and mpi4py. CPU preparation requires AmberTools/ACPYPE/Open Babel or
OpenFF, according to the selected model.

```bash
export FELIS_REPO=/absolute/path/to/felis
cd "$FELIS_REPO"
conda env create -f extensions/felis_workflows/environments/simulation.yml
conda env create -f extensions/felis_workflows/environments/receptor-gaff2.yml
conda env create -f extensions/felis_workflows/environments/sage.yml
conda run -n felis python -m pip install -e .
conda run -n felis python -m pip install -e extensions/felis_workflows
conda activate felis
felis-workflow verify-upstream
```

The CLI itself needs only Python and PyYAML. Planning and dry runs do not require
GPUs or the molecular toolchains. Worker scripts put this checkout first on
`PYTHONPATH`, then select the appropriate interpreter through the site profile.
Each worker environment needs PyYAML. Installing the extension into every
environment is optional when using the generated scripts.

Copy a site YAML to your own configuration directory and edit it. Easley's
starting values come from the existing scripts. Hopper and generic Slurm have
partition placeholders that intentionally fail validation until filled in.
Use `sinfo` and your allocation's documented GPU/account policy to choose them.
The backend supports both `--gpus=1` and `--gres=gpu:1` (including GPU types).
No cluster access or partition entitlement is assumed.

Set `repo`, environment command prefixes, account/partitions and walltimes.
`bootstrap` may name a shell file that loads modules or initializes Conda.
Python runners are argv lists, for example `["/opt/envs/felis/bin/python"]` or
`[conda, run, --no-capture-output, -n, felis, python]`. These names and paths occur
only in site settings. The parent shell needs the Slurm client commands in PATH.

```bash
felis-workflow doctor --site /path/to/my-easley.yaml
```

Each simulation task uses **one full NVIDIA GPU**. The launcher retains Slurm's
GPU visibility, resolves its CUDA UUID, and optionally creates a private MPS
server. It never clears the scheduler GPU mask or stops another job's MPS
server. MIG is not qualified. Preparation uses four MPI ranks; the array rank
count is a site setting, bounded by the pinned engine's 48-context limit.
OpenMPI is the supported MPI launcher; an arbitrary scheduler/MPI replacement
is not inferred from the hostname.

## Run an example

Use `campaigns/tyk2.yaml` for the provided TYK2/ejm_31 input files. Materialize
any required Git LFS inputs first. The input topologies are copied together with
their full include graph; missing includes and LFS pointers are rejected.

```bash
wf="$FELIS_REPO/extensions/felis_workflows"
run=/shared/project/runs/tyk2-gaff2-validation
felis-workflow plan --campaign "$wf/campaigns/tyk2.yaml" \
  --forcefield "$wf/forcefields/gaff2.yaml" \
  --protocol "$wf/protocols/validation.yaml" --output "$run"
felis-workflow submit --run "$run" --site /path/to/my-easley.yaml --dry-run
felis-workflow prepare --run "$run" --site /path/to/my-easley.yaml
felis-workflow submit --run "$run" --site /path/to/my-easley.yaml
felis-workflow status --run "$run"
felis-workflow analyze --run "$run"
```

`prepare` runs CPU receptor/ligand preparation in the calling allocation, one
ligand at a time. Use a CPU allocation if your site's login-node policy requires
one. Inspect `receptor/audit.json`, ligand `parameters.json`/`validation.json` and
the preparation logs before submission. `submit` schedules GPU system assembly
and equilibration, solvent/complex arrays, then CPU analysis with dependencies.
Every bound ligand and fully interacting partner is checked after full-system
assembly. GROMACS preprocessing must succeed without `-maxwarn`.

`--dry-run` writes inspectable shell scripts and submission arguments under
`executions/` and submits nothing; it can run before molecular preparation.
The local backend uses the same workers, waits for completion, and limits array
concurrency to `local.gpus`. Choose `sites/local-gpu.yaml` on a standalone server.

Select Sage by changing only the force-field argument to
`forcefields/sage-2.3.0.yaml` and use a new run directory. This preserves the
supplied Sage workflow's **AshGC** charge model; it does not substitute GAFF2
charges. The bundled OFFXML and its declared model hash are checked.

| Profile | Sampling per lambda state, each leg | Replicates | Role |
| --- | --- | --- | --- |
| `smoke.yaml` | 0.1 ns; coarse e05/v18 ladder | 1 | Pipeline check |
| `validation.yaml` | 1 ns; full e29/v45 ladder | 1 | Topology, overlap and runtime validation |
| `production.yaml` | 10 ns; full e29/v45 ladder | 3 | Starting production protocol |

All use 3 ns of Boresch preparation, r02 restraints, 298.15 K and 2 fs steps.
Durations are per state, not total GPU runtime; overlapping boundary states
are simulated in adjacent groups. For longer production, copy the protocol and
increase `solvent_ns`/`complex_ns` (e.g. 50 ns), choose a new name/seed, and plan a
new run. Set site walltimes from measured validation performance. A duration
label does not establish convergence.

## Prepare a new receptor and ligands

Copy `campaigns/new-protein.yaml`. Supply a complete receptor PDB and bound
ligand SDFs in the **same coordinate frame**. The workflow does not dock ligands,
build missing loops, choose a pH, select biological assemblies, or trim domains.

Declare chain/segment count, terminal caps (`charged` or `ace_nme`), expected
disulfides and HETATM handling. Disulfides can be detected with an explicit
expected count or given as pairs of original residue IDs such as `A:57`.
Ambiguous pairing fails. Histidine states come from existing HID/HIE/HIP labels
or Reduce's hydrogens, with explicit `histidine_overrides` available. Other
protonation choices belong in the input residue names and ligand structures.
LEaP applies ff14SB and explicit disulfide bonds; unexpected heavy-atom creation
fails. Caps and residue renumbering are recorded in the receptor audit.

Alternatively provide `gro` and `top` (plus `include_dirs` if needed) for a
receptor whose chemistry has already been prepared. The user is responsible for
that topology's force field, protonation and retained cofactors. This route
checks readability and atom counts but cannot infer chemical correctness from
filenames. The supported science profile assumes compatible ff14SB/AMBER
nonbonded conventions.

Every ligand SDF must contain one connected molecule, explicit hydrogens,
defined stereochemistry, a 3D bound pose and the declared formal charge.
Parameterization preserves the source atom order and pose. Native OpenMM
parameters, exported ITP parameters, energies, forces, bonded terms, exclusions
and hydrogen constraints must agree before the ligand is accepted.
For GAFF2, ACPYPE's integer-charge balancing is reconstructed independently in
the Amber validation reference and recorded, including changed atoms and charge
deltas. Residuals above 0.01 e fail; comparison tolerances are not relaxed to
hide a model difference. The original Amber prmtop is retained.
The final GAFF2 ITP is exported through ParmEd to retain Amber bonded-parameter
precision beyond ACPYPE's GROMACS writer. Exactly zero-amplitude torsions may
be omitted; every nonzero term and the energy/force round trip are checked.

`campaigns/9opz-sucralose.yaml` carries the original campaign's **two chains,
ACE/NME caps and sixteen disulfides**. These are construct-specific assumptions,
not defaults for new proteins. The supplied archives did not contain the actual
prepared receptor or bound ligand poses; fill the template's input paths with
your reviewed structures. The archived generated sucralose test conformer is
not a replacement for a receptor-aligned bound pose.

## Orthosteric ligand–PAM coupling

Use `campaigns/coupling.yaml`, adapting its receptor chemistry, L and P structures,
formal charges and site-distance threshold. Both ligands use the selected ligand
force field. The four calculations are:

| Calculation | Alchemical target | Fully interacting partner |
| --- | --- | --- |
| `L_in_R` | Orthosteric ligand L | None |
| `P_in_R` | PAM P | None |
| `L_in_RP` | L | P |
| `P_in_RL` | P | L |

The partner is a separate cofactor in the complex, excluded from the alchemical
atom mask and absent from the solvent leg. It remains mobile; no unaccounted
partner restraint is introduced. Every stored checkpoint frame and replica is
checked for proximity to receptor atoms surrounding the partner's initial site.
An occupancy failure keeps the raw ABFE but excludes that replicate from
conditional coupling estimates. This diagnostic cannot exclude excursions
between frames or establish equilibration of receptor conformations.

With `reuse_solvent: true`, each target's solvent leg is shared between its binary
and conditional ABFEs **within a replicate**. This gives four complex and two
solvent legs per replicate. Set false to sample four independent solvent legs.

\[
\Delta G_c^{(L)}=G(L\mid RP)-G(L\mid R),\quad
\Delta G_c^{(P)}=G(P\mid RL)-G(P\mid R)
\]

Negative coupling favors co-binding. Closure is
`G(L|R) + G(P|RL) - G(P|R) - G(L|RP)`, equivalently the P route minus the L route.
`analyze` reports both routes and closure, with standard errors from matched
independent replicate differences. It does not sum dependent leg errors as if
independent, or automatically average the two routes. One replicate has no
estimated replicate standard error. This is thermodynamic binding coupling,
not a prediction of receptor signaling efficacy.

Charged targets need a reviewed finite-size/electrostatic correction appropriate
to the box, boundary conditions and alchemical convention. The extension reports
raw charged ABFEs and withholds their coupling interpretation until documented
external corrections are supplied. `analyze --corrections corrections.json`
accepts additive kcal/mol corrections per `calculation/rN`:

```json
{"L_in_R/r1": {"value_kcal_mol": 0.25, "provenance": "method, inputs, script version and report path"}}
```

Do not copy this illustrative number. Correction uncertainty is not propagated.
Review overlap, equilibration, replicate agreement, restraints, occupancy and
cycle closure before interpreting any result.

## Resume and provenance

After failed/preempted jobs have stopped:

```bash
felis-workflow resume --run "$run" --site /path/to/my-easley.yaml --dry-run
felis-workflow resume --run "$run" --site /path/to/my-easley.yaml
```

Resume checks the original site's queue before opening trajectory files, then
checks creation markers, stored iteration targets, replica assignments and
checkpoint coordinates. Only incomplete groups are resubmitted. Waiting
finalizers/dependencies from the old attempt are cancelled after active writers
are excluded. A timeout or `scancel` does not itself guarantee Slurm requeue;
use `resume` after inspecting job state. Corrupt files are not deleted or silently
restarted. Partial receptor preparation requires inspection and moving the
partial directory aside before retrying.

Resume on the same site/backend, with the original run path and compatible
runtime. Portability means a campaign can be started on another configured
site; arbitrary live checkpoint migration across GPU/MPI/runtime versions is
not promised. Requested integrator seeds are recorded, but upstream
OpenMMTools also uses automatic random streams. Bitwise trajectory replay is
not promised across processes, hardware or restarts.

The run records scientific settings and input hashes (`science.json` and its
lock), parameterization provenance, receptor audit, assembled parameter/mask
checks, runtime package/source fingerprints, per-attempt site snapshots and
submitted job IDs. Successful process exit alone never certifies completion.
`status` reads manifests and finalized artifacts, not active NetCDF files.
Native MBAR/restraint outputs remain under each calculation's `work/<target>/analysis/`;
the combined report is `analysis/report.json` and `analysis/abfe.tsv`.

See [migration](docs/migration.md), [validation](docs/validation.md) and
[source attribution](docs/sources.md).
