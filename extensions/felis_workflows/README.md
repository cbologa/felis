# Portable FELIS workflows

Derived physical endpoint MD and evidence-only validation qualification are
documented separately in [the endpoint workflow guide](../felis_endpoint/README.md).
They do not change this package's ABFE workflow version or runtime fingerprint.

This extension separates molecular inputs, force fields, sampling protocols and
execution sites. The FELIS base revision is
`4d2556bfe09753ff63ddf549c3838a517f173b12`. Four narrowly reviewed core
files differ from that base; their base and approved Git blob IDs and rationales
are recorded in `upstream.lock.json`. `verify-upstream` checks the exact approved
blobs and requires every other upstream-owned path to match the base revision.

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

Do not edit the pinned core files without updating and reviewing the approved
patch manifest. An edited worktree or staged blob outside the recorded hashes
fails verification. The numeric configuration patch validates explicitly listed
fields at loading; the integrator also retains a defensive float conversion.

The CLI itself needs only Python and PyYAML. Planning and dry runs do not require
GPUs or the molecular toolchains. Worker scripts put this checkout first on
`PYTHONPATH`, then select the appropriate interpreter through the site profile.
Each worker environment needs PyYAML. Installing the extension into every
environment is optional when using the generated scripts.

Set `site.repo` to the absolute checkout used for workers (or use
`${FELIS_REPO}` in the site file). `plan` and `verify-upstream` select a checkout
with `--repo`, `FELIS_REPO`, or a recognizable source/editable installation; an
unidentifiable installed package fails with a request for an explicit checkout.
Runtime `FELIS_REPO` and `--repo` values must be absolute; a relative path in a
site file is resolved relative to that file before reaching the runtime layer.
The CLI must itself import `felis_workflows` from that checkout (use its editable
installation or put its `extensions/felis_workflows/src` on `PYTHONPATH`); an
already loaded implementation from another location fails. Conflicting
selections fail. Generated workers export absolute `FELIS_REPO` and
put `extensions/felis_workflows/src` and the repository root first on
`PYTHONPATH`. Empty and relative inherited entries are removed, so an arbitrary
shell directory cannot select a different `felis`, `felis_workflows`, or bundled
`bytemol`. Pre-imported modules from another checkout fail. The selected
checkout passes `verify-upstream` before new FELIS or bytemol code is imported.
FELIS stages enter their calculation's `work/<target>` directory explicitly,
where native `prepare/...` paths are relative; array processes use that same
directory as their subprocess CWD.

Copy a site YAML to your own configuration directory and edit it. Easley's
starting values come from the existing scripts. Hopper and generic Slurm have
partition placeholders that intentionally fail validation until filled in.
Use `sinfo` and your allocation's documented GPU/account policy to choose them.
The backend supports both `--gpus=1` and `--gres=gpu:1` (including GPU types).
No cluster access or partition entitlement is assumed.

Set `repo`, environment command prefixes, account/partitions and walltimes.
`bootstrap` may name a shell file that loads modules or initializes Conda.
For Slurm, `slurm.partition` selects the A/B arrays, `slurm.analysis_partition`
selects the CPU finalizer, and optional `slurm.prep_partition` selects the
GPU system-preparation job (defaulting to the array partition). `prepare`,
which builds the receptor and ligand on CPU, runs in the allocation from which
you invoke it; the site profile does not request that allocation.
The shipped GPU system-preparation profiles request **48G**; array and analysis
memory settings are separate.
Python runners are argv lists, for example `["/opt/envs/felis/bin/python"]` or
`[conda, run, --no-capture-output, -n, felis, python]`. These names and paths occur
only in site settings. The parent shell needs the Slurm client commands in PATH.

```bash
felis-workflow doctor --site /path/to/my-easley.yaml
```

`doctor` reports the configured checkout, Git revision, Python executable,
Python implementation/version, and resolved FELIS, workflow, and bundled
bytemol source files for the CLI and each configured Python runner. It checks
source identity without allocating a GPU. `runtime.json` also records these
identities alongside the existing package versions and source hashes. On
continuation, stable Python implementation/version, package versions, and
executable source hashes must match preparation. Absolute checkout and Python
paths and Git revision remain diagnostic, so relocation with identical content
does not by itself invalidate a run. Each invocation still validates that
imports originate in its configured checkout. Older runtime snapshots without
source identity cannot be safely continued by this version.

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
the preparation logs before submission. `submit` schedules independent GPU
system assembly and equilibration for each calculation/replica, then its
solvent/complex arrays and CPU analysis with same-replica dependencies. Slurm
arrays have no workflow-imposed `%N` limit, and preparation jobs have no
artificial inter-replica dependencies; Slurm account/QoS controls concurrency.
Every bound ligand and fully interacting partner is checked after full-system
assembly. GROMACS preprocessing must succeed without `-maxwarn`.

## Stage artifacts and restart

The frozen `science.json` and input hashes define the scientific plan. Global
receptor and ligand preparation commits `prepared.json`. Each calculation/replica
commits `assembly.ok.json` after `makebox`, assembly validation and both GROMACS
preprocessing checks. Its preproduction job then runs `boresch_em`,
`boresch_npt`, `boresch_post_process`, `sysA_em` and `sysB_em`, committing
`equil.ok.json` after production starts and configurations validate. This
manifest identifies the exact assembly manifest and deterministic preparation
seed; it does not repeatedly hash the large Boresch NPT trajectory.
Every solvent/complex group has a checkpointed NetCDF trajectory and a small
terminal `completion/<stem>.json` record written only after quiescent semantic
validation. `finalized.json` commits the calculation result and analysis files
after its own and any shared solvent groups validate. A production group
identifies the exact equilibration manifest and its seed. The stage graph
records these dependencies, including same-replica shared solvent ownership.
Replicas never share prepared or equilibrated states.

Terminal manifests use a versioned schema, science/task identity, dependency
identity, producer attempt, stable runtime compatibility, and SHA-256 hashes of
small immutable outputs. They are published atomically once; identical content
may be reused, while conflicting or corrupted records fail. Checkpointed `.nc`
files are never treated as immutable hash-verifiable outputs. Resume uses the
native FELIS/OpenMMTools checkpoint and verifies its stored iteration target,
replica mapping, checkpoint interval and finite coordinates after establishing
that no scheduler job can still write it. An interrupted equilibration reuses
validated assembly without running `makebox` again. Completed equilibration is
retained when only production groups need restarting. Resume selects the
incomplete preproduction chain, groups and finalizers with their dependencies.
A stale completion record or corrupt checkpoint is an
error; it is not silently regenerated. `resume --dry-run` leaves pending jobs
untouched.

`status` reads manifests and job/attempt history without opening NetCDF, so it
remains safe while writers are active. It checks that a recorded trajectory
still exists, without reading or hashing it. Group records shown there are
explicitly *unverified* until a quiescent probe. Shared solvent groups report
their owner's completion records, even while the consuming calculation is still
being prepared. Overall calculation state advances through preparation,
equilibration, sampling, recorded groups, and finalization. Interrupted
equilibration and finalization are reported separately.

Each `executions/<attempt>/` retains an immutable attempt/site snapshot and
the canonical workflow stage graph identity. After the resume probe, its
`task_graph.json` records the exact tasks selected for that attempt, with its
own hash; a resume subset has a different task graph from the full workflow.
Generated script identities, exact submission commands/dependencies, and job
IDs remain inspectable as jobs are successfully submitted. Previous attempts
remain history when resuming. New completion manifests record the worker's
current source diagnostics; checkout paths, interpreter paths, Git revision,
and timestamps do not determine continuation compatibility. Continuation
compares source hashes, package versions, and stable Python identity.

The PR4 stage model is explicitly versioned in `science.json`. Pre-PR4 plans,
including PR3 `prep.ok.json` markers representing the entire old preproduction
chain, cannot be resumed or reclassified; plan a new run directory.
Pre-PR3 terminal markers without this manifest schema cannot certify completed
artifacts. They fail with an actionable legacy-artifact error instead of being
automatically blessed. An unrecorded trajectory can receive a completion record
only after a quiescent semantic validation under a compatible runtime. A partial
FELIS preparation stays partial until its terminal artifact is validated and
committed; initialized systems retain their original absolute run mount path.

`--dry-run` writes inspectable shell scripts and submission arguments under
`executions/` and submits nothing; it can run before molecular preparation.
The local backend uses the same workers, waits for completion, and limits array
concurrency to `local.gpus`. Choose `sites/local-gpu.yaml` on a standalone server.

Select Sage by changing only the force-field argument to
`forcefields/sage-2.3.0.yaml` and use a new run directory. This preserves the
supplied Sage workflow's **AshGC** charge model; it does not substitute GAFF2
charges. The bundled OFFXML and its declared model hash are checked.

| Profile | Sampling per state, each leg | Lambda states A/B | GPU groups A/B | Replicates | Role |
| --- | --- | --- | --- | --- | --- |
| `smoke.yaml` | 0.1 ns; e05/v18 | 22/29 | 6/8 | 1 | Quick pipeline check |
| `validation.yaml` | 1 ns; e05/v18 | 22/29 | 6/8 | 1 | Short end-to-end validation |
| `full-ladder-validation.yaml` | 1 ns; e29/v45 | 73/80 | 25/30 | 1 | Check complete production ladder with short sampling |
| `production.yaml` | 10 ns; e29/v45 | 73/80 | 25/30 | 3 | Independent 10 ns Boresch equilibration per calculation/replica |

Smoke and both validation profiles use 3 ns of Boresch equilibration;
production uses 10 ns (2000 iterations at 5 ps each) per calculation/replica.
All use r02 restraints, 298.15 K and 2 fs steps.
Durations are per state, not total GPU runtime; overlapping boundary states
are simulated in adjacent groups. The 3 ns GPU equilibration is still required
for smoke and short validation; fewer ABFE states do not shorten it. The
previous portable sucralose run planned with `validation.yaml` used the full
73/80-state ladder; its frozen science settings do not change when this template
changes. Plan a new run directory to choose a different protocol. A successful
short validation checks the end-to-end path, not the production-ladder overlap
or a converged affinity. For longer production, copy the protocol and
increase `solvent_ns`/`complex_ns` (e.g. 50 ns), choose a new name/seed, and plan a
new run. Set site walltimes from measured validation performance. A duration
label does not establish convergence.

Production A/B sampling remains 10 ns per alchemical state across the 73/80
ladder and 25/30 groups.
Boresch equilibration occurs once per calculation/replica preproduction
chain, not once per lambda state.

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

New production groups record an immutable initialization intent before the
native sampler can create NetCDF files. A separate ready record, bound to that
intent and the native creation marker, is committed before any MPI rank starts
production. These records contain the science, group, seed/equilibration and
runtime identities; they are not simulation-completion manifests.

If preemption interrupts initialization before the creation marker is written,
the worker preserves the uncommitted trajectory/checkpoint files in
`calculations/<calculation>/initialization/<stem>/<id>/` and retries from the
validated equilibrated start. The quiescent resume probe only reports that group
as incomplete; it does not move files. If the marker was written but readiness
was not committed, the worker restores the initialized iteration-zero sampler.
A marked trajectory is never passed to the native NetCDF-error delete/recreate
fallback: restoration errors preserve all existing files and stop the task.
Ready groups still require semantic checkpoint validation, and completion still
requires the exact iteration target and a simulation-group manifest.

Existing PR4/PR5 marked trajectories retain their artifact format. An unmarked
trajectory with no initialization intent is still rejected; this protocol does
not fabricate provenance for existing orphaned campaign data. The existing
runtime source-fingerprint checks remain enforced. Installing this patch is
not authorization to resume an older live run with changed executable sources;
any such upgrade/recovery needs a separate reviewed compatibility plan.

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
