# Derived endpoint MD and ABFE qualification

This is a **separate** Python distribution beside `felis_workflows`. It leaves
ABFE `science.json`, the `0.2.0` workflow version, PR4 terminal artifacts,
alchemical stages, and the PR4 source fingerprint unchanged. Existing PR4 runs
remain readable and resumable with the same simulation code and runtime.

Install into the simulation interpreter (OpenMM and MDTraj are needed for MD):

```bash
python -m pip install -e extensions/felis_workflows -e 'extensions/felis_endpoint[simulation]'
```

The configured site's `python.simulation` runner must have these packages.
`felis-endpoint doctor --site /absolute/site.yaml` reports CLI and simulation
runner source/runtime identity without allocating a GPU. A runner from another
checkout or with changed executable code cannot continue a planned endpoint run.

## Physical endpoint plan

Create a new configuration file with **explicit** starting structures. Example:

```yaml
schema_version: 1
name: ligand-pocket-physical-md
source_run: /absolute/path/to/finished-or-partial-abfe-run
seed: 81723
replicas: 3
duration_ns: 50
report_interval_ps: 50
temperature_K: 298.15
timestep_fs: 2
pressure_bar: 1
starts:
  - id: chosen-complex-start
    calculation: ligand_in_receptor/r1
    topology: /absolute/path/to/prepare/sysB.top
    include_dirs: []  # List directories for any GROMACS #include dependencies.
    pdb: /absolute/path/to/explicitly-selected-full-solvated-complex.pdb
```

Alternatively replace `pdb` with explicit `trajectory`, `trajectory_topology`
(PDB), and zero-based `frame` for an ordinary physical-MD trajectory. FELIS
replica-exchange `.nc` group output is **not** an ordinary frame-indexed
trajectory: an ABFE iteration also requires a thermodynamic-state to
sampler-state mapping. Direct `.nc` frame extraction and `source_group` are
rejected. Extract and inspect the intended fully solvated complex as a PDB
with a separate state-aware procedure, then provide that explicit PDB here.
No clusters, medoids, ligands, frames, or poses are selected automatically.
Relative input paths resolve beside the configuration file, never against the
invoking shell directory. `source_run` and the site checkout must be absolute.

```bash
felis-endpoint plan --config endpoint.yaml --site /absolute/site.yaml \
  --output /scratch/endpoint-runs/ligand-pocket-1
felis-endpoint submit --run /scratch/endpoint-runs/ligand-pocket-1 \
  --site /absolute/site.yaml --dry-run
felis-endpoint submit --run /scratch/endpoint-runs/ligand-pocket-1 \
  --site /absolute/site.yaml
felis-endpoint status --run /scratch/endpoint-runs/ligand-pocket-1
felis-endpoint resume --run /scratch/endpoint-runs/ligand-pocket-1 \
  --site /absolute/site.yaml --dry-run
```

The endpoint output **must be outside the Git checkout**. It freezes a copy of
each PDB/frame and the complete topology include graph, their SHA-256 hashes,
source run/calculation and trajectory/frame/time attribution, deterministic
start-specific replica seeds for initial velocities, the Langevin integrator,
and the Monte Carlo barostat, exact protocol step/frame counts, and simulation
runner identity. The default is 50 ns with a 2 fs timestep: 25 million steps
and 1,000 frames/checkpoints at 50 ps. Each start/replica gets a separate
directory. Its physical NPT system uses 298.15 K, one bar and one GPU; custom
forces, positional/Boresch restraints and alchemical forces are rejected.

Slurm submits one independent job per task with no `%N` throttle. A run-level
`execution.lock` serializes task selection, attempt creation, and scheduler
submission, preventing concurrent submit/resume commands from duplicating jobs.
The scheduler and QOS determine concurrency. Attempts freeze their selected task graph,
site, exact command, and script hash. Local execution uses one configured GPU
selector per concurrent worker. Exit status alone never marks completion:
the worker checks the expected DCD frame count and final frame, checkpoint,
state and step count before writing `completed.json`.

`status` does not open an active DCD. A completed record is reported as
`recorded_unverified` without reading its trajectory; a quiescent resume probe
checks frames semantically. A matching atomic progress/checkpoint pair resumes
from its saved step. A partial DCD without a matching checkpoint, a corrupt
checkpoint, or an extra DCD frame is `blocked`: inspect/repair or make a new
endpoint plan. The worker never silently starts such output from step zero.
No endpoint files are added to ABFE's stage graph or finalization products.

## Parameterized endpoint networks

`felis-endpoint analyze-networks --config network.yaml --site /absolute/site.yaml
--output /scratch/endpoint-analysis/network-1` reads explicit PDB/DCD or other
MDTraj-supported trajectories and frame lists/ranges, optionally filtered by
`time_range_ns: {start: 0, stop: 10}`. Set `frame_interval_ps` explicitly when
the trajectory format does not preserve physical frame times. The configuration names
the ligand residue and its donor/acceptor atoms, each receptor site and its
polar atoms, water aliases, loose/strict distance and angle cutoffs, and an
optional per-trajectory frame-to-cluster TSV (`frame`, `cluster`). Example:

```yaml
schema_version: 1
ligand: {resname: LIG, resid: 10, donors: [O1], acceptors: [O1]}
sites:
  - {id: candidate-site, resname: ASN, resid: 42, donors: [ND2], acceptors: [OD1], water_bridge: true}
water_resnames: [HOH, SOL, WAT]
loose: {distance_A: 3.5, angle_deg: 135}
strict: {distance_A: 3.2, angle_deg: 150}
trajectories:
  - replica: independent-1
    topology: /absolute/path/to/start.pdb
    trajectory: /absolute/path/to/trajectory.dcd
    frames: {start: 0, stop: 1000, stride: 10}
    cluster_tsv: /absolute/path/to/cluster-assignments.tsv
```

It writes detailed geometry, direct H-bond, water-bridge, and per-replica and
per-cluster TSVs, plus analysis provenance. The exploratory sucrose scripts
in `easley-sync-2026-09-25` informed these tables; their fixed receptor network,
replica labels, poses, and frame windows are not defaults. Analyze only
quiescent trajectories. `water_bridge: true` specifically means a water donor
hydrogen-bonding to a configured **site acceptor**, with a water/ligand link;
it requires at least one site acceptor. Site-donor-to-water bridges are not
classified by this schema. Direct ligand/site H-bonds support both directions.
Generated analysis files belong outside Git.

## ABFE validation qualification

```bash
felis-endpoint qualify --run /absolute/abfe-run --site /absolute/site.yaml \
  --output /scratch/reports/qualification.json
```

The report inventories planned calculation/replica coverage, validated PR4
preparation/equilibration/group/finalization records, native A/B/R free-energy
tables and `sys_abfe.tsv`, and the complete native `A_converge_table.tsv` and
`B_converge_table.tsv` rows when present. It exposes missing/corrupt evidence
and counts a replica complete only when every planned calculation in that
replica has terminal finalization and all required native tables. It never
declares scientific convergence from a scheduler exit code or an
automatic threshold. Convergence, overlap, restraint, and receptor-state
judgments remain the investigator's responsibility.

The four ABFE protocols remain distinct: `smoke.yaml` (0.1 ns, coarse ladder),
`validation.yaml` (1 ns, 22/29 states), `full-ladder-validation.yaml` (1 ns,
73/80 states), and `production.yaml` (three replicas, independent 10 ns
Boresch equilibration and 10 ns/state). Endpoint MD does not change them.
