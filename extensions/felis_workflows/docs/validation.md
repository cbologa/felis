# Verification and deployment qualification

The extension is ready for code review and site qualification. Automated CPU
checks do not establish GPU compatibility, receptor correctness or convergence.
The initial verification, including replay of recorded real GAFF2 tool outputs,
is recorded in [verification.json](verification.json). Rerun the checks after
changing workflow profiles or code.

## Checks performed during implementation

* All 4,035 upstream-owned paths matched the pinned revision, including modes,
  symlinks and staged content. The retained legacy Slurm files are fork additions.
* Unit/integration tests cover four-ABFE endpoint definitions; native 73/80-state
  lambda ladders and overlapping group boundaries; distinct replicate seed
  assignments; two shared solvent legs per replicate; dependency construction;
  missing-group recovery; active-writer exclusion; non-mutating resume previews;
  site/science separation; input hashes; recursive topology includes; upstream
  content/mode/symlink/index changes; disulfide ambiguity and HETATM handling.
* An independently constructed OpenMM/ITP ethane fixture exercises charges,
  bonds, angles, periodic torsions, 1-4 terms, energies, forces and hydrogen
  constraints. Deliberately changed charges, torsions and atom order fail.
* A real OpenMMTools 0.26.0 replica-exchange simulation on the Reference platform
  writes a partial NetCDF/checkpoint pair, resumes from storage and reaches the
  stored target. Completion checks reject a different requested iteration target.
* Coupling tests check sign, cycle closure, common solvent cancellation,
  matched-replicate uncertainty, missing cycles, partner-occupancy failures and
  charged-target correction requirements.
* The provided TYK2 campaign plans successfully, imports its receptor GRO/TOP
  into OpenMM, and renders Slurm submission scripts with valid shell syntax.
  Its complete CLI `prepare` command also passed with real GAFF2 parameterization.
* Real ACPYPE 2023.10.27/AM1-BCC/GAFF2 output for TYK2/ejm_31 was generated with
  the ACPYPE wheel's bundled AmberTools 22 tools. Its Amber topology and final
  ParmEd export pass the independent parameter/energy/force/constraint checks.
  The test exposed ACPYPE's charge balancing and bonded-constant rounding;
  these are now handled explicitly without loosening comparison tolerances.

The CPU test environment used Python 3.12, NumPy 1.26.4, OpenMM 8.6.1,
ParmEd 4.3.1 and RDKit 2026.03.6. The OpenMMTools reporter integration used
0.26.0; mpiplus was imported from its v0.0.2 source because its packaging uses an
API removed in Python 3.12. Scientific environment recipes use Python 3.11.
The CI workflow is configured for Python 3.11/OpenMM 8.4.0.
These are CPU results, not measured Easley/Hopper timings.

Run the checks from the repository root:

```bash
python -m pytest -q extensions/felis_workflows/tests
felis-workflow verify-upstream
```

Tests need pytest, PyYAML, NumPy, RDKit and OpenMM. The real reporter test also
needs OpenMMTools and its dependencies and skips if they are absent. The CI
workflow installs those dependencies. GAFF2 regression tests also need ParmEd.
OpenMM's GROMACS reader can emit resource warnings
in its own code; the numerical checks still run.

## Checks still required on each deployment

1. Resolve the site's environment specifications and record `conda list
   --explicit`. Check the desired GPU partition, account, memory limits and
   walltimes. `doctor` checks configuration and command discovery, not allocation
   eligibility or scientific convergence.
2. Run TYK2 CPU preparation with each selected force field. Sage's generalized
   adapter has not been rerun in a fresh OpenFF environment here; the original
   Sage workflow and its prior test reports remain in the supplied archive.
   Its native/export checks execute automatically during preparation.
3. First run `smoke.yaml` (0.1 ns on 22/29 states, split into 6/8 groups)
   for an end-to-end pipeline check on a previously successful ligand. Use
   `validation.yaml` for 1 ns on that coarse ladder. Run
   `full-ladder-validation.yaml` (1 ns on 73/80 states, 25/30 groups) when
   checking the production lambda schedule. Check the saved assembled-system
   parameter/atom-mask report, GROMACS logs, MPS isolation, memory use,
   native overlap/convergence plots and checkpoint cadence. Finishing the
   coarse calculation does not establish the accuracy or convergence of the
   production protocol.
4. Interrupt one test array task through the site's normal job controls. Once
   all trajectory writers have stopped, preview and run `resume`; verify that
   completed groups are retained and the interrupted group reaches its original
   target. Do not test preemption on a valuable production run first.
5. Review the actual 9OPZ receptor and bound poses before that campaign. For a
   coupling campaign, inspect both ternary setups, the target/partner atom masks,
   partner occupancy, receptor states, standard-state restraints and cycle
   closure. A fixed 6-Angstrom occupancy threshold is a template choice, not a
   universal binding-site definition.

## Known-system acceptance on Easley

For a sucralose/Sage run, record the frozen campaign, force-field and protocol
from `science.json`; validate the receptor audit, ligand parameter validation,
both assembled-leg atom masks and the `grompp_A.log`/`grompp_B.log` checks.
Require `prep.ok.json`, every A and B trajectory at its stored iteration target,
`finalized.json`, and an `analysis/report.json` with no missing calculation.
Check the actual Slurm exit states and retry incomplete groups only after all
previous trajectory writers stop. A Slurm completion state by itself does not
establish that the physical calculation is complete.

Compare the resulting diagnostics and free-energy estimate with the prior
sucralose/Sage run only after checking receptor construct, ligand pose, force
field and protocol differences. A coarse or short run can produce an end-to-end
estimate while still having poor overlap or sampling. Keep that qualification
separate from using a new receptor or ligand for scientific interpretation.

The PDB-to-receptor path requires AmberTools `pdb4amber`/Reduce/LEaP and has not
been executed against the user's complete 9OPZ construct here because those
input coordinates were not supplied. No Slurm jobs or production GPU simulations
were launched during implementation. Production duration must be chosen from
sampling diagnostics, not merely from successful workflow completion.
