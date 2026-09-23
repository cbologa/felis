# Verification and deployment qualification

The extension is ready for code review and site qualification. Automated CPU
checks do not establish GPU compatibility, receptor correctness or convergence.
The final local run passed **28 tests**, including replay of recorded real GAFF2
tool outputs. Machine-readable results are in [verification.json](verification.json).

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
3. Run a smoke campaign on one allocated full GPU, then the full-ladder
   validation profile. Check the saved assembled-system parameter/atom-mask
   report, GROMACS logs, MPS isolation, memory use, native overlap/convergence
   plots and checkpoint cadence.
4. Interrupt one test array task through the site's normal job controls. Once
   all trajectory writers have stopped, preview and run `resume`; verify that
   completed groups are retained and the interrupted group reaches its original
   target. Do not test preemption on a valuable production run first.
5. Review the actual 9OPZ receptor and bound poses before that campaign. For a
   coupling campaign, inspect both ternary setups, the target/partner atom masks,
   partner occupancy, receptor states, standard-state restraints and cycle
   closure. A fixed 6-Angstrom occupancy threshold is a template choice, not a
   universal binding-site definition.

The PDB-to-receptor path requires AmberTools `pdb4amber`/Reduce/LEaP and has not
been executed against the user's complete 9OPZ construct here because those
input coordinates were not supplied. No Slurm jobs or production GPU simulations
were launched during implementation. Production duration must be chosen from
sampling diagnostics, not merely from successful workflow completion.
