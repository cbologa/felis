# Migration and upstream ownership

The base boundary is every tracked path in ByteDance-Seed/felis commit
`4d2556bfe09753ff63ddf549c3838a517f173b12`, including content, symlinks and
executable bits. `upstream.lock.json` records the base commit/tree plus a small
approved patch manifest: path, original Git blob, exact patched Git blob, and
rationale. The four approved core files implement explicit numeric configuration
normalization and preserve the defensive OpenMM constraint-tolerance conversion.
`verify-upstream` validates the manifest against the base tree, checks the
worktree and index against each approved hash, and requires all other upstream
paths to remain exact. Correctly materialized LFS objects are accepted against
their upstream object hashes. A new patch or upstream update requires explicit
manifest review and compatibility tests.

The existing fork has changes to upstream-owned files: the example ABFE config,
packaging metadata and residue XML files, plus a missing residue-reference ignore
file. The accepted **exact pinned revision** policy restores those paths to the
upstream revision. This is a reconciliation of fork changes, not new engine
patches. The old fork commits remain in history. Existing `scripts/slurm` files
are retained unchanged for reference; new campaigns use the portable extension.

| Previous input | New home/responsibility |
| --- | --- |
| `scripts/slurm` | Shared execution backends plus `sites/*.yaml`; original files retained |
| `scripts/ff` | `parameterization/gaff2.py` and `forcefields/gaff2.yaml` |
| `9opz_campaign` | General receptor preparation and campaign templates |
| `scripts/9opz_sage_v4_workflow` | General Sage adapter, pinned OFFXML, environment and validation |
| Embedded cluster paths/partitions | Site YAML/bootstrap only |
| Per-script sampling edits | Protocol YAML, frozen by a new plan |

Original supplied ZIPs are retained in `legacy/` for audit and rollback reference.
Do not run them unchanged: they contain old absolute paths and narrower molecular
assumptions. The extension reuses their cap geometry, GAFF parsing and Sage
validation logic while generalizing molecular inputs and cofactor handling.

Start new run directories. Existing trajectory folders and hand-written progress
markers do not contain the new manifests and cannot be adopted safely by renaming
files. They remain analyzable using their original recorded environment/scripts.

The process-local launcher adapter replaces only job dispatch and MPS ownership.
Native box building, equilibration, lambda recipes/group boundaries, replica
exchange, MBAR and Boresch corrections are used from the pinned checkout. Array
analysis calls native analysis functions directly instead of fabricating engine
`.done` files to pretend that a Slurm array was a single-node launch.
