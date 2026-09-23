# Real GAFF2 regression fixture

Generated from pinned upstream `examples/abfe/input/ejm_31.sdf` with ACPYPE
2023.10.27, its bundled AmberTools 22, `-c bcc -a gaff2 -n 0 -o all`, on
2026-09-23. These are tool outputs, not hand-constructed ligand parameters.

The fixture reproduces two conversion issues: ACPYPE balances a roughly -0.002 e
Amber net-charge residual, and its GMX writer rounds some bonded constants.
ParmEd's more precise export omits three exactly zero-amplitude torsions.
The regression test runs the adapter's remaining processing and independent
OpenMM energy/force/parameter validation against these recorded tool outputs.
This does not replace qualification of AmberTools 24 or arbitrary new ligands.
