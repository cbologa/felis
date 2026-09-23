# Sources and attribution

* FELIS: ByteDance Ltd. and/or affiliates, Apache-2.0, pinned at
  https://github.com/ByteDance-Seed/felis/tree/4d2556bfe09753ff63ddf549c3838a517f173b12.
  Its existing source notices and root LICENSE are unchanged.
* User-provided `ff.zip`: adapted ACPYPE/GROMACS extraction helpers.
* User-provided `9opz_campaign.zip`: adapted receptor terminal-cap geometry and
  explicit receptor audit/preparation requirements.
* User-provided `9opz_sage_v4_workflow.zip`: adapted Sage conversion, force/energy
  validation, OFFXML/model pins, GROMACS check settings and Sage environment
  specifications. Original archives are retained under `legacy/`.
* Bundled OpenFF `openff_unconstrained-2.3.0.offxml` and its MIT license are in
  `src/felis_workflows/data/`. The OFFXML SHA256 and declared NAGL model SHA256
  are in `parameterization/topology.py`. NAGL obtains the model through its own
  package/cache; pre-cache it if compute nodes have no network access.
* OpenMMTools reporter API and implementation:
  https://openmmtools.readthedocs.io/en/stable/api/generated/openmmtools.multistate.MultiStateReporter.html
  and https://github.com/choderalab/openmmtools/tree/0.26.0.
* ACPYPE CLI/output contract inspected in release 2023.10.27:
  https://github.com/alanwilter/acpype.

The supplied Sage explicit Linux environment is preserved as an installation
record, not a promise that arbitrary CUDA simulation environments are portable.
Capture `conda list --explicit` after each site's environment has been qualified.
