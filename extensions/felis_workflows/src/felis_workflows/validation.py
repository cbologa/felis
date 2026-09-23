"""Validation of the assembled complex and persisted simulation progress."""
from pathlib import Path
import json

from .common import WorkflowError, read
from .planning import calculation, workdir


def validate_assembled(root, science, calc):
    from openmm import app
    from .parameterization.validation import compare_signatures, load_export, signatures
    root = Path(root)
    work = workdir(root, calc)
    result = {}
    for leg in "AB":
        top = app.GromacsTopFile(str(work / f"prepare/sys{leg}.top"))
        system = top.createSystem(nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False, removeCMMotion=False)
        atom_ids = read(work / f"prepare/sys{leg}_atom_ids.json")
        selected = read(work / f"prepare/sys{leg}_ab_ligatoms.json")["ab"]["ligatoms"]
        if selected != atom_ids["ligands"]["M00"][0] or len(selected) != len(set(selected)):
            raise WorkflowError("Alchemical atom selection does not identify exactly the target")
        members = [(calc["target"], selected)]
        if leg == "B":
            for index, ligand in enumerate(calc["partners"]):
                ids = atom_ids["cofactors"][f"C{index:02d}"][0]
                if set(ids) & set(selected):
                    raise WorkflowError("Partner entered the alchemical atom selection")
                members.append((ligand, ids))
        elif atom_ids.get("cofactors"):
            raise WorkflowError("Solvent leg unexpectedly contains cofactors")
        result[leg] = {}
        for ligand, ids in members:
            _, reference = load_export(root / "parameters" / ligand)
            compare_signatures(signatures(reference), signatures(system, ids), allow_mass_rounding=True)
            result[leg][ligand] = {"indices": ids, "parameters_verified": True,
                                  "alchemical": ligand == calc["target"]}
    return result


def iteration_status(root, science, calc, leg, unit, reporter_factory=None):
    import numpy as np
    if reporter_factory is None:
        from openmmtools.multistate import MultiStateReporter
        reporter_factory = MultiStateReporter
    owner = calculation(science, calc["solvent_owner"]) if leg == "A" else calc
    nc = workdir(root, owner) / "trj" / f'{unit["stem"]}.nc'
    if not nc.exists():
        return {"complete": False, "reason": "missing trajectory", "stem": unit["stem"]}
    marker = nc.with_suffix(".create_done")
    if not marker.exists():
        raise WorkflowError(f"Trajectory without FELIS creation marker: {nc}")
    reporter = reporter_factory(str(nc), open_mode="r", checkpoint_interval=unit["checkpoint_interval"])
    try:
        last = int(reporter.read_last_iteration(last_checkpoint=False))
        checkpoint = int(reporter.read_last_iteration(last_checkpoint=True))
        target = int(reporter.read_dict("options")["number_of_iterations"])
        if target != unit["iterations"] or last > target:
            raise WorkflowError(f"Stored iteration target differs for {nc}")
        states = np.asarray(reporter.read_replica_thermodynamic_states(iteration=last)).reshape(-1)
        valid = sorted(states.tolist()) == list(range(len(unit["ilam"])))
        if not valid or not 0 <= checkpoint <= last:
            raise WorkflowError(f"Corrupt replica/checkpoint state in {nc}")
        if reporter.checkpoint_interval != unit["checkpoint_interval"]:
            raise WorkflowError(f"Checkpoint interval differs for {nc}")
        samples = reporter.read_sampler_states(iteration=checkpoint)
        if samples is None or len(samples) != len(unit["ilam"]) or any(
                not np.isfinite(np.asarray(s.positions)).all() for s in samples):
            raise WorkflowError(f"Missing or invalid checkpoint coordinates in {nc}")
        # If the checkpoint lags, native from_storage may replay later work.
        return {"complete": last == target, "last_iteration": last, "last_checkpoint": checkpoint,
                "stored_target": target, "stem": unit["stem"], "trajectory": str(nc)}
    finally:
        reporter.close()


def partner_occupancy(root, science, calc):
    """Check each partner against a fixed receptor binding-site atom set.

    Minimum-image distances are evaluated for every stored position frame and
    every replica in each complex group. This is an occupancy diagnostic, not
    a confinement potential or a proof of equilibrium sampling.
    """
    if not calc["partners"]:
        return {"passed": True, "partners": {}}
    import numpy as np
    from openmm import app, unit as u
    from openmmtools.multistate import MultiStateReporter
    from .planning import units
    work = workdir(root, calc)
    ids = read(work / "prepare/sysB_atom_ids.json")
    protein = [int(i) for copies in ids["protein_heavy"].values() for group in copies for i in group]
    if not protein:
        raise WorkflowError("No protein heavy atoms for partner occupancy checks")
    gro = app.GromacsGroFile(str(work / "prepare/sysB.gro"))
    xyz = np.asarray(gro.positions.value_in_unit(u.nanometer))
    topology = app.GromacsTopFile(str(work / "prepare/sysB.top")).topology
    heavy = {a.index for a in topology.atoms() if a.element and a.element.atomic_number > 1}
    def min_dist(points, site, box):
        delta = points[:, None, :] - site[None, :, :]
        if box is not None and np.linalg.det(box) > 0:
            fractions = delta @ np.linalg.inv(box)
            delta = (fractions - np.rint(fractions)) @ box
        return float(np.linalg.norm(delta, axis=-1).min())
    sites = {}
    for index, name in enumerate(calc["partners"]):
        partner = [i for i in ids["cofactors"][f"C{index:02d}"][0] if i in heavy]
        if not partner:
            raise WorkflowError("Partner has no heavy atoms")
        # Fix the reference site to receptor atoms near the initial bound pose.
        near = [i for i in protein if np.min(np.linalg.norm(xyz[partner] - xyz[i], axis=1)) < .6]
        if not near:
            raise WorkflowError(f"Partner {name} has no receptor contacts in the starting pose")
        sites[name] = {"partner": partner, "site": near, "maximum_minimum_distance_angstrom": 0.0, "frames": 0}
    threshold = calc["partner_site_max_distance_angstrom"]
    for item in units(root, science, calc, "B"):
        nc = work / "trj" / f'{item["stem"]}.nc'
        reporter = MultiStateReporter(str(nc), open_mode="r", checkpoint_interval=item["checkpoint_interval"])
        try:
            last = int(reporter.read_last_iteration(last_checkpoint=True))
            for iteration in range(0, last + 1, item["checkpoint_interval"]):
                for state in reporter.read_sampler_states(iteration=iteration):
                    positions = np.asarray(state.positions.value_in_unit(u.nanometer))
                    box = None if state.box_vectors is None else np.asarray(state.box_vectors.value_in_unit(u.nanometer))
                    for entry in sites.values():
                        distance = 10 * min_dist(positions[entry["partner"]], positions[entry["site"]], box)
                        entry["maximum_minimum_distance_angstrom"] = max(entry["maximum_minimum_distance_angstrom"], distance)
                        entry["frames"] += 1
        finally:
            reporter.close()
    passed = all(v["frames"] > 0 and v["maximum_minimum_distance_angstrom"] <= threshold for v in sites.values())
    return {"passed": passed, "threshold_angstrom": threshold, "partners": sites,
            "scope": "checkpoint frames/all replicas; excursions between stored frames are not excluded"}
