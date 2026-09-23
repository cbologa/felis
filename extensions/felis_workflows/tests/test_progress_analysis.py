from copy import deepcopy
import math
import pytest

from felis_workflows.analysis import coupling
from felis_workflows.common import WorkflowError
from felis_workflows.planning import units, workdir
from felis_workflows.validation import iteration_status


def test_checkpoint_resume_requires_exact_stored_target(cycle):
    root, science = cycle
    calc = science["calculations"][0]
    unit = units(root, science, calc, "B")[0]
    nc = workdir(root, calc) / "trj" / (unit["stem"] + ".nc")
    nc.parent.mkdir(parents=True)
    nc.touch()
    with pytest.raises(WorkflowError, match="creation marker"):
        iteration_status(root, science, calc, "B", unit, object)
    nc.with_suffix(".create_done").touch()
    class Reporter:
        last, checkpoint, target = 1950, 1950, 2000
        checkpoint_interval = 50
        def __init__(self, *args, **kwargs): pass
        def read_last_iteration(self, last_checkpoint): return self.checkpoint if last_checkpoint else self.last
        def read_dict(self, _): return {"number_of_iterations": self.target}
        def read_replica_thermodynamic_states(self, **kwargs): return list(range(len(unit["ilam"])))
        def read_sampler_states(self, **kwargs):
            from types import SimpleNamespace
            return [SimpleNamespace(positions=[[0., 0., 0.]])] * len(unit["ilam"])
        def close(self): pass
    assert not iteration_status(root, science, calc, "B", unit, Reporter)["complete"]
    Reporter.last = 2000
    assert iteration_status(root, science, calc, "B", unit, Reporter)["complete"]
    Reporter.target = 2500
    with pytest.raises(WorkflowError, match="target differs"):
        iteration_status(root, science, calc, "B", unit, Reporter)


def cycle_records(science):
    records = {}
    for c in science["calculations"]:
        # Large common per-replicate solvent offsets cancel in matched differences.
        shift = 100 * c["replica"]
        values = {"L_in_R": -5, "P_in_R": -2, "L_in_RP": -8, "P_in_RL": -5}
        records[c["key"]] = {"dG_kcal_mol": values[c["id"]] + shift,
                              "occupancy_passed": True, "formal_charge": 0}
    return records


def test_coupling_sign_closure_and_correlated_uncertainty(cycle):
    _, science = cycle
    records = cycle_records(science)
    result = coupling(science, records)
    assert result["via_L"]["mean_kcal_mol"] == -3
    assert result["via_P"]["mean_kcal_mol"] == -3
    assert result["closure"]["mean_kcal_mol"] == 0
    assert result["via_L"]["standard_error_kcal_mol"] == 0
    records["P_in_RL/r2"]["dG_kcal_mol"] += 2
    result = coupling(science, records)
    assert result["closure"]["mean_kcal_mol"] == pytest.approx(2/3)


def test_failed_occupancy_and_uncorrected_charge_excluded(cycle):
    _, science = cycle
    records = cycle_records(science)
    records["L_in_RP/r1"]["occupancy_passed"] = False
    records["P_in_R/r2"]["formal_charge"] = 1
    result = coupling(science, records)
    assert result["via_L"]["n"] == 1
    assert result["via_L"]["standard_error_kcal_mol"] is None
    assert len(result["excluded"]) == 2
