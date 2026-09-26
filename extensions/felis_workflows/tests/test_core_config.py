"""Regression tests for the approved core FELIS configuration patch."""
import math

import pytest
import yaml

from felis.configs import GlobalKeys
from felis.protocols.abfe.config_types import ABFEInputConfig
from felis.utils.omm.omm_system import get_integrator


@pytest.mark.parametrize("literal", ["1e-8", "'1e-8'", "1.0e-8", "0.00000001", "1"])
def test_constraint_tolerance_from_yaml_is_typed_before_openmm(tmp_path, literal):
    path = tmp_path / "global.yaml"
    path.write_text(f"integrator:\n  constraint_tol: {literal}\n")
    keys = GlobalKeys()
    keys.update_by_cfg(str(path))
    keys.check()
    assert type(keys.integrator.constraint_tol) is float
    expected = float(yaml.safe_load(path.read_text())["integrator"]["constraint_tol"])
    integrator = get_integrator(keys)
    assert integrator.getConstraintTolerance() == expected


@pytest.mark.parametrize("tkv", ["f:integrator.constraint_tol:1e-8", "s:integrator.constraint_tol:1e-8"])
def test_tkv_override_is_typed(tkv):
    keys = GlobalKeys()
    keys.update_by_tkv(tkv)
    keys.check()
    assert type(keys.integrator.constraint_tol) is float
    assert get_integrator(keys).getConstraintTolerance() == 1e-8


@pytest.mark.parametrize("literal", ["'not-a-number'", "true", "false", ".nan", ".inf", "-.inf",
                                     "'NaN'", "'Infinity'", "0", "-1e-8", "null"])
def test_bad_constraint_tolerance_fails_at_ingestion(tmp_path, literal):
    path = tmp_path / "global.yaml"
    path.write_text(f"integrator:\n  constraint_tol: {literal}\n")
    with pytest.raises(ValueError, match="constraint_tol"):
        GlobalKeys().update_by_cfg(str(path))


@pytest.mark.parametrize("value", ["bad", "nan", "inf", "-inf", "0", "-1e-8"])
def test_bad_tkv_tolerance_fails_at_ingestion(value):
    with pytest.raises(ValueError, match="constraint_tol"):
        GlobalKeys().update_by_tkv(f"s:integrator.constraint_tol:{value}")


def test_explicit_global_numeric_fields_and_non_numeric_options(tmp_path):
    path = tmp_path / "global.yaml"
    path.write_text("""integrator:
  dt_ps: '0.002'
  nstep_per_snapshot: '2500'
  minimize: fire2
posres:
  atoms: ['0', 1]
  k_kcal: 25
boresch:
  k_r_a_dih_kcal: ['2', 80, 80.0]
ab:
  lam_list: [['0', 1, '0.5']]
openmm:
  checkpoint_interval: '50'
""")
    keys = GlobalKeys()
    keys.update_by_cfg(str(path))
    keys.check()
    assert (type(keys.integrator.dt_ps), type(keys.integrator.nstep_per_snapshot)) == (float, int)
    assert keys.integrator.minimize == 4
    assert keys.posres.atoms == [0, 1]
    assert keys.boresch.k_r_a_dih_kcal == [2.0, 80.0, 80.0]
    assert keys.ab.lam_list == [[0.0, 1.0, 0.5]]
    assert keys.openmm.checkpoint_interval == 50
    for value in ("true", ".nan", ".inf", "'wrong'"):
        path.write_text(f"openmm:\n  checkpoint_interval: {value}\n")
        with pytest.raises(ValueError, match="checkpoint_interval"):
            GlobalKeys().update_by_cfg(str(path))


def test_abfe_file_numeric_fields_are_typed(tmp_path):
    path = tmp_path / "abfe.yaml"
    path.write_text("""pro_ionic_strength: '1e-1'
md_eq_nsnapshots: '2e3'
md_checkpoint_interval: 50.0
k_r_a_dih: ['2', 80, 80.0]
supplementary_vdw_lambda_list: ['0', 0.5, 1]
""")
    cfg = ABFEInputConfig.from_file(str(path))
    assert type(cfg.pro_ionic_strength) is float and cfg.pro_ionic_strength == 0.1
    assert type(cfg.md_eq_nsnapshots) is int and cfg.md_eq_nsnapshots == 2000
    assert type(cfg.md_checkpoint_interval) is int and cfg.md_checkpoint_interval == 50
    assert cfg.k_r_a_dih == [2.0, 80.0, 80.0]
    assert cfg.supplementary_vdw_lambda_list == [0.0, 0.5, 1.0]


@pytest.mark.parametrize("field,value", [
    ("pro_ionic_strength", "'bad'"), ("pro_ionic_strength", "true"),
    ("pro_ionic_strength", ".nan"), ("pro_ionic_strength", ".inf"),
    ("pro_ionic_strength", "-.inf"), ("md_eq_nsnapshots", "'2.5'"),
    ("md_eq_nsnapshots", "false"), ("md_eq_nsnapshots", ".nan"),
    ("md_eq_nsnapshots", "null"),
    ("k_r_a_dih", "[2, true, 80]"),
])
def test_abfe_rejects_invalid_numeric_fields(tmp_path, field, value):
    path = tmp_path / "abfe.yaml"
    path.write_text(f"{field}: {value}\n")
    with pytest.raises(ValueError, match=field):
        ABFEInputConfig.from_file(str(path))


def test_defensive_openmm_boundary_still_accepts_external_numeric_string():
    keys = GlobalKeys()
    keys.integrator.constraint_tol = "1e-8"  # Bypasses the config ingestion APIs.
    assert math.isclose(get_integrator(keys).getConstraintTolerance(), 1e-8)
