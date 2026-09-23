from copy import deepcopy
from pathlib import Path
import sys
import pytest

EXTENSION = Path(__file__).resolve().parents[1]
REPO = EXTENSION.parents[1]
sys.path.insert(0, str(EXTENSION / "src"))

from felis_workflows.common import read, write
from felis_workflows.planning import plan


@pytest.fixture
def extension():
    return EXTENSION


@pytest.fixture
def repo():
    return REPO


@pytest.fixture
def site(tmp_path):
    value = read(EXTENSION / "sites/easley.yaml")
    value["repo"] = str(REPO)
    for role in value["python"]:
        value["python"][role] = [sys.executable]
    path = tmp_path / "site.json"
    write(path, value)
    return path


@pytest.fixture
def cycle(tmp_path, monkeypatch):
    import felis_workflows.planning as planning
    # Test integrity separately. Here the fixture isolates planning semantics.
    monkeypatch.setattr(planning, "verify_upstream", lambda _: None)
    value = read(EXTENSION / "campaigns/coupling.yaml")
    pdb = tmp_path / "receptor.pdb"
    pdb.write_text("REMARK planning fixture; not a physical receptor\n")
    value["receptor"]["pdb"] = str(pdb)
    for ligand in value["ligands"].values():
        ligand["sdf"] = str(REPO / "examples/abfe/input/ejm_31.sdf")
    campaign = tmp_path / "campaign.json"
    write(campaign, value)
    root = tmp_path / "run"
    plan(campaign, EXTENSION / "forcefields/gaff2.yaml", EXTENSION / "protocols/production.yaml", root, REPO)
    return root, read(root / "science.json")
