from pathlib import Path
import os
import subprocess
import pytest

from felis_workflows.common import WorkflowError
from felis_workflows.integrity import verify_upstream
from felis_workflows.preparation.receptor import clean_source, disulfide_pairs
from felis_workflows.preparation.caps import pdb_new


def test_real_upstream_integrity(repo):
    result = verify_upstream(repo)
    assert result["verified_paths"] > 100


def test_upstream_file_mode_symlink_and_index_guards(tmp_path, monkeypatch):
    import felis_workflows.integrity as integrity
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], stderr=subprocess.DEVNULL).decode().strip()
    git("init")
    path = tmp_path / "engine.py"
    path.write_text("original\n")
    (tmp_path / "alias").symlink_to("engine.py")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    monkeypatch.setattr(integrity, "UPSTREAM_COMMIT", git("rev-parse", "HEAD"))
    (tmp_path / "extension.py").write_text("allowed")
    assert verify_upstream(tmp_path)["verified_paths"] == 2
    path.write_text("changed\n")
    git("add", "engine.py")
    path.write_text("original\n")
    with pytest.raises(WorkflowError, match="Staged upstream"):
        verify_upstream(tmp_path)
    git("reset", "HEAD", "engine.py")
    path.chmod(0o755)
    with pytest.raises(WorkflowError, match="mode changed"):
        verify_upstream(tmp_path)
    path.chmod(0o644)
    (tmp_path / "alias").unlink()
    (tmp_path / "alias").write_text("engine.py")
    with pytest.raises(WorkflowError, match="symlink replaced"):
        verify_upstream(tmp_path)


def test_disulfide_ambiguity_and_explicit_pairing():
    import numpy as np
    groups = [(f"A:{i}", [pdb_new(i, "SG", "CYS", "A", i, np.array([x, 0., 0.]), "S")])
              for i, x in [(1, 0.), (2, 2.), (3, 2.2)]]
    with pytest.raises(WorkflowError, match="Ambiguous"):
        disulfide_pairs(groups, {"mode": "detect", "expected": 1})
    assert disulfide_pairs(groups, {"mode": "explicit", "expected": 1, "pairs": [["A:1", "A:2"]]}) == [("A:1", "A:2")]


def test_unexpected_cofactor_is_not_silently_removed(tmp_path):
    path = tmp_path / "source.pdb"
    path.write_text(pdb_new(1, "FE", "HEM", "A", 1, [0, 0, 0], "FE").replace("ATOM  ", "HETATM"))
    with pytest.raises(WorkflowError, match="HETATM"):
        clean_source(path, {"expected_chains": 1, "remove_hetatm": []})
