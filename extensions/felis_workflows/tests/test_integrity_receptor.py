from pathlib import Path
import os
import subprocess
import pytest

from felis_workflows.common import WorkflowError, write
from felis_workflows.integrity import verify_upstream
from felis_workflows.preparation.receptor import clean_source, disulfide_pairs
from felis_workflows.preparation.caps import pdb_new


def test_real_upstream_integrity(repo):
    result = verify_upstream(repo)
    assert result["verified_paths"] > 100
    assert result["approved_patches"] == [
        "felis/configs/_config_tools.py", "felis/configs/global_keys.py",
        "felis/protocols/abfe/config_types.py", "felis/utils/omm/omm_system.py",
    ]


def _fixture_manifest(repo, base, patches):
    tree = subprocess.check_output(["git", "-C", str(repo), "rev-parse", f"{base}^{{tree}}"], text=True).strip()
    write(repo / "extensions/felis_workflows/upstream.lock.json",
          {"commit": base, "tree": tree, "approved_patches": patches})


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
    base = git("rev-parse", "HEAD")
    monkeypatch.setattr(integrity, "UPSTREAM_COMMIT", base)
    _fixture_manifest(tmp_path, base, [])
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


def test_approved_patch_must_match_worktree_and_index(tmp_path, monkeypatch):
    import felis_workflows.integrity as integrity
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()
    git("init")
    approved = tmp_path / "core.py"
    other = tmp_path / "other.py"
    approved.write_text("base\n")
    other.write_text("unchanged\n")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "base")
    base = git("rev-parse", "HEAD")
    monkeypatch.setattr(integrity, "UPSTREAM_COMMIT", base)
    base_blob = git("rev-parse", f"{base}:core.py")
    approved.write_text("approved\n")
    patched_blob = git("hash-object", "core.py")
    _fixture_manifest(tmp_path, base, [{"path": "core.py", "base_blob": base_blob,
                                       "patched_blob": patched_blob, "rationale": "Regression fix"}])
    assert verify_upstream(tmp_path)["approved_patches"] == ["core.py"]
    git("add", "core.py")
    assert verify_upstream(tmp_path)["approved_patches"] == ["core.py"]
    git("update-index", "--chmod=+x", "core.py")
    with pytest.raises(WorkflowError, match="Staged upstream"):
        verify_upstream(tmp_path)
    git("update-index", "--chmod=-x", "core.py")
    assert verify_upstream(tmp_path)["approved_patches"] == ["core.py"]
    git("rm", "--cached", "core.py")
    with pytest.raises(WorkflowError, match="Staged upstream"):
        verify_upstream(tmp_path)
    git("add", "core.py")
    assert verify_upstream(tmp_path)["approved_patches"] == ["core.py"]
    approved.write_text("unreviewed\n")
    with pytest.raises(WorkflowError, match="content changed"):
        verify_upstream(tmp_path)
    git("add", "core.py")
    approved.write_text("approved\n")
    with pytest.raises(WorkflowError, match="Staged upstream"):
        verify_upstream(tmp_path)
    git("add", "core.py")
    other.write_text("unreviewed\n")
    with pytest.raises(WorkflowError, match="content changed"):
        verify_upstream(tmp_path)


def test_patch_manifest_rejects_wrong_base_blob(tmp_path, monkeypatch):
    import felis_workflows.integrity as integrity
    subprocess.run(["git", "init", str(tmp_path)], check=True, stdout=subprocess.DEVNULL)
    (tmp_path / "core.py").write_text("base\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-m", "base"],
                   check=True, stdout=subprocess.DEVNULL)
    base = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(integrity, "UPSTREAM_COMMIT", base)
    _fixture_manifest(tmp_path, base, [{"path": "core.py", "base_blob": "0" * 40,
                                       "patched_blob": "1" * 40, "rationale": "Wrong base"}])
    with pytest.raises(WorkflowError, match="Incorrect upstream base blob"):
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
