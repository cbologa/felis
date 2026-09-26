"""Verify every upstream-owned path, including symlinks and executable bits."""
from pathlib import Path
import hashlib
import os
import re
import stat
import subprocess

from . import UPSTREAM_COMMIT
from .common import WorkflowError, read


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def index_entry(repo, path):
    """Return the complete index entry for a literal upstream path, if unique."""
    output = git(repo, "--literal-pathspecs", "ls-files", "--stage", "-z", "--", path)
    records = [record for record in output.split(b"\0") if record]
    if len(records) != 1:
        return None
    try:
        header, recorded_path = records[0].split(b"\t", 1)
        mode, blob, stage = header.decode("ascii").split()
    except (ValueError, UnicodeDecodeError):
        return None
    if recorded_path != os.fsencode(path):
        return None
    return mode, blob, stage


def approved_patches(repo, base_entries):
    """Validate the reviewed patch manifest against the pinned base tree."""
    manifest = read(repo / "extensions/felis_workflows/upstream.lock.json")
    tree = git(repo, "rev-parse", f"{UPSTREAM_COMMIT}^{{tree}}").decode().strip()
    if manifest.get("commit") != UPSTREAM_COMMIT or manifest.get("tree") != tree:
        raise WorkflowError("Approved patch manifest does not match the pinned upstream base")
    patches = {}
    for patch in manifest.get("approved_patches", []):
        if not isinstance(patch, dict) or set(patch) != {"path", "base_blob", "patched_blob", "rationale"}:
            raise WorkflowError("Approved patch entries need path, base_blob, patched_blob and rationale")
        path = patch["path"]
        if not isinstance(path, str) or path not in base_entries or path in patches:
            raise WorkflowError(f"Unknown or duplicate approved upstream path: {path!r}")
        base_mode, base_kind, base_blob = base_entries[path]
        if base_kind != "blob" or base_mode not in {"100644", "100755"} or patch["base_blob"] != base_blob:
            raise WorkflowError(f"Incorrect upstream base blob for {path}")
        patched = patch["patched_blob"]
        if not isinstance(patched, str) or not re.fullmatch(r"[0-9a-f]{40}", patched) or patched == base_blob:
            raise WorkflowError(f"Invalid approved patched blob for {path}")
        if not isinstance(patch["rationale"], str) or not patch["rationale"].strip():
            raise WorkflowError(f"Missing patch rationale for {path}")
        patches[path] = patched
    return patches


def verify_upstream(repo):
    repo = Path(repo).resolve()
    entries = git(repo, "ls-tree", "-rz", UPSTREAM_COMMIT).split(b"\0")
    base_entries = {}
    for entry in filter(None, entries):
        header, name = entry.split(b"\t", 1)
        base_entries[os.fsdecode(name)] = tuple(header.decode().split())
    patches = approved_patches(repo, base_entries)
    failures, count = [], 0
    for entry in filter(None, entries):
        header, name = entry.split(b"\t", 1)
        mode, kind, expected = header.decode().split()
        relative = os.fsdecode(name)
        path = repo / relative
        count += 1
        try:
            st = path.lstat()
            if mode == "120000":
                if not stat.S_ISLNK(st.st_mode):
                    raise WorkflowError("symlink replaced")
                content = os.fsencode(os.readlink(path))
            elif kind == "blob":
                if not stat.S_ISREG(st.st_mode):
                    raise WorkflowError("regular file replaced")
                if bool(st.st_mode & 0o111) != (mode == "100755"):
                    raise WorkflowError("executable mode changed")
                content = path.read_bytes()
                # LFS materialization may differ in the worktree while the Git
                # blob is the original pointer. Compare to its recorded SHA256.
                oid = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
                if oid != expected:
                    pointer = git(repo, "cat-file", "blob", expected)
                    if pointer.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
                        fields = dict(line.split(b" ", 1) for line in pointer.splitlines()[1:])
                        if fields.get(b"oid") == b"sha256:" + hashlib.sha256(content).hexdigest().encode() and int(fields[b"size"]) == len(content):
                            continue
            else:
                raise WorkflowError(f"Unsupported upstream tree entry {kind}")
            actual = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            if actual != patches.get(relative, expected):
                raise WorkflowError("content changed")
        except (OSError, ValueError) as error:
            failures.append(f"{relative}: {error}")
    if failures:
        raise WorkflowError("Upstream integrity failed:\n" + "\n".join(failures[:30]))
    # Also catch staged changes masked by a restored working copy.
    staged = git(repo, "diff", "--cached", "--name-only", "-z", UPSTREAM_COMMIT).split(b"\0")
    upstream_names = {os.fsencode(name) for name in base_entries}
    modified = upstream_names.intersection(staged)
    unexpected = {os.fsdecode(name) for name in modified if os.fsdecode(name) not in patches}
    for name in modified:
        path = os.fsdecode(name)
        if path in patches:
            if index_entry(repo, path) != (base_entries[path][0], patches[path], "0"):
                unexpected.add(path)
    if unexpected:
        raise WorkflowError(f"Staged upstream modifications: {sorted(unexpected)}")
    return {"commit": UPSTREAM_COMMIT, "verified_paths": count, "approved_patches": sorted(patches)}
