"""Verify every upstream-owned path, including symlinks and executable bits."""
from pathlib import Path
import hashlib
import os
import stat
import subprocess

from . import UPSTREAM_COMMIT
from .common import WorkflowError


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def verify_upstream(repo):
    repo = Path(repo).resolve()
    entries = git(repo, "ls-tree", "-rz", UPSTREAM_COMMIT).split(b"\0")
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
            if actual != expected:
                raise WorkflowError("content changed")
        except (OSError, ValueError) as error:
            failures.append(f"{relative}: {error}")
    if failures:
        raise WorkflowError("Upstream integrity failed:\n" + "\n".join(failures[:30]))
    # Also catch staged changes masked by a restored working copy.
    staged = git(repo, "diff", "--cached", "--name-only", "-z", UPSTREAM_COMMIT).split(b"\0")
    upstream_names = {entry.split(b"\t", 1)[1] for entry in filter(None, entries)}
    modified = upstream_names.intersection(staged)
    if modified:
        raise WorkflowError(f"Staged upstream modifications: {sorted(os.fsdecode(x) for x in modified)}")
    return {"commit": UPSTREAM_COMMIT, "verified_paths": count}
