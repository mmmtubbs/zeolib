#!/usr/bin/env python3
"""
provenance.py — which zeolib produced this result?

Added 2026-08-25, when `zeolib/` was put under git. The motivating evidence:
four shipped copies of zeolib had silently diverged from master —
`Foundations/f0_fau_rebaseline/pkg/zeolib`, `Foundations/f1_naform/pkg_FAU`,
`.../pkg_MOR`, `MOR/pipeline/stage1a_v2/ship_rank/zeolib` — each differing in
6-8 modules, with no record of which library version generated which numbers.
Those copies stay frozen (README rule 4, they ARE the record of what ran);
this module makes every FUTURE package say so for itself.

The two halves, and why they are separate:

  * `version_info()` reads the LIVE git checkout. It only means anything on a
    machine that has the repo — the desktop. Call it at PACKAGING time.
  * `read_stamp()` reads a `ZEOLIB_VERSION.json` file that packaging left
    behind. That is the only thing that works on the cluster, where the
    shipped `zeolib/` has no `.git` at all.

So the flow is: desktop `write_stamp(pkg_dir)` -> ship -> anything asking
"which zeolib was this?" later calls `read_stamp(pkg_dir)`.

Rule 7 (no silent fallbacks) applies with a wrinkle. A shipped copy genuinely
has no repo, so `version_info()` cannot simply raise. Instead it never invents
a plausible-looking answer: with no git it returns `vcs="none"` and
`sha=None`, which no caller can mistake for a real commit. Callers that are
generating RESULTS OF RECORD should use `require_clean()`, which is the loud
path: it raises on a missing repo, an unknown commit, or a dirty tree.

Stdlib only (subprocess/json/datetime) so it imports on the cluster and in
python environments without ase/spglib/numpy.
"""
import json
import os
import subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))

STAMP_NAME = "ZEOLIB_VERSION.json"


def _git(args, repo_dir):
    """Run a git command in repo_dir. Returns stripped stdout, or None if git
    is absent / this is not a repo / the command fails. Never raises."""
    try:
        out = subprocess.run(["git", "-C", repo_dir] + list(args),
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except (OSError, ValueError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.decode("utf-8", "replace").strip()


def is_repo(repo_dir=None):
    """True if repo_dir is inside a git work tree."""
    repo_dir = repo_dir or HERE
    return _git(["rev-parse", "--is-inside-work-tree"], repo_dir) == "true"


def git_sha(repo_dir=None, short=True):
    """Current commit SHA, or None when there is no repo (a shipped copy)."""
    repo_dir = repo_dir or HERE
    args = ["rev-parse", "--short=9", "HEAD"] if short else ["rev-parse", "HEAD"]
    return _git(args, repo_dir)


def is_dirty(repo_dir=None):
    """
    True if the work tree differs from HEAD in any way that would SHIP.

    Untracked files count as dirty (`--untracked-files=normal`): a new module
    that has not been committed is still copied into a package by
    shutil.copytree, so a SHA that omits it does not describe what ran.
    .gitignore'd paths (__pycache__, .DS_Store) are excluded by git itself.

    None when there is no repo — deliberately NOT False, so that 'no repo'
    can never be read as 'clean'.
    """
    repo_dir = repo_dir or HERE
    if not is_repo(repo_dir):
        return None
    status = _git(["status", "--porcelain", "--untracked-files=normal"], repo_dir)
    if status is None:
        return None
    return status != ""


def version_info(repo_dir=None):
    """
    Everything known about the zeolib the CALLER is importing right now.

    Always returns the same key set, so a manifest schema is stable:
        zeolib_version : __init__.__version__ (hand-set, coarse)
        vcs            : "git" | "none"
        sha            : 9-char commit, or None with vcs="none"
        sha_full       : 40-char commit, or None
        branch         : branch name, "HEAD" when detached, or None
        describe       : `git describe --tags --always --dirty`, or None
        dirty          : True/False, or None when vcs="none"
        source         : absolute path of the zeolib actually imported
        stamped_utc    : ISO-8601 Z timestamp of THIS call

    On the cluster (shipped copy, no .git) this reports vcs="none" — that is
    correct and expected; use read_stamp() there instead.
    """
    repo_dir = repo_dir or HERE
    try:
        from zeolib import __version__ as ver
    except Exception:
        ver = None

    info = {"zeolib_version": ver,
            "vcs": "none",
            "sha": None,
            "sha_full": None,
            "branch": None,
            "describe": None,
            "dirty": None,
            "source": HERE,
            "stamped_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}

    if not is_repo(repo_dir):
        return info

    info["vcs"] = "git"
    info["sha"] = git_sha(repo_dir, short=True)
    info["sha_full"] = git_sha(repo_dir, short=False)
    info["branch"] = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir)
    info["describe"] = _git(["describe", "--tags", "--always", "--dirty"], repo_dir)
    info["dirty"] = is_dirty(repo_dir)
    return info


def stamp_line(info=None):
    """One-line human form, for sbatch/NOTES headers and log banners.

    Examples:
        zeolib 0.1.0 @ 7f3c2591b (main, clean) 2026-08-25T18:40:02Z
        zeolib 0.1.0 @ 7f3c2591b (main, DIRTY) 2026-08-25T18:40:02Z
        zeolib 0.1.0 @ NO-GIT-REPO 2026-08-25T18:40:02Z
    """
    info = info or version_info()
    if info.get("vcs") != "git" or not info.get("sha"):
        return "zeolib %s @ NO-GIT-REPO %s" % (info.get("zeolib_version"),
                                               info.get("stamped_utc"))
    return "zeolib %s @ %s (%s, %s) %s" % (
        info.get("zeolib_version"), info["sha"], info.get("branch"),
        "DIRTY" if info.get("dirty") else "clean", info.get("stamped_utc"))


def require_clean(repo_dir=None, what="this package"):
    """
    The loud path, for anything generating RESULTS OF RECORD.

    Raises RuntimeError unless zeolib is a git checkout at a known commit with
    no uncommitted tracked changes. Returns the version_info dict otherwise.

    A dirty tree is fatal on purpose: a SHA that does not describe the code
    that actually ran is worse than no SHA, because it reads as trustworthy.
    """
    info = version_info(repo_dir)
    if info["vcs"] != "git":
        raise RuntimeError(
            "zeolib is not a git checkout (%s), so %s cannot be stamped with a "
            "commit. Run from the versioned zeolib, or pass require_clean=False "
            "and accept an unidentified library version." % (info["source"], what))
    if not info["sha"]:
        raise RuntimeError("zeolib git repo has no HEAD commit yet; commit "
                           "before packaging %s." % what)
    if info["dirty"]:
        raise RuntimeError(
            "zeolib work tree is DIRTY: %s would be stamped %s, but that commit "
            "does not describe the code being shipped. Commit (or stash) zeolib "
            "first." % (what, info["sha"]))
    return info


def write_stamp(pkg_dir, repo_dir=None, require_clean_tree=False, extra=None):
    """
    Write `ZEOLIB_VERSION.json` into pkg_dir and return the info dict.

    Call this from every packaging step, right after the `zeolib/` copy is
    made, so the shipped tree carries its own identity to the cluster.

    require_clean_tree=True routes through require_clean() (raises on a
    missing repo / dirty tree). `extra` merges caller context into the file
    (e.g. {"package": "stage1a_v2 ship_rank", "model": "na1500-polar"}).

    Written through fileio.write_lf — this file travels to Linux.
    """
    if require_clean_tree:
        info = require_clean(repo_dir, what=os.path.basename(os.path.abspath(pkg_dir)))
    else:
        info = version_info(repo_dir)
    if extra:
        info = dict(info)
        info.update(extra)
    from zeolib import fileio
    fileio.write_lf(os.path.join(pkg_dir, STAMP_NAME),
                    json.dumps(info, indent=1, sort_keys=True) + "\n")
    return info


def read_stamp(pkg_dir):
    """
    Read back the stamp a packaging step left in pkg_dir.

    This is the cluster-side / after-the-fact question "which zeolib made
    this?". Missing file is FATAL (rule 7) rather than an "unknown" dict: an
    unstamped package predates this module, and pretending otherwise is
    exactly the ambiguity this module exists to remove.
    """
    path = os.path.join(pkg_dir, STAMP_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "no %s in %s — this package was not stamped (it predates "
            "zeolib.provenance, or its packaging step skipped write_stamp)."
            % (STAMP_NAME, pkg_dir))
    with open(path, "r") as fh:
        return json.load(fh)


if __name__ == "__main__":
    print(stamp_line())
