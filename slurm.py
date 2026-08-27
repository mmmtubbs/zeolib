"""
slurm.py — sbatch and submit-script generation, cluster-profile aware.

Two profiles (pass as `profile=` to every builder; PRONGHORN is the default):

  PRONGHORN (default — long / high-priority work):
    jobs start immediately, max walltime 14 days. CP2K 2026 singularity image
    launched via the container's OWN mpirun on a single node — NEVER srun
    (PMIx mismatch hangs / silently serialises) [global CLAUDE.md]. Submission
    routes through MOR/partition.sh psub (s1 free-first, overflow to paid s2)
    with a 50-job throttle.

  PERLMUTTER (NERSC — VOLUME of short jobs, e.g. 300+ single-point DFTs):
    handles hundreds of queued jobs without manual throttling and nodes are
    fast (128 ranks/node), but jobs may queue 1–2 days and max walltime is
    2 days. Shifter image docker:cp2k/cp2k:2022.1, launched via srun (correct
    THERE — the never-srun rule is Pronghorn-specific). No #SBATCH --account:
    forcing one trips "No available node hour balance"; NERSC falls back to
    the default repo. NEVER set --time-min: CP2K reads it as an effective
    walltime and self-terminates right after the first SCF energy print,
    hours before the real --time cap (2026-07-14 incident: every job in a
    resubmitted batch quit within ~1h of a 20h walltime). For hundreds of
    homogeneous jobs prefer ONE job array (array_manifest_text +
    array_sbatch_text, 2026-07-17) over a per-dir submit loop.
    *** PROVENANCE WARNING: the Perlmutter image is CP2K 2022.1, NOT the
    validated Pronghorn 2026 container. Fine for MLIP training frames and
    screens; do NOT pool its absolute energies with Pronghorn-2026 numbers in
    a ranking without a cross-image validation (PIPELINE.md: never pool across
    protocols). ***

Conventions encoded for both: OMP_NUM_THREADS=1, shell redirect '> out' (never
CP2K's -o flag), timestamped submit lines. Perlmutter text is byte-faithful to
the 369 shipped wsp perlmutter.sbatch files (na_placement_3al wanderer SP
offload, the working precedent); Pronghorn text to na_placement_multicomp.
Write everything with fileio.write_lf (CRLF breaks cluster bash).
"""

import json
import os

# ── Pronghorn constants ─────────────────────────────────────────────────────
CP2K_SIF = "/apps/cp2k/cp2k_psmp2026.1-rocky9-spack.sif"
CP2K_EXE = "cp2k.psmp"
S1_PARTITION = "cpu-s1-vessel-0"; S1_ACCOUNT = "cpu-s1-vessel-0"   # free, 6 nodes
S2_PARTITION = "cpu-s2-core-0";   S2_ACCOUNT = "cpu-s2-vessel-0"   # paid overflow
MAX_CONCURRENT_JOBS = 50   # Pronghorn standard (feedback_cluster_job_throttle)

# ── Perlmutter constants ────────────────────────────────────────────────────
PERLMUTTER_IMAGE = "docker:cp2k/cp2k:2022.1"   # NOT the 2026 protocol — see header


# ── Cluster identity (login account + scratch root) ─────────────────────────
# Deliberately NOT hardcoded: zeolib is a public repo, and a login name paired
# with its host and work path is site reconnaissance even though it grants no
# access on its own (both clusters need a key + MFA). Everything else about a
# profile is a public protocol convention and stays inline.
#
# Resolution order, first hit wins:
#   1. the profile attribute, if a caller set it explicitly (tests do this)
#   2. $ZEOLIB_<PROFILE>_HOST / $ZEOLIB_<PROFILE>_BASE
#   3. ~/.config/zeolib/clusters.json, e.g.
#        {"pronghorn": {"host": "user@cluster.example.edu",
#                       "base": "/scratch/user/MOR"}}
#   4. FATAL — never a placeholder (rule 7). A silently wrong host would
#      generate a ship.sh that pushes your work to nowhere, or somewhere else.

NO_BASE = "<no-default-base>"   # sentinel: this cluster HAS no default base

_CLUSTER_CONF = os.path.join(os.path.expanduser("~"), ".config", "zeolib",
                             "clusters.json")


def _from_conf(profile_name, field):
    """Look up clusters.json. Absent file / key -> None (the caller raises)."""
    try:
        with open(_CLUSTER_CONF) as fh:
            return json.load(fh).get(profile_name, {}).get(field)
    except (IOError, OSError, ValueError):
        return None


def _identity(profile, field, env_suffix):
    attr = getattr(profile, "host" if field == "host" else "remote_base", None)
    if attr is not None and attr != NO_BASE:
        return attr
    if attr == NO_BASE:
        return NO_BASE
    env = os.environ.get("ZEOLIB_%s_%s" % (profile.name.upper(), env_suffix))
    if env:
        return env
    conf = _from_conf(profile.name, field)
    if conf:
        return conf
    raise RuntimeError(
        "%s %s is not configured. Set $ZEOLIB_%s_%s, or add it to %s as "
        '{"%s": {"%s": "..."}}. It is not hardcoded because zeolib is public.'
        % (profile.name, field, profile.name.upper(), env_suffix,
           _CLUSTER_CONF, profile.name, field))


def resolve_host(profile):
    """The `user@host` this profile ships to. Raises if unconfigured."""
    return _identity(profile, "host", "HOST")


def resolve_remote_base(profile):
    """The profile's remote work root, or None where it has none by design
    (PERLMUTTER: scratch varies, so callers pass an absolute remote_dir)."""
    base = _identity(profile, "base", "BASE")
    return None if base == NO_BASE else base


class ClusterProfile(object):
    """Attribute bag for one cluster's conventions."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


PRONGHORN = ClusterProfile(
    name="pronghorn",
    max_walltime="14-00:00:00",
    default_walltime="2-00:00:00",
    ntasks=32,
    sbatch_name="cp2k.sbatch",
    host=None,          # site identity — see "Cluster identity" above
    remote_base=None,   # NOT ~ (reference_pronghorn_transfer); configured, not hardcoded
)

PERLMUTTER = ClusterProfile(
    name="perlmutter",
    max_walltime="2-00:00:00",
    default_walltime="01:00:00",     # sized for the volume-of-single-points use case
    ntasks=128,
    sbatch_name="perlmutter.sbatch",
    host=None,               # site identity — see "Cluster identity" above
    remote_base=NO_BASE,     # scratch path varies — pass an absolute remote_dir
)


def _walltime_seconds(t):
    """Parse SLURM walltime 'D-HH:MM:SS' / 'HH:MM:SS' / 'MM:SS' / minutes."""
    t = str(t)
    days = 0
    if "-" in t:
        d, t = t.split("-", 1)
        days = int(d)
    parts = [int(x) for x in t.split(":")]
    if len(parts) == 1:        # bare number = minutes (SLURM convention)
        h, m, s = 0, parts[0], 0
    elif len(parts) == 2:      # MM:SS
        h, (m, s) = 0, parts
    else:
        h, m, s = parts
    return ((days * 24 + h) * 60 + m) * 60 + s


def _check_walltime(walltime, profile):
    if _walltime_seconds(walltime) > _walltime_seconds(profile.max_walltime):
        raise ValueError("walltime %s exceeds %s max %s"
                         % (walltime, profile.name, profile.max_walltime))


def cp2k_run_line(inp, out=None, profile=PRONGHORN, ntasks=None,
                  srun_flags=None):
    """
    One CP2K launch line (shell redirect, never -o).

    srun_flags: extra srun flags for the PERLMUTTER branch, e.g. the
    resource slice of a node-packed step ("--exact --nodes 1 --ntasks 32 ...").
    Default None renders byte-identically to the shipped wsp fixtures.
    """
    if out is None:
        out = inp.rsplit(".", 1)[0] + ".out"
    if profile.name == "perlmutter":
        return "srun %sshifter --entrypoint cp2k -i %s > %s 2>&1" % (
            (srun_flags.strip() + " ") if srun_flags else "", inp, out)
    return ("singularity exec $SIF mpirun -np %d %s %s > %s 2>&1"
            % (ntasks or profile.ntasks, CP2K_EXE, inp, out))


def sbatch_text(job_name, run_lines, profile=PRONGHORN, walltime=None,
                ntasks=None, output="cp2k_%j.out",
                qos="regular", partition=S2_PARTITION, account=S2_ACCOUNT,
                image=None):
    """
    A CP2K sbatch script for the given profile. run_lines: shell lines (use
    cp2k_run_line with the SAME profile; $SIF is defined on Pronghorn).
    partition/account apply to Pronghorn only (header default s2 — the submit
    script overrides per-job via partition.sh psub when available); Perlmutter
    takes neither (no -A by design, see module header).
    qos: Perlmutter QOS (default "regular"; pass "debug" for a fast smoke job —
    debug starts almost immediately and caps at 30 min, ideal for confirming the
    shifter image + inputs get past setup into SCF before a real run). Ignored on
    Pronghorn. The "regular" default keeps byte-parity with the wsp fixtures.
    image: Perlmutter shifter image, default PERLMUTTER_IMAGE (CP2K 2022.1).
    Override ONLY with a deliberate provenance decision — the whole point of the
    default is that every NERSC job is traceable to one image. Added 2026-08-19
    for tests/cp2k_image_parity, which runs the same inputs under 2022.1 and a
    2026-matching tag to measure the cross-image difference directly.
    Raises ValueError if walltime exceeds the cluster's cap (14 d / 2 d).
    """
    walltime = walltime or profile.default_walltime
    _check_walltime(walltime, profile)
    if profile.name == "perlmutter":
        head = """\
#!/bin/bash
#SBATCH --image %s
#SBATCH --job-name=%s
#SBATCH --nodes 1
#SBATCH --ntasks-per-node %d
#SBATCH --cpus-per-task 2
#SBATCH --constraint cpu
#SBATCH --qos %s
#SBATCH --time %s
#SBATCH --output %s
#
export OMP_NUM_THREADS=1
""" % (image or PERLMUTTER_IMAGE, job_name, ntasks or profile.ntasks,
       qos, walltime, output)
    else:
        head = """\
#!/bin/bash
#SBATCH --job-name=%s
#SBATCH --nodes=1
#SBATCH --ntasks=%d
#SBATCH --cpus-per-task=1
#SBATCH --hint=compute_bound
#SBATCH --account=%s
#SBATCH --partition=%s
#SBATCH --time=%s
#SBATCH --output=%s
#
# 2026 CP2K: container's OWN mpirun on a single node (NOT srun). See global CLAUDE.md.
export OMP_NUM_THREADS=1
SIF=%s
""" % (job_name, ntasks or profile.ntasks, account, partition, walltime,
       output, CP2K_SIF)
    return head + "\n".join(run_lines) + "\n"


def array_manifest_text(job_dirs):
    """
    The manifest for a Perlmutter job array: one job dir per line, in
    submission order. Array task i runs line i (1-based — sed's numbering,
    matching `--array 1-N` in array_sbatch_text). Write next to the array
    sbatch with fileio.write_lf as `job_manifest.txt` (the array_sbatch_text
    default). Paths are relative to the submit dir, exactly as passed.
    """
    job_dirs = list(job_dirs)
    if not job_dirs:
        raise ValueError("array_manifest_text: empty job_dirs")
    for d in job_dirs:
        if "\n" in d or not d.strip():
            raise ValueError("array_manifest_text: bad job dir %r" % d)
    return "\n".join(job_dirs) + "\n"


def array_sbatch_text(job_name, run_lines, n_jobs, profile=PERLMUTTER,
                      walltime=None, ntasks=None, qos="regular",
                      manifest="job_manifest.txt", max_running=None,
                      output="cp2k_%A_%a.out"):
    """
    ONE job-array sbatch replacing a whole per-dir submit loop — the
    Perlmutter bundling for hundreds of homogeneous jobs (2026-07-17; NERSC
    prefers arrays over sbatch loops). Each array task reads its line of the
    manifest (see array_manifest_text), cd's into that job dir, and runs
    run_lines there — so run_lines are the SAME dir-relative lines you'd put
    in a per-dir sbatch (use cp2k_run_line(profile=PERLMUTTER)). Per-task
    resources (1 node, ntasks ranks, walltime) equal the old per-job sbatch.

    PERLMUTTER ONLY. Pronghorn raises: an array carries uniform
    partition/account directives, so it cannot express partition.sh psub's
    per-job s1-free-first routing + paid-s2 overflow, and its pending tasks
    all count against the 50-job queue-depth standard at once
    (feedback_cluster_job_throttle). Keep submit_script_text there.

    n_jobs      : task count — an int, or the job_dirs list itself (len used;
                  pass the same list given to array_manifest_text).
    max_running : optional running-task cap, rendered as `--array 1-N%%cap`.
                  Default None — NERSC manages large queues.
    output      : SLURM log per task, opened in the SUBMIT dir (%A=array job
                  id, %a=task id). The CP2K .out still lands in the job dir
                  via run_lines' shell redirect, as always.

    Usage: write manifest + this text (fileio.write_lf) into the package top
    dir, then on Perlmutter:  sbatch perlmutter_array.sbatch
    Rerun failed tasks only:  sbatch --array=17,42 perlmutter_array.sbatch
    (a command-line --array overrides the header; --array=1 gives a
    single-task trial, but Marcus submits full batches directly — a broken
    setup fails every task fast and one scancel of the array job id clears
    the rest [feedback-no-smoke-tests]). A missing manifest line is FATAL,
    never skipped (zeolib rule 7).
    """
    if profile.name != "perlmutter":
        raise ValueError(
            "array_sbatch_text is perlmutter-only: a job array cannot express "
            "Pronghorn's psub s1/s2 routing or queue-depth throttle — use "
            "submit_script_text(profile=PRONGHORN)")
    if not isinstance(n_jobs, int):
        n_jobs = len(n_jobs)
    if n_jobs < 1:
        raise ValueError("array_sbatch_text: n_jobs must be >= 1")
    walltime = walltime or profile.default_walltime
    _check_walltime(walltime, profile)
    array_spec = "1-%d" % n_jobs
    if max_running is not None:
        array_spec += "%%%d" % max_running
    head = """\
#!/bin/bash
#SBATCH --image %s
#SBATCH --job-name=%s
#SBATCH --array %s
#SBATCH --nodes 1
#SBATCH --ntasks-per-node %d
#SBATCH --cpus-per-task 2
#SBATCH --constraint cpu
#SBATCH --qos %s
#SBATCH --time %s
#SBATCH --output %s
#
# Task i runs line i (1-based) of the manifest, in that job dir.
export OMP_NUM_THREADS=1
MANIFEST="${SLURM_SUBMIT_DIR}/%s"
JOB_DIR="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")"
if [ -z "$JOB_DIR" ]; then
    echo "FATAL: no manifest line ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}" >&2
    exit 1
fi
cd "${SLURM_SUBMIT_DIR}/${JOB_DIR}"
echo "($(date '+%%F %%T')) array task ${SLURM_ARRAY_TASK_ID} -> [${JOB_DIR}]"
""" % (PERLMUTTER_IMAGE, job_name, array_spec, ntasks or profile.ntasks,
       qos, walltime, output, manifest)
    return head + "\n".join(run_lines) + "\n"


def pack_groups(job_dirs, per_node):
    """
    Chunk job dirs into groups of `per_node` for packed_array_*: group i is
    ONE array task = ONE node running its members side by side. Order is
    preserved, so the caller sorts first — group members share a node and the
    task only ends when its SLOWEST member does, so grouping LIKE with LIKE
    (same stage, same guest) is what keeps a node from idling.
    """
    dirs = list(job_dirs)
    if not dirs:
        raise ValueError("pack_groups: empty job_dirs")
    if int(per_node) < 1:
        raise ValueError("pack_groups: per_node must be >= 1")
    per_node = int(per_node)
    return [dirs[i:i + per_node] for i in range(0, len(dirs), per_node)]


def packed_array_manifest_text(groups):
    """
    Manifest for a PACKED Perlmutter job array: one GROUP per line, its job
    dirs space-separated (task i runs line i, 1-based). Job dirs may not
    contain whitespace — the runner word-splits the line. Write with
    fileio.write_lf as `job_manifest.txt`.
    """
    groups = [list(g) for g in groups]
    if not groups or not any(groups):
        raise ValueError("packed_array_manifest_text: empty groups")
    lines = []
    for g in groups:
        if not g:
            raise ValueError("packed_array_manifest_text: empty group")
        for d in g:
            if not d.strip() or len(d.split()) != 1:
                raise ValueError(
                    "packed_array_manifest_text: bad job dir %r (whitespace "
                    "is the field separator)" % d)
        lines.append(" ".join(g))
    return "\n".join(lines) + "\n"


def packed_array_sbatch_text(job_name, inp, n_groups, per_node, out=None,
                             profile=PERLMUTTER, walltime=None,
                             ntasks_per_job=32, cpus_per_task=2,
                             mem_per_job="110G", qos="regular",
                             manifest="job_manifest.txt", max_running=None,
                             output="cp2k_%A_%a.out", image=None):
    """
    A job array whose every task PACKS `per_node` CP2K jobs onto ONE node,
    each as its own `--exact` srun step at `ntasks_per_job` ranks — the
    node-hour-efficient Perlmutter shape (Foundations Perlmutter migration,
    2026-08-25).

    WHY (measured, not assumed): NERSC bills by NODE-hour, and CP2K scales
    badly past ~32 ranks on our cells. `tests/cp2k_image_parity` ran the SAME
    MOR Si15 energy+force at 32 and at 128 ranks on one Perlmutter node:
    728.8 s vs 540.1 s, i.e. 4x the cores bought 1.35x the speed (34%
    efficiency). Four packed 32-rank jobs therefore cost 4/1.35 = 2.9x FEWER
    node-hours than the same four run one-per-node at 128 ranks, for the same
    science. `array_sbatch_text` (one job per node) remains correct for jobs
    that genuinely need a whole node.

    Each task: read its manifest line (a GROUP of job dirs), launch every
    member as a background subshell that cd's into its dir and sruns CP2K
    there, then `wait`. The task's exit status is nonzero if ANY member
    failed (rule 7 — a packed failure must not hide behind its group).
    Its wall clock is the group's SLOWEST member, so pack like with like.

    inp/out    : the CP2K input each member runs (uniform across the array —
                 that is what makes a group homogeneous); out defaults to
                 inp with a .out suffix.
    n_groups   : task count — an int, or the groups list itself (len used).
    per_node   : members per group; ntasks-per-node = per_node*ntasks_per_job.
    mem_per_job: --mem for each step; with --exact SLURM does not divide node
                 memory for you. 4x110G fits a 512 GB Perlmutter CPU node.

    PERLMUTTER ONLY (same reason as array_sbatch_text: an array cannot carry
    psub's per-job s1/s2 routing).
    Rerun one failed group:  sbatch --array=17 perlmutter_array.sbatch
    """
    if profile.name != "perlmutter":
        raise ValueError(
            "packed_array_sbatch_text is perlmutter-only: a job array cannot "
            "express Pronghorn's psub s1/s2 routing or queue-depth throttle")
    if not isinstance(n_groups, int):
        n_groups = len(n_groups)
    if n_groups < 1:
        raise ValueError("packed_array_sbatch_text: n_groups must be >= 1")
    per_node = int(per_node)
    if per_node < 1:
        raise ValueError("packed_array_sbatch_text: per_node must be >= 1")
    walltime = walltime or profile.default_walltime
    _check_walltime(walltime, profile)
    array_spec = "1-%d" % n_groups
    if max_running is not None:
        array_spec += "%%%d" % max_running
    step_flags = ("--exact --nodes 1 --ntasks %d --cpus-per-task %d "
                  "--cpu-bind=cores --mem=%s"
                  % (ntasks_per_job, cpus_per_task, mem_per_job))
    run_line = cp2k_run_line(inp, out, profile=profile, srun_flags=step_flags)
    return """\
#!/bin/bash
#SBATCH --image %s
#SBATCH --job-name=%s
#SBATCH --array %s
#SBATCH --nodes 1
#SBATCH --ntasks-per-node %d
#SBATCH --cpus-per-task %d
#SBATCH --constraint cpu
#SBATCH --qos %s
#SBATCH --time %s
#SBATCH --output %s
#
# PACKED array: task i runs the WHOLE group on line i of the manifest —
# %d CP2K jobs side by side on ONE node, %d ranks each (node-hour efficient:
# CP2K scales at ~34%% efficiency from 32 to 128 ranks on our cells).
export OMP_NUM_THREADS=1
MANIFEST="${SLURM_SUBMIT_DIR}/%s"
GROUP="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")"
if [ -z "$GROUP" ]; then
    echo "FATAL: no manifest line ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}" >&2
    exit 1
fi
PIDS=""
for JOB_DIR in $GROUP; do
    (
        cd "${SLURM_SUBMIT_DIR}/${JOB_DIR}" || exit 1
        echo "($(date '+%%F %%T')) task ${SLURM_ARRAY_TASK_ID} start [${JOB_DIR}]"
        %s
        rc=$?
        echo "($(date '+%%F %%T')) task ${SLURM_ARRAY_TASK_ID} done [${JOB_DIR}] rc=${rc}"
        exit $rc
    ) &
    PIDS="$PIDS $!"
done
RC=0
for p in $PIDS; do
    wait "$p" || RC=1
done
exit $RC
""" % (image or PERLMUTTER_IMAGE, job_name, array_spec,
       per_node * ntasks_per_job, cpus_per_task, qos, walltime, output,
       per_node, ntasks_per_job, manifest, run_line)


def submit_script_text(job_dirs, sbatch_name=None, profile=PRONGHORN,
                       partition_sh="../../partition.sh", header_note="",
                       image=None):
    """
    A submit-all script for the given profile, one timestamped line per job.
    Pronghorn: partition.sh psub (throttle + s1-first routing) when present,
    else a plain 50-cap squeue-poll throttle + sbatch.
    Perlmutter: plain sbatch — NERSC manages large queues, no manual throttle
    (that's the point of using it for volume).
    Callers should still smoke-test one job before releasing the rest.
    """
    sbatch_name = sbatch_name or profile.sbatch_name
    note = ("# " + header_note + "\n") if header_note else ""
    if profile.name == "perlmutter":
        head = """\
#!/bin/bash
# Generated by zeolib.slurm (perlmutter profile). NERSC manages large queues -
# no manual throttle. Image must be pulled once first:
#   shifterimg pull %s
# SMOKE-TEST one job first. NB: CP2K 2022.1 image - keep provenance separate
# from Pronghorn-2026 energies (see zeolib/slurm.py header).
%sset -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
submit_one() { sbatch "$1"; }

""" % (image or PERLMUTTER_IMAGE, note)
    else:
        head = """\
#!/bin/bash
# Generated by zeolib.slurm. Throttled (max %d concurrent); s1-first routing via
# partition.sh psub when available. SMOKE-TEST one job first.
%sset -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "%s" ]; then
    source "%s"
    submit_one() { psub "$1"; }
else
    echo "WARNING: %s not found - plain sbatch with local throttle (no s1/s2 routing)" >&2
    throttle() {
        while [ "$(squeue -u $USER -h | wc -l)" -ge %d ]; do sleep 60; done
    }
    submit_one() { throttle; sbatch "$1"; }
fi

""" % (MAX_CONCURRENT_JOBS, note, partition_sh, partition_sh, partition_sh,
       MAX_CONCURRENT_JOBS)
    lines = []
    for d in job_dirs:
        lines.append('cd "%s" && submit_one %s && cd "$SCRIPT_DIR" && '
                     'echo "($(date \'+%%F %%T\')) submitted [%s]"' % (d, sbatch_name, d))
    return head + "\n".join(lines) + "\n"


# ── transfer scripts (ship / copy-back) ─────────────────────────────────────
# CP2K restart files: large, numerous, regenerable — never pulled back
# (reference_pronghorn_transfer standard).
COPYBACK_EXCLUDES = (
    "*.wfn", "*.wfn.bak-*", "*-RESTART.wfn", "*-RESTART.wfn.bak-*",
    "*.restart", "*.restart.bak-*",
)

# The transfer scripts are ENVIRONMENT-ADAPTIVE (2026-07-10, feedback: Marcus's
# machine is now Windows Git Bash, which has ssh + tar but NO rsync — the old
# rsync-only scripts silently failed). Each script prefers rsync when present
# (Mac/Linux — keeps incremental transfer) and otherwise falls back to
# tar-over-ssh, which needs only ssh + tar (both in Git Bash, on the cluster,
# and on macOS). Excludes use the same '--exclude=GLOB' syntax for rsync and
# GNU tar, so one list drives both paths.


def _resolve_remote(remote_dir, profile):
    """Absolute remote path: verbatim if remote_dir is absolute, else joined
    onto the profile's remote_base. Raises for a relative dir on a base-less
    profile (PERLMUTTER)."""
    if remote_dir.startswith("/"):
        return remote_dir.rstrip("/")
    base = resolve_remote_base(profile)
    if base:
        return base.rstrip("/") + "/" + remote_dir.strip("/")
    raise ValueError(
        "%s profile has no remote_base — pass an absolute remote_dir"
        % profile.name)


def ship_script_text(remote_dir, profile=PRONGHORN):
    """
    A self-contained `ship.sh` that PUSHES the package it lives in UP to the
    cluster (creating the remote dir). Ship one in every cluster-bound package
    so Marcus just runs `bash ship.sh` — no hand-crafted rsync/scp per machine
    (feedback_windows_transfer). Run LOCALLY with VPN + 2FA up.

    Environment-adaptive: rsync if available, else tar-over-ssh (Git Bash-safe).
    No excludes — a freshly generated package has no wavefunctions/restarts.
    remote_dir: as for copy_back_script_text (absolute, or under remote_base).
    Write with fileio.write_lf(..., executable=True).
    """
    remote_path = _resolve_remote(remote_dir, profile)
    host = resolve_host(profile)
    return ("""\
#!/usr/bin/env bash
# Generated by zeolib.slurm — PUSH this package UP to %s.
# Run LOCALLY (VPN + 2FA already up), from anywhere:  bash ship.sh
# Prefers rsync; falls back to tar-over-ssh where rsync is absent (Git Bash).
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
HOST="%s"
REMOTE="%s"
echo "ship  $SRC"
echo "to    $HOST:$REMOTE/"
ssh "$HOST" "mkdir -p '$REMOTE'"
if command -v rsync >/dev/null 2>&1; then
  rsync -avz "$SRC"/ "$HOST:$REMOTE/"
else
  echo "(rsync not found; using tar over ssh)"
  tar czf - -C "$SRC" . | ssh "$HOST" "tar xzf - -C '$REMOTE'"
fi
""" % (host, host, remote_path))


def copy_back_script_text(remote_dir, profile=PRONGHORN, extra_excludes=(),
                          dest="."):
    """
    A self-contained `copy_back.sh` that PULLS a shipped package's results back
    from the cluster into the local package dir. Ship one in EVERY cluster-bound
    package (feedback_ship_copyback_script) so Marcus just runs `bash
    copy_back.sh` — he never has to ask Claude for the transfer command.

    Direction is cluster -> LOCAL and the script runs on the LOCAL machine:
    Pronghorn/Perlmutter cannot reach the client, so a push from the cluster is
    not possible. VPN + 2FA stay Marcus's manual step
    (reference_pronghorn_transfer); this only removes the "what's the command?"
    round-trip.

    Environment-adaptive: rsync when present (incremental — re-run cheaply as
    jobs finish), else tar-over-ssh (Git Bash-safe). The tar path re-pulls the
    whole tree each run and suppresses "file changed as we read it" warnings so
    a still-running campaign doesn't abort the pull.

    remote_dir : source under the profile's remote_base (e.g.
        "tests/na_placement_multicomp"); pass an absolute path (leading '/') to
        use it verbatim (required for PERLMUTTER, whose base is unset).
    dest       : where results land, resolved relative to the script's OWN
        location at run time (default '.', i.e. merge back into the package dir),
        so it works from Mac or Windows Git Bash wherever the tree lives.
    extra_excludes : extra globs on top of the CP2K restart excludes, for other
        large/regenerable outputs ([[feedback-large-file-cleanup]]).

    Targeted path, WITHOUT --delete (never clobber the diverged remote).
    Write the returned text with fileio.write_lf(..., executable=True).
    """
    remote_path = _resolve_remote(remote_dir, profile)
    host = resolve_host(profile)
    excludes = tuple(COPYBACK_EXCLUDES) + tuple(extra_excludes)
    exc_rsync = " \\\n".join("    --exclude='%s'" % e for e in excludes)
    exc_tar = " \\\n".join("    --exclude='%s'" % e for e in excludes)
    return ("""\
#!/usr/bin/env bash
# Generated by zeolib.slurm — PULL results back from %s into this package.
# Run LOCALLY (VPN + 2FA already up), from anywhere:  bash copy_back.sh
# Never deletes; skips CP2K restart files (large/regenerable). Prefers rsync;
# falls back to tar-over-ssh where rsync is absent (Git Bash).
set -euo pipefail
DEST="$(cd "$(dirname "$0")" && cd %s && pwd)"
HOST="%s"
REMOTE="%s"
echo "pull  $HOST:$REMOTE/"
echo "into  $DEST"
if command -v rsync >/dev/null 2>&1; then
  rsync -avz \\
%s \\
    "$HOST:$REMOTE/" "$DEST"/
else
  echo "(rsync not found; using tar over ssh — e.g. Git Bash on Windows)"
  ssh "$HOST" "tar czf - -C '$REMOTE' --warning=no-file-changed --ignore-failed-read \\
%s \\
    ." | tar xzf - -C "$DEST"
fi
""" % (host, dest, host, remote_path, exc_rsync, exc_tar))
