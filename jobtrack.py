"""
jobtrack.py — the program-wide cluster job ledger: what is running, when it
should finish, and which drained waves never came home.

WHY (2026-09-27, Marcus): "once the job drains and I run squeue, it's empty
and I struggle to remember which thread I need to pick up." squeue forgets a
job the moment it ends; the Ops Board remembers, but costs a Claude session to
refresh. This module needs no Claude at all: ONE ssh (password + OTP once)
runs `sacct` + `squeue` on the cluster, and everything else — grouping jobs
into threads, finish estimates, and checking which results are already on
this machine — happens locally from files.

Design decisions, each for a reason:
  * The record is `sacct`, not a submit-time log. sacct's WorkDir names the
    package a job belongs to, so waves submitted by any route (psub loops,
    hand resubmits, rescue scripts) are all seen, with nothing to remember.
    A local ledger (`ledger.json`) merges every snapshot, so history outlives
    sacct's window.
  * A THREAD is the package dir: the nearest local ancestor of the job's dir
    that holds a `copy_back.sh`. That is the unit Marcus pulls, and the
    package's NOTES.md "Next:" line is the resume instruction shown with it.
  * "Came home" is decided from the LOCAL files, not from memory: a job is
    `home` when its local dir holds a non-input file written at or after the
    job's own end (mtimes survive both rsync -a and tar). The Slurm log name
    (cp2k_<id>.out) is NOT used — targeted pulls (f5 ladder, 2026-09-17)
    bring energy-force.out without it, so the log name under-counts.
  * Only the LATEST job per WorkDir counts; resubmits supersede earlier jobs.
  * A snapshot without its `#END` marker is refused (a dropped connection
    must never read as "all jobs vanished").
  * Remote workdirs map to local paths by prefix rules held in the tracker's
    `config.json` (site paths are configuration, never in this public repo —
    slurm.py "Cluster identity").

Entry point: `bash jobtrack/jobs.sh` (fetch + report) or
`python -m zeolib.jobtrack report` (offline: re-check what is home).
"""

import html
import json
import os
import re
import statistics
import sys
import time

FORMAT_TAG = "#ZEOJT 1"
ACTIVE = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "REQUEUED",
          "RESIZING", "SUSPENDED", "REQUEUE_HOLD", "REQUEUE_FED"}
BAD = {"FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL",
       "DEADLINE", "PREEMPTED", "CANCELLED", "REVOKED"}
# Files a package ships or a human edits: never evidence that a job's output
# came home. Everything else in a job dir is a product.
INPUT_EXT = (".inp", ".sbatch", ".sh", ".py", ".md", ".inc", ".json")
INPUT_NAMES = {"BASIS", "POTENTIALS", "dftd3.dat", "coords.xyz",
               "ZEOLIB_VERSION.json", "PRODUCT_INFO"}
HOME_SLACK = 1800   # s: the last output write precedes Slurm's End by seconds
                    # (epilog/cleanup); 30 min absorbs that and clock drift
START_SLACK = 120
SACCT_FIELDS = ("JobIDRaw,JobID,JobName,Partition,State,Submit,Start,End,"
                "Elapsed,Timelimit,NodeList,ExitCode,WorkDir")


# ── remote side ─────────────────────────────────────────────────────────────

def remote_script(days=21):
    """Bash run ON the cluster (`ssh host bash -s`). Prints one snapshot.
    Times as epoch seconds via SLURM_TIME_FORMAT so no timezone guessing;
    the parser still accepts ISO stamps if a site ignores it."""
    return r"""export SLURM_TIME_FORMAT=%%s
echo '%(tag)s'
echo "#NOW $(date +%%s)"
echo "#HOST $(hostname)"
echo "#DAYS %(days)d"
echo '#SACCT'
sacct -u "$USER" -X -P -n -S "$(date -d '-%(days)d days' +%%F)" -E now -o %(fields)s
echo '#SQUEUE'
squeue -u "$USER" -h -o '%%A|%%T|%%S|%%r'
echo '#LOOPS'
for p in $(pgrep -u "$USER" -f 'submit[^ ]*\.sh|psub' 2>/dev/null); do
  echo "$p|$(ps -o etimes= -p "$p" 2>/dev/null | tr -d ' ')|$(readlink "/proc/$p/cwd" 2>/dev/null)|$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)"
done
echo '#END'
""" % {"tag": FORMAT_TAG, "days": int(days), "fields": SACCT_FIELDS}


# ── parsing ─────────────────────────────────────────────────────────────────

def parse_time(s):
    """Epoch int from an epoch string or ISO stamp; None for Unknown/None/N/A."""
    s = (s or "").strip()
    if not s or s in ("Unknown", "None", "N/A", "0"):
        return None
    if s.isdigit():
        return int(s)
    try:
        return int(time.mktime(time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")))
    except ValueError:
        return None


def parse_duration(s):
    """Seconds from Slurm's [D-]HH:MM:SS / MM:SS / MM.SS forms; None if
    UNLIMITED/Partition_Limit/blank."""
    s = (s or "").strip()
    m = re.match(r"^(?:(\d+)-)?(\d+):(\d+)(?::(\d+))?(?:\.\d+)?$", s)
    if not m:
        return None
    d, a, b, c = m.groups()
    if c is None:           # MM:SS, or D-HH:MM
        h, mi, se = (int(a), int(b), 0) if d else (0, int(a), int(b))
    else:
        h, mi, se = int(a), int(b), int(c)
    return int(d or 0) * 86400 + h * 3600 + mi * 60 + se


def parse_snapshot(text):
    """Snapshot text -> dict(now, host, days, jobs{id: rec}, loops[]).
    Raises ValueError on a truncated or foreign snapshot."""
    lines = text.splitlines()
    if FORMAT_TAG not in [l.strip() for l in lines]:
        raise ValueError("not a jobtrack snapshot (no %r line)" % FORMAT_TAG)
    if "#END" not in [l.strip() for l in lines]:
        raise ValueError("snapshot is TRUNCATED (no #END) — refusing to ingest; "
                         "a dropped link must not read as 'all jobs gone'")
    snap = {"now": None, "host": "", "days": None, "jobs": {}, "loops": []}
    queue = {}
    sec = None
    for raw in lines:
        l = raw.rstrip("\r")
        if l.startswith("#"):
            key, _, val = l.partition(" ")
            if key == "#NOW":
                snap["now"] = parse_time(val)
            elif key == "#HOST":
                snap["host"] = val.strip()
            elif key == "#DAYS":
                snap["days"] = int(val)
            else:
                sec = key
            continue
        if sec == "#SACCT":
            f = l.split("|")
            if len(f) < 13 or not f[0].strip().isdigit():
                continue          # module-load chatter etc.
            wd = "|".join(f[12:]).strip()
            snap["jobs"][f[0].strip()] = {
                "id": f[0].strip(), "jobid": f[1], "name": f[2],
                "partition": f[3], "state": f[4].split()[0] if f[4] else "",
                "submit": parse_time(f[5]), "start": parse_time(f[6]),
                "end": parse_time(f[7]), "elapsed": parse_duration(f[8]),
                "limit": parse_duration(f[9]), "nodes": f[10],
                "exit": f[11], "workdir": wd.rstrip("/")}
        elif sec == "#SQUEUE":
            f = l.split("|")
            if len(f) >= 4 and f[0].strip().isdigit():
                queue[f[0].strip()] = {"state": f[1], "est_start": parse_time(f[2]),
                                       "reason": f[3]}
        elif sec == "#LOOPS":
            f = l.split("|", 3)
            if len(f) == 4 and f[0].strip().isdigit():
                snap["loops"].append({"pid": f[0], "etime": f[1],
                                      "cwd": f[2].rstrip("/"), "cmd": f[3].strip()})
    for jid, q in queue.items():
        j = snap["jobs"].get(jid)
        if j is not None:
            j["state"] = q["state"] or j["state"]
            if j["state"] == "PENDING":
                j["est_start"] = q["est_start"]
                j["reason"] = q["reason"]
    if snap["now"] is None:
        raise ValueError("snapshot has no #NOW")
    return snap


# ── ledger ──────────────────────────────────────────────────────────────────

def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (IOError, OSError, ValueError):
        return default


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def merge(ledger, snap, cluster="pronghorn"):
    """Fold a snapshot into the ledger. Jobs absent from the snapshot keep
    their last known record (they aged out of sacct's window); jobs the
    snapshot knows overwrite theirs, keeping any `home` verdict."""
    jobs = ledger.setdefault("jobs", {})
    for jid, rec in snap["jobs"].items():
        old = jobs.get(jid, {})
        rec = dict(rec, cluster=cluster, seen=snap["now"])
        if old.get("home") and old.get("home_end") == rec.get("end"):
            rec["home"], rec["home_end"] = old["home"], old["home_end"]
        jobs[jid] = rec
    ledger["last_fetch"] = snap["now"]
    if snap.get("days"):   # earliest instant any snapshot could see
        cs = snap["now"] - snap["days"] * 86400
        ledger["coverage_start"] = min(ledger.get("coverage_start", cs), cs)
    ledger["host"] = snap["host"]
    ledger["loops"] = snap["loops"]
    return ledger


# ── local mapping + home check ──────────────────────────────────────────────

def to_local(workdir, prefixes, root):
    """Remote dir -> local path via the first matching (remote_prefix,
    local_rel) rule, or None."""
    for rp, lp in prefixes:
        rp = rp.rstrip("/") + "/"
        if (workdir + "/").startswith(rp):
            rel = workdir[len(rp):]
            return os.path.normpath(os.path.join(root, lp, rel))
    return None


def find_package(local_dir, root):
    """Nearest ancestor (inclusive) holding copy_back.sh, stopping at root."""
    root = os.path.normpath(root)
    d = os.path.normpath(local_dir)
    while d.startswith(root) and d != root:
        if os.path.isfile(os.path.join(d, "copy_back.sh")):
            return d
        d = os.path.dirname(d)
    return None


def _products(local_dir, since):
    """Newest mtime among non-input files at or after `since`, else None."""
    try:
        names = os.listdir(local_dir)
    except OSError:
        return None
    best = None
    for n in names:
        if n in INPUT_NAMES or n.endswith(INPUT_EXT):
            continue
        try:
            st = os.stat(os.path.join(local_dir, n))
        except OSError:
            continue
        if st.st_mtime >= since and (best is None or st.st_mtime > best):
            best = st.st_mtime
    return best


def home_state(job, local_dir):
    """'home' | 'partial' | 'missing' | 'nodir' | 'n/a' (never started)."""
    if job.get("start") is None:
        return "n/a"
    if local_dir is None or not os.path.isdir(local_dir):
        return "nodir"
    newest = _products(local_dir, job["start"] - START_SLACK)
    if newest is None:
        return "missing"
    end = job.get("end")
    if end is None or job.get("state") in ACTIVE:
        return "partial"
    return "home" if newest >= end - HOME_SLACK else "partial"


def next_step(pkg_dir, root):
    """The last 'Next:' line of the nearest NOTES.md (package dir or up to two
    parents) — the thread's own resume instruction. '' when none."""
    d = pkg_dir
    for _ in range(3):
        p = os.path.join(d, "NOTES.md")
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8", errors="replace") as fh:
                    hits = [l.strip() for l in fh
                            if re.search(r"\bNext:", l)]
            except OSError:
                hits = []
            if hits:
                return re.sub(r"^.*?\bNext:\s*", "", hits[-1])[:240]
            return ""
        if os.path.normpath(d) == os.path.normpath(root):
            break
        d = os.path.dirname(d)
    return ""


def submit_dirs(pkg_dir):
    """{script_name: (mtime, [subdir,...])} from the package's generated
    submit*.sh (`cd "X"` lines) — what each wave is meant to contain. The
    mtime matters: rescue scripts re-target dirs an earlier wave already used,
    so only a job submitted AFTER the script was written counts for it."""
    out = {}
    try:
        names = sorted(n for n in os.listdir(pkg_dir)
                       if n.startswith("submit") and n.endswith(".sh"))
    except OSError:
        return out
    for n in names:
        try:
            fp = os.path.join(pkg_dir, n)
            with open(fp, encoding="utf-8", errors="replace") as fh:
                txt = fh.read()
            mt = os.path.getmtime(fp)
        except OSError:
            continue
        dirs = list(dict.fromkeys(re.findall(r'cd "([^"$]+)"', txt)))
        if dirs:
            out[n] = (mt, dirs)
    return out


# ── threads ─────────────────────────────────────────────────────────────────

ABORT_SECONDS = 600


def _aborted(j):
    """CANCELLED before doing real work (never started, or < 10 min)."""
    return (j.get("state") == "CANCELLED"
            and (j.get("start") is None or (j.get("elapsed") or 0) < ABORT_SECONDS))


def _loop_targets(loop):
    """Remote dirs a live submit loop is working through: its cwd plus every
    path-like token of its command line (a combined loop like
    `for p in f3_rebuild/pkg_A f3_rebuild/pkg_B; do bash $p/submit...` runs
    from the parent dir and names its packages only in the command)."""
    import posixpath
    cwd = loop.get("cwd", "")
    out = {cwd}
    for tok in re.split(r"[\s;'\"]+", loop.get("cmd", "")):
        if "/" in tok and not tok.startswith("-") and "$" not in tok.split("/")[0]:
            full = posixpath.normpath(posixpath.join(cwd, tok))
            out.add(posixpath.dirname(full) if tok.endswith(".sh") else full)
    return out


def build_threads(ledger, prefixes, root, now=None, acks=None, check_home=True):
    """Group the ledger's jobs into threads (packages) and classify each.
    Returns a list of thread dicts, most urgent first."""
    now = now or time.time()
    acks = acks or {}
    latest = {}
    for j in ledger.get("jobs", {}).values():
        wd = j.get("workdir") or "?"
        cur = latest.get(wd)
        # a quick-cancelled job is an aborted duplicate (the 2026-09-21 probe
        # resubmit, cancelled at ~30 s): it never outranks a real job
        key = (not _aborted(j), j.get("submit") or 0, int(j["id"]))
        if cur is None or key > (not _aborted(cur), cur.get("submit") or 0,
                                 int(cur["id"])):
            latest[wd] = j
    threads = {}
    pkg_cache = {}
    for wd, j in latest.items():
        loc = to_local(wd, prefixes, root)
        if loc is not None and loc not in pkg_cache:
            pkg_cache[loc] = find_package(loc, root)
        pkg = pkg_cache.get(loc) if loc else None
        if pkg:
            key = os.path.relpath(pkg, root).replace(os.sep, "/")
            rel = os.path.relpath(loc, pkg).replace(os.sep, "/")
            remote_pkg = wd[:len(wd) - len(rel)].rstrip("/") if rel != "." else wd
        else:
            key = "(unmapped) " + os.path.dirname(wd)
            remote_pkg = os.path.dirname(wd)
        t = threads.setdefault(key, {"key": key, "pkg": pkg, "remote": remote_pkg,
                                     "jobs": []})
        if check_home and not j.get("home") == "home":
            st = home_state(j, loc)
            if st == "home":
                j["home"], j["home_end"] = "home", j.get("end")
            j["home_now"] = st
        else:
            j["home_now"] = j.get("home") or "?"
        t["jobs"].append(j)
    out = []
    for t in threads.values():
        _classify(t, ledger, now, root)
        a = acks.get(t["key"])
        t["acked"] = bool(a and a.get("upto", 0) >= t["last_activity"])
        t["ack_note"] = a.get("note", "") if a else ""
        out.append(t)
    rank = {"ATTENTION": 0, "PULL": 1, "RUNNING": 2, "HOME": 3}
    out.sort(key=lambda t: (t["acked"], rank[t["status"]], -t["last_activity"]))
    return out


def _classify(t, ledger, now, root):
    js = t["jobs"]
    c = {"running": 0, "pending": 0, "done": 0, "bad": 0, "home": 0,
         "partial": 0, "missing": 0}
    for j in js:
        s = j.get("state", "")
        if s == "RUNNING" or s == "COMPLETING":
            c["running"] += 1
        elif s in ACTIVE:
            c["pending"] += 1
        elif s in BAD:
            c["bad"] += 1
        else:
            c["done"] += 1
        h = j.get("home_now")
        if s not in ACTIVE and j.get("start") is not None:
            if h == "home":
                c["home"] += 1
            elif h == "partial":
                c["partial"] += 1
            else:
                c["missing"] += 1
    t["counts"] = c
    ended = [j["end"] for j in js if j.get("end") and j.get("state") not in ACTIVE]
    t["last_end"] = max(ended) if ended else None
    t["first_submit"] = min((j.get("submit") or now) for j in js)
    t["last_activity"] = max([j.get("submit") or 0 for j in js] + ended)
    # typical runtime from this thread's own completed jobs
    runs = [j["elapsed"] for j in js if j.get("state") == "COMPLETED"
            and j.get("elapsed")]
    typ = statistics.median(runs) if len(runs) >= 3 else None
    likely, bound = [], []
    for j in js:
        s = j.get("state")
        if s in ("RUNNING", "COMPLETING") and j.get("start"):
            if j.get("limit"):
                bound.append(j["start"] + j["limit"])
            if typ:
                likely.append(j["start"] + typ)
        elif s in ACTIVE:
            st = j.get("est_start")
            if st and j.get("limit"):
                bound.append(st + j["limit"])
            if st and typ:
                likely.append(st + typ)
    t["eta_likely"] = max(likely) if likely else None
    t["eta_bound"] = max(bound) if bound else None
    t["typical_run"] = typ
    t["pending_unestimated"] = sum(1 for j in js if j.get("state") == "PENDING"
                                   and not j.get("est_start"))
    # waves still inside a submit loop (never handed to Slurm yet)
    t["unsubmitted"] = []
    recent = t["last_activity"] >= now - 3 * 86400
    t["unsubmitted_info"] = []
    if t["pkg"]:
        subs = {}
        for j in ledger.get("jobs", {}).values():   # ALL jobs, not just latest
            wd = j.get("workdir")
            if wd and wd.startswith(t["remote"] + "/"):
                subs[wd] = max(subs.get(wd, 0), j.get("submit") or 0)
        for script, (mt, dirs) in submit_dirs(t["pkg"]).items():
            rds = [t["remote"] + "/" + d.strip("/") for d in dirs]
            hit = sum(1 for r in rds if subs.get(r, 0) >= mt - 60)
            if not 0 < hit < len(rds) or mt < ledger.get("coverage_start", 0):
                continue          # wave not started, or older than sacct's view
            # only a CURRENT wave can have a live-or-dead loop; on an old
            # thread the gap is usually work done elsewhere or skipped by a
            # DONE guard, so it is shown but does not raise ATTENTION
            (t["unsubmitted"] if recent or c["running"] + c["pending"]
             else t["unsubmitted_info"]).append((script, len(rds) - hit, len(rds)))
    rp = t["remote"]
    t["loops"] = [l for l in ledger.get("loops", [])
                  if any(x == rp or x.startswith(rp + "/") for x in _loop_targets(l))]
    t["next"] = next_step(t["pkg"], root) if t["pkg"] else ""
    active = c["running"] + c["pending"]
    waiting = sum(n for _, n, _ in t["unsubmitted"])
    if active or (waiting and t["loops"]):
        t["status"] = "RUNNING"
    elif c["bad"] or (waiting and not t["loops"]):
        t["status"] = "ATTENTION"
    elif c["missing"] or c["partial"]:
        t["status"] = "PULL"
    else:
        t["status"] = "HOME"


# ── rendering ───────────────────────────────────────────────────────────────

def _t(ts, with_time=True):
    if not ts:
        return "?"
    return time.strftime("%a %m-%d %H:%M" if with_time else "%a %m-%d",
                         time.localtime(ts))


def _ago(ts, now):
    if not ts:
        return "?"
    d = now - ts
    if d < 0:
        return "in %s" % _span(-d)
    return "%s ago" % _span(d)


def _span(sec):
    sec = int(sec)
    if sec >= 86400:
        return "%.1f d" % (sec / 86400.0)
    if sec >= 3600:
        return "%.1f h" % (sec / 3600.0)
    return "%d min" % (sec // 60)


LABEL = {"ATTENTION": "⚠ NEEDS A LOOK", "PULL": "⬇ READY TO PULL",
         "RUNNING": "▶ RUNNING / QUEUED", "HOME": "✓ HOME"}


def thread_lines(t, now):
    c = t["counts"]
    n = len(t["jobs"])
    parts = []
    if c["running"]:
        parts.append("%d running" % c["running"])
    if c["pending"]:
        parts.append("%d queued" % c["pending"])
    fin = c["done"] + c["bad"]
    if fin:
        parts.append("%d finished (%d home%s)" % (
            fin, c["home"], ", %d partial" % c["partial"] if c["partial"] else ""))
    if c["bad"]:
        bad = {}
        for j in t["jobs"]:
            if j.get("state") in BAD:
                bad[j["state"]] = bad.get(j["state"], 0) + 1
        parts.append("FAILED: " + ", ".join("%d %s" % (v, k) for k, v in sorted(bad.items())))
    out = ["%s  [%d job%s]" % (t["key"], n, "" if n == 1 else "s"),
           "    " + "; ".join(parts)]
    for script, left, tot in t["unsubmitted"]:
        live = "loop alive" if t["loops"] else "NO submit loop seen"
        out.append("    %s: %d of %d not yet handed to Slurm (%s)" % (script, left, tot, live))
    for script, left, tot in t.get("unsubmitted_info", []):
        out.append("    (%s: %d of %d dirs never ran from here — elsewhere, or skipped)"
                   % (script, left, tot))
    if t["status"] == "RUNNING":
        eta = []
        if t["eta_likely"] and t["eta_likely"] >= now:
            eta.append("likely ~%s" % _t(t["eta_likely"]))
        elif t["eta_likely"]:
            eta.append("running past this thread's median job (%s)"
                       % _span(t["typical_run"]))
        if t["eta_bound"]:
            eta.append("walltime bound %s" % _t(t["eta_bound"]))
        if t["pending_unestimated"]:
            eta.append("%d queued with no start estimate" % t["pending_unestimated"])
        if eta:
            out.append("    finish: " + "; ".join(eta))
    elif t["last_end"]:
        out.append("    drained %s (%s)" % (_ago(t["last_end"], now), _t(t["last_end"])))
    if t["pkg"] and (c["missing"] or c["partial"]):
        out.append("    pull: bash %s/copy_back.sh   (%d finished job%s not home)" % (
            t["key"], c["missing"] + c["partial"],
            "" if c["missing"] + c["partial"] == 1 else "s"))
    if t["next"]:
        out.append("    next (NOTES.md): " + t["next"])
    if t["acked"]:
        out.append("    (acknowledged%s)" % (": " + t["ack_note"] if t["ack_note"] else ""))
    return out


def render_text(threads, ledger, now, home_days=7):
    lf = ledger.get("last_fetch")
    head = ["Cluster jobs — as of %s (%s)%s" % (
        _t(lf), _ago(lf, now), "  ⚠ STALE: re-run jobs.sh" if lf and now - lf > 12 * 3600 else "")]
    body = []
    for l in ledger.get("loops", []):
        body.append("submit loop alive on %s for %s: %s" % (
            ledger.get("host", "?"), _span(int(l.get("etime") or 0)), l["cmd"][:200]))
    for status in ("ATTENTION", "PULL", "RUNNING", "HOME"):
        ts = [t for t in threads if t["status"] == status and not t["acked"]]
        if status == "HOME":
            ts = [t for t in ts if t["last_activity"] >= now - home_days * 86400]
        if not ts:
            continue
        body.append("")
        body.append("%s (%d)" % (LABEL[status], len(ts)))
        for t in ts:
            body.extend(thread_lines(t, now))
    acked = [t for t in threads if t["acked"]]
    if acked:
        body.append("")
        body.append("(%d acknowledged thread(s) hidden — `report --all` shows them)" % len(acked))
    if not body:
        body = ["", "No jobs in the ledger yet."]
    return "\n".join(head + body) + "\n"


def render_html(threads, ledger, now, home_days=7):
    lf = ledger.get("last_fetch")
    stale = lf and now - lf > 12 * 3600
    esc = html.escape
    sec = []
    tally = {}
    for status in ("ATTENTION", "PULL", "RUNNING", "HOME"):
        ts = [t for t in threads if t["status"] == status and not t["acked"]]
        if status == "HOME":
            ts = [t for t in ts if t["last_activity"] >= now - home_days * 86400]
        tally[status] = len(ts)
        if not ts:
            continue
        cards = []
        for t in ts:
            ls = thread_lines(t, now)
            cards.append('<div class="card %s"><div class="k">%s</div><pre>%s</pre></div>'
                         % (status.lower(), esc(ls[0]), esc("\n".join(l[4:] for l in ls[1:]))))
        sec.append('<h2 class="%s">%s <span>%d</span></h2>%s'
                   % (status.lower(), esc(LABEL[status]), len(ts), "".join(cards)))
    chips = "".join('<div class="chip %s"><b>%d</b>%s</div>' % (s.lower(), tally.get(s, 0), esc(LABEL[s]))
                    for s in ("ATTENTION", "PULL", "RUNNING", "HOME"))
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cluster Jobs</title><style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e2e1dc;
--att:#b3261e;--pull:#9a5b00;--run:#1f5fa8;--home:#2e7d32}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9993;
--card:#20201e;--line:#34332f;--att:#f2877f;--pull:#e7b566;--run:#8ab8f0;--home:#8fcf93}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:980px;margin:0 auto}h1{font-size:20px;margin:0 0 4px}
.asof{color:var(--mut);margin-bottom:16px}.stale{color:var(--att);font-weight:600}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px}
.chip{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 12px}
.chip b{font-size:18px;margin-right:6px}
h2{font-size:15px;margin:22px 0 8px;letter-spacing:.02em}h2 span{color:var(--mut);font-weight:400}
.card{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);
border-radius:6px;padding:10px 12px;margin:8px 0}
.card .k{font-weight:600;word-break:break-all}
.card pre{margin:4px 0 0;white-space:pre-wrap;word-break:break-word;font:13px/1.45 ui-monospace,Consolas,monospace;color:var(--fg)}
.attention{border-left-color:var(--att)}h2.attention,.chip.attention b{color:var(--att)}
.pull{border-left-color:var(--pull)}h2.pull,.chip.pull b{color:var(--pull)}
.running{border-left-color:var(--run)}h2.running,.chip.running b{color:var(--run)}
.home{border-left-color:var(--home)}h2.home,.chip.home b{color:var(--home)}
footer{color:var(--mut);font-size:13px;margin-top:28px}
</style></head><body><main>
<h1>Cluster jobs</h1><div class="asof">Snapshot %s (%s) from %s%s</div>
<div class="chips">%s</div>%s
<footer>Generated by zeolib.jobtrack %s. Refresh: <code>bash jobtrack/jobs.sh</code> (one ssh) ·
after a copy_back: <code>bash jobtrack/jobs.sh --offline</code> (no ssh).</footer>
</main></body></html>
""" % (esc(_t(lf)), esc(_ago(lf, now)), esc(ledger.get("host", "?")),
       ' <span class="stale">— STALE, re-run jobs.sh</span>' if stale else "",
       chips, "".join(sec) or "<p>No jobs in the ledger yet.</p>", esc(_t(now)))


# ── CLI ─────────────────────────────────────────────────────────────────────

def default_config(root):
    """First-run config: the PRONGHORN base from zeolib.slurm's configured
    identity, with the FoundationsCampaign/ remote subtree mapping to the
    repo root (how Foundations packages are shipped)."""
    from zeolib import slurm
    base = slurm.resolve_remote_base(slurm.PRONGHORN).rstrip("/")
    return {"prefixes": [[base + "/FoundationsCampaign/", ""], [base + "/", ""]],
            "home_days": 7}


def report(tdir, root, show_all=False, quiet=False):
    ledger = load_json(os.path.join(tdir, "ledger.json"), {"jobs": {}})
    cfg_path = os.path.join(tdir, "config.json")
    cfg = load_json(cfg_path, None)
    if cfg is None:
        cfg = default_config(root)
        save_json(cfg_path, cfg)
    acks = {} if show_all else load_json(os.path.join(tdir, "acks.json"), {})
    now = time.time()
    threads = build_threads(ledger, cfg["prefixes"], root, now=now, acks=acks)
    save_json(os.path.join(tdir, "ledger.json"), ledger)   # keeps `home` verdicts
    text = render_text(threads, ledger, now, cfg.get("home_days", 7))
    with open(os.path.join(tdir, "STATUS.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("```\n" + text + "```\n")
    with open(os.path.join(tdir, "jobs.html"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_html(threads, ledger, now, cfg.get("home_days", 7)))
    if not quiet:
        sys.stdout.write(text)
    return threads


def ingest(tdir, root, snap_path):
    with open(snap_path, encoding="utf-8", errors="replace") as fh:
        snap = parse_snapshot(fh.read())
    lp = os.path.join(tdir, "ledger.json")
    ledger = merge(load_json(lp, {"jobs": {}}), snap)
    save_json(lp, ledger)
    return snap


def main(argv=None):
    import argparse
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(prog="python -m zeolib.jobtrack")
    ap.add_argument("--dir", default=os.path.join(here, "jobtrack"),
                    help="tracker dir (ledger, config, reports)")
    ap.add_argument("--root", default=here, help="local repo root (Zeolites/)")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("remote-script"); r.add_argument("--days", type=int, default=21)
    i = sub.add_parser("ingest"); i.add_argument("snapshot")
    p = sub.add_parser("report"); p.add_argument("--all", action="store_true")
    a = sub.add_parser("ack", help="hide a thread until it gets new jobs")
    a.add_argument("thread"); a.add_argument("--note", default="")
    args = ap.parse_args(argv)
    os.makedirs(args.dir, exist_ok=True)
    try:   # Windows consoles default to cp1252, which cannot print the ⚠/▶ marks
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    if args.cmd == "remote-script":
        # bytes, not text: Windows text mode would emit CRLF into remote bash
        sys.stdout.buffer.write(remote_script(args.days).encode("utf-8"))
    elif args.cmd == "ingest":
        snap = ingest(args.dir, args.root, args.snapshot)
        print("ingested %d jobs from %s" % (len(snap["jobs"]), snap["host"]))
        report(args.dir, args.root)
    elif args.cmd == "report":
        report(args.dir, args.root, show_all=args.all)
    elif args.cmd == "ack":
        ap_ = os.path.join(args.dir, "acks.json")
        acks = load_json(ap_, {})
        acks[args.thread] = {"upto": time.time(), "note": args.note}
        save_json(ap_, acks)
        print("acknowledged %s — hidden until it gets new jobs" % args.thread)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
