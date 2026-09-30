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
  * A home thread is split (2026-09-29): HOME, NOT PROCESSED until its
    NOTES.md carries a `Conclusion (YYYY-MM-DD):` dated on/after the last
    job's End, or its work dir gains a non-.md file written after that End
    (`processed_evidence` says why NOTES edits alone do not count). Cards
    show the NOTES.md `Why:` line (else its title) and the Conclusion.
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

LOGIN_NODE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def remote_script(days=21, login_nodes=()):
    """Bash run ON the cluster (`ssh host bash -s`). Prints one snapshot.
    Times as epoch seconds via SLURM_TIME_FORMAT so no timezone guessing;
    the parser still accepts ISO stamps if a site ignores it.

    SUBMIT LOOPS ON EVERY LOGIN NODE (2026-09-30). A process list only shows
    the node the ssh landed on, and Pronghorn round-robins login-0/login-1:
    three 09-29/30 snapshots from login-0 saw NO loop while the probe loop ran
    on login-1 the whole time, and the 09:48 one from login-1 missed the two
    rebuild loops on login-0. So the loop probe runs locally AND over
    `ssh -n -o BatchMode=yes` on each other node in `login_nodes` (the
    tracker's config.json — site identity stays out of this public repo;
    Marcus verified node-to-node ssh needs no prompt, 2026-09-30). `-n` is
    load-bearing: this script arrives on stdin, and an ssh without it would
    swallow the rest of it. Each node reports `#LOOPHOST <node> local|ok|
    UNREACHABLE`, so an unchecked node is SAID, never silently empty. The
    pgrep pattern is bracketed (`submi[t]`, `psu[b]`) so the probe's own
    command lines (bash -c / ssh carrying the pattern text) never match it.
    Lines are `L|host|pid|ppid|etime|cwd|cmd`; login-shell chatter is
    filtered by the `L|` prefix."""
    nodes = [n for n in login_nodes if n]
    bad = [n for n in nodes if not LOGIN_NODE_RE.match(n)]
    if bad:
        raise ValueError("login_nodes: refusing unsafe node name(s) %r" % bad)
    return r"""export SLURM_TIME_FORMAT=%%s
echo '%(tag)s'
echo "#NOW $(date +%%s)"
echo "#HOST $(hostname)"
echo "#DAYS %(days)d"
echo '#SACCT'
sacct -u "$USER" -X -P -n -S "$(date -d '-%(days)d days' +%%F)" -E now -o %(fields)s
echo '#SQUEUE'
squeue -u "$USER" -h -o '%%A|%%T|%%S|%%r'
JT_LOOPS='for p in $(pgrep -u "$USER" -f "submi[t][^ ]*\.sh|psu[b]" 2>/dev/null); do echo "L|$(hostname -s)|$p|$(ps -o ppid= -p "$p" 2>/dev/null | tr -d " ")|$(ps -o etimes= -p "$p" 2>/dev/null | tr -d " ")|$(readlink "/proc/$p/cwd" 2>/dev/null)|$(tr "\0" " " < "/proc/$p/cmdline" 2>/dev/null)"; done'
JT_SELF=$(hostname -s)
echo '#LOOPS2'
echo "#LOOPHOST $JT_SELF local"
bash -c "$JT_LOOPS"
for h in %(nodes)s; do
  [ "$h" = "$JT_SELF" ] && continue
  if JT_OUT=$(ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$h" "$JT_LOOPS" 2>/dev/null); then
    echo "#LOOPHOST $h ok"
    printf '%%s\n' "$JT_OUT" | grep '^L|'
  else
    echo "#LOOPHOST $h UNREACHABLE"
  fi
done
echo '#END'
""" % {"tag": FORMAT_TAG, "days": int(days), "fields": SACCT_FIELDS,
       "nodes": " ".join(nodes)}


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
    snap = {"now": None, "host": "", "days": None, "jobs": {}, "loops": [],
            "loop_hosts": {}}
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
            elif key == "#LOOPHOST":
                h, _, st = val.strip().partition(" ")
                snap["loop_hosts"][h] = st.strip()
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
        elif sec == "#LOOPS":       # pre-2026-09-30 snapshots: one node, no ppid
            f = l.split("|", 3)
            if len(f) == 4 and f[0].strip().isdigit():
                snap["loops"].append({"pid": f[0], "etime": f[1],
                                      "cwd": f[2].rstrip("/"), "cmd": f[3].strip()})
        elif sec == "#LOOPS2":
            f = l.split("|", 6)
            if len(f) == 7 and f[0] == "L" and f[2].strip().isdigit():
                snap["loops"].append({"host": f[1].strip(), "pid": f[2].strip(),
                                      "ppid": f[3].strip(), "etime": f[4].strip(),
                                      "cwd": f[5].rstrip("/"), "cmd": f[6].strip()})
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
    ledger["loop_hosts"] = snap.get("loop_hosts", {})
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


def notes_path(pkg_dir, root):
    """The thread's NOTES.md: nearest in the package dir or up to two parents
    (never above root), else None."""
    d = pkg_dir
    for _ in range(3):
        p = os.path.join(d, "NOTES.md")
        if os.path.isfile(p):
            return p
        if os.path.normpath(d) == os.path.normpath(root):
            break
        d = os.path.dirname(d)
    return None


def _notes_lines(pkg_dir, root):
    p = notes_path(pkg_dir, root)
    if not p:
        return []
    try:
        with open(p, encoding="utf-8", errors="replace") as fh:
            return fh.read().splitlines()
    except OSError:
        return []


def next_step(pkg_dir, root):
    """The last 'Next:' line of the thread's NOTES.md — its own resume
    instruction. '' when none."""
    hits = [l.strip() for l in _notes_lines(pkg_dir, root) if re.search(r"\bNext:", l)]
    return re.sub(r"^.*?\bNext:\s*", "", hits[-1])[:240] if hits else ""


# One-sentence NOTES.md fields (2026-09-29, Marcus: home cards say why the
# jobs ran; analysed cards say the most important conclusion). Line-anchored
# (optional "- " / "**") so prose that merely contains "why:" never matches;
# the LAST line wins, like Next:. The Conclusion carries its date because it
# is also processed-evidence: a rerun package sharing its parent's NOTES.md
# (mace_volume_states_si11/pkg_rerun) must not inherit an older conclusion.
_WHY = re.compile(r"^\s*(?:[-*]\s+)?(?:\*\*)?Why:(?:\*\*)?\s*(\S.*)$")
_CONC = re.compile(r"^\s*(?:[-*]\s+)?(?:\*\*)?Conclusion\s*\((\d{4}-\d{2}-\d{2})\):"
                   r"(?:\*\*)?\s*(\S.*)$")


def thread_why(pkg_dir, root):
    """(sentence, from_title): the last `Why:` line of the thread's NOTES.md;
    failing that its `# ` title (usually phrased as the question), flagged
    from_title=True; ("", False) when there is no NOTES.md."""
    lines = _notes_lines(pkg_dir, root)
    hits = [m.group(1).strip() for m in map(_WHY.match, lines) if m]
    if hits:
        return hits[-1][:300], False
    for l in lines:
        if l.startswith("# "):
            return l[2:].strip()[:300], True
    return "", False


def thread_conclusion(pkg_dir, root):
    """(date 'YYYY-MM-DD', sentence) of the last `Conclusion (date):` line of
    the thread's NOTES.md, else None."""
    hits = [m for m in map(_CONC.match, _notes_lines(pkg_dir, root)) if m]
    return (hits[-1].group(1), hits[-1].group(2).strip()[:400]) if hits else None


def processed_evidence(pkg_dir, since, root):
    """(what, when) showing a HOME thread's results were worked on after
    `since` (its last job's End), else None -> "not processed yet".

    Evidence (2026-09-29, Marcus: "pulled Home but haven't been analyzed"),
    either of:
      * a `Conclusion (YYYY-MM-DD):` line in its NOTES.md dated on/after the
        End's date — the analysis written down;
      * a file in its work dir (the package's parent, where harvest/collect
        scripts write CSVs, tables, relaxed/ ...) modified at or after
        `since`. Pulled outputs keep their cluster mtimes, so End is a fair
        floor.
    NOT evidence: NOTES.md or any other .md edit (adding the `Why:` line to an
    unanalysed thread must not mark it analysed), the package itself, any
    sibling subtree holding its own copy_back.sh (another thread's pulled
    output — dissociation_probe/ contains na_control/pkg), __pycache__/.pyc
    and dot-dirs. A false positive (an unrelated edit in the work dir) or a
    false negative (analysis written elsewhere) is visible on the card:
    processed threads name their evidence; `ack` clears the category."""
    c = thread_conclusion(pkg_dir, root)
    if c and c[0] >= time.strftime("%Y-%m-%d", time.localtime(since)):
        return notes_path(pkg_dir, root), time.mktime(time.strptime(c[0], "%Y-%m-%d"))
    work = os.path.dirname(os.path.normpath(pkg_dir))
    if os.path.normpath(work) == os.path.normpath(root):
        return None      # a package at the repo root has no work dir of its own
    pkg = os.path.normpath(pkg_dir)
    for dp, dn, fn in os.walk(work):
        dn[:] = [x for x in dn if x != "__pycache__" and not x.startswith(".")
                 and os.path.normpath(os.path.join(dp, x)) != pkg
                 and not os.path.isfile(os.path.join(dp, x, "copy_back.sh"))]
        for f in fn:
            if f.endswith((".pyc", ".md")):
                continue
            p = os.path.join(dp, f)
            try:
                m = os.path.getmtime(p)
            except OSError:
                continue
            if m >= since:
                return p, m
    return None


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
    cmd = loop.get("cmd", "")
    # 2026-09-30: expand a simple `for v in a b c;` word list into `$v`/`${v}`
    # tokens (`cd f3_rebuild/pkg_$p` names 12 packages), and resolve a `../x`
    # token against the other named dirs too (`cd pkg_A && ...; cd ../pkg_B`
    # is relative to pkg_A, not to the loop's cwd). Extra wrong paths are
    # harmless: they match no package.
    words = {v: vals.split() for v, vals in
             re.findall(r"\bfor\s+(\w+)\s+in\s+([^;]*);", cmd)}
    toks = []
    for tok in re.split(r"[\s;'\"()&|]+", cmd):
        if "/" not in tok or tok.startswith("-"):
            continue
        hit = [v for v in words if re.search(r"\$\{?%s(?!\w)\}?" % v, tok)]
        if hit:
            pat = r"\$\{?%s(?!\w)\}?" % hit[0]
            toks += [re.sub(pat, w, tok) for w in words[hit[0]]]
        else:
            toks.append(tok)
    def place(base, tok):
        full = posixpath.normpath(posixpath.join(base, tok))
        return posixpath.dirname(full) if tok.endswith(".sh") else full
    out = {cwd}
    for tok in toks:
        if "$" not in tok.split("/")[0]:
            out.add(place(cwd, tok))
    for tok in [t for t in toks if t.startswith("..")]:
        out |= {place(b, tok) for b in list(out)}
    return out


def _loop_scripts(loop):
    """Basenames of the submit scripts a live loop runs or will run — the
    waves it is holding, even before any of their jobs reaches Slurm."""
    return {t.rsplit("/", 1)[-1] for t in re.split(r"[\s;'\"()&|]+", loop.get("cmd", ""))
            if t.endswith(".sh")}


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
    # 2026-09-30: a package a live loop is working through has a thread even
    # before its first job reaches Slurm (the throttle may hold it for days)
    for l in ledger.get("loops", []):
        for tgt in _loop_targets(l):
            loc = to_local(tgt, prefixes, root)
            pkg = find_package(loc, root) if loc and os.path.isdir(loc) else None
            if not pkg:
                continue
            key = os.path.relpath(pkg, root).replace(os.sep, "/")
            if key not in threads:
                rel = os.path.relpath(loc, pkg).replace(os.sep, "/")
                threads[key] = {"key": key, "pkg": pkg, "jobs": [],
                                "remote": tgt[:len(tgt) - len(rel)].rstrip("/")
                                if rel != "." else tgt}
    out = []
    for t in threads.values():
        _classify(t, ledger, now, root)
        a = acks.get(t["key"])
        t["acked"] = bool(a and a.get("upto", 0) >= t["last_activity"])
        t["ack_note"] = a.get("note", "") if a else ""
        out.append(t)
    rank = {"ATTENTION": 0, "PULL": 1, "PROCESS": 2, "RUNNING": 3, "HOME": 4}
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
    t["first_submit"] = min(((j.get("submit") or now) for j in js), default=now)
    t["last_activity"] = max([j.get("submit") or 0 for j in js] + ended, default=now)
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
    rp = t["remote"]
    t["loops"] = [l for l in ledger.get("loops", [])
                  if any(x == rp or x.startswith(rp + "/") for x in _loop_targets(l))]
    named = set()
    for l in t["loops"]:
        named |= _loop_scripts(l)
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
            if hit >= len(rds) or mt < ledger.get("coverage_start", 0):
                continue          # wave complete, or older than sacct's view
            if hit == 0 and script not in named:
                continue          # not started, and no live loop is holding it
            if hit == 0:          # held by a live loop behind the throttle
                t["unsubmitted"].append((script, len(rds), len(rds)))
                continue
            # only a CURRENT wave can have a live-or-dead loop; on an old
            # thread the gap is usually work done elsewhere or skipped by a
            # DONE guard, so it is shown but does not raise ATTENTION
            (t["unsubmitted"] if recent or c["running"] + c["pending"]
             else t["unsubmitted_info"]).append((script, len(rds) - hit, len(rds)))
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
    t["processed_by"] = None
    if t["status"] == "HOME" and t["pkg"] and t["last_end"]:
        t["processed_by"] = processed_evidence(t["pkg"], t["last_end"], root)
        if t["processed_by"] is None:
            t["status"] = "PROCESS"
        else:
            p, m = t["processed_by"]
            t["processed_by"] = (os.path.relpath(p, root).replace(os.sep, "/"), m)
    t["why"], t["why_from_title"] = (thread_why(t["pkg"], root) if t["pkg"]
                                     else ("", False))
    t["conclusion"] = thread_conclusion(t["pkg"], root) if t["pkg"] else None


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


def shell_path(path):
    """Absolute local path as Git Bash takes it, double-quoted for a paste
    (the Drive root has a space): C:\\a b\\c -> "/c/a b/c". POSIX paths pass
    through, still quoted."""
    p = os.path.abspath(path)
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m:
        p = "/%s/%s" % (m.group(1).lower(), m.group(2).replace("\\", "/"))
    return shell_path_arg(p)


def pull_command(t):
    """Paste-ready full-path pull for a thread with finished jobs not home
    (copy_back.sh runs from anywhere), else ""."""
    c = t["counts"]
    if not (t["pkg"] and (c["missing"] or c["partial"])):
        return ""
    return "bash " + shell_path(os.path.join(t["pkg"], "copy_back.sh"))


def thread_commands(t, tdir):
    """[(label, paste-ready full-path command)] for a thread's HTML card."""
    out = []
    pc = pull_command(t)
    if pc:
        n = t["counts"]["missing"] + t["counts"]["partial"]
        out.append(("pull %d job%s" % (n, "" if n == 1 else "s"), pc))
    if t["status"] == "PROCESS":
        out.append(("done processing", "bash %s --ack %s" % (
            shell_path(os.path.join(tdir, "jobs.sh")), shell_path_arg(t["key"]))))
    return out


def shell_path_arg(s):
    """A plain argument double-quoted for bash (no path conversion): the four
    characters special inside double quotes get a backslash."""
    bs = chr(92)
    return '"%s"' % "".join(bs + ch if ch in (bs, '"', "$", "`") else ch for ch in s)


STATUSES = ("ATTENTION", "PULL", "PROCESS", "RUNNING", "HOME")
LABEL = {"ATTENTION": "⚠ NEEDS A LOOK", "PULL": "⬇ READY TO PULL",
         "PROCESS": "◆ HOME, NOT PROCESSED", "RUNNING": "▶ RUNNING / QUEUED",
         "HOME": "✓ HOME"}


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
    if not t["jobs"]:
        parts.append("nothing handed to Slurm yet")
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
    if t["status"] in ("PROCESS", "HOME") and t["pkg"]:
        if t["why"]:
            out.append("    why: %s%s" % (t["why"], "   (NOTES.md title — no Why: line)"
                                          if t["why_from_title"] else ""))
        else:
            out.append("    why: (no NOTES.md for this thread)")
    if t["status"] == "PROCESS":
        out.append("    not processed: no dated Conclusion in NOTES.md and no work-dir "
                   "file written since the last job ended")
        out.append('    done? bash jobtrack/jobs.sh --ack "%s"' % t["key"])
    elif t.get("processed_by"):
        cc = t["conclusion"]
        if cc and t["processed_by"][0].endswith("NOTES.md"):
            out.append("    processed: dated Conclusion line in %s" % t["processed_by"][0])
        else:
            out.append("    processed: %s (%s)" % (t["processed_by"][0], _t(t["processed_by"][1])))
        out.append("    conclusion (%s): %s" % cc if cc else
                   "    conclusion: (none recorded — add a `Conclusion (YYYY-MM-DD):` line to NOTES.md)")
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
    loops = ledger.get("loops", [])
    pids = {(l.get("host"), l["pid"]) for l in loops}
    for l in loops:      # a loop's own children (the running submit_*.sh) are not repeated
        if l.get("ppid") and (l.get("host"), l["ppid"]) in pids:
            continue
        body.append("submit loop alive on %s for %s: %s" % (
            l.get("host") or ledger.get("host", "?"), _span(int(l.get("etime") or 0)),
            l["cmd"][:200]))
    lh = ledger.get("loop_hosts")
    if not lh:
        body.append("(submit loops checked on %s only — set login_nodes in "
                    "jobtrack/config.json to check every login node)" % ledger.get("host", "?"))
    for h, st in sorted((lh or {}).items()):
        if st not in ("local", "ok"):
            body.append("⚠ submit loops on %s NOT checked (%s) — loops there are "
                        "invisible to this report" % (h, st))
    for status in STATUSES:
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


def render_html(threads, ledger, now, home_days=7, tdir="jobtrack"):
    lf = ledger.get("last_fetch")
    stale = lf and now - lf > 12 * 3600
    esc = html.escape
    sec = []
    tally = {}
    for status in STATUSES:
        ts = [t for t in threads if t["status"] == status and not t["acked"]]
        if status == "HOME":
            ts = [t for t in ts if t["last_activity"] >= now - home_days * 86400]
        tally[status] = len(ts)
        if not ts:
            continue
        cards = []
        for t in ts:
            ls = thread_lines(t, now)
            prose = ("    why: ", "    conclusion")
            body = [l[4:] for l in ls[1:]
                    if not l.startswith(("    pull: ", "    done? ") + prose)]
            say = "".join('<p class="say"><b>%s</b> %s</p>' % (esc(h + ":"), esc(v.strip()))
                          for h, _, v in (l[4:].partition(": ") for l in ls[1:]
                                          if l.startswith(prose)))
            pull = "".join('<div class="cmd"><span class="lbl">%s:</span><code>%s</code>'
                           '<button type="button" onclick="cp(this)">Copy</button></div>'
                           % (esc(lb), esc(cmd)) for lb, cmd in thread_commands(t, tdir))
            cards.append('<div class="card %s"><div class="k">%s</div>%s<pre>%s</pre>%s</div>'
                         % (status.lower(), esc(ls[0]), say, esc("\n".join(body)), pull))
        sec.append('<h2 class="%s">%s <span>%d</span></h2>%s'
                   % (status.lower(), esc(LABEL[status]), len(ts), "".join(cards)))
    chips = "".join('<div class="chip %s"><b>%d</b>%s</div>' % (s.lower(), tally.get(s, 0), esc(LABEL[s]))
                    for s in STATUSES)
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cluster Jobs</title><style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e2e1dc;
--att:#b3261e;--pull:#9a5b00;--proc:#6a3fa0;--run:#1f5fa8;--home:#2e7d32}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9993;
--card:#20201e;--line:#34332f;--att:#f2877f;--pull:#e7b566;--proc:#c3a2ec;--run:#8ab8f0;--home:#8fcf93}}
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
.process{border-left-color:var(--proc)}h2.process,.chip.process b{color:var(--proc)}
.running{border-left-color:var(--run)}h2.running,.chip.running b{color:var(--run)}
.home{border-left-color:var(--home)}h2.home,.chip.home b{color:var(--home)}
footer{color:var(--mut);font-size:13px;margin-top:28px}
.say{margin:6px 0 0;font-size:14px}.say b{color:var(--mut);font-weight:600}
.cmd{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px}
.cmd .lbl{color:var(--mut);font-size:13px}
.cmd code{flex:1 1 320px;min-width:0;background:var(--bg);border:1px solid var(--line);border-radius:4px;
padding:4px 8px;font:13px/1.4 ui-monospace,Consolas,monospace;word-break:break-all;user-select:all}
.cmd button{font:inherit;font-size:13px;padding:3px 10px;border:1px solid var(--line);border-radius:4px;
background:var(--card);color:var(--fg);cursor:pointer}.cmd button:hover{border-color:var(--mut)}
</style></head><body><main>
<h1>Cluster jobs</h1><div class="asof">Snapshot %s (%s) from %s%s</div>
<div class="chips">%s</div>%s
<footer>Generated by zeolib.jobtrack %s. Refresh: <code>bash jobtrack/jobs.sh</code> (one ssh) ·
after a copy_back: <code>bash jobtrack/jobs.sh --offline</code> (no ssh).</footer>
</main><script>
function cp(b){var c=b.previousElementSibling,t=c.textContent;
function ok(){b.textContent="Copied";setTimeout(function(){b.textContent="Copy"},1500)}
function fb(){var r=document.createRange();r.selectNodeContents(c);var s=getSelection();
s.removeAllRanges();s.addRange(r);try{document.execCommand("copy")?ok():b.textContent="Ctrl+C"}
catch(e){b.textContent="Ctrl+C"}}
if(navigator.clipboard&&window.isSecureContext){navigator.clipboard.writeText(t).then(ok,fb)}else{fb()}}
</script></body></html>
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
        fh.write(render_html(threads, ledger, now, cfg.get("home_days", 7), tdir))
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
        cfg = load_json(os.path.join(args.dir, "config.json"), {})
        sys.stdout.buffer.write(remote_script(
            args.days, cfg.get("login_nodes", [])).encode("utf-8"))
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
        report(args.dir, args.root, quiet=True)   # jobs.html/STATUS.md drop it now
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
