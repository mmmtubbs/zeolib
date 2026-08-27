# zeolib version control — setup, and the two-machine model

Created 2026-08-25 alongside `git init`. Rule + rationale live in
**README.md rule 8**; this file is only the *how*.

**Status: LIVE and PUBLIC.** Remote is `git@github.com:mmmtubbs/zeolib.git`,
pushed 2026-08-25 (`$ZEOLITES` below = wherever the Zeolites tree lives on the
machine you are on).

Because it is public, **cluster login identities are configured, never
committed** — see "Cluster identity" in `slurm.py`. Set them per machine:

```
mkdir -p ~/.config/zeolib && cat > ~/.config/zeolib/clusters.json <<'JSON'
{"pronghorn": {"host": "you@your.cluster.edu", "base": "/your/work/path"},
 "perlmutter": {"host": "you@your.hpc.facility"}}
JSON
chmod 600 ~/.config/zeolib/clusters.json
```

This file is machine-local and outside Drive, so the Windows PC needs its own
copy (or the `$ZEOLIB_*_HOST` / `$ZEOLIB_*_BASE` env vars). Unconfigured is
FATAL by design — a silently wrong host would ship a package to nowhere. Steps 1-2 below are kept as the record of how it was set
up; Step 3 is the standing two-machine rule and still applies daily.

## What is already done

- `git init` in `Zeolites/zeolib/`, branch `main`, repo-local identity set
  (global git config deliberately untouched).
- `.gitignore` — `__pycache__/`, `.DS_Store`, and Google Drive conflict copies
  (`* (1).*`), which already exist elsewhere in this project.
- `.gitattributes` — `eol=lf` on every text type, on every platform. This is
  the rule-6 CRLF guard: a clone on the Windows PC cannot reintroduce the
  CRLF-breaks-Pronghorn failure.
- Three commits: the selftest-green baseline, `provenance.py`, and this file.
- SSH auth to GitHub (Step 2) and the `origin` remote, pushed.

## Step 1 — create the empty GitHub repo (needs your account)

Two ways:

- **Claude on the web** is connected to GitHub and can see your repos — ask it
  to create an empty private repo named `zeolib`. (That connection does NOT
  reach this Mac; the push below still happens locally.)
- Or the web UI: <https://github.com/new>

Either way: name `zeolib`, **Private**, and **do not** add a README,
.gitignore, or licence — the repo already has history here, and an
initialising commit would force a merge on the first push.

Public + a Zenodo DOI is a paper-time decision, not now.

## Step 2 — auth this Mac by SSH (done 2026-08-25)

There is no `gh` CLI on this Mac (Homebrew is older than macOS 15 and its
directories need a `sudo chown` to repair), so the GitHub CLI route was
skipped. SSH was chosen over an HTTPS Personal Access Token for two reasons:
tokens expire and would need redoing, and a token has to be typed into an
interactive `git push` prompt — which means Claude cannot run the push for
you. With SSH, the secret never leaves this machine and the push is a normal
command.

Already done here:

- `~/.ssh/id_ed25519_github` + `.pub` — an ed25519 key dedicated to GitHub,
  passphraseless (add one later with `ssh-keygen -p -f ~/.ssh/id_ed25519_github`).
- A `Host github.com` block appended to `~/.ssh/config` (`IdentitiesOnly yes`,
  so it never offers this key to Pronghorn). Prior config backed up alongside.

The **public** key was pasted into <https://github.com/settings/keys> ("New
SSH key", type Authentication). Public keys are safe to share; the private key
never leaves this Mac. To re-do this on another machine, generate a NEW key
there rather than copying this one.

```
cat ~/.ssh/id_ed25519_github.pub
```

GitHub's host key was verified against its published ed25519 fingerprint
(`SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU`) before being added to
`known_hosts` — worth repeating rather than blind-accepting on any new machine.

What was run (`ssh -T` greets you by username and exits 1 — that is success):

```
ssh -T git@github.com
cd "$ZEOLITES/zeolib"
git remote add origin git@github.com:mmmtubbs/zeolib.git
git push -u origin main
```

Day to day from here it is just `git push` — `main` tracks `origin/main`.

## Step 3 — the Windows PC does NOT clone

This matters, and it is the opposite of normal advice.

`zeolib/` lives inside Google Drive, so `.git` syncs to the PC along with the
working tree. **The PC already has the repository** the moment Drive settles —
including the repo-local identity in `.git/config`. Do not `git clone` a second
copy: you would end up with two working trees for one set of files.

So the model is:

- **Drive** is how the two machines share the repo (as it already shares
  everything else).
- **GitHub** is the durable remote: off-machine backup, history that survives
  a Drive accident, and the future publication path.
- Push from whichever machine you are on. There is no pull-before-work step,
  because Drive has already delivered the commits.

### The discipline this needs

Exactly the rule already in force — *one machine at a time, let Drive finish
syncing before switching* — now extended to `.git`:

1. Commit before you stop working.
2. `git push` (cheap, and it is what makes the next point safe).
3. Let Drive go quiet before opening the project on the other machine.

### If Drive mangles `.git`

A half-synced pack file can corrupt the repo (`git status` starts erroring).
Because GitHub has the history, this costs nothing:

```
cd "$ZEOLITES"
mv zeolib zeolib_broken
git clone git@github.com:mmmtubbs/zeolib.git zeolib
```

Then diff any uncommitted work out of `zeolib_broken/` and delete it. This is
why putting `.git` inside Drive is an acceptable risk *once a remote exists* —
and why Step 2 is worth doing promptly rather than eventually.

## Working rules

- **Commit before packaging.** `provenance.write_stamp()` records the commit
  into each package; a dirty tree stamps `DIRTY`, and
  `provenance.require_clean()` refuses outright. See README rule 8.
- **After any change: `python zeolib/selftest.py` must pass** (README rule 3)
  — the selftest is what makes a green commit meaningful.
- **Scope stays `zeolib/`.** The surrounding Zeolites tree is 27 GB of
  structures, models, and CP2K output. Never `git add` upward.
- Tag releases when a paper is written (`git tag -a v1.0 -m "Paper 1"`);
  `provenance.version_info()["describe"]` picks tags up automatically.
