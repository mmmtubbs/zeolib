# zeolib version control — setup, and the two-machine model

Created 2026-08-25 alongside `git init`. Rule + rationale live in
**README.md rule 8**; this file is only the *how*, including the steps that
need Marcus's GitHub account and so could not be done for him.

## What is already done

- `git init` in `Zeolites/zeolib/`, branch `main`, repo-local identity set
  (global git config deliberately untouched).
- `.gitignore` — `__pycache__/`, `.DS_Store`, and Google Drive conflict copies
  (`* (1).*`), which already exist elsewhere in this project.
- `.gitattributes` — `eol=lf` on every text type, on every platform. This is
  the rule-6 CRLF guard: a clone on the Windows PC cannot reintroduce the
  CRLF-breaks-Pronghorn failure.
- Two commits: the selftest-green baseline, then `provenance.py`.

## Step 1 — create the empty GitHub repo (needs your account)

There is no `gh` CLI on this Mac, so use the web UI:
<https://github.com/new>

- Name: `zeolib`
- **Private** (public + a Zenodo DOI is a paper-time decision, not now)
- **Do not** add a README, .gitignore, or licence — the repo already has
  history, and an initialising commit would force a merge on the first push.

## Step 2 — add the remote and push

There is no SSH key on either machine, so use HTTPS. GitHub will ask for a
Personal Access Token (Settings -> Developer settings -> Tokens, `repo` scope)
— your account password will not work.

```
cd "$ZEOLITES/zeolib"
git remote add origin https://github.com/<your-user>/zeolib.git
git push -u origin main
```

To avoid retyping the token: `git config --global credential.helper osxkeychain`
(macOS) / `manager` (Windows) before pushing.

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
git clone https://github.com/<your-user>/zeolib.git zeolib
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
