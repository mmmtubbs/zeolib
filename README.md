# zeolib — the shared zeolite-pipeline library

*The single canonical home for code that used to be re-implemented per job/test.
Created 2026-07-03 from a survey of MOR/ + MOR/tests/: framework loading was
copied ~4×, CP2K templates embedded in ~28 scripts, energy parsing in ~33,
sbatch writers in ~28, MIC geometry in ~15. Every function here was extracted
from the NEWEST VALIDATED source (named per module below), and
`selftest.py` pins behaviour against real repo data — including byte-parity of
generated CP2K inputs with the staged na_placement_multicomp production inputs.*

## Rules (also in Zeolites/CLAUDE.md, which is loaded every session)

1. **New scripts import zeolib instead of copying helpers.** If you're about to
   paste a `mic()`, a CP2K template, an `ENERGY| Total FORCE_EVAL` grep, or an
   sbatch string into a script — stop, import it, or add it here first.
2. **New common functionality goes here in the same change** that first uses
   it, with a provenance note in the docstring and a selftest check.
3. **After ANY zeolib change: `python zeolib/selftest.py` must pass.**
4. **Frozen code is never retrofitted**: completed tests (`MOR/tests/*` with
   DONE status), `MOR/pipeline_archive/`, `MOR/oldbinding/`, `MOR/olddata/`,
   `FAU/` keep their inlined copies — they are the provenance record of what
   actually ran. Living scripts (`MOR/pipeline/`, `MOR/binding/`) migrate when
   next touched, keeping their public behaviour identical.
5. **Cluster shipping**: a dir that gets scp'd to Pronghorn and runs zeolib-
   importing python there must ship `zeolib/` alongside (and insert its path).
   Submit-side shell scripts are generated text (no python needed); anything
   run by the cluster *system* python must remain python-2-safe — zeolib is
   python-3-only, so it never runs under the system python.
6. **Everything that ships to the cluster is written via `fileio.write_lf`**
   (CRLF from Windows breaks Pronghorn bash and CP2K).
7. **No silent fallbacks: a missing resource is FATAL, never approximated.**
   (2026-07-08: archived mor_core silently substituted approximate UFF params
   when its RANGE import missed — ~3e-4 kcal energy shift, caught only by the
   selftest parity pin. Frozen code keeps its fallbacks; callers must guard.)
8. **zeolib is a git repo** (since 2026-08-25; scope = `zeolib/` only — the
   surrounding 27 GB Zeolites tree is NOT versioned). Commit before packaging:
   every cluster package stamps the commit that built it via
   `provenance.write_stamp(pkg_dir)`, and `provenance.require_clean()` refuses
   to stamp a dirty tree, because a SHA that doesn't describe the shipped code
   is worse than no SHA. `.gitattributes` pins LF on every platform, so a
   clone on the Windows PC cannot reintroduce the CRLF failure of rule 6.

## Import pattern

```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")))  # adjust ..s -> Zeolites/
from zeolib import cp2k, framework, geometry, slurm, fileio, constants
```

## Module map (with provenance)

| Module | Contents | Extracted from |
|---|---|---|
| `constants.py` | Unit conversions (Ha→eV, Ha/Bohr→eV/Å, bar→eV/Å³), production protocol numbers (1500/100/1e-7/LBFGS), GTH basis+potential table (13 elements), guest multiplicities, 16 MOR combos + `COMBOS_BY_FRAMEWORK`/`combo_names`/`n_al_for_ratio` (Foundations: + the 17 historical FAU combos, FAUn = Si:Al n, verified vs FullOpt_out compositions; charge balance selftest-pinned) | `tests/na_training_set/collect_train.py`, `pipeline_archive/stage1a_v1/cell-opt-example.inp`, `binding/setup_fullopt.py`; FAU map Foundations 2026-07-10 |
| `geometry.py` | MIC vector/dist/all-pairs, `unwrap`, `centroid_unwrapped` (the PBC-centroid bug fix), min/mean pair distances, `wrap_to_cell`, `cell_matrix`, `perp_widths`. Cell always explicit, in TWO dispatch forms (Foundations 2026-07-10): (3,) box lengths = the orthorhombic fast path (behaviour unchanged, pins intact) or a (3,3) matrix = general cells (FAU rhombohedral; fractional round + exact ±1-image search — round-only is inexact at α=60°; parity vs `ase.geometry.get_distances` pinned) | `tests/na_placement_multicomp/common.py`, `binding/run_range_all.py`, `pipeline_archive/stage1a_v1/mor_core.py`; general-cell path new (Foundations) |
| `framework.py` | Baseline registry (cell read from embedded Lattice + cross-checked → enforces the "update xyz + constant together" rule; **per-baseline `BASELINE_SYMPREC`** — 0.30 is MOR-tuned, not universal), `load_framework` (T-orbits, Löwenstein graph), `loewenstein_ok`, Al samplers, space-group op helpers (`map_atom_under_op` residual FIXED 2026-07-07 — the test-lineage copy dropped the translation, spuriously rejecting t≠0 ops). **Stage-1a v2 enumeration layer (2026-07-07):** `site_permutations`, `canonical_arrangement` (canonical form + exact degeneracy g), `enumerate_al_arrangements` (exhaustive symmetry-distinct Löwenstein classes; MOR n_al=3/4/6 → 844/7,136/173,833), `sample_al_arrangements`, `arrangement_invariants` (stratification/Dedeček axes; single-orbit-safe for FAU). **Foundations layer (2026-07-10):** general (3,3) cells (loader raise dropped, `fw['cellmat']` always set, MOR (3,)-cell behaviour unchanged), `load_structure` (light loader for relaxed decorated structures — no orbits/4-coordination assert), `t_rings` (no-shortcut T-ring enumeration; FAU 16 6-rings / 36 4-rings pinned), `idealized_primitive` (spglib structure repair — the frozen 500-Ry `Fau-All_Si.xyz` is P1-broken, Fd-3m only at symprec ~1.0; every new FAU calc seeds from its idealized form) | `tests/na_placement_multicomp/common.py`, `pipeline_archive/stage1a_v1/mor_core.py`; enumeration new (design memory `project_stage1a_v2_design`); Foundations layer new |
| `cation.py` | Extra-framework cation model absorbed from `mor_core.py` at the stage-1a v2 rework (the absorption point named below): UFF-LJ + reduced-charge-Coulomb `cation_energy_grad` (ENERGY parity with mor_core pinned at 1e-9; Coulomb-gradient factor-1/2 bug FIXED; as-run eV/kcal units quirk kept deliberately — see docstrings), per-axis image expansion (`images="exact"` wrap-invariant default / `"legacy"` = mor_core ±1 parity; general (3,3) cells use explicit lattice images over `perp_widths`, legacy rejected), `relax_cations`, `place_cations_near_al`, `seed_cation_sets` (deterministic multi-seed Na layer, Hausdorff dedupe), `assemble_atoms` (cations-last convention), **`exchange_na`** (Foundations f2 baseline exchange: Ag/Cu 1:1 at Na positions, Pb/Bi deterministic greedy MIC grouping → centroid + UFF relax; a pre-DFT nudge, NOT a siting search — stage1b_v1 stays DO-NOT-RESURRECT) | `pipeline_archive/stage1a_v1/mor_core.py` (2026-07-07); mor_core keeps its copy as v0/v1 provenance; exchange new (Foundations 2026-07-10) |
| `cp2k.py` | Input builders: `cell_opt_input` / `geo_opt_input` / `energy_force_input` (+`dft_section`, `subsys_section`, `kind_blocks`); RKS defaults byte-identical to the staged multicomp production inputs; UKS/multiplicity for guests; `angles=`/`symmetry=` cell kwargs (FAU `60/60/60` + `RHOMBOHEDRAL`, the validated FAU-era form) and `geo_opt_input(fixed_atoms=(first,last))` → the v0/FAU `&FIXED_ATOMS LIST a..b` frozen-framework screen (contiguous range only — cations-then-guest-last assembly makes every freeze a prefix); `wfn_restart=` → SCF_GUESS RESTART + WFN_RESTART_FILE_NAME (the v0 screen→full-opt wavefunction hand-off); `scf_outer=(inner, outer)` → inner MAX_SCF + &OUTER_SCF loop at unchanged EPS_SCF (the SCF-robustness rescue variant; default renders byte-identically); `rescue_scf_text(inp_text)` → the same rescue applied POST-HOC to an already-rendered input (byte-parity with the builders pinned; refuses double application) — for cluster-side supervisors that can't reconstruct builder args. `strip_wfn_restart(inp_text)` → the INVERSE post-hoc edit (drop WFN_RESTART_FILE_NAME, SCF_GUESS back to ATOMIC; byte-parity with `wfn_restart=None` pinned, raises when there is nothing to strip) — for a full-opt whose screen wavefunction is not travelling with it (Foundations Perlmutter migration 2026-08-25). Parsers: `final_energy_ha`, `program_ended_ok`, `opt_completed`, `run_time_seconds` (wall-clock duration of a COMPLETED run — CP2K's own T I M I N G row, banner-timestamp fallback; **None** for a killed/running job, never a partial time, so walltime budgets are never biased low), `read_forces_au`, `read_stress_ase_ev_ang3` (CP2K→ASE **sign flip**), `read_last_cell_abc`, `read_last_cell_vectors` (full 3×3, rhombohedral cell-opts) | `tests/na_placement_multicomp/setup_dft.py`, `pipeline_archive/stage1a_v1/*-example.inp`, `tests/na_training_set/collect_train.py`; constraints/angles Foundations 2026-07-10 (from `oldbinding/setup_all.py` + `FAU/.../opt1.inp`) |
| `slurm.py` | Cluster-profile-aware `sbatch_text` / `cp2k_run_line` / `submit_script_text` / `ship_script_text` + `copy_back_script_text` (the per-package `ship.sh` pusher + `copy_back.sh` puller Marcus runs locally — ship BOTH with EVERY package; **environment-adaptive**: rsync when present, else tar-over-ssh, because Windows Git Bash has ssh+tar but no rsync). **PRONGHORN** (default: long/high-priority; starts immediately, 14-day cap; container **mpirun, never srun**; OMP=1; partition.sh `psub` s1-first + 50-cap throttle; timestamped lines). **PERLMUTTER** (volume of short jobs, e.g. 300+ SP DFTs: hundreds queue fine, fast nodes, but 1–2 day queue latency + 2-day cap, enforced; shifter `docker:cp2k/cp2k:2022.1` + srun, no `--account`, no `--time-min` (CP2K reads it as an effective walltime and quits after the first SCF energy — 2026-07-14 incident), no throttle; `sbatch_text(qos=)` default `"regular"`, `"debug"` for a fast ~immediate ≤30-min smoke job; **`array_manifest_text` + `array_sbatch_text`** — bundle hundreds of homogeneous jobs into ONE job array (manifest line i ↔ task i, cd + run in that dir; smoke via `sbatch --array=1`, rerun failures via `--array=i,j`); Perlmutter-only — Pronghorn raises, since an array's uniform directives cannot express psub's s1-free-first/s2-overflow routing or the queue-depth throttle). **`pack_groups` + `packed_array_manifest_text` + `packed_array_sbatch_text`** (2026-08-25) are the NODE-HOUR-efficient variant for a metered allocation: each array task packs N CP2K jobs onto ONE node as `--exact` srun steps of 32 ranks each, because CP2K runs at ~34% parallel efficiency from 32→128 ranks on our cells (728.8 s vs 540.1 s, `tests/cp2k_image_parity`), so 4-packed costs 2.9× fewer node-hours than 4×128-rank; a task's wall clock is its SLOWEST member, so pack like with like, and any member's failure fails the task. ⚠ Perlmutter image = CP2K **2022.1**, not the validated 2026 container — never pool its absolute energies with Pronghorn-2026 rankings without cross-image validation | `tests/na_placement_multicomp/setup_dft.py`, `MOR/partition.sh`, global CLAUDE.md; Perlmutter: `tests/na_placement_3al/setup_perlmutter_wsp.py` (369 shipped sbatch files = byte-parity fixtures) |
| `fileio.py` | `write_lf`, `has_crlf`, `write_coords_inc`/`read_coords_inc`; **`write_csv_lf`** (2026-08-25) — `csv`'s excel dialect terminates lines with CRLF and that wins over the handle's `newline=`, so the natural `open(p,'w',newline='\n')` + `DictWriter` still writes CRLF; every Foundations result CSV predating this has CRLF endings; **XYZ frames (Foundations 2026-08-17)** — `read_xyz_frames(path, last_only=)` (multi-frame CP2K `*-pos-1.xyz` trajectories; `last_only` = the converged geometry; a truncated trailing frame RAISES per rule 7, since that is what a walltime-killed job leaves) and `write_extxyz(path, frames, lattice=)` (one multi-frame extended-XYZ that OVITO/ASE/VMD open as a flipbook; free text goes into a quoted `info="..."` key so CP2K comment lines can't break key=value parsing) | CRLF working-agreement memory; format from all setup scripts; XYZ frames new (Foundations `communication/`) |
| `molecules.py` | Canonical home of the 7 guest templates (`GUESTS`, `MOL_RADIUS`, `guest_positions`/`guest_atoms`/`guest_elements`) — iodides I2/HI/CH3I + confounders H2O/Cl2/NO2/NO3, origin-centred idealized seeds; key parity with `constants.MULTIPLICITY` and composition parity with the frozen FAU BindingMolecules pinned | `MOR/oldbinding/setup_all.py` MOLECULES verbatim (Foundations 2026-07-10) |
| `placement.py` | Hand-template guest placement: `build_void_grid` (0.5 Å fractional grid, MIC clearance), `snap_to_void` (R_PROBE 1.2), `place_guest` (≤300 random orientations vs the 1.6 Å clash floor), `random_rotation`, `min_dist_mol_fw`, `ring_normal_sites` (FAU ± ring-normal anchors, driven by `framework.t_rings` not file row order). Orthorhombic results identical to v0 (a shipped Ag_5 SITE_INFO position is reproduced in selftest); general cells via geometry dispatch | `MOR/oldbinding/setup_all.py` + `test_placement.py` (Foundations 2026-07-10) |
| `maceenv.py` | `ensure_mace_env` (correct `~/mace_env_MOR_pipeline` path + `MACE_PYTHON` override), **model registry** `MODELS`/`resolve_model` (canonical home `Zeolites/models/`, `$ZEOLIB_MODELS_DIR` override; fine-tuned `na1500-std`/`na1500-polar` + foundations `mp0-medium`/`polar-1-m`; each entry names its cluster **`env` (GPU) and `env_cpu`** conda envs for sbatch generators), `get_calc(name)` unified factory, `prep_atoms` (polar info keys), legacy `get_mace_calc` | supersedes `MOR/mace_utils.py` + `mor_core.ensure_mace_env` (stale `../mace_env`); registry 2026-07-07 (models moved out of `tests/finetune_1500/`); `env_cpu` 2026-07-08 (Si15 CPU-route launch) |
| `provenance.py` | `version_info`/`git_sha`/`is_dirty`/`stamp_line` (which zeolib is running), `write_stamp`/`read_stamp` (a `ZEOLIB_VERSION.json` that travels WITH a package, the only form that works cluster-side where the shipped copy has no `.git`), `require_clean` (the loud path for results of record — raises on missing repo, no HEAD, or dirty tree; untracked files count as dirty because copytree ships them). Stdlib-only, so it imports under any cluster python | new 2026-08-25, prompted by four shipped zeolib copies having silently diverged with no version record |
| `selftest.py` | Checks against real repo data — byte-parity of generated CP2K inputs, parser pins on real 1500-Ry outputs, cation parity vs the archived v1 mor_core (it prints its own count; don't restate counts here — doc-hygiene rule) | — |

## Deliberately NOT in zeolib (yet)

- **RANGE wrappers** (`run_range_all.py` config): one live call site; the
  known-good RANGE kwargs are documented in project memory. Extract when a
  second consumer appears (e.g. the FAU+ generalisation).
- **Analysis/plotting** (Spearman tooling, eval_analyze): still evolving per
  test; extract once the pattern stabilises.

## Maintenance log

- 2026-08-25 — **zeolib put under version control** (`git init` in `zeolib/`,
  branch `main`, initial commit taken from a selftest-green working state).
  Motivation was provenance, not backup: four shipped copies —
  `Foundations/f0_fau_rebaseline/pkg/zeolib`, `f1_naform/pkg_FAU`,
  `.../pkg_MOR`, `MOR/pipeline/stage1a_v2/ship_rank/zeolib` — had diverged
  from master in 6-8 modules each, with no record of which version produced
  which numbers. Those stay frozen (rule 4). New `provenance.py` + rule 8 make
  every future package self-identifying; `s2_package.py` now writes
  `ZEOLIB_VERSION.json` beside its `zeolib/` copy and records the same sha in
  `manifest.json`. `.gitignore` (pycache/.DS_Store/Drive `(1)` conflict copies)
  and `.gitattributes` (LF everywhere) added. Selftest +19.
- 2026-07-03 — v0.1.0 created (survey + extraction + selftest). Consumers: none
  yet; first planned consumer = na_placement_multicomp eval fixes / the
  post-rebaseline binding re-run.
- 2026-07-03 — slurm.py gained cluster profiles (PRONGHORN default, PERLMUTTER
  for volume-of-short-jobs; Marcus's routing guidance in the module header);
  Perlmutter renders byte-checked against the shipped wsp sbatch files;
  walltime caps (14 d / 2 d) enforced. Selftest now 30 checks.
- 2026-07-07 — slurm.py gained `copy_back_script_text` (+ profile `host`/
  `remote_base`): generates a self-contained `copy_back.sh` that PULLS a
  package's results back from the cluster into its own dir, so Marcus just runs
  `bash copy_back.sh` — ship one in every cluster-bound package
  (feedback_ship_copyback_script). Skips CP2K restart files, no `--delete`,
  run-anywhere dest resolution. Selftest now 33 checks.
- 2026-07-07 — Stage-1a v2 layer: maceenv model registry (models moved to
  `Zeolites/models/`), framework.py enumeration layer + per-baseline symprec
  + map_atom_under_op translation fix, NEW cation.py (mor_core absorption;
  energy parity pinned, Coulomb-grad ×1/2 + legacy wrap-dependence fixed),
  cp2k.read_stress_ase_ev_ang3 `pattern` arg (mixed opt+SP dirs). First
  consumer: `MOR/pipeline/stage1a_v2/`. Selftest now 55 checks.
- 2026-07-08 — maceenv registry entries gained `env_cpu` (CPU conda env per
  model: `mace` / `mace-polar`) so sbatch generators can emit CPU-fallback
  jobs — first consumer: stage1a_v2 Si15 CPU-route launch (GPU queue jammed).
  `fileio.write_lf` now writes UTF-8 explicitly (Windows cp1252 gotcha).
  Selftest now 57 checks.
- 2026-07-08 — v1 stage-1a chain archived to `MOR/pipeline_archive/stage1a_v1/`
  (pipeline/ cleanup): selftest's mor_core parity fixture and the provenance
  paths above now point there. mor_core stays importable as a frozen fixture.
- 2026-07-10 — Foundations layer (first consumer: `Foundations/` — the Paper-0
  FAU+MOR baseline binding pipeline; see `Foundations/FOUNDATIONS.md`):
  geometry general-cell dispatch (+`wrap_to_cell`/`cell_matrix`/`perp_widths`),
  framework general cells + `load_structure` + `t_rings` +
  `idealized_primitive` (frozen FAU all-Si xyz is P1-broken — probed
  2026-07-10, idealization is the documented seed path), cp2k
  `angles`/`symmetry` + `fixed_atoms` constraint + `read_last_cell_vectors`,
  NEW `molecules.py` (v0 guest templates) + `placement.py` (v0 void-grid
  port; shipped SITE_INFO reproduced), cation general cells + `exchange_na`,
  constants `COMBOS_BY_FRAMEWORK` (FAUn = Si:Al n, 17 FAU combos). All
  orthorhombic defaults byte-identical (multicomp/mor_core/wsp pins intact).
  Also cp2k `wfn_restart=` (the v0 screen→full-opt SCF hand-off, f3).
  Selftest now 103 checks.
- 2026-07-10 — slurm transfer scripts made ENVIRONMENT-ADAPTIVE + added
  `ship_script_text` (the upload counterpart to `copy_back_script_text`):
  both prefer rsync, else fall back to tar-over-ssh, because Marcus's machine
  is now Windows Git Bash (ssh + tar, no rsync — the old rsync-only scripts
  silently failed). copy_back excludes now render in both branches; ship.sh
  mkdir -p's the remote. Foundations packages emit both via
  `flib.write_transfer_scripts`. Selftest now 105 checks (copy_back pins
  updated to the host/remote-var form).
- 2026-07-17 — cp2k builders gained `scf_outer=(inner, outer)` (threaded through
  `dft_section` and all three input builders): inner MAX_SCF plus an
  `&OUTER_SCF` loop at the SAME EPS_SCF — the standard CP2K recipe for OT
  ATOMIC-guess non-convergence, criterion unchanged so rescued energies stay
  protocol-comparable. First consumer: `MOR/pipeline/stage1a_v2/
  s3b_rescue_scf.py` (dft_run1: 11/110 jobs aborted "SCF run NOT converged",
  incl. the MACE-#1 class). Defaults byte-identical (parity pins intact);
  selftest +5 (three builders, default-clean, EXT_RESTART+wfn+outer combo).
- 2026-07-17 — slurm.py gained `array_manifest_text` + `array_sbatch_text`:
  Perlmutter job-array bundling — one sbatch + a job-dir manifest replaces the
  N-line per-dir submit loop (NERSC prefers arrays; single scancel, per-task
  `--array` reruns, optional `%cap` running limit). Missing manifest line is
  FATAL (rule 7). Deliberately Perlmutter-only: arrays carry uniform
  partition/account directives, so Pronghorn keeps `submit_script_text` +
  partition.sh psub (s1-first routing / queue-depth throttle). Existing
  renderers untouched (wsp byte-parity pins intact). Selftest +6. No consumer
  yet — next Perlmutter campaign should use it.
- 2026-07-22 — **`opt_completed` now accepts BOTH CP2K convergence banners.**
  CP2K's L-BFGS can exit on its OWN criteria (WANTED_PROJ_GRADIENT +
  WANTED_REL_F_ERROR) printing "Specific L-BFGS convergence criteria …
  satisfied …. run CONVERGED!" and NEVER printing the standard
  "OPTIMIZATION COMPLETED". Found in the live Foundations campaign: every FAU
  f1 cell-opt converged that way (~82 steps) and was misread as
  non-convergent, while MOR's took the standard path (~193 steps) — both in
  one wave, so both banners must count. Match is case-sensitive so the SCF's
  own lowercase "SCF run converged in N steps" can never satisfy a GEOMETRY
  check. Selftest +4 (both banners, SCF-only negative, unconverged negative).
  Consumer-side companion: `Foundations/supervisor.py --recheck` re-judges
  jobs already written off under the old parser.
- 2026-08-24 — cp2k.py gained `run_time_seconds` (needed to size the Foundations
  f3 walltimes from measured post-loosening screen durations; two independent
  sources cross-checked in selftest against `MFI/AIMD_out/md.out`).
- 2026-07-21 — cp2k.py gained `rescue_scf_text` (post-hoc SCF-rescue transform
  of a rendered input; byte-parity with the builders' `scf_outer=` pinned for
  all three builders + UKS/wfn variants; raises on double application). First
  consumer: `Foundations/supervisor.py` (the vacation campaign supervisor on
  Pronghorn — rescues jobs without reconstructing builder args). Walltime-death
  resubmits deliberately need NO new slurm code: `cp2k_run_line("<proj>-1.restart",
  out="<proj>.out")` + `sbatch_text` compose the restart sbatch from existing
  pinned primitives. Selftest +7.
- 2026-07-14 — `sbatch_text` gained a `qos=` kwarg (Perlmutter only; default
  `"regular"` keeps wsp byte-parity, `"debug"` renders `#SBATCH --qos debug` for
  a fast ~immediate smoke job). First consumer: Foundations f0 ships a
  standalone `smoke/` dir (`flib.write_smoke_job`; debug QOS, 10 min, cell-opt
  only) so the shifter image + inputs can be validated past setup→SCF before the
  real wave. Pronghorn branch unaffected. Selftest +1 (qos default/override).
