"""
cp2k.py — CP2K input generation and output parsing for the production protocol.

Generation (stdlib only): builds CELL_OPT / GEO_OPT / ENERGY_FORCE inputs from
the production protocol (constants.py: LBFGS · CUTOFF 1500 · REL_CUTOFF 100 ·
EPS_SCF 1e-7 · PBE-D3 · DZVP-MOLOPT-SR-GTH). Text is lifted verbatim from
tests/na_placement_multicomp/setup_dft.py (RKS two-stage) and
MOR/pipeline_archive/stage1a_v1/cell-opt-example.inp + final-sp-example.inp
(UKS variants; archived 2026-07-08 with the v1 chain) — the
default RKS CELL_OPT/GEO_OPT render byte-identically to the multicomp inputs
(selftest-enforced). Write results with fileio.write_lf ONLY (CRLF kills CP2K).

Parsing: energy / forces / stress readers lifted from
tests/na_training_set/collect_train.py — including the CP2K->ASE stress SIGN
FLIP (CP2K's printed stress tensor is sign-opposite to the ASE/MACE convention;
verified on the na_training_set). Works for both the old 5.1 and the 2026 image
('ENERGY| Total FORCE_EVAL' is unchanged; only the final-energy *label* line
changed to 'energy [hartree]' in 2026 — don't grep for the old label).
"""
import glob
import os
import re

from .constants import (BASIS_SET, GTH_POTENTIAL, CUTOFF_RY, REL_CUTOFF_RY,
                        EPS_SCF, BAR_TO_EV_ANG3, HA_BOHR_TO_EV_ANG, HARTREE_TO_EV)

# ═════════════════════════════════════════════════════════════════════════════
# Input generation
# ═════════════════════════════════════════════════════════════════════════════

def kind_blocks(elements):
    """&KIND blocks (4-space indent) for the given elements, in given order."""
    out = []
    for el in elements:
        if el not in GTH_POTENTIAL:
            raise KeyError("element %r not in constants.GTH_POTENTIAL — add it "
                           "there (with provenance) rather than inlining" % el)
        out.append("    &KIND %s\n      BASIS_SET %s\n      POTENTIAL %s\n"
                   "    &END KIND\n" % (el, BASIS_SET, GTH_POTENTIAL[el]))
    return "".join(out)


def dft_section(basis_rel="../../../", charge=0, uks=False, multiplicity=None,
                cutoff=CUTOFF_RY, rel_cutoff=REL_CUTOFF_RY, eps_scf=EPS_SCF,
                aspc=True, wfn_restart=None, scf_outer=None,
                ot_minimizer="CG", ot_preconditioner="FULL_ALL",
                diagonalization=False, mixing_alpha=0.4):
    """
    The &DFT section (2-space base indent), production protocol defaults.
    basis_rel: relative prefix from the job dir to MOR/ (where BASIS,
    POTENTIALS, dftd3.dat live), e.g. '../../../' for MOR/tests/<test>/<cfg>/.
    aspc: include WF_INTERPOLATION ASPC (optimisation runs); False for single
    points (matches final-sp-example.inp).
    wfn_restart: path to a -RESTART.wfn to seed the SCF (SCF_GUESS RESTART) —
    the v0 screen->full-opt wavefunction hand-off (oldbinding/setup_all.py
    wfn_line). Default None renders byte-identically. [Foundations 2026-07-10]
    SCF SOLVER KNOBS (all default to the production settings, so the rendered
    input is byte-identical unless a caller opts in). Added 2026-08-20 for
    tests/scf_openshell: the UKS guests (NO2/NO3, multiplicity 2) stall in OT —
    592 iterations without converging on a 3-ATOM molecule
    (tests/cp2k_image_parity) — and every geometry step of a Foundations f3
    NO2/NO3 relaxation pays that. All of these converge to the SAME `eps_scf`,
    so converged energies stay protocol-comparable (the argument already
    accepted for scf_outer in s3b_rescue_scf).
      ot_minimizer      : "CG" (default) or "DIIS". LINESEARCH is emitted only
                          for CG, where it applies.
      ot_preconditioner : "FULL_ALL" (default) or e.g. "FULL_SINGLE_INVERSE",
                          the standard escalation for stubborn OT.
      diagonalization   : True replaces &OT with &DIAGONALIZATION + Broyden
                          &MIXING. OT is built for large gapped systems; an
                          isolated open-shell radical is where it is weakest.
      mixing_alpha      : Broyden ALPHA, diagonalization only.

    scf_outer: None (default, byte-identical: one OT run, MAX_SCF 3000) or
    (inner, outer) — inner MAX_SCF per OT run plus an &OUTER_SCF loop of up to
    `outer` restarts at the same eps_scf, refreshing the FULL_ALL
    preconditioner from the current wavefunction each restart. The standard
    CP2K robustness recipe for OT non-convergence; the convergence criterion
    is UNCHANGED, so converged energies remain protocol-comparable. Added
    2026-07-17 for the stage1a_v2 dft_run1 SCF rescue (11/110 jobs aborted
    "SCF run NOT converged" from the ATOMIC guess at 1500 Ry).
    """
    spin = ""
    if uks:
        spin = "    UKS\n"
        if multiplicity is not None:
            spin += "    MULTIPLICITY %d\n" % multiplicity
    wfn = ""
    guess = "ATOMIC"
    if wfn_restart:
        wfn = "    WFN_RESTART_FILE_NAME %s\n" % wfn_restart
        guess = "RESTART"
    max_scf, outer = 3000, ""
    if scf_outer:
        max_scf = int(scf_outer[0])
        outer = ("      &OUTER_SCF\n        EPS_SCF %s\n        MAX_SCF %d\n"
                 "      &END OUTER_SCF\n" % (eps_scf, int(scf_outer[1])))
    qs_extra = "      WF_INTERPOLATION ASPC\n      EXTRAPOLATION_ORDER 1\n" if aspc else ""
    if diagonalization:
        solver = ("      &DIAGONALIZATION T\n        ALGORITHM STANDARD\n"
                  "      &END DIAGONALIZATION\n"
                  "      &MIXING T\n        METHOD BROYDEN_MIXING\n"
                  "        ALPHA %.2f\n        NBROYDEN 8\n"
                  "      &END MIXING\n" % float(mixing_alpha))
    else:
        ls = "        LINESEARCH 3PNT\n" if ot_minimizer.upper() == "CG" else ""
        solver = ("      &OT T\n        PRECONDITIONER %s\n"
                  "        MINIMIZER %s\n%s      &END OT\n"
                  % (ot_preconditioner, ot_minimizer, ls))
    return """\
  &DFT
%s    BASIS_SET_FILE_NAME %sBASIS
    POTENTIAL_FILE_NAME %sPOTENTIALS
    CHARGE %d
%s
    &MGRID
      CUTOFF %d
      REL_CUTOFF %d
    &END MGRID

    &QS
      EPS_DEFAULT 1.0E-12
      EPS_PGF_ORB 1.0E-6
%s    &END QS

    &SCF
      EPS_SCF %s
      SCF_GUESS %s
      MAX_SCF %d
%s%s    &END SCF

    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
      &VDW_POTENTIAL
        DISPERSION_FUNCTIONAL PAIR_POTENTIAL
        &PAIR_POTENTIAL
          TYPE DFTD3
          REFERENCE_FUNCTIONAL PBE
          PARAMETER_FILE_NAME %sdftd3.dat
          R_CUTOFF 8.0
        &END PAIR_POTENTIAL
      &END VDW_POTENTIAL
    &END XC
  &END DFT
""" % (spin, basis_rel, basis_rel, charge, wfn, cutoff, rel_cutoff, qs_extra,
       eps_scf, guess, max_scf, solver, outer, basis_rel)


def strip_wfn_restart(inp_text):
    """
    Drop a rendered input's wavefunction hand-off: remove
    WFN_RESTART_FILE_NAME and put SCF_GUESS back to ATOMIC. Byte-parity with
    `wfn_restart=None` is selftest-pinned, so the result is exactly the input
    the builders would have produced without the hand-off.

    Provenance: the Foundations Perlmutter migration (2026-08-25). An f3
    full-opt seeds its SCF from the sibling screen's `screen-RESTART.wfn`; a
    full-opt migrating to NERSC while its screen stays on Pronghorn would
    reference a wavefunction that is not travelling (a `.wfn` is a large
    binary written by a DIFFERENT CP2K build — not something to move across
    images), and a missing WFN_RESTART_FILE_NAME is FATAL to CP2K. Stripping
    it costs the first geometry step's SCF iterations from the ATOMIC guess
    and changes no converged energy (same EPS_SCF, same minimum).
    Raises if the text has no restart to strip (rule 7 — a silent no-op here
    would ship the fatal input).
    """
    if "WFN_RESTART_FILE_NAME" not in inp_text:
        raise ValueError("strip_wfn_restart: no WFN_RESTART_FILE_NAME in input")
    lines = [ln for ln in inp_text.splitlines(True)
             if "WFN_RESTART_FILE_NAME" not in ln]
    out = "".join(lines)
    if "      SCF_GUESS RESTART\n" not in out:
        raise ValueError("strip_wfn_restart: no 'SCF_GUESS RESTART' to reset")
    return out.replace("      SCF_GUESS RESTART\n", "      SCF_GUESS ATOMIC\n")


def rescue_scf_text(inp_text, scf_outer=(50, 60)):
    """
    Transform an already-rendered production input into its SCF-rescue
    variant: inner MAX_SCF plus an &OUTER_SCF loop at the SAME EPS_SCF —
    byte-identical to what the builders emit for the equivalent scf_outer=
    (selftest-pinned against all three builders, RKS and UKS). Exists so a
    cluster-side supervisor can rescue a job WITHOUT reconstructing its
    builder arguments (the s3b_rescue_scf recipe; (50, 60) is that script's
    validated default — convergence criterion unchanged, energies stay
    protocol-comparable). Raises if the text already carries &OUTER_SCF
    (bounded one-rescue policy — never silently re-apply, rule 7) or if the
    &SCF block doesn't match the production shape. [Foundations 2026-07-21]
    """
    if "&OUTER_SCF" in inp_text:
        raise ValueError("rescue_scf_text: input already has &OUTER_SCF")
    i0 = inp_text.find("    &SCF\n")
    i1 = inp_text.find("    &END SCF\n", i0)
    if i0 < 0 or i1 < 0:
        raise ValueError("rescue_scf_text: no production &SCF section found")
    scf = inp_text[i0:i1]
    m = re.search(r"^      EPS_SCF (\S+)$", scf, re.M)
    if not m:
        raise ValueError("rescue_scf_text: no EPS_SCF in &SCF section")
    scf2, n = re.subn(r"^      MAX_SCF \d+$",
                      "      MAX_SCF %d" % int(scf_outer[0]), scf, flags=re.M)
    if n != 1:
        raise ValueError("rescue_scf_text: expected exactly one MAX_SCF in "
                         "&SCF, found %d" % n)
    outer = ("      &OUTER_SCF\n        EPS_SCF %s\n        MAX_SCF %d\n"
             "      &END OUTER_SCF\n" % (m.group(1), int(scf_outer[1])))
    return inp_text[:i0] + scf2 + outer + inp_text[i1:]


def subsys_section(cell_abc, elements, coords_file="coords.inc",
                   angles=(90.0, 90.0, 90.0), symmetry="ORTHORHOMBIC"):
    """&SUBSYS with ABC + ALPHA_BETA_GAMMA cell (%.4f / %.3f), @INCLUDE'd
    coords, KIND blocks. Defaults render byte-identically to the orthorhombic
    pre-Foundations output (selftest-pinned); FAU passes angles=(60,60,60),
    symmetry='RHOMBOHEDRAL' — the exact cell block of the validated FAU-era
    inputs (FAU/Old/Old CellOpt/cell-opt.inp). [Foundations 2026-07-09]"""
    A, B, C = float(cell_abc[0]), float(cell_abc[1]), float(cell_abc[2])
    al, be, ga = (float(x) for x in angles)
    return """\
  &SUBSYS
    &CELL
      ABC %.4f %.4f %.4f
      ALPHA_BETA_GAMMA %.3f %.3f %.3f
      SYMMETRY %s
    &END CELL
    &COORD
@INCLUDE '%s'
    &END COORD
%s  &END SUBSYS
&END FORCE_EVAL
""" % (A, B, C, al, be, ga, symmetry, coords_file, kind_blocks(elements))


def cell_opt_input(cell_abc, elements, basis_rel="../../../", project="cell-opt",
                   charge=0, uks=False, multiplicity=None,
                   optimizer="LBFGS", max_iter=3000,
                   keep_angles=True, keep_symmetry=False,
                   angles=(90.0, 90.0, 90.0), symmetry="ORTHORHOMBIC",
                   scf_outer=None, **dft_kw):
    """
    Variable-cell relax, production protocol. Defaults reproduce the
    na_placement_multicomp stage-1 input byte-for-byte: MOTION-level per-step
    TRAJECTORY/FORCES/STRESS/CELL prints (training-frame harvest),
    RESTART_HISTORY OFF, STRESS_TENSOR Analytical.
    keep_symmetry: ONLY for bare-framework re-baselines — an Al/Na decoration
    lowers the space group, so decorated cells must relax with KEEP_ANGLES only.
    angles/symmetry: passed to subsys_section (FAU rhombohedral support).
    """
    keep = ""
    if keep_symmetry:
        keep += "    KEEP_SYMMETRY TRUE\n"
    if keep_angles:
        keep += "    KEEP_ANGLES TRUE\n"
    head = """\
&GLOBAL
  PROJECT %s
  RUN_TYPE CELL_OPT
  PRINT_LEVEL LOW
&END GLOBAL

&MOTION
  &CELL_OPT
    MAX_ITER %d
    OPTIMIZER %s
%s  &END CELL_OPT
  &PRINT
    &TRAJECTORY
      FORMAT XYZ
    &END TRAJECTORY
    &FORCES
      FORMAT XYZ
    &END FORCES
    &STRESS ON
    &END STRESS
    &CELL
    &END CELL
    &RESTART_HISTORY OFF
    &END RESTART_HISTORY
  &END PRINT
&END MOTION

&FORCE_EVAL
  METHOD Quickstep
  STRESS_TENSOR Analytical
""" % (project, max_iter, optimizer, keep)
    return head + dft_section(basis_rel=basis_rel, charge=charge, uks=uks,
                              multiplicity=multiplicity, scf_outer=scf_outer, **dft_kw) \
                + "\n" + subsys_section(cell_abc, elements,
                                        angles=angles, symmetry=symmetry)


def geo_opt_input(cell_abc, elements, basis_rel="../../../", project="geo-opt",
                  charge=0, uks=False, multiplicity=None,
                  optimizer="BFGS", max_iter=1000,
                  restart_from="cell-opt-1.restart",
                  angles=(90.0, 90.0, 90.0), symmetry="ORTHORHOMBIC",
                  fixed_atoms=None, wfn_restart=None, scf_outer=None,
                  coords_file="coords.inc",
                  max_force="1.0E-4", rms_force="7.0E-5",
                  max_dr="1.0E-3", rms_dr="7.0E-4", **dft_kw):
    """
    Fixed-cell tight GEO_OPT (MAX_FORCE 1e-4), stage 2 of the two-stage
    relaxation; restarts pos+cell from the CELL_OPT minimum via &EXT_RESTART
    (restart_from=None for a standalone geo-opt from coords.inc). Defaults
    reproduce the na_placement_multicomp stage-2 input byte-for-byte.
    Ranking energy = final_energy_ha('geo-opt.out').

    max_force / rms_force / max_dr / rms_dr: GEO_OPT convergence, defaulting to
    the production values (byte-identical). Loosen ONLY for a step whose output
    is a GEOMETRY, never an energy — the Foundations f3 *screens* are the case
    this was added for (2026-08-20): measured over 391 completed screens, 57-69%
    of every screen's geometry steps are spent grinding from a 1e-3 gradient
    down to 1e-4, on a frozen-framework pre-relaxation whose only products are a
    starting geometry and a wavefunction for the full-opt. The full-opt keeps
    the tight defaults, so E(ZM) is unaffected. Loosen all four TOGETHER: the
    step-size criteria gate convergence too, so relaxing the force limits alone
    does not stop the crawl.

    coords_file: the @INCLUDE'd geometry, default 'coords.inc'. Point it
    elsewhere to run a standalone geo-opt from a DIFFERENT start without
    clobbering the job dir's original coords.inc, which is the provenance of
    what was first submitted. [added 2026-08-19 for the dft_run1 SCF rescue:
    six configs had a CONVERGED cell-opt whose .restart had since been cleaned
    off the cluster, so their geo-opt start was rebuilt from the surviving
    cell-opt trajectory into coords_cellopt.inc instead of re-running the
    cell-opt.]

    fixed_atoms: (first, last) 1-based inclusive — emits &CONSTRAINT/
    &FIXED_ATOMS/LIST first..last, the frozen-framework constrained screen of
    the v0 binding pipeline (MOR/oldbinding/setup_all.py SCREEN_MOTION; range
    form from FAU/BindingEnergies opt1.inp). Contiguous-range-only is a
    deliberate constraint: the assembly convention (framework, then cations,
    then guest LAST) makes every freeze a prefix range, and anything else in a
    Foundations input is a bug. [Foundations 2026-07-09]
    """
    constraint = ""
    if fixed_atoms is not None:
        first, last = int(fixed_atoms[0]), int(fixed_atoms[1])
        if first < 1 or last < first:
            raise ValueError("fixed_atoms must be (first, last), 1-based "
                             "inclusive, got %r" % (fixed_atoms,))
        constraint = """\
  &CONSTRAINT
    &FIXED_ATOMS
      LIST %d..%d
    &END FIXED_ATOMS
  &END CONSTRAINT
""" % (first, last)
    head = """\
&GLOBAL
  PROJECT %s
  RUN_TYPE GEO_OPT
  PRINT_LEVEL LOW
&END GLOBAL

&MOTION
  &GEO_OPT
    OPTIMIZER %s
    MAX_ITER %d
    MAX_FORCE %s
    RMS_FORCE %s
    MAX_DR %s
    RMS_DR %s
  &END GEO_OPT
%s  &PRINT
    &TRAJECTORY
      FORMAT XYZ
    &END TRAJECTORY
    &FORCES
      FORMAT XYZ
    &END FORCES
    &RESTART_HISTORY OFF
    &END RESTART_HISTORY
  &END PRINT
&END MOTION

&FORCE_EVAL
  METHOD Quickstep
""" % (project, optimizer, max_iter, max_force, rms_force,
       max_dr, rms_dr, constraint)
    body = head + dft_section(basis_rel=basis_rel, charge=charge, uks=uks,
                              multiplicity=multiplicity,
                              wfn_restart=wfn_restart, scf_outer=scf_outer, **dft_kw) \
                + "\n" + subsys_section(cell_abc, elements,
                                        coords_file=coords_file,
                                        angles=angles, symmetry=symmetry)
    if restart_from:
        body += """
&EXT_RESTART
  RESTART_FILE_NAME %s
  RESTART_POS
  RESTART_CELL
  RESTART_DEFAULT FALSE
&END EXT_RESTART
""" % restart_from
    return body


def energy_force_input(cell_abc, elements, basis_rel="../../../",
                       project="energy-force", charge=0, uks=False,
                       multiplicity=None, restart_from=None,
                       angles=(90.0, 90.0, 90.0), symmetry="ORTHORHOMBIC",
                       scf_outer=None, **dft_kw):
    """
    ENERGY_FORCE single point printing forces + stress to files (the pristine
    E/F/stress training frame — final-sp-example.inp shape). restart_from: a
    cell-opt .restart to pull pos+cell from (final-SP-after-cell-opt); None
    runs on coords.inc as given. angles/symmetry: FAU rhombohedral support.
    """
    body = """\
&GLOBAL
  PROJECT %s
  RUN_TYPE ENERGY_FORCE
  PRINT_LEVEL MEDIUM
&END GLOBAL

&FORCE_EVAL
  METHOD Quickstep
  STRESS_TENSOR Analytical
  &PRINT
    &FORCES
      FILENAME forces
    &END FORCES
    &STRESS_TENSOR
      FILENAME stress
    &END STRESS_TENSOR
  &END PRINT
""" % project
    body += dft_section(basis_rel=basis_rel, charge=charge, uks=uks,
                        multiplicity=multiplicity, aspc=False,
                        scf_outer=scf_outer, **dft_kw) \
          + "\n" + subsys_section(cell_abc, elements,
                                  angles=angles, symmetry=symmetry)
    if restart_from:
        body += """
&EXT_RESTART
  RESTART_FILE_NAME %s
  RESTART_DEFAULT F
  RESTART_POS T
  RESTART_CELL T
&END EXT_RESTART
""" % restart_from
    return body


# ═════════════════════════════════════════════════════════════════════════════
# Output parsing
# ═════════════════════════════════════════════════════════════════════════════

# CP2K stress-print units by version (tests/cp2k_image_parity, 2026-08-20):
# 2022.1 -> GPa, 2025.2 and 2026.1 -> bar. Values are otherwise identical.
STRESS_UNIT_TO_BAR = {"bar": 1.0, "GPa": 1.0e4, "MPa": 10.0, "Pa": 1.0e-5}

ENERGY_TAG = "ENERGY| Total FORCE_EVAL"   # stable across 5.1 and 2026 images


def energies_ha(out_path):
    """All 'ENERGY| Total FORCE_EVAL' values (Ha), in order."""
    es = []
    for ln in open(out_path, errors="replace"):
        if ENERGY_TAG in ln:
            es.append(float(ln.split()[-1]))
    return es


def final_energy_ha(out_path):
    """Last 'ENERGY| Total FORCE_EVAL' (Ha), or None if absent."""
    es = energies_ha(out_path)
    return es[-1] if es else None


def final_energy_ev(out_path):
    e = final_energy_ha(out_path)
    return None if e is None else e * HARTREE_TO_EV


def program_ended_ok(out_path):
    """True iff CP2K printed 'PROGRAM ENDED' (run finished, not walltime-killed)."""
    if not os.path.exists(out_path):
        return False
    return any("PROGRAM ENDED" in ln for ln in open(out_path, errors="replace"))


RUNTIME_ROW = re.compile(r"^ CP2K\s+\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s*$")
_STAMP = re.compile(r"PROGRAM (STARTED|ENDED) AT\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")


def run_time_seconds(out_path):
    """
    Wall-clock seconds CP2K actually ran, or None if the run did not finish.

    Two independent sources, preferred in order:

    1. the ``CP2K`` row of the closing T I M I N G table — its last column is
       the maximum total time over ranks, CP2K's own wall-clock number;
    2. the ``PROGRAM STARTED AT`` / ``PROGRAM ENDED AT`` banner timestamps.

    Both are present in every completed run and agree to well under a second
    (pinned in selftest on MFI/AIMD_out/md.out: 292233.2 s vs the 3 d 09:10:34
    banner delta). (2) exists as the fallback because the timing table can be
    suppressed (&GLOBAL PRINT_LEVEL LOW) while the banners never are.

    Returns None — never a partial time — for a run that was walltime-killed or
    is still going: neither marker is written until CP2K exits cleanly, so a
    number here always means "this is how long a COMPLETE run took". Callers
    sizing a walltime need exactly that; treating a killed job's elapsed time as
    a duration would bias any budget downward. [2026-08-24]
    """
    if not os.path.exists(out_path):
        return None
    started = ended = None
    timed = None
    with open(out_path, errors="replace") as fh:
        for ln in fh:
            m = RUNTIME_ROW.match(ln.rstrip("\n"))
            if m:
                timed = float(m.group(1))
                continue
            m = _STAMP.search(ln)
            if m:
                if m.group(1) == "STARTED":
                    started = m.group(2)
                else:
                    ended = m.group(2)
    if timed is not None:
        return timed
    if started and ended:
        from datetime import datetime
        fmt = "%Y-%m-%d %H:%M:%S.%f"
        return (datetime.strptime(ended, fmt)
                - datetime.strptime(started, fmt)).total_seconds()
    return None


def opt_completed(out_path):
    """
    True iff the geometry/cell optimisation reported convergence.

    CP2K ends a CONVERGED optimisation via either of TWO banners, and both
    count:

    * ``OPTIMIZATION COMPLETED`` — the standard exit, when the geometry
      criteria (MAX_DR / MAX_FORCE / RMS_* / pressure) are all satisfied.
    * ``run CONVERGED!`` — the **L-BFGS optimiser's own** criteria
      (WANTED_PROJ_GRADIENT + WANTED_REL_F_ERROR) firing FIRST. CP2K prints
      a "Specific L-BFGS convergence criteria ... satisfied .... run
      CONVERGED!" block, reevaluates the energy at the minimum, and stops
      WITHOUT ever printing the standard banner.

    The second path was found on the Foundations f1 FAU cell-opts
    (2026-07-22): every FAU Si11 config converged this way at ~82 steps and
    was misread as non-convergent, triggering pointless rescues. MOR
    cell-opts of the same wave took the standard path (~193 steps), so the
    two coexist in one campaign — which is why BOTH must be accepted.

    NB the match is CASE-SENSITIVE on purpose: the SCF's own
    ``*** SCF run converged in N steps ***`` is lowercase and must NEVER
    satisfy a geometry-convergence check.
    """
    if not os.path.exists(out_path):
        return False
    return any(("OPTIMIZATION COMPLETED" in ln) or ("run CONVERGED!" in ln)
               for ln in open(out_path, errors="replace"))


def read_forces_au(job_dir):
    """
    Forces (Ha/Bohr) from the ENERGY_FORCE print file (*forces*1_0.xyz /
    *-forces-*.xyz) in job_dir: list of (fx, fy, fz), or None. Multiply by
    constants.HA_BOHR_TO_EV_ANG for eV/Å. [collect_train.py]
    """
    fs = (glob.glob(os.path.join(job_dir, "*forces*1_0.xyz"))
          or glob.glob(os.path.join(job_dir, "*-forces-*.xyz")))
    if not fs:
        return None
    # A RERUN IN THE SAME DIR APPENDS: CP2K does not truncate this file, so a
    # job submitted twice leaves 2*N rows. Take the LAST block, matching
    # final_energy_ha's "last energy wins" — concatenating them silently
    # produced a 294-row force array for a 147-atom system
    # (tests/cp2k_image_parity, 2026-08-20; duplicate submissions are a
    # recurring incident here, so this must be handled, not assumed away).
    blocks, rows = [], []
    for ln in open(fs[0]):
        if "ATOMIC FORCES" in ln:
            if rows:
                blocks.append(rows)
            rows = []
            continue
        p = ln.split()
        if len(p) == 6 and p[0].isdigit():     # "<atom> <kind> <El> fx fy fz"
            rows.append((float(p[3]), float(p[4]), float(p[5])))
    if rows:
        blocks.append(rows)
    return blocks[-1] if blocks else None


def read_stress_ase_ev_ang3(job_dir, pattern="*stress*"):
    """
    3x3 stress from the stress print file (glob `pattern`), row-major 9 floats
    in eV/Å³, ASE/MACE SIGN convention. CP2K's printed stress is sign-OPPOSITE
    to ASE (verified on the na_training_set: compressed start cell, positive
    physical pressure, prints +Tr in CP2K but must be -Tr in ASE) — NEGATES.
    If your MACE config uses 'virial' instead: virial = -stress_ASE * V.
    Pass a tighter pattern (e.g. 'energy-force*stress*') when the dir also
    holds a MOTION per-step stress print from an optimisation stage.
    [collect_train.py; pattern arg added for stage1a_v2 mixed dirs 2026-07-07]
    """
    fs = glob.glob(os.path.join(job_dir, pattern))
    if not fs:
        return None
    L = open(fs[0], errors="replace").readlines()
    # Same append-on-rerun issue as read_forces_au: scan for the LAST tensor,
    # not the first, so energy/forces/stress all describe the same run.
    hits = [i for i, x in enumerate(L) if "Analytical stress tensor" in x]
    for i in hits[-1:]:
        x = L[i]
        if True:
            # CP2K CHANGED THIS UNIT BETWEEN VERSIONS: 2022.1 prints [GPa],
            # 2025.2/2026.1 print [bar] (same numbers, 1 GPa = 1e4 bar).
            # Discovered 2026-08-20 in tests/cp2k_image_parity, where the
            # identical input printed both. The old code required "[bar]" and
            # returned None otherwise — i.e. stress silently VANISHED from any
            # 2022.1 harvest, which is what the July Perlmutter migration would
            # have produced. Unknown unit RAISES (rule 7), never returns None.
            m = re.search(r"\[([^\]]+)\]", x)
            unit = m.group(1).strip() if m else None
            if unit not in STRESS_UNIT_TO_BAR:
                raise ValueError(
                    "%s: unrecognised stress unit %r in %r — add it to "
                    "STRESS_UNIT_TO_BAR rather than guessing"
                    % (fs[0], unit, x.strip()))
            to_bar = STRESS_UNIT_TO_BAR[unit]
            try:                    # rows i+2..i+4 = 'STRESS| x/y/z <xx> <xy> <xz>'
                vals = []
                for r in (2, 3, 4):
                    vals += [-float(v) * to_bar * BAR_TO_EV_ANG3
                             for v in L[i + r].split()[-3:]]
                return vals
            except (IndexError, ValueError):
                return None
    return None


def read_last_cell_abc(cell_file):
    """
    Final orthorhombic ABC (Å) from a MOTION &CELL print file (project-1.cell):
    last data row, columns 2/6/10 are the diagonal a_x b_y c_z.
    """
    last = None
    for ln in open(cell_file):
        if not ln.lstrip().startswith("#") and ln.split():
            last = ln.split()
    if last is None:
        return None
    return [float(last[2]), float(last[6]), float(last[10])]


def read_last_cell_vectors(cell_file):
    """
    Final FULL 3x3 cell (Å) from a MOTION &CELL print file: last data row,
    columns 2..10 = Ax Ay Az Bx By Bz Cx Cy Cz (rows = lattice vectors, ASE
    convention — equals diag(read_last_cell_abc) for an orthorhombic run,
    selftest-pinned). Needed for FAU's rhombohedral cell-opts, where the
    off-diagonals carry the cell. [Foundations 2026-07-09]
    """
    last = None
    for ln in open(cell_file):
        if not ln.lstrip().startswith("#") and ln.split():
            last = ln.split()
    if last is None:
        return None
    v = [float(x) for x in last[2:11]]
    return [v[0:3], v[3:6], v[6:9]]
