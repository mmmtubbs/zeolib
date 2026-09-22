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
                   scf_outer=None, pressure_tolerance=None, **dft_kw):
    """
    Variable-cell relax, production protocol. Defaults reproduce the
    na_placement_multicomp stage-1 input byte-for-byte: MOTION-level per-step
    TRAJECTORY/FORCES/STRESS/CELL prints (training-frame harvest),
    RESTART_HISTORY OFF, STRESS_TENSOR Analytical.
    keep_symmetry: ONLY for bare-framework re-baselines — an Al/Na decoration
    lowers the space group, so decorated cells must relax with KEEP_ANGLES only.
    angles/symmetry: passed to subsys_section (FAU rhombohedral support).

    pressure_tolerance (bar, 2026-09-17): emits PRESSURE_TOLERANCE. None keeps
    CP2K's default, which is what every production Stage-1a input used and what
    the selftest pins byte-for-byte — pass a value ONLY for a deliberate
    convergence study. Added for MOR/tests/cell_treatment_si11, which measured
    that repeat CELL_OPT of the SAME Na basin scatters by mean 6.9 / max 33.1
    kJ/mol while every run reports CONVERGED: with the default the run is
    accepted anywhere in a 200-bar-wide pressure window, and the arm that
    tightens this separates "tolerance too loose" from "stress numerically
    noisy" (README STATUS 2026-09-17).
    """
    keep = ""
    if keep_symmetry:
        keep += "    KEEP_SYMMETRY TRUE\n"
    if keep_angles:
        keep += "    KEEP_ANGLES TRUE\n"
    if pressure_tolerance is not None:
        keep += "    PRESSURE_TOLERANCE %g\n" % float(pressure_tolerance)
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


def read_input_cell(inp_path):
    """
    The cell a CP2K INPUT declares: {'abc': [a, b, c] (Å), 'angles':
    [alpha, beta, gamma] (deg), 'matrix': 3x3 rows = lattice vectors (Å)}.

    Reads the first `&CELL` block's ABC and ALPHA_BETA_GAMMA (absent -> 90°,
    CP2K's own default). The matrix is `ase.geometry.cellpar_to_cell`, which
    is the orientation CP2K itself constructs from ABC + angles (verified for
    the rhombohedral FAU cell against the FAU-era cell-opt.out, FOUNDATIONS.md
    §2). Needed wherever the geometry of a FIXED-cell job (geo-opt, frozen
    screen, full-opt, energy-force) is written out with its cell: those runs
    print no `.cell` file, and the input's 4-decimal ABC is the cell that
    actually ran — not the f2 relaxed cell it was rounded from.

    No silent fallback (rule 7): an input with no ABC line RAISES, since a
    guessed cell would put every atom in the wrong place.

    Provenance: Foundations 2026-09-22, geometry export for the advisor pack
    (`communication/export_geometries.py`).
    """
    txt = open(inp_path).read()
    # The SUBSYS cell, not the first &CELL in the file: a CELL_OPT input also
    # has a MOTION/PRINT `&CELL ... &END CELL` print key with no ABC in it.
    sub = re.search(r"&SUBSYS\b(.*?)&END\s+SUBSYS", txt, re.S | re.I)
    m = re.search(r"&CELL\b(.*?)&END\s+CELL", sub.group(1) if sub else txt,
                  re.S | re.I)
    if not m:
        raise ValueError("%s: no &CELL block" % inp_path)
    blk = m.group(1)
    a = re.search(r"^\s*ABC\s+(?:\[\w+\]\s+)?(\S+)\s+(\S+)\s+(\S+)", blk,
                  re.M | re.I)
    if not a:
        raise ValueError("%s: &CELL has no ABC line" % inp_path)
    g = re.search(r"^\s*ALPHA_BETA_GAMMA\s+(?:\[\w+\]\s+)?(\S+)\s+(\S+)"
                  r"\s+(\S+)", blk, re.M | re.I)
    abc = [float(x) for x in a.groups()]
    ang = [float(x) for x in g.groups()] if g else [90.0, 90.0, 90.0]
    from ase.geometry.cell import cellpar_to_cell
    M = cellpar_to_cell(abc + ang)
    return {"abc": abc, "angles": ang, "matrix": [list(map(float, r)) for r in M]}


def opt_frame_cells(inp_path, n_frames, cell_file=None, restart_cell=None):
    """
    The cell of EVERY frame of an optimisation's `<project>-pos-1.xyz`, as a
    list of n_frames 3x3 matrices (rows = lattice vectors, Å).

    * CELL_OPT (`cell_file` = its `<project>-1.cell`): frame k sits at .cell
      row k, and the one extra LAST frame is CP2K's re-print of the final
      geometry (identical positions to the frame before it — verified on all
      33 Foundations f2 cell-opts), so it takes the last row's cell. The
      trajectory does NOT contain the starting geometry: frame 1's energy is
      the SECOND energy evaluation in the .out (MOR Cu_5 f2: the first,
      -2105.3546 Ha, is the input structure; frame 1 is -2105.5643). So
      n_frames must be rows + 1, and anything else RAISES — a restarted,
      appended cell-opt would misalign silently otherwise.
    * Fixed-cell jobs (GEO_OPT, screen, full-opt, mol-only): every frame is at
      one cell. That is the input's ABC UNLESS the input restarts its cell
      from ANOTHER job (`&EXT_RESTART` + `RESTART_CELL` naming a different
      project's restart — the f2 geo-opt reads `cell-opt-1.restart`, whose
      input ABC is still the cell-opt's STARTING cell). Then the caller must
      pass `restart_cell` (e.g. `read_last_cell_vectors` of that cell-opt),
      or this RAISES. A restart from the job's OWN `<project>-1.restart` (a
      resumed fixed-cell run) keeps the input cell.

    Provenance: Foundations 2026-09-22, trajectory export for the advisor
    report — the f2 geo-opt's input ABC (17.8882 Å for MOR Cu_5) differs from
    the cell it actually ran at (17.5709 Å), which is how this was found.
    """
    txt = open(inp_path).read()
    base = read_input_cell(inp_path)["matrix"]
    if cell_file is not None:
        rows = []
        for ln in open(cell_file):
            if ln.lstrip().startswith("#") or not ln.split():
                continue
            v = [float(x) for x in ln.split()[2:11]]
            rows.append([v[0:3], v[3:6], v[6:9]])
        if n_frames != len(rows) + 1:
            raise ValueError("%s: %d trajectory frames but %d .cell rows "
                             "(expected rows + 1)" % (cell_file, n_frames,
                                                      len(rows)))
        return rows + [rows[-1]]
    m = re.search(r"&EXT_RESTART(.*?)&END\s+EXT_RESTART", txt, re.S | re.I)
    if m and re.search(r"^\s*RESTART_CELL(\s+(T|TRUE|\.TRUE\.))?\s*$",
                       m.group(1), re.M | re.I):
        f = re.search(r"RESTART_FILE_NAME\s+(\S+)", m.group(1), re.I)
        proj = re.search(r"^\s*PROJECT(?:_NAME)?\s+(\S+)", txt, re.M | re.I)
        own = (f and proj and os.path.basename(f.group(1))
               == "%s-1.restart" % proj.group(1))
        if not own:
            if restart_cell is None:
                raise ValueError("%s restarts its cell from %s — pass "
                                 "restart_cell" % (inp_path,
                                                   f.group(1) if f else "?"))
            base = [list(map(float, r)) for r in restart_cell]
    return [base] * n_frames


# ── Job-directory hygiene (Foundations 2026-09-03) ─────────────────────────
# A package generator that can be RE-RUN over a tree where some jobs have
# already finished has exactly one dangerous failure mode: rewriting the
# inputs while leaving the old outputs. The dir then looks converged, and the
# stale result is silently adopted. That is what happened between the two
# 2026-08-17 Foundations f3 gen runs (245 dirs), so both halves of the guard
# live here now: clear the products whenever inputs are rewritten, and never
# trust an output older than the inputs sitting beside it.
# (Foundations/f3_guests.py keeps its own inlined copy — it is mid-campaign on
# the cluster and is migrated when next touched, per zeolib README rule 4.)
GENERATED_SUFFIXES = (".out", ".wfn", ".restart", ".Hessian",
                      "-1.cell", "-pos-1.xyz", "-frc-1.xyz")
GENERATED_PREFIXES = ("cp2k_", "slurm-")


def clear_generated(job_dir, suffixes=GENERATED_SUFFIXES,
                    prefixes=GENERATED_PREFIXES):
    """
    Delete CP2K products from `job_dir` so a rewritten job starts clean.
    Returns the number of files removed; a missing dir is 0, not an error.

    Call this on EVERY dir whose inputs you rewrite. Leaving the old `.out`
    behind is not a cosmetic problem: the next pass reads it as a converged
    result for inputs it never saw.
    """
    if not os.path.isdir(job_dir):
        return 0
    n = 0
    for f in os.listdir(job_dir):
        if f.endswith(suffixes) or f.startswith(prefixes):
            try:
                os.remove(os.path.join(job_dir, f))
                n += 1
            except OSError:
                pass
    return n


def output_is_current(job_dir, out_name, extra_inputs=("coords.inc",)):
    """
    True if `out_name` was written AFTER every input now in `job_dir`
    (any `*.inp` plus `extra_inputs`). False when the output is missing.

    Timestamps, not geometry: comparing a trajectory's first frame against
    coords.inc looks appealing but is invalid — CP2K's first printed frame
    differs from the input by ~0.1 A on a perfectly clean run, so no tolerance
    separates "clean" from "stale".
    """
    out = os.path.join(job_dir, out_name)
    if not os.path.exists(out):
        return False
    t_out = os.path.getmtime(out)
    for f in os.listdir(job_dir):
        if f.endswith(".inp") or f in tuple(extra_inputs):
            if os.path.getmtime(os.path.join(job_dir, f)) > t_out:
                return False
    return True


_ABS_SPIN = re.compile(r"Integrated absolute spin density:\s+([-\d.Ee+]+)")


def read_abs_spin_density(out_path):
    """
    The LAST ``Integrated absolute spin density`` value (electrons) printed by a
    spin-unrestricted run, or None when the line never appears (an RKS run, or
    a job that died before its first SCF finished).

    This is the integral of |rho_alpha - rho_beta| over the cell: the direct,
    basis-independent measure of how much unpaired spin the solution carries. A
    bare-UKS closed-shell system that stayed closed-shell prints ~1e-9; a
    broken-symmetry one prints O(1).

    Provenance: Foundations f5 (2026-09-17). The all-UKS rescore of the
    Foundations table needed a diagnostic that separates a GENUINE
    broken-symmetry solution from a cold-start SCF that merely landed on a
    worse one, because on a multiplicity-1 term E_UKS <= E_RKS makes a positive
    dE provably an instrument fault. Spin settles it: the 24 real instabilities
    all carry O(1) spin while every cold-start artifact carries exactly 0.
    """
    if not os.path.exists(out_path):
        return None
    val = None
    with open(out_path, errors="replace") as fh:
        for ln in fh:
            m = _ABS_SPIN.search(ln)
            if m:
                val = float(m.group(1))
    return val


def read_mulliken_spin(out_path):
    """
    Per-atom Mulliken populations from the LAST ``Mulliken Population
    Analysis`` block: list of ``(index, element, net_charge, spin_moment)``,
    or None when no block is present.

    LAST block wins, matching `final_energy_ha` and `read_forces_au` — a dir
    re-run in place holds several, and only the final one describes the
    reported energy.

    The spin-unrestricted header is ``Atomic population (alpha,beta) Net charge
    Spin moment`` (6 numeric columns after the element/kind); an RKS run prints
    only ``Atomic population  Net charge`` (2 columns) and then spin_moment
    comes back None per row, so a caller summing |spin| must guard for it
    rather than read a zero that was never computed.

    Provenance: Foundations f5 (2026-09-17), together with
    `read_abs_spin_density` — the per-atom breakdown is what attributes an
    instability to specific atoms (e.g. Cu 1.09 + Cl 0.30 for the Cu(I)->Cl2
    charge transfer), which the integrated number alone cannot do.
    """
    if not os.path.exists(out_path):
        return None
    blocks, rows, inside = [], [], False
    with open(out_path, errors="replace") as fh:
        for ln in fh:
            if "Mulliken Population Analysis" in ln:
                if rows:
                    blocks.append(rows)
                rows, inside = [], True
                continue
            if not inside:
                continue
            if "Total charge and spin" in ln or "Total charge" in ln:
                inside = False
                if rows:
                    blocks.append(rows)
                    rows = []
                continue
            p = ln.split()
            # "<idx> <El> <kind> <alpha> <beta> <net> <spin>" (UKS) or
            # "<idx> <El> <kind> <pop> <net>" (RKS)
            if len(p) >= 5 and p[0].isdigit() and not p[1].isdigit():
                try:
                    if len(p) >= 7:
                        rows.append((int(p[0]), p[1], float(p[5]),
                                     float(p[6])))
                    else:
                        rows.append((int(p[0]), p[1], float(p[4]), None))
                except ValueError:
                    pass
    if rows:
        blocks.append(rows)
    return blocks[-1] if blocks else None
