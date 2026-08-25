"""
constants.py — unit conversions and project-wide fixed tables.

Unit-conversion values are the exact ones the na_training_set harvest was built
with (tests/na_training_set/collect_train.py) — do not "improve" them, training
data consistency depends on them.
"""

# ── Unit conversions ────────────────────────────────────────────────────────
HARTREE_TO_EV = 27.211386245988        # Ha -> eV
HA_BOHR_TO_EV_ANG = 51.42208619        # Ha/Bohr -> eV/Å (forces)
BAR_TO_EV_ANG3 = 1.0e-4 / 160.2176634  # bar -> GPa (1e-4) -> eV/Å³ (stress)
EV_TO_KJ_PER_MOL = 96.48533212331
HARTREE_TO_KJ_PER_MOL = HARTREE_TO_EV * EV_TO_KJ_PER_MOL
KCAL_TO_KJ = 4.184

# ── CP2K production protocol (2026-06-23/25, tests/opt_convergence) ─────────
# CELL_OPT: LBFGS · CUTOFF 1500 · REL_CUTOFF 100 · EPS_SCF 1e-7 · PBE-D3 ·
# DZVP-MOLOPT-SR-GTH · GTH. Single source; cp2k.py builds inputs from these.
CUTOFF_RY = 1500
REL_CUTOFF_RY = 100
EPS_SCF = "1.0E-7"
CELL_OPT_OPTIMIZER = "LBFGS"

# ── Basis / pseudopotential table (PBE GTH) ─────────────────────────────────
# Provenance: MOR/pipeline_archive/stage1a_v1/cell-opt-example.inp (archived
# 2026-07-08 with the v1 chain) + MOR/binding/setup_fullopt.py.
BASIS_SET = "DZVP-MOLOPT-SR-GTH"
GTH_POTENTIAL = {
    "Si": "GTH-PBE-q4",  "Al": "GTH-PBE-q3",  "O":  "GTH-PBE-q6",
    "H":  "GTH-PBE-q1",  "Na": "GTH-PBE-q9",  "C":  "GTH-PBE-q4",
    "N":  "GTH-PBE-q5",  "I":  "GTH-PBE-q7",  "Cl": "GTH-PBE-q7",
    "Ag": "GTH-PBE-q11", "Cu": "GTH-PBE-q11", "Pb": "GTH-PBE-q4",
    "Bi": "GTH-PBE-q5",
}

# ── Guest molecules and spin multiplicities ─────────────────────────────────
MULTIPLICITY = {
    "I2": 1, "HI": 1, "H2O": 1, "CH3I": 1, "Cl2": 1, "NO2": 2, "NO3": 2,
}
MOLECULES = sorted(MULTIPLICITY)

# ── Cation-exchanged MOR combos (dir names under MOR/Al-MOR/) ───────────────
# Combo dir = "<cation>_<ratio>", e.g. Ag_4.33. Ratios are Si:Al.
COMBOS = {
    "Ag": ["4.33", "5", "7", "11", "15"],
    "Cu": ["4.33", "5", "7", "11", "15"],
    "Pb": ["5", "7", "11"],
    "Bi": ["4.33", "7", "15"],
}
COMBO_NAMES = [c + "_" + r for c in sorted(COMBOS) for r in COMBOS[c]]

# ── Per-framework combo map (Foundations, 2026-07-09) ───────────────────────
# The historical charge-balanced (cation, Si:Al) matrices for both frameworks.
# FAU dir names FAUn encode Si:Al = n — verified 2026-07-09 against the
# FAU/FullOpt_out compositions (n_al 16/12/8/6/4 for ratios 2/3/5/7/11).
# Pb2+ needs n_al even; Bi3+ needs n_al % 3 == 0 — which is exactly why the
# historical maps are 16 (MOR) and 17 (FAU) combos rather than 20.
# COMBOS/COMBO_NAMES above stay as the MOR aliases for frozen consumers.
COMBOS_BY_FRAMEWORK = {
    "MOR": COMBOS,
    "FAU": {
        "Ag": ["2", "3", "5", "7", "11"],
        "Cu": ["2", "3", "5", "7", "11"],
        "Pb": ["2", "3", "5", "7", "11"],
        "Bi": ["3", "7"],
    },
}
N_T = {"MOR": 48, "FAU": 48}   # T sites per cell


def combo_names(framework):
    """['Ag_15', ...] for a framework, sorted like COMBO_NAMES."""
    cm = COMBOS_BY_FRAMEWORK[framework]
    return [c + "_" + r for c in sorted(cm) for r in cm[c]]


def n_al_for_ratio(framework, ratio):
    """
    Al count for a Si:Al ratio string on the framework's N_T-site cell:
    n_al = N_T / (r + 1), exactness enforced. "4.33" is the historical dir
    label for exactly 13/3 (39 Si / 9 Al on 48 T).
    """
    from fractions import Fraction
    r = Fraction(13, 3) if ratio == "4.33" else Fraction(ratio)
    n = Fraction(N_T[framework]) / (r + 1)
    if n.denominator != 1:
        raise ValueError("Si:Al %s does not divide the %d-T %s cell"
                         % (ratio, N_T[framework], framework))
    return int(n)
