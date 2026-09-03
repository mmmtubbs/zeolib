"""
molecules.py — the canonical home of the 7 guest-molecule template geometries
(iodides I2 / HI / CH3I, confounders H2O / Cl2 / NO2 / NO3).

Provenance: lifted VERBATIM from MOR/oldbinding/setup_all.py MOLECULES +
MOL_RADIUS (the v0 binding pipeline; the same numbers were re-inlined in
MOR/binding/run_range_all.py and setup_molonly.py). These are idealized
pre-optimisation seeds — every production protocol relaxes them (guest
geo-opt in the framework, mol-only geo-opt for E(M)); FAU-era
BindingMolecules/*.xyz agree in composition. Extracted for Foundations
(2026-07-09) so no script re-inlines them again.

Spin multiplicities live in constants.MULTIPLICITY (NO2/NO3 = 2, rest 1) —
key parity is selftest-pinned.
"""
import numpy as np

from .geometry import mic_dist

# name -> list of (element, x, y, z), centred at the origin
GUESTS = {
    "I2": [("I", 0.000, 0.000, -1.335),
           ("I", 0.000, 0.000, 1.335)],
    "HI": [("H", 0.000, 0.000, 0.000),
           ("I", 0.000, 0.000, 1.609)],
    "H2O": [("O", 0.000, 0.000, 0.000),
            ("H", 0.000, 0.757, -0.586),
            ("H", 0.000, -0.757, -0.586)],
    "CH3I": [("C", 0.000, 0.000, 0.000),
             ("I", 0.000, 0.000, 2.140),
             ("H", 1.026, 0.000, -0.363),
             ("H", -0.513, 0.888, -0.363),
             ("H", -0.513, -0.888, -0.363)],
    "Cl2": [("Cl", 0.000, 0.000, -0.995),
            ("Cl", 0.000, 0.000, 0.995)],
    "NO2": [("N", 0.000, 0.000, 0.000),
            ("O", 1.097, 0.000, -0.477),
            ("O", -1.097, 0.000, -0.477)],
    "NO3": [("N", 0.000, 0.000, 0.000),
            ("O", 1.240, 0.000, 0.000),
            ("O", -0.620, 1.074, 0.000),
            ("O", -0.620, -1.074, 0.000)],
}

# effective radius (Å) for the void-clearance pre-filter [setup_all.py]
MOL_RADIUS = {
    "I2": 2.0, "HI": 1.6, "H2O": 1.4, "CH3I": 2.0,
    "Cl2": 1.8, "NO2": 1.6, "NO3": 1.6,
}


def guest_positions(name):
    """(elements list, (n,3) positions) of the origin-centred template."""
    rows = GUESTS[name]
    return [r[0] for r in rows], np.array([r[1:] for r in rows], float)


def guest_atoms(name, cell=None):
    """ASE Atoms of the origin-centred template; cell/pbc set if given."""
    from ase import Atoms
    elems, pos = guest_positions(name)
    at = Atoms(symbols=elems, positions=pos)
    if cell is not None:
        at.set_cell(cell); at.set_pbc(True)
    return at


def guest_natoms(name):
    """
    Number of ATOMS in the guest template.

    Use this, never ``len(guest_elements(name))`` — guest_elements returns the
    UNIQUE elements for CP2K &KIND blocks (NO3 -> ['N','O'], I2 -> ['I']), so
    it undercounts every guest except HI. Getting that wrong silently slices
    the wrong trailing atoms out of an assembled framework+cations+guest
    structure, which then reads as framework contact at ~1.1 Å when it is
    really an intramolecular bond (Foundations 2026-08-17).
    """
    return len(GUESTS[name])


def guest_elements(name):
    """Unique elements of the guest in first-appearance order (KIND-block
    order) — NOT per-atom symbols and NOT an atom count; see guest_natoms."""
    seen = []
    for el, *_ in GUESTS[name]:
        if el not in seen:
            seen.append(el)
    return seen


# ── Adsorption-state classifier (Foundations 2026-09-03) ────────────────────
# Covalent radii (Å, Cordero 2008) for the guest elements only — used to define
# which template atom pairs are BONDS, never to place anything.
_COV_RADIUS = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "Cl": 1.02, "I": 1.39}
BOND_FACTOR = 1.25      # a template pair is bonded if d <= 1.25 * (r_i + r_j)
STRETCH_DISSOCIATED = 1.30   # a bond stretched to > 1.30 * template length = broken


def template_bonds(name):
    """
    Bonded atom-index pairs of the guest template, [(i, j, d0), ...] with d0
    the template bond length. Defined by covalent-radius sum * BOND_FACTOR on
    the idealized template, which yields exactly the chemical bonds (I2 1, HI
    1, CH3I 4, H2O 2, Cl2 1, NO2 2, NO3 3) and no H-H / O-O / I-H contacts.
    Selftest-pinned.
    """
    elems, pos = guest_positions(name)
    out = []
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d <= BOND_FACTOR * (_COV_RADIUS[elems[i]] + _COV_RADIUS[elems[j]]):
                out.append((i, j, d))
    if not out:
        raise ValueError("%s: template has no bonds — check _COV_RADIUS" % name)
    return out


def adsorption_state(name, guest_pos, cell, stretch=STRETCH_DISSOCIATED):
    """
    Classify a relaxed guest as 'molecular' or 'dissociated' from its OWN bond
    lengths: every template bond is re-measured (minimum-image, general cell)
    in `guest_pos` (the guest's atoms in template order — the trailing
    guest_natoms(name) rows of an assembled structure), and the guest is
    'dissociated' if any bond exceeds `stretch` x its template length.

    Why 1.30: the Foundations f3 winners that stayed intact re-measure at
    1.02-1.04 x template (HI 1.645/1.609, CH3I 2.198/2.140, Cl2 2.049/1.990,
    H2O 0.999/0.957) while the FAU Ag_11 HI winner that lost its proton to a
    framework O sits at 1.41 (2.27 Å). Nothing bound vibrationally reaches
    +30 %. The ratio is returned so the threshold can be revisited without
    recomputing anything.

    Returns dict(state, max_ratio, bond=(i, j, elem_i, elem_j, d, d0)) for the
    most-stretched bond. Raises if guest_pos has the wrong atom count (rule 7:
    a wrong slice must never read as an intact molecule).
    """
    elems, _ = guest_positions(name)
    guest_pos = np.asarray(guest_pos, float)
    if guest_pos.shape != (len(elems), 3):
        raise ValueError("%s: expected %d guest atoms, got %s"
                         % (name, len(elems), guest_pos.shape))
    worst = None
    for i, j, d0 in template_bonds(name):
        d = float(mic_dist(guest_pos[i], guest_pos[j], cell))
        r = d / d0
        if worst is None or r > worst[0]:
            worst = (r, (i, j, elems[i], elems[j], d, d0))
    ratio, bond = worst
    return dict(state="dissociated" if ratio > stretch else "molecular",
                max_ratio=ratio, bond=bond)
