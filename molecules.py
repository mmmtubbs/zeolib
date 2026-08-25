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
