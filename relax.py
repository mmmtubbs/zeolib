"""
relax.py — ASE (MLIP) relaxation with an EXPLICIT pressure criterion.

Why this exists. An ASE cell filter (FrechetCellFilter / ExpCellFilter) turns
the stress into pseudo-forces on the cell, scaled by `1/cell_factor`
(default cell_factor = number of atoms). Running FIRE with `fmax` then stops a
cell relax once |stress| < fmax * cell_factor / V — for a 148-atom Na-MOR cell
at fmax 0.05 that is ~4.3 KILObar, ~40x looser than CP2K's default
PRESSURE_TOLERANCE of 100 bar. MOR/tests/mace_volume_states_si11 (2026-09-24)
lost its two central readouts to exactly this: relaxes started 1.3 kbar from
MACE equilibrium stopped after ~19 steps without moving the cell, and relaxes
from 2800 A^3 stopped ~60 A^3 short. `stage1a_v2/mace_rank.py --cell variable`
used the same default.

`relax()` therefore stops ONLY when both the real atomic forces are below
`fmax` AND every relaxed stress component is below `pressure_tol_bar`, and a
variable-cell call with no pressure tolerance RAISES (rule 7: the loose
default must never come back silently). Cell angles are held by default
(`keep_angles=True`, the filter's diagonal-strain mask) = CP2K KEEP_ANGLES, the
Stage-1a DFT cell-opt protocol.

ASE is imported lazily, so `import zeolib` stays ASE-free.
"""
import time

import numpy as np

CELL_MODES = ("fixed", "variable")
_DIAG_MASK = [1, 1, 1, 0, 0, 0]          # Voigt: xx yy zz relaxed, shears held


def _bar():
    from ase import units
    return units.bar                      # eV/A^3 per bar


def cell_filter(atoms, keep_angles=True):
    """The ASE cell filter (Frechet, ExpCellFilter on older ASE), diagonal
    strain only when keep_angles."""
    try:
        from ase.filters import FrechetCellFilter as F
    except ImportError:                   # ASE < 3.23
        from ase.constraints import ExpCellFilter as F
    return F(atoms, mask=_DIAG_MASK if keep_angles else None)


def max_stress_bar(atoms, keep_angles=True):
    """Largest |stress| component (bar) over the components a relax moves:
    the three normal stresses when angles are held, all six otherwise."""
    s = np.asarray(atoms.get_stress(voigt=True))
    comps = s[:3] if keep_angles else s
    return float(np.abs(comps).max() / _bar())


def pressure_bar(atoms):
    """Hydrostatic pressure (bar), positive = compressed (wants to expand)."""
    return float(-np.asarray(atoms.get_stress(voigt=True))[:3].mean() / _bar())


def relax(atoms, cell_mode="fixed", fmax=0.05, maxstep=1000,
          pressure_tol_bar=None, keep_angles=True):
    """
    FIRE-relax `atoms` in place (calculator attached). Returns a dict:
    conv, nsteps, fmax_atoms (eV/A, real atomic forces), and for a variable
    cell also max_stress_bar, pressure_bar, pressure_tol_bar.

    cell_mode "fixed": atoms only, converged when fmax_atoms < fmax.
    cell_mode "variable": atoms + cell (angles held unless keep_angles=False),
    converged only when fmax_atoms < fmax AND max_stress_bar < pressure_tol_bar.
    pressure_tol_bar is REQUIRED for "variable" — no default, on purpose.
    """
    from ase.optimize import FIRE
    if cell_mode not in CELL_MODES:
        raise ValueError("cell_mode must be one of %s, got %r" % (CELL_MODES, cell_mode))
    if cell_mode == "variable" and (pressure_tol_bar is None or pressure_tol_bar <= 0):
        raise ValueError("variable-cell relax needs an explicit pressure_tol_bar "
                         "(> 0): the filter's fmax criterion alone accepts ~kbar "
                         "residual stress (zeolib/relax.py docstring)")
    target = cell_filter(atoms, keep_angles) if cell_mode == "variable" else atoms

    def done():
        fa = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
        if fa >= fmax:
            return False
        return (cell_mode == "fixed"
                or max_stress_bar(atoms, keep_angles) < pressure_tol_bar)

    t0 = time.time()
    dyn = FIRE(target, logfile=None)
    conv = done()
    if not conv:
        for _ in dyn.irun(fmax=0.0, steps=maxstep):   # fmax 0: we decide
            if done():
                conv = True
                break
    out = dict(conv=bool(conv), nsteps=int(dyn.nsteps),
               fmax_atoms=float(np.linalg.norm(atoms.get_forces(), axis=1).max()),
               wall_s=round(time.time() - t0, 2))
    if cell_mode == "variable":
        out.update(max_stress_bar=max_stress_bar(atoms, keep_angles),
                   pressure_bar=pressure_bar(atoms),
                   pressure_tol_bar=float(pressure_tol_bar))
    return out
