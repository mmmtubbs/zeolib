"""
placement.py — hand-template guest placement: void grid, void snap,
clash-checked random orientations, and ring-normal candidate sites.

Provenance: MOR/oldbinding/setup_all.py + test_placement.py — the v0 "5
hand-picked sites" screen pipeline (GRID_SPACING 0.5, R_PROBE 1.2, CLASH 1.6,
N_ORI 300) — generalized to arbitrary cells through zeolib.geometry dispatch.
For an orthorhombic cell the grid points, snapped sites and clash decisions
are identical to v0's (selftest reproduces a shipped SITE_INFO position).
ring_normal_sites is the FAU/BindingEnergies/prepare_binding.ipynb ±normal
idea, driven by framework.t_rings instead of the notebook's file row order.
Extracted for Foundations f3 (2026-07-09).
"""
import numpy as np

from .geometry import mic_all, unwrap, cell_matrix

GRID_SPACING = 0.5   # Å, void-grid pitch
R_PROBE = 1.2        # Å, min framework clearance for a "void" grid point
CLASH_DIST = 1.6     # Å, guest-atom / framework clash floor
N_ORI = 300          # random orientations tried per site


def build_void_grid(fw_pos, cell, spacing=GRID_SPACING, chunk=4096):
    """
    (grid_pts (N,3) cartesian, min_dist (N,)): a fractional-space grid over
    the cell with, per point, the MIC distance to the nearest framework atom.
    Orthorhombic cells reproduce v0's linspace(0, L, ceil(L/spacing),
    endpoint=False) grid exactly. Build once per structure and reuse across
    guests/sites (the v0 pattern).
    """
    M = cell_matrix(cell)
    n = [max(1, int(np.ceil(np.linalg.norm(M[i]) / spacing))) for i in range(3)]
    fr = [np.arange(ni) / float(ni) for ni in n]
    gx, gy, gz = np.meshgrid(fr[0], fr[1], fr[2], indexing="ij")
    grid_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1) @ M
    fw_pos = np.asarray(fw_pos, float)
    min_dist = np.empty(len(grid_pts))
    for lo in range(0, len(grid_pts), chunk):
        hi = min(lo + chunk, len(grid_pts))
        min_dist[lo:hi] = mic_all(grid_pts[lo:hi], fw_pos, cell).min(axis=1)
    return grid_pts, min_dist


def snap_to_void(target, grid_pts, min_dist, cell, r_probe=R_PROBE):
    """
    Nearest grid point to `target` (MIC) with framework clearance > r_probe.
    Returns (point (3,), clearance) or (None, None) if no void point exists.
    """
    d = mic_all(np.asarray(target, float)[None, :], grid_pts, cell)[0]
    d[min_dist <= r_probe] = np.inf
    idx = int(np.argmin(d))
    if np.isinf(d[idx]):
        return None, None
    return grid_pts[idx].copy(), float(min_dist[idx])


def snap_to_void_k(target, grid_pts, min_dist, cell, r_probe=R_PROBE,
                   k=1, max_shift=None):
    """
    The k nearest grid points to `target` (MIC) with clearance > r_probe,
    ordered nearest-first, optionally restricted to within `max_shift` of the
    target. Returns [(point, clearance), ...] (possibly empty).

    Why k > 1 exists (Foundations 2026-08-17): `snap_to_void` returns the
    single NEAREST qualifying point, so one unlucky snap kills a site with no
    retry — and with the v0 constants it snapped to points that could never
    succeed (see place_guest). Ordering nearest-first keeps the anchor close
    to the template target, which is what keeps the five MOR sites physically
    DISTINCT; `max_shift` makes that a hard guarantee rather than a tendency.
    """
    d = mic_all(np.asarray(target, float)[None, :], grid_pts, cell)[0]
    bad = min_dist <= r_probe
    if max_shift is not None:
        bad = bad | (d > float(max_shift))
    d = np.where(bad, np.inf, d)
    order = np.argsort(d)[:max(1, int(k))]
    return [(grid_pts[i].copy(), float(min_dist[i]))
            for i in order if np.isfinite(d[i])]


def random_rotation(rng):
    """Uniform random rotation matrix from a normalized quaternion
    (numpy Generator). [setup_all.py, verbatim]"""
    q = rng.standard_normal(4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def min_dist_mol_fw(mol_pos, fw_pos, cell):
    """Smallest MIC distance between any guest atom and any framework atom."""
    return float(mic_all(np.asarray(mol_pos, float),
                         np.asarray(fw_pos, float), cell).min())


def place_guest(target, guest_name, fw_pos, cell, rng, grid=None,
                n_ori=N_ORI, clash=CLASH_DIST, r_probe=R_PROBE,
                n_anchors=1, max_shift=None):
    """
    v0 placement recipe for one (site, guest): snap `target` to the void
    grid, then try up to n_ori random orientations of the origin-centred
    guest template until no guest atom sits within `clash` of the framework.

    grid: (grid_pts, min_dist) from build_void_grid, to reuse across calls;
    built on the fly if None. Returns a dict:
      status    'ok' | 'no void' | 'clash'
      mol_pos   (n,3) placed guest positions (None unless 'ok')
      site      snapped anchor point (None only for 'no void')
      clearance framework clearance of the anchor (None only for 'no void')

    DEFAULTS ARE THE v0 RECIPE and are byte-parity pinned — do not change
    them. `n_anchors`/`max_shift` are the opt-in Foundations 2026-08-17
    widening, added after 291 of 1004 f3 sites (29%) were skipped:

    the guest is placed with its CENTROID on the anchor, but the anchor only
    had to clear R_PROBE=1.2 Å while every guest atom must clear
    CLASH_DIST=1.6 Å. Any guest with an atom at its centroid (NO3 0.00 Å,
    CH3I 0.21, NO2 0.32, H2O 0.39) is therefore pre-doomed at such an
    anchor, no matter how many orientations are tried — which is exactly why
    skip rate tracked centroid occupancy (I2, hollow at 1.34 Å, skipped 6%;
    the rest 35-39%) instead of molecular size. Passing
    `r_probe=CLASH_DIST` removes the doomed anchors; `n_anchors>1` retries
    the next-nearest void points instead of giving up on one bad snap; and
    `max_shift` bounds how far the anchor may travel from the template
    target, which is what keeps distinct template sites distinct.
    """
    from .molecules import guest_positions
    if grid is None:
        grid = build_void_grid(fw_pos, cell)
    pts, mind = grid
    if n_anchors == 1 and max_shift is None:
        site, clearance = snap_to_void(target, pts, mind, cell,
                                       r_probe=r_probe)          # v0 path
        cands = [] if site is None else [(site, clearance)]
    else:
        cands = snap_to_void_k(target, pts, mind, cell, r_probe=r_probe,
                               k=n_anchors, max_shift=max_shift)
    if not cands:
        return dict(status="no void", mol_pos=None, site=None, clearance=None)
    _, geom = guest_positions(guest_name)
    for site, clearance in cands:
        for _ in range(n_ori):
            R = random_rotation(rng)
            mol_pos = geom @ R.T + site
            if min_dist_mol_fw(mol_pos, fw_pos, cell) >= clash:
                return dict(status="ok", mol_pos=mol_pos, site=site,
                            clearance=clearance)
    site, clearance = cands[0]
    return dict(status="clash", mol_pos=None, site=site, clearance=clearance)


def ring_normal_sites(fw, ring, offset=2.0):
    """
    The two candidate guest anchors for a T-ring: ring centroid ± offset
    along the ring-plane normal (SVD least-variance direction of the
    unwrapped ring T positions). ring: tuple of T-site indices
    (framework.t_rings output). Returns (2,3) cartesian, unwrapped —
    snap_to_void wraps implicitly via MIC.
    """
    pos = unwrap(fw["pos"][list(ring)], fw["cell"])
    cen = pos.mean(axis=0)
    _, _, vt = np.linalg.svd(pos - cen)
    n = vt[2]
    return np.array([cen + offset * n, cen - offset * n])
