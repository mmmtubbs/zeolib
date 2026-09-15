"""
geometry.py — minimum-image-convention (MIC) geometry helpers.

Every function takes the cell explicitly, in either of two forms:

  * length-3 array of box LENGTHS (Å) — the orthorhombic fast path; behaviour
    identical to the pre-Foundations orthorhombic-only module (round-MIC).
  * (3,3) cell MATRIX (rows = lattice vectors, ASE convention) — the general
    path (Foundations 2026-07-09, needed for FAU's rhombohedral α=60° cell).
    A diagonal matrix is detected and routed to the fast path. The general MIC
    is fractional rounding followed by an exact minimisation over the 27
    neighbour images: rounding alone is NOT exact for skewed cells (FAU's
    primitive cell is the fcc lattice, the classic counterexample), while the
    ±1 image shell is exact for any cell whose skew is zeolite-like (parity
    with ase.geometry.get_distances pinned in selftest).

Provenance: tests/na_placement_multicomp/common.py (mic, mic_all),
MOR/binding/run_range_all.py (centroid_unwrapped — the PBC centroid bug fix),
MOR/pipeline_archive/stage1a_v1/mor_core.py (mic_vec, min/mean pair
distances). General-cell dispatch new for Foundations (FAU leg).
`species_match_max_disp` / `same_structure` (permutation-invariant structure
comparison) new for the Stage-1a v2 top-K carry-k dedup, 2026-09-10.
"""
import numpy as np

_SHIFTS27 = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1)
                      for k in (-1, 0, 1)], float)


def _cell(cell):
    """Dispatch: ('ortho', (3,) lengths) or ('general', (3,3) matrix)."""
    c = np.asarray(cell, float)
    if c.shape == (3,):
        return "ortho", c.ravel()
    if c.shape == (3, 3):
        d = np.diag(c)
        if np.allclose(c, np.diag(d), atol=1e-10):
            return "ortho", d.copy()
        return "general", c
    raise ValueError("cell must be 3 box lengths or a (3,3) matrix, got %r"
                     % (cell,))


def cell_matrix(cell):
    """The (3,3) cell matrix for either cell form (diag for box lengths)."""
    kind, c = _cell(cell)
    return np.diag(c) if kind == "ortho" else c.copy()


def perp_widths(cell):
    """Perpendicular width of the cell along each lattice direction
    (V / |a_j x a_k| — the face-to-face distance), which is what determines
    how many periodic images a cutoff sphere needs in a skewed cell."""
    M = cell_matrix(cell)
    V = abs(np.linalg.det(M))
    return np.array([V / np.linalg.norm(np.cross(M[(i + 1) % 3], M[(i + 2) % 3]))
                     for i in range(3)])


def cell_abc_angles(cell):
    """(a, b, c) lengths and (alpha, beta, gamma) angles in degrees — the pair
    CP2K's &CELL wants as ABC / ALPHA_BETA_GAMMA. Crystallographic convention:
    alpha is the angle between b and c, beta between c and a, gamma between a
    and b. Orthorhombic cells come back as exactly 90/90/90; FAU's rhombohedral
    primitive cell as 60/60/60. [added 2026-08-19 for tests/cp2k_image_parity,
    which rebuilds CP2K inputs from harvested extxyz frames whose cells are
    stored as (3,3) matrices]"""
    M = cell_matrix(cell)
    L = np.linalg.norm(M, axis=1)
    if np.any(L <= 0):
        raise ValueError("degenerate cell: zero-length lattice vector")
    ang = np.empty(3)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        c = float(M[j] @ M[k] / (L[j] * L[k]))
        ang[i] = np.degrees(np.arccos(min(1.0, max(-1.0, c))))
    return L, ang


def mic_vec(a, b, cell):
    """Minimum-image displacement vector b-a."""
    d = np.asarray(b, float) - np.asarray(a, float)
    kind, c = _cell(cell)
    if kind == "ortho":
        return d - np.round(d / c) * c
    f = d @ np.linalg.inv(c)
    f -= np.round(f)
    cands = (f + _SHIFTS27) @ c
    return cands[int(np.argmin(np.einsum("ij,ij->i", cands, cands)))]


def mic_dist(a, b, cell):
    """Minimum-image distance |b-a|."""
    return float(np.linalg.norm(mic_vec(a, b, cell)))


def mic_all(P, Q, cell):
    """All pairwise MIC distances. P:(m,3) Q:(n,3) -> (m,n)."""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    kind, c = _cell(cell)
    D = P[:, None, :] - Q[None, :, :]
    if kind == "ortho":
        D -= np.round(D / c) * c
        return np.linalg.norm(D, axis=2)
    inv = np.linalg.inv(c)
    F = D @ inv
    F -= np.round(F)
    best = None
    for s in _SHIFTS27:
        V = (F + s) @ c
        d2 = np.einsum("mnk,mnk->mn", V, V)
        best = d2 if best is None else np.minimum(best, d2)
    return np.sqrt(best)


def unwrap(positions, cell, ref=0):
    """
    Copy of `positions` with every atom shifted by lattice vectors to the
    periodic image nearest positions[ref]. Use before averaging/fitting a
    group that may straddle a boundary.
    """
    pos = np.asarray(positions, float).copy()
    kind, c = _cell(cell)
    if kind == "ortho":
        d = pos - pos[ref]
        pos -= np.round(d / c) * c
        return pos
    r = pos[ref].copy()
    for i in range(len(pos)):
        pos[i] = r + mic_vec(r, pos[i], c)
    return pos


def wrap_to_cell(positions, cell):
    """Positions wrapped into the home cell (fractional coords in [0,1)).
    Orthorhombic path is `positions % lengths`, exactly the pre-Foundations
    inline wraps it replaces."""
    pos = np.asarray(positions, float)
    kind, c = _cell(cell)
    if kind == "ortho":
        return pos % c
    f = pos @ np.linalg.inv(c)
    return (f - np.floor(f)) @ c


def wrap_preserving_groups(positions, cell, groups=()):
    """
    Wrap into the home cell WITHOUT breaking bonded groups.

    Free atoms (framework, single-atom cations) wrap individually. Each group
    in `groups` — a sequence of index sequences, e.g. one guest molecule — is
    unwrapped relative to its first atom and then shifted by the SINGLE
    lattice translation that brings its centroid inside the cell. The group
    stays intact, where a naive per-atom wrap would tear a molecule straddling
    a cell face into fragments at opposite corners.

    Provenance: Foundations 2026-08-17, writing wrapped structure sets for
    review — the f3 framework+cation+guest structures are exactly this case.
    Same hazard class as the mol_centroid PBC bug that `centroid_unwrapped`
    exists to fix.
    """
    pos = np.asarray(positions, float).copy()
    grouped = set()
    for g in groups:
        grouped.update(int(i) for i in g)
    free = [i for i in range(len(pos)) if i not in grouped]
    if free:
        pos[free] = wrap_to_cell(pos[free], cell)
    for g in groups:
        idx = [int(i) for i in g]
        if not idx:
            continue
        whole = unwrap(pos[idx], cell, ref=0)
        cen = whole.mean(axis=0)
        shift = wrap_to_cell(cen.reshape(1, 3), cell)[0] - cen
        pos[idx] = whole + shift
    return pos


def centroid_unwrapped(positions, cell, ref=0):
    """
    Centroid of a group of atoms, unwrapped relative to positions[ref].

    Without unwrapping, an atom at x=0.1 and its image at x=A+0.1 give a
    centroid halfway across the cell, so two identical placements can appear
    >5 Å apart (the mol_centroid bug fixed in binding/run_range_all.py).
    """
    return unwrap(positions, cell, ref=ref).mean(axis=0)


def min_pair_dist(positions, cell):
    """Smallest MIC distance among all pairs; inf for <2 atoms."""
    pos = np.asarray(positions, float)
    if len(pos) < 2:
        return np.inf
    D = mic_all(pos, pos, cell)
    iu = np.triu_indices(len(pos), k=1)
    return float(D[iu].min())


COLLAPSE_FLOOR_ANG = 0.7


def is_collapsed(positions, cell, floor=COLLAPSE_FLOOR_ANG):
    """True when two atoms sit closer than `floor` Å (MIC) — i.e. the structure
    is unphysical, not merely strained.

    Provenance (Stage-1a Si15 std re-rank, 2026-08-19 / diagnosed 2026-08-24):
    an MLIP can have a spurious infinitely-deep attractor at zero separation,
    and a relaxation can fall into it. Two of 844 candidates finished with a
    min pair distance of **0.019 / 0.016 Å** and energies of ~-1.3e9 eV against
    a physical -50,992 — and because the Stage-1a score is min-over-Na-seeds,
    the most negative garbage automatically wins its candidate and takes rank #1.
    A behavioural guard (`conv & nsteps < maxstep`, mace_rank.seed_trusted) does
    catch the observed cases, which ran away and hit the step cap; this is the
    orthogonal STRUCTURAL guard, for a collapse that settles into the attractor
    and reports converged.

    Default floor 0.7 Å is far below any real contact (shortest here is Si-O at
    ~1.58 Å; an O-H would be ~0.97), so it can only fire on genuine collapse.
    """
    return bool(min_pair_dist(positions, cell) < float(floor))


# Two placements of the same composition count as ONE structure when a
# per-species atom matching maps one onto the other inside this MIC
# displacement (Å). Same number as cation.seed_cation_sets' `dedupe_tol`,
# which asks the same question of the UFF SEEDS that same_structure asks of
# the RELAXED geometries — one tolerance, one home.
DEDUPE_TOL_ANG = 0.75


def _species_groups(syms_a, syms_b):
    """{element: (indices into a, indices into b)} — None when the two
    structures do not even have the same composition."""
    ia, ib = {}, {}
    for d, syms in ((ia, syms_a), (ib, syms_b)):
        for i, s in enumerate(np.asarray(syms).ravel().tolist()):
            d.setdefault(str(s), []).append(i)
    if set(ia) != set(ib) or any(len(ia[k]) != len(ib[k]) for k in ia):
        return None
    return dict((k, (ia[k], ib[k])) for k in ia)


def species_match_max_disp(syms_a, pos_a, syms_b, pos_b, cell):
    """
    Largest MIC atom displacement between two structures under the best
    (minimum-TOTAL-distance) matching of like species — a
    PERMUTATION-INVARIANT structural distance. Exactly 0.0 for two structures
    that differ only by a relabeling of identical atoms; `inf` when the two do
    not have the same composition.

    Atoms of one species are physically indistinguishable, so an index-order
    comparison (`max |pos_a - pos_b|`) reports two labelings of ONE structure
    as different placements. This is the number to log; `same_structure` is
    the tolerance test, and it does NOT use this value (see there).

    Provenance: Stage-1a v2 Si11 ranking, 2026-09-10. mace_rank's DFT carry-k
    kept the K lowest-energy Na SEEDS per arrangement with no geometric test,
    so seeds that relax into one basin ship as separate CP2K jobs. Measured
    over MOR/pipeline/stage1a_v2/ship_rank_si11/Si11/topk_geoms.extxyz (6,954
    arrangements, 7,465 within-arrangement pairs): 1,687 pairs (22.6 %, in
    20 % of arrangements) sit below DEDUPE_TOL_ANG — 330 of them ONLY under a
    permutation-invariant test, and another 143 only once MIC is applied.
    """
    grp = _species_groups(syms_a, syms_b)
    if grp is None:
        return np.inf
    from scipy.optimize import linear_sum_assignment
    pa, pb = np.asarray(pos_a, float), np.asarray(pos_b, float)
    worst = 0.0
    for el, (ja, jb) in sorted(grp.items()):
        D = mic_all(pa[ja], pb[jb], cell)
        r, c = linear_sum_assignment(D)
        worst = max(worst, float(D[r, c].max()))
    return worst


def same_structure(syms_a, pos_a, syms_b, pos_b, cell, tol=DEDUPE_TOL_ANG):
    """
    True iff SOME matching of like species maps structure a onto structure b
    with EVERY atom moved less than `tol` Å (MIC) — the permutation-invariant
    "these two are the same placement" test. Different compositions are never
    the same structure.

    EXACT, not a heuristic. Minimising the TOTAL displacement (what
    `species_match_max_disp` reports) can in principle return a matching whose
    worst atom exceeds `tol` while a feasible matching exists, so the decision
    is a bipartite PERFECT MATCHING on the "within tol" graph: the assignment
    is run over the 0/1 cost `D >= tol`, and a total cost of 0 means every
    atom found a partner inside `tol`. The cheap Hausdorff necessary condition
    (every atom must have SOME partner within tol) runs first and rejects
    virtually every non-match without touching scipy.

    NOT invariant under framework SYMMETRY operations: two placements related
    by a space-group operation of the host are physically equivalent and
    degenerate in energy, but are different point sets and stay distinct here
    (Stage-1a Si11 c07066 seeds 7 and 15 are such a pair — identical MACE
    energy to the last float32 bit, 3.68 Å apart under this metric).

    Provenance: as species_match_max_disp.
    """
    grp = _species_groups(syms_a, syms_b)
    if grp is None:
        return False
    tol = float(tol)
    pa, pb = np.asarray(pos_a, float), np.asarray(pos_b, float)
    mats = []
    for el, (ja, jb) in sorted(grp.items()):
        D = mic_all(pa[ja], pb[jb], cell)
        if D.size and (D.min(axis=1).max() >= tol or D.min(axis=0).max() >= tol):
            return False                 # Hausdorff: some atom has no partner
        mats.append(D)
    from scipy.optimize import linear_sum_assignment
    for D in mats:
        if not D.size:
            continue
        cost = (D >= tol).astype(float)
        r, c = linear_sum_assignment(cost)
        if cost[r, c].sum() > 0.0:       # no perfect matching inside tol
            return False
    return True


def mean_pair_dist(positions, cell):
    """Mean MIC distance over all pairs; inf for <2 atoms."""
    pos = np.asarray(positions, float)
    if len(pos) < 2:
        return np.inf
    D = mic_all(pos, pos, cell)
    iu = np.triu_indices(len(pos), k=1)
    return float(D[iu].mean())
