"""
framework.py — zeolite framework loading, T-site orbits, Löwenstein graph,
Al-placement sampling, and space-group symmetry helpers.

Provenance: tests/na_placement_multicomp/common.py (loader / symmetry — the
newest validated copy of the lineage common.py -> na_placement_3al -> here) and
MOR/pipeline_archive/stage1a_v1/mor_core.py (pre-archive: MOR/pipeline/) (Al samplers). Cell-aware everywhere: the cell is read
from the xyz's embedded Lattice and carried in the returned dict — no module
globals, so a new framework or a re-baselined cell needs no code edits.

The official baseline registry pins the expected cell for known frameworks; a
mismatch between registry and the file's embedded Lattice raises, enforcing the
"update all_Si.xyz + cell constant together" convention (PIPELINE.md §7).

spglib is imported lazily (only load_framework(symmetry=True) / get_symmetry_ops
need it) so collect/analyze scripts run in cluster envs without spglib.
"""
import os
import numpy as np
from ase.io import read

from .geometry import mic_dist, cell_matrix

_ZEOLITES_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SYMPREC = 0.30       # fallback tolerance; per-baseline values in BASELINE_SYMPREC
BOND_CUTOFF = 2.2    # Å, Si/Al–O bond

# Per-baseline symmetry tolerance. 0.30 was tuned so relaxed all-Si MOR reads
# as Cmcm (tests/symmetry_test) — it is a MOR-specific number, NOT a universal
# constant, so a new framework registers its own value here (framework-
# agnosticism requirement, 2026-07-07).
# FAU 0.10 (Foundations f0; re-confirmed 2026-07-15 on the Perlmutter/2022.1
# rebaseline): the relaxed all-Si FAU reads as Fd-3m / single T-orbit across
# the whole scanned range 0.01–0.30, so 0.10 is chosen for a comfortable
# image-matching tolerance (2*symprec = 0.20 Å), not because tighter fails.
BASELINE_SYMPREC = {"MOR": 0.30, "FAU": 0.10}

# ── Official baselines ──────────────────────────────────────────────────────
# name -> (xyz path, expected cell). The cell here is a CHECK, not the source:
# the loader reads the embedded Lattice and verifies it matches. Expected cell
# is (3,) box lengths for an orthorhombic baseline (MOR) or the full (3,3)
# matrix for a general cell (FAU rhombohedral — registered by Foundations f0
# once the 1500-Ry re-baseline lands; see Foundations/FOUNDATIONS.md).
BASELINES = {
    # 1500-Ry LBFGS re-baseline, 2026-07-03 (tests/cell_rebaseline_1500), V=2800.1 Å³
    "MOR": (os.path.join(_ZEOLITES_ROOT, "MOR", "All-Si_cellopt", "all_Si.xyz"),
            np.array([17.8481395059, 20.6994211544, 7.5791549307])),
    # 1500-Ry LBFGS KEEP_SYMMETRY rebaseline, 2026-07-15 (Foundations f0 on
    # Perlmutter / CP2K 2022.1 after the migration; FOUNDATIONS.md §3,
    # checkpoint M1). Rhombohedral Fd-3m, a=b=c=17.2307 Å, α=β=γ=60°;
    # E=-1735.7507 Ha; duplicate-run scatter 0.00-0.01 kJ/mol (vs 22-24 at
    # 500 Ry). Seed = idealized_primitive of the frozen P1-broken
    # FAU/Old/Old CellOpt/Fau-All_Si.xyz. Supersedes the scrapped 2026-07-10
    # Pronghorn/CP2K-2026 cell (a=17.2262, E=-1735.7506).
    "FAU": (os.path.join(_ZEOLITES_ROOT, "Foundations", "baselines",
                         "FAU_allSi_1500", "all_si_fau.xyz"),
            np.array([[17.2306867650,  0.0000000000,  0.0000000000],
                      [ 8.6153433825, 14.9222124631,  0.0000000000],
                      [ 8.6153433825,  4.9740708210, 14.0687968307]])),
}

# Superseded cells — provenance only (interpreting old runs), NEVER for new work.
HISTORICAL_CELLS = {
    "MOR_500cell":      np.array([18.3007240358, 20.5396313634, 7.5659322147]),  # CUTOFF-500 Pulay-inflated
    "MOR_broken_5e-7":  np.array([18.435, 20.767, 7.610]),                       # loose-SCF, broken
}


def load_framework(xyz_path=None, cell=None, symprec=None, symmetry=True,
                   baseline="MOR"):
    """
    Load an all-Si framework. Default: the official MOR baseline.

    Returns a dict:
      atoms (ASE), cell (3,), pos, syms, frac, nums, si_idx, o_idx,
      sio  (T-site idx -> its 4 bridging-O indices),
      loew (T-site idx -> set of Löwenstein neighbours, i.e. T sharing an O),
      and, when symmetry=True (needs spglib):
      t_of (T idx -> 'T1'..), by_type ('T1' -> sorted [idx, ...]).

    cell: only needed for an xyz without an embedded Lattice. For a registered
    baseline, the embedded Lattice must match the registry (raises otherwise).
    symprec: None -> the baseline's registered tolerance (BASELINE_SYMPREC),
    falling back to SYMPREC. The value used is stored as fw['symprec'] so the
    symmetry helpers below stay consistent with the loader.
    """
    if symprec is None:
        symprec = BASELINE_SYMPREC.get(baseline if xyz_path is None else None,
                                       SYMPREC)
    expected = None
    if xyz_path is None:
        xyz_path, expected = BASELINES[baseline]
    at = read(xyz_path)
    if cell is not None:
        at.set_cell(np.asarray(cell, float)); at.set_pbc(True)
    if at.cell.volume < 1e-6:
        raise ValueError("%s has no embedded Lattice; pass cell= explicitly" % xyz_path)
    at.set_pbc(True)
    M = np.array(at.cell[:], float)
    L = at.cell.lengths().copy()
    ortho = np.allclose(M, np.diag(L), atol=1e-8)
    # cell in geometry-dispatch form: (3,) lengths when orthorhombic (the
    # pre-Foundations behaviour every MOR consumer relies on), else the full
    # (3,3) matrix (FAU rhombohedral, 2026-07-09). fw['cellmat'] is always the
    # (3,3) matrix.
    cell = L if ortho else M
    if expected is not None:
        exp = np.asarray(expected, float)
        if exp.shape == (3,) and not ortho:
            raise ValueError(
                "baseline %s registers an orthorhombic cell but %s embeds a "
                "non-diagonal Lattice %r" % (baseline, xyz_path, M))
        got = L if exp.shape == (3,) else M
        if not np.allclose(got, exp, atol=1e-4):
            raise ValueError(
                "Embedded Lattice of %s (%s) does not match the registered baseline cell "
                "(%s). If a re-baseline landed, update zeolib.framework.BASELINES together "
                "with the xyz (PIPELINE.md §7)." % (xyz_path, got, exp))

    syms = np.array(at.get_chemical_symbols())
    pos = at.get_positions().copy()
    si_idx = np.where(syms == "Si")[0]
    o_idx = np.where(syms == "O")[0]

    # Si–O connectivity and Löwenstein graph (two T neighbours iff shared bridging O).
    sio = {int(i): [int(j) for j in o_idx if mic_dist(pos[i], pos[j], cell) < BOND_CUTOFF]
           for i in si_idx}
    for i in si_idx:
        assert len(sio[int(i)]) == 4, "Si %d has %d O neighbours" % (i, len(sio[int(i)]))
    loew = {int(i): set() for i in si_idx}
    for k, i in enumerate(si_idx):
        for j in si_idx[k + 1:]:
            if set(sio[int(i)]) & set(sio[int(j)]):
                loew[int(i)].add(int(j)); loew[int(j)].add(int(i))

    fw = dict(atoms=at, cell=cell, cellmat=M, pos=pos, syms=syms, si_idx=si_idx,
              o_idx=o_idx, sio=sio, loew=loew, frac=at.get_scaled_positions().copy(),
              nums=np.array(at.numbers), symprec=symprec)

    if symmetry:
        import spglib
        ds = spglib.get_symmetry_dataset((at.cell[:], at.get_scaled_positions(), at.numbers),
                                         symprec=symprec)
        eq = ds.equivalent_atoms
        reps = sorted(set(int(eq[i]) for i in si_idx))
        tlab = {rep: "T%d" % (k + 1) for k, rep in enumerate(reps)}
        t_of = {int(i): tlab[int(eq[i])] for i in si_idx}
        fw["t_of"] = t_of
        fw["by_type"] = {lab: sorted(int(i) for i in si_idx if t_of[int(i)] == lab)
                         for lab in sorted(set(tlab.values()))}
    return fw


def loewenstein_ok(sites, loew):
    """True iff no two T-sites in `sites` are Löwenstein neighbours (no Al–O–Al)."""
    s = list(sites)
    return all(s[j] not in loew[s[i]] for i in range(len(s)) for j in range(i + 1, len(s)))


def load_structure(xyz_path, cell=None):
    """
    Light loader for RELAXED / DECORATED structures (cation forms, guest
    complexes): atoms, syms, pos, frac, cell (geometry-dispatch form: (3,)
    lengths if orthorhombic else (3,3) matrix), cellmat, nums — and nothing
    else. No T-orbits, no Löwenstein graph, no Si-4-O coordination assert: a
    relaxed decorated cell may legitimately violate the pristine-framework
    assumptions load_framework enforces. New for Foundations f2/f3
    (2026-07-09).

    cell: only needed for an xyz without an embedded Lattice.
    """
    at = read(xyz_path)
    if cell is not None:
        at.set_cell(cell_matrix(cell)); at.set_pbc(True)
    if at.cell.volume < 1e-6:
        raise ValueError("%s has no embedded Lattice; pass cell= explicitly" % xyz_path)
    at.set_pbc(True)
    M = np.array(at.cell[:], float)
    L = at.cell.lengths().copy()
    c = L if np.allclose(M, np.diag(L), atol=1e-8) else M
    return dict(atoms=at, cell=c, cellmat=M, pos=at.get_positions().copy(),
                syms=np.array(at.get_chemical_symbols()),
                nums=np.array(at.numbers),
                frac=at.get_scaled_positions().copy())


def idealized_primitive(xyz_path, cellpar=None, symprec=1.0):
    """
    spglib-IDEALIZED primitive cell of a (possibly symmetry-broken) periodic
    structure: standardize_cell(to_primitive=True, idealize) at `symprec`,
    re-expressed on the canonical cellpar_to_cell orientation of the DETECTED
    primitive cell parameters. Returns ASE Atoms (cell + pbc set).

    cellpar: [a, b, c, alpha, beta, gamma] for an xyz without an embedded
    Lattice (the cell the coordinates were built for). symprec: how much
    symmetry breaking to forgive — deliberately LARGE by default; this is a
    structure-repair tool, not a symmetry probe.

    Provenance: Foundations f0 FAU bootstrap (2026-07-10). The frozen 500-Ry
    era FAU/Old/Old CellOpt/Fau-All_Si.xyz is P1-broken — Fd-3m only emerges
    at symprec ~1.0, and no single rhombohedral cell heals all its boundary
    bonds — so every new FAU calculation seeds from its idealized form
    (exact Fd-3m #227, one T-orbit at 1e-4, clean Si 4-coordination).
    """
    import spglib
    from ase import Atoms
    from ase.geometry.cell import cellpar_to_cell, cell_to_cellpar
    at = read(xyz_path)
    if cellpar is not None:
        at.set_cell(cellpar_to_cell(list(cellpar))); at.set_pbc(True)
    if at.cell.volume < 1e-6:
        raise ValueError("%s has no embedded Lattice; pass cellpar=" % xyz_path)
    at.set_pbc(True)
    out = spglib.standardize_cell(
        (np.array(at.cell[:]), at.get_scaled_positions(), at.numbers),
        to_primitive=True, no_idealize=False, symprec=symprec)
    if out is None:
        raise ValueError("spglib could not idealize %s at symprec %g"
                         % (xyz_path, symprec))
    lat, frac, nums = out
    M = cellpar_to_cell(cell_to_cellpar(np.asarray(lat)))
    return Atoms(numbers=nums, scaled_positions=np.asarray(frac) % 1.0,
                 cell=M, pbc=True)


def t_rings(fw, size=6):
    """
    T-site rings of length `size` from the Löwenstein adjacency (two T
    neighbours share a bridging O). A cycle counts as a RING iff the graph
    distance between every pair of its members equals their distance along
    the cycle (the standard no-shortcut ring criterion), so a 6-cycle around
    two fused 4-rings is rejected. Returns a sorted list of tuples, each in
    cycle order starting from its smallest member.

    New for Foundations FAU guest sites (2026-07-09) — replaces the FAU-era
    prepare_binding.ipynb row-order ring heuristic, which depended on the
    file's atom ordering. Symmetry-blind: works with symmetry=False loads.
    """
    loew = fw["loew"]
    nodes = sorted(loew)
    # all-pairs shortest path over the T graph (48 nodes — BFS is instant)
    dist = {}
    for s in nodes:
        d = {s: 0}
        frontier = [s]
        while frontier:
            nxt = []
            for u in frontier:
                for w in loew[u]:
                    if w not in d:
                        d[w] = d[u] + 1
                        nxt.append(w)
            frontier = nxt
        dist[s] = d

    found = set()
    rings = []

    def dfs(path):
        last = path[-1]
        if len(path) == size:
            if path[0] in loew[last]:
                key = frozenset(path)
                if key in found:
                    return
                for i in range(size):
                    for j in range(i + 1, size):
                        dc = min(j - i, size - (j - i))
                        if dist[path[i]].get(path[j], size) < dc:
                            return          # shortcut exists — not a ring
                found.add(key)
                rings.append(tuple(path))
            return
        for w in sorted(loew[last]):
            if w <= path[0] or w in path:   # smallest member anchors the cycle
                continue
            if dist[path[0]].get(w, size) > size - len(path):
                continue                    # can't close in the steps left
            dfs(path + [w])

    for v in nodes:
        dfs([v])
    return sorted(rings)


# ── Al placement sampling (from mor_core.py) ────────────────────────────────
def random_al_placement(si_idx, loew, n_al, rng=None):
    """Random Löwenstein-valid Al placement. Returns sorted list or None."""
    import random as _random
    rng = rng or _random
    candidates = list(si_idx)
    rng.shuffle(candidates)
    chosen, excluded = [], set()
    for c in candidates:
        c = int(c)
        if c in excluded:
            continue
        chosen.append(c)
        excluded.update(loew[c]); excluded.add(c)
        if len(chosen) == n_al:
            return sorted(chosen)
    return None


def mutate_al_arrangement(al_sites, n_swap, si_idx, loew, max_attempts=500, rng=None):
    """Swap n_swap Al sites for new Löwenstein-valid ones. Sorted list or None."""
    import random as _random
    rng = rng or _random
    al_set = set(al_sites)
    si_set = set(map(int, si_idx))
    for _ in range(max_attempts):
        to_remove = set(rng.sample(sorted(al_set), min(n_swap, len(al_set))))
        kept = al_set - to_remove
        forbidden = set(kept)
        for site in kept:
            forbidden.update(loew[site])
        candidates = sorted(si_set - forbidden)
        rng.shuffle(candidates)
        new_al = set(kept)
        local_forbidden = set(forbidden)
        for c in candidates:
            if c in local_forbidden:
                continue
            new_al.add(c)
            local_forbidden.add(c); local_forbidden.update(loew[c])
            if len(new_al) == len(al_set):
                return sorted(new_al)
    return None


# ── Space-group symmetry operations ─────────────────────────────────────────
def get_symmetry_ops(fw, symprec=None):
    """(rotations, translations) of the framework space group.
    symprec=None -> the tolerance the framework was loaded with (fw['symprec'])."""
    import spglib
    if symprec is None:
        symprec = fw.get("symprec", SYMPREC)
    at = fw["atoms"]
    sym = spglib.get_symmetry((at.cell[:], at.get_scaled_positions(), at.numbers),
                              symprec=symprec)
    return sym["rotations"], sym["translations"]


def map_atom_under_op(R, t, fw, idx, tol=None):
    """
    Index of the atom that framework atom `idx` maps onto under op (R,t), or
    None if the best match is further than tol (default 2*symprec: the source
    and target both deviate up to symprec from the ideal positions spglib
    matched at).
    Fix vs the tests/na_placement_* lineage (2026-07-07): the residual there
    was computed WITHOUT the op's translation t, so any op with t != 0 could
    be spuriously rejected; the match itself (argmin) always included t.
    """
    frac, nums = fw["frac"], fw["nums"]
    M = fw.get("cellmat")
    if M is None:                       # pre-Foundations fw dicts
        M = cell_matrix(fw["cell"])
    if tol is None:
        tol = 2.0 * fw.get("symprec", SYMPREC)
    g = (R @ frac[idx]) + t
    same = np.where(nums == nums[idx])[0]
    d = frac[same] - g
    d -= np.round(d)
    cart_res = np.linalg.norm(d @ M, axis=1)
    k = int(np.argmin(cart_res))
    if cart_res[k] > tol:
        return None
    return int(same[k])


def map_point_under_op(R, t, cart_point, cell):
    """Apply op (R,t) to a Cartesian point (via fractional); wrapped Cartesian.
    cell: 3 box lengths or a (3,3) matrix (geometry dispatch forms)."""
    M = cell_matrix(cell)
    f = np.asarray(cart_point, float) @ np.linalg.inv(M)
    g = (R @ f) + t
    g -= np.floor(g)
    return g @ M


def is_identity(R, t):
    return np.array_equal(R, np.eye(3, dtype=int)) and np.allclose(t, 0.0, atol=1e-6)


# ── Symmetry-distinct Al-arrangement enumeration (Stage-1a v2, 2026-07-07) ──
# The space group acting on T-site SUBSETS: canonical forms give exact dedupe
# and exact configurational degeneracy g (the Boltzmann multiplicity of a
# labeled arrangement in this cell). Framework-agnostic: everything derives
# from spglib ops + the Löwenstein graph; no site labels are assumed (FAU's
# single T-orbit is a supported case).

def site_permutations(fw, symprec=None):
    """
    The space-group action on T sites as deduped permutation dicts
    {raw_atom_idx -> raw_atom_idx}. Ops that act identically on the T sites
    are merged. Raises if any op fails to map a T site onto a T site (wrong
    symprec) or if the identity is missing.
    """
    rots, trans = get_symmetry_ops(fw, symprec)
    si = [int(i) for i in fw["si_idx"]]
    si_set = set(si)
    seen, perms = set(), []
    for R, t in zip(rots, trans):
        m = {}
        for i in si:
            j = map_atom_under_op(R, t, fw, i)
            if j is None or j not in si_set:
                raise ValueError(
                    "symmetry op does not map T site %d onto a T site — "
                    "symprec %.3g inconsistent with the loaded framework"
                    % (i, fw.get("symprec", SYMPREC)))
            m[i] = int(j)
        key = tuple(m[i] for i in si)
        if len(set(key)) != len(si):
            raise ValueError("symmetry op is not a bijection on T sites")
        if key not in seen:
            seen.add(key)
            perms.append(m)
    if not any(all(m[i] == i for i in si) for m in perms):
        raise ValueError("identity op missing from site permutations")
    return perms


def canonical_arrangement(sites, perms):
    """
    (canonical_form, degeneracy) of a T-site subset under the site
    permutations: canonical_form = lexicographically smallest sorted image
    tuple; degeneracy = number of DISTINCT images = the number of equivalent
    labeled arrangements in this cell (exact Boltzmann multiplicity g).
    """
    base = tuple(sorted(int(x) for x in sites))
    images = {tuple(sorted(p[i] for i in base)) for p in perms}
    return min(images), len(images)


def enumerate_al_arrangements(fw, n_al, perms=None, limit=3000000):
    """
    EXHAUSTIVE enumeration of Löwenstein-valid, symmetry-distinct Al
    arrangements: DFS over increasing T-site indices with Löwenstein pruning;
    a leaf is kept iff it IS its own canonical form, so each equivalence class
    appears exactly once. Returns [(sites_tuple, degeneracy), ...].

    Feasible for small n_al (MOR: n_al=3 -> 17k leaves, n_al=4 -> ~120k);
    raises RuntimeError once more than `limit` valid subsets are visited —
    the caller falls back to sample_al_arrangements for Al-rich ratios.
    """
    if perms is None:
        perms = site_permutations(fw)
    si = sorted(int(i) for i in fw["si_idx"])
    loew = fw["loew"]
    out, chosen = [], []
    forb = {i: 0 for i in si}          # Löwenstein-block counts along the DFS path
    state = {"visited": 0}

    def dfs(start):
        if len(chosen) == n_al:
            state["visited"] += 1
            if state["visited"] > limit:
                raise RuntimeError(
                    "enumerate_al_arrangements: >%d Löwenstein-valid subsets "
                    "for n_al=%d — space too large, use sample_al_arrangements"
                    % (limit, n_al))
            base = tuple(chosen)
            canon, g = canonical_arrangement(base, perms)
            if canon == base:
                out.append((base, g))
            return
        need = n_al - len(chosen)
        for k in range(start, len(si) - need + 1):
            c = si[k]
            if forb[c]:
                continue
            chosen.append(c)
            for nb in loew[c]:
                forb[nb] += 1
            dfs(k + 1)
            for nb in loew[c]:
                forb[nb] -= 1
            chosen.pop()

    dfs(0)
    return out


def sample_al_arrangements(fw, n_al, n_want, perms=None, rng=None,
                           exclude=None, max_tries=None):
    """
    Random Löwenstein-valid arrangements, deduped by canonical form.
    Returns [(canonical_sites_tuple, degeneracy), ...] in draw order.
    exclude: set of canonical tuples to skip (e.g. already-DFT'd arrangements).
    Note: draws are uniform over LABELED arrangements, so classes are hit
    ~proportionally to their degeneracy — fine for coverage sampling; use
    enumerate_al_arrangements when exact uniform-over-classes matters.
    """
    import random as _random
    if perms is None:
        perms = site_permutations(fw)
    rng = rng or _random
    seen = set(exclude or ())
    out, tries = [], 0
    max_tries = max_tries or n_want * 200
    while len(out) < n_want and tries < max_tries:
        tries += 1
        al = random_al_placement(fw["si_idx"], fw["loew"], n_al, rng=rng)
        if al is None:
            continue
        canon, g = canonical_arrangement(al, perms)
        if canon in seen:
            continue
        seen.add(canon)
        out.append((canon, g))
    return out


def arrangement_invariants(fw, sites):
    """
    Cheap symmetry-invariant descriptors of an Al arrangement — the strata for
    coverage sampling and the axes for Dedeček-style validation. All framework-
    agnostic: t_multiset degrades gracefully to a single label for a one-orbit
    framework (FAU).
      t_multiset      : sorted T-orbit labels, e.g. 'T1|T1|T3' ('' if no orbits)
      n_second_shell  : # Al pairs on T sites sharing a common T neighbour
                        (Al-O-T-O-Al — the close-pair / Dedeček-proxy count)
      min_alal, mean_alal : MIC Al-Al distance stats (Å)
    """
    s = sorted(int(x) for x in sites)
    inv = {}
    inv["t_multiset"] = ("|".join(sorted(fw["t_of"][i] for i in s))
                         if "t_of" in fw else "")
    loew = fw["loew"]
    inv["n_second_shell"] = sum(
        1 for a in range(len(s)) for b in range(a + 1, len(s))
        if loew[s[a]] & loew[s[b]])
    if len(s) >= 2:
        d = [mic_dist(fw["pos"][s[a]], fw["pos"][s[b]], fw["cell"])
             for a in range(len(s)) for b in range(a + 1, len(s))]
        inv["min_alal"] = round(min(d), 3)
        inv["mean_alal"] = round(float(np.mean(d)), 3)
    else:
        inv["min_alal"] = inv["mean_alal"] = None
    return inv
