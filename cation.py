"""
cation.py — extra-framework cation model: UFF LJ + reduced-charge Coulomb
energy/gradient, L-BFGS relax, near-Al initial placement, and multi-seed
generation for the Stage-1a Na layer.

Provenance: absorbed from MOR/pipeline_archive/stage1a_v1/mor_core.py (pre-archive: MOR/pipeline/) at the Stage-1a v2 rework
(2026-07-07 — the absorption point named in README "Deliberately NOT in
zeolib (yet)"). The ENERGY is numerically IDENTICAL to mor_core for MOR
(selftest pins parity at 1e-9 kcal); the Coulomb GRADIENT fixes mor_core's
inherited factor-1/2 bug (see cation_energy_grad — energies and, to first
order, minima were unaffected, so past placements stand). The code is
framework-agnostic:

  * cell comes from the zeolib framework dict, never a module constant;
  * the mor_core hard-coded ±z image expansion (c < 2*cutoff) generalizes to
    per-axis treatment: axes with L >= 2*cutoff use single-image MIC, shorter
    axes get explicit ±1 images. For MOR this reproduces mor_core exactly.
    NOTE (inherited, documented not "fixed"): with ±1 images an axis with
    L < cutoff truncates the (2L-cutoff, cutoff) interaction shell — for MOR
    (c=7.58, cutoff 8) that is the 7.16–8 Å z-tail of a cutoff-SHIFTED
    potential, negligible and exactly what every stage-1a result used;
  * MOR's hard-coded void slots (initial_cation_positions) are NOT absorbed —
    place_cations_near_al is the framework-agnostic path.

mor_core.py keeps its own copy as the provenance of past runs (never
retrofitted); new code imports this module.
"""
import itertools

import numpy as np

from .geometry import (mic_dist, mic_vec, mic_all, centroid_unwrapped,
                       cell_matrix, perp_widths, wrap_to_cell)

# ── Physical / model constants (mor_core values, unchanged) ─────────────────
COULOMB_K = 332.0637          # kcal/mol·Å/e²
LJ_CUTOFF = 8.0               # Å
COUL_CUTOFF = 8.0             # Å
MAX_OPT_STEPS = 100

# LJ params exactly as RANGE's get_UFF_para returns them (hardcoded so nothing
# here needs RANGE). ⚠ AS-RUN UNITS QUIRK, reproduced deliberately: these eps
# are UFF values in eV (kcal/23.06), while COULOMB_K is kcal-scale — so the
# as-validated model's LJ term is ~23x weaker relative to Coulomb than textbook
# UFF. Every stage-1a placement to date ran with exactly these numbers
# (mor_core._UFF_FALLBACK matches get_UFF_para); parity > unit purity here.
UFF_PARAMS = {
    'Na': (0.0013058979664136582, 2.658), 'Si': (0.017432701424664864, 3.826),
    'Al': (0.021899701611365553, 4.008), 'O': (0.0026014316632526047, 3.118),
    'Ag': (0.0015650047057814474, 2.805), 'Cu': (0.00021764966106894304, 3.114),
    'Pb': (0.0287504838002499, 3.828), 'Bi': (0.022459372168399976, 3.893),
}
# Reduced partial charges (standard zeolite electrostatic model)
Q_FW = {'Si': 2.4, 'Al': 1.4, 'O': -1.2}
Q_CAT = {'Na': 1.0, 'Ag': 1.0, 'Cu': 1.0, 'Pb': 2.0, 'Bi': 3.0}


def _uff(el):
    try:
        return UFF_PARAMS[el]
    except KeyError:
        raise KeyError("no UFF params for %r in zeolib.cation.UFF_PARAMS" % el)


# ── Framework precomputation ────────────────────────────────────────────────
def precompute_fw(fw, al_sites=(), cutoff=LJ_CUTOFF, images="exact"):
    """
    Precompute the fixed-framework side of the cation energy for framework
    dict `fw` (zeolib.framework.load_framework or load_structure) with Al
    already substituted at `al_sites`. al_sites=None reads Al from EXPLICIT
    'Al' symbols instead (relaxed decorated structures — Foundations f2).
    Returns a dict consumed by cation_energy_grad/relax_cations.

    Orthorhombic cells — image policy per axis: L >= 2*cutoff -> MIC; shorter
    axes get explicit images and no MIC. images=
      "exact"  (default): ceil(cutoff/L) images per short axis — the full
               cutoff sphere is covered, so the energy is WRAP-INVARIANT
               (E(x) == E(x + L), pinned by selftest);
      "legacy": ±1 images = mor_core's ±z expansion, byte-parity with every
               as-run stage-1a placement. Its truncation of the
               (2L-cutoff, cutoff) shell makes the energy depend on whether a
               coordinate is wrapped — kcal-scale for MOR's c=7.58 < cutoff=8
               (measured 2.5 kcal on a relaxed 3-Na config) — so use it only
               to reproduce/compare against old runs.

    General (3,3) cells (FAU rhombohedral, Foundations 2026-07-09): per-axis
    MIC is invalid for skewed cells, so ALL axes use explicit lattice-vector
    images, ±ceil(cutoff/perp_width) per axis — the cutoff sphere around any
    home-cell cation is fully covered (keep cations wrapped; relax_cations
    returns them wrapped). images="legacy" has no meaning here and raises.
    """
    c = np.asarray(fw["cell"], float)
    ortho = c.shape == (3,)
    if ortho:
        L = c
        mic_axes = L >= 2.0 * cutoff
        short = [i for i in range(3) if not mic_axes[i]]
        if images == "legacy":
            m = {i: 1 for i in short}
        else:
            m = {i: int(np.ceil(cutoff / L[i])) for i in short}
        if short:
            ranges = [[k for j in range(m[i] + 1) for k in ((j, -j) if j else (0,))]
                      for i in short]
            shifts = np.array(
                [[(s[short.index(i)] if i in short else 0) for i in range(3)]
                 for s in itertools.product(*ranges)], float)
        else:
            shifts = np.zeros((1, 3))
        shifts = shifts * L
    else:
        if images == "legacy":
            raise ValueError("images='legacy' is a mor_core/MOR parity mode; "
                             "it has no meaning for a general cell")
        M = cell_matrix(c)
        w = perp_widths(c)
        mic_axes = np.array([False, False, False])
        m = [int(np.ceil(cutoff / w[i])) for i in range(3)]
        shifts = np.array([list(s) for s in itertools.product(
            *[range(-mi, mi + 1) for mi in m])], float) @ M

    if al_sites is None:
        fw_mask = np.isin(fw["syms"], ["Si", "O", "Al"])
        eff = fw["syms"][fw_mask].tolist()
    else:
        fw_mask = np.isin(fw["syms"], ["Si", "O"])
        fw_idx = np.where(fw_mask)[0]
        fw_syms = fw["syms"][fw_mask].tolist()
        al_set = set(int(a) for a in al_sites)
        eff = ['Al' if int(i) in al_set else s for i, s in zip(fw_idx, fw_syms)]
    fw_pos = fw["pos"][fw_mask]
    n_img = len(shifts)
    fw_exp = np.vstack([fw_pos + sh for sh in shifts])
    eps = np.tile(np.array([_uff(s)[0] for s in eff]), n_img)
    sig = np.tile(np.array([_uff(s)[1] for s in eff]), n_img)
    chg = np.tile(np.array([Q_FW[s] for s in eff]), n_img)
    return dict(cell=c, ortho=ortho, mic_axes=mic_axes, fw_exp=fw_exp,
                eps=eps, sig=sig, chg=chg, cutoff=float(cutoff))


# ── Energy + gradient (mor_core.lj_cat_energy_grad, cell-generalized) ───────
def cation_energy_grad(cat_flat, pre, cat_sym):
    """
    UFF LJ + cutoff-shifted Coulomb energy (kcal/mol) and gradient w.r.t.
    cation positions; framework fixed. Cation–framework uses pre's image/MIC
    policy; cation–cation uses full 3D MIC (as mor_core).
    """
    eps_c, sig_c = _uff(cat_sym)
    q_c = Q_CAT[cat_sym]
    cell = pre["cell"]
    ortho = pre.get("ortho", True)
    mic = pre["mic_axes"]
    fw_exp, fw_eps, fw_sig, fw_chg = pre["fw_exp"], pre["eps"], pre["sig"], pre["chg"]
    CUT = pre["cutoff"]
    n_cat = len(cat_flat) // 3
    cat_pos = np.asarray(cat_flat, float).reshape(n_cat, 3)
    E = 0.0
    grad = np.zeros_like(cat_pos)

    for k in range(n_cat):
        dvec = cat_pos[k] - fw_exp
        if ortho:
            for ax in range(3):
                if mic[ax]:
                    dvec[:, ax] -= np.round(dvec[:, ax] / cell[ax]) * cell[ax]
        # general cells: no per-axis MIC — coverage is entirely explicit images
        r2 = np.einsum('ij,ij->i', dvec, dvec)

        mask = (r2 < CUT**2) & (r2 > 0.09)
        if mask.any():
            rm2 = r2[mask]
            eps_ij = np.sqrt(eps_c * fw_eps[mask])
            sig_ij = (sig_c + fw_sig[mask]) / 2.0
            sr6 = (sig_ij**2 / rm2) ** 3
            sr6c = (sig_ij / CUT) ** 6
            E += (4.0 * eps_ij * ((sr6**2 - sr6) - (sr6c**2 - sr6c))).sum()
            dEdr2 = 4.0 * eps_ij * (-12.0 * sr6**2 + 6.0 * sr6) / rm2
            grad[k] += (dEdr2[:, np.newaxis] * dvec[mask]).sum(axis=0)

        cmask = (r2 < CUT**2) & (r2 > 0.09)
        if cmask.any():
            rm2c = r2[cmask]
            r = np.sqrt(rm2c)
            q_j = fw_chg[cmask]
            E += (COULOMB_K * q_c * q_j * (1.0 / r - 1.0 / CUT)).sum()
            # dE/dx = -K q q dvec / r^3. FIX vs mor_core (2026-07-07): its
            # Coulomb gradient carried dE/d(r^2) but multiplied by dvec (not
            # 2*dvec), yielding HALF the true gradient — energies were never
            # affected, and minima of a uniformly scaled field coincide, so
            # past placements stand; selftest pins energy parity + FD-grad.
            dEdxc = -COULOMB_K * q_c * q_j / (r * rm2c)
            grad[k] += (dEdxc[:, np.newaxis] * dvec[cmask]).sum(axis=0)

    for i in range(n_cat):
        for j in range(i + 1, n_cat):
            if ortho:
                dv = cat_pos[i] - cat_pos[j]
                dv -= np.round(dv / cell) * cell
            else:
                dv = mic_vec(cat_pos[j], cat_pos[i], cell)
            r2 = max(float(dv @ dv), 1e-6)
            r = float(np.sqrt(r2))
            if r2 < CUT**2:
                sr6 = (sig_c**2 / r2) ** 3
                sr6c = (sig_c / CUT) ** 6
                E += 4.0 * eps_c * ((sr6**2 - sr6) - (sr6c**2 - sr6c))
                dEdr2 = 4.0 * eps_c * (-12.0 * sr6**2 + 6.0 * sr6) / r2
                g = dEdr2 * dv
                grad[i] += g
                grad[j] -= g
            if r2 < CUT**2:
                E += COULOMB_K * q_c * q_c * (1.0 / r - 1.0 / CUT)
                g = -COULOMB_K * q_c * q_c / (r * r2) * dv   # same 1/2 fix
                grad[i] += g
                grad[j] -= g

    return E, grad.flatten()


def relax_cations(cat_init, pre, cat_sym='Na'):
    """L-BFGS-B relax of cation positions (framework fixed). Returns
    (positions wrapped into the cell, E_kcal). mor_core.optimize_cation +
    relax_na_uff merged (the Al substitution now lives in precompute_fw)."""
    from scipy.optimize import minimize

    def fun_jac(x):
        return cation_energy_grad(x, pre, cat_sym)

    cat_init = np.asarray(cat_init, float)
    res = minimize(fun_jac, cat_init.flatten(), method='L-BFGS-B', jac=True,
                   options={'maxiter': MAX_OPT_STEPS, 'ftol': 1e-8, 'gtol': 1e-4})
    wrapped = wrap_to_cell(res.x.reshape(cat_init.shape[0], 3), pre["cell"])
    # report the energy AT the returned (wrapped) coordinates — identical to
    # res.fun for images="exact" (wrap-invariant), and keeps E consistent with
    # the coordinates handed back under "legacy" truncation
    e_w, _ = cation_energy_grad(wrapped.flatten(), pre, cat_sym)
    return wrapped, float(e_w)


# ── Initial placement (mor_core.place_na_near_al, cell-generalized) ─────────
def _mic_round(D, cell):
    """Row-wise round-only MIC displacement (exact for orthorhombic — the
    pre-Foundations inline code; fractional-round approximation for general
    cells, which is fine for the local <3.2 Å placement heuristics it serves:
    round-MIC error only appears near half-cell distances)."""
    c = np.asarray(cell, float)
    D = np.asarray(D, float)
    if c.shape == (3,):
        return D - np.round(D / c) * c
    inv = np.linalg.inv(c)
    F = D @ inv
    return (F - np.round(F)) @ c


def place_cations_near_al(fw, al_sites, offset=2.3, local_cut=3.2,
                          rng=None):
    """
    Framework-agnostic initial cation guess: one cation per Al, ~offset Å off
    one of the Al's bridging O, pushed into the local void (away from the
    framework mass within local_cut Å). Deterministic (rng=None): each cation
    picks, among its Al's four bridging O, the candidate farthest from the
    already-placed (mor_core's greedy spreading). With rng: the bridging O is
    drawn at random instead — the diversity source for multi-seed generation.
    Returns (n_al, 3). Intended as the START for relax_cations.
    """
    cell, pos, sio = fw["cell"], fw["pos"], fw["sio"]
    placed = []
    for al in al_sites:
        al_p = pos[int(al)]
        cands = []
        for o in sio[int(al)]:
            o_p = pos[o]
            dvec = _mic_round(pos - o_p, cell)
            r = np.linalg.norm(dvec, axis=1)
            near = (r > 0.1) & (r < local_cut)
            if near.any():
                outward = -(dvec[near] / r[near, None]).mean(axis=0)
            else:
                outward = _mic_round(o_p - al_p, cell)
            nrm = np.linalg.norm(outward)
            if nrm < 1e-6:
                continue
            cand = o_p + offset * outward / nrm
            spread = min((mic_dist(cand, p, cell) for p in placed),
                         default=np.inf)
            cands.append((spread, cand))
        if rng is not None:
            placed.append(wrap_to_cell(cands[rng.randrange(len(cands))][1], cell))
        else:
            placed.append(wrap_to_cell(max(cands, key=lambda t: t[0])[1], cell))
    return np.array(placed)


def cation_set_distance(a, b, cell):
    """Symmetric Hausdorff MIC distance between two same-size cation sets —
    the dedupe metric for seed generation (cations are indistinguishable, so
    per-index RMSD would over-count permuted placements as distinct)."""
    a, b = np.asarray(a), np.asarray(b)
    d = np.array([[mic_dist(p, q, cell) for q in b] for p in a])
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def seed_cation_sets(fw, al_sites, n_seeds, cat_sym='Na', rng=None,
                     dedupe_tol=0.75, max_draws=None, sort=True):
    """
    Up to n_seeds DISTINCT UFF-relaxed cation placements for one Al
    arrangement — the Na-layer seeds of the Stage-1a successive-halving
    search. Seed 0 is the deterministic greedy-spread placement; further
    seeds draw random bridging-O starts (place_cations_near_al with rng),
    relax under the same UFF/Coulomb model, and are kept if farther than
    dedupe_tol (Hausdorff MIC) from every kept set. Returns
    [(positions (n_al,3), E_kcal), ...] sorted by energy — pass sort=False
    to keep DRAW order, which is stable under extension: re-calling with the
    same rng seed and a larger n_seeds reproduces the earlier seeds at the
    same indices (what the successive-halving runner's resume relies on).
    """
    import random as _random
    rng = rng or _random.Random(0)
    pre = precompute_fw(fw, al_sites)
    kept = []

    def try_add(start):
        p, e = relax_cations(start, pre, cat_sym)
        if any(cation_set_distance(p, q, fw["cell"]) < dedupe_tol
               for q, _ in kept):
            return False
        kept.append((p, e))
        return True

    try_add(place_cations_near_al(fw, al_sites))
    draws = 0
    max_draws = max_draws or n_seeds * 8
    while len(kept) < n_seeds and draws < max_draws:
        draws += 1
        try_add(place_cations_near_al(fw, al_sites, rng=rng))
    if sort:
        kept.sort(key=lambda t: t[1])
    return kept


def exchange_na(struct, cat_sym, images="exact"):
    """
    Baseline cation exchange (Foundations f2, 2026-07-09): replace the Na
    layer of a RELAXED Na-form structure with `cat_sym` at formal charge
    Q_CAT[cat_sym], then relax the new cations under the UFF/Coulomb model
    with the framework FIXED.

    This is a pre-DFT nudge for a baseline paper, NOT a cation-siting search
    (the archived stage1b_v1 search is DO-NOT-RESURRECT — audit F1/F2); the
    downstream DFT cell-opt -> geo-opt is the corrector.

      q = 1 (Ag, Cu): one cation starts at each Na position.
      q > 1 (Pb, Bi): the n_al Na are partitioned into n_al/q groups by a
        deterministic greedy MIC clustering — seed = the remaining Na farthest
        from its nearest remaining neighbour (most isolated first), joined by
        its q-1 MIC-nearest — and each cation starts at its group's unwrapped
        centroid. If a start lands badly, place_cations_near_al on the Al
        sites is the documented fallback (caller's choice).

    struct: dict from framework.load_structure (syms/pos/cell; Al explicit,
    Na present). Returns (cat_pos (n_cat,3) wrapped, E_kcal, groups) with
    groups = tuples of Na atom indices into struct's atom list.
    """
    syms = np.asarray(struct["syms"])
    cell = struct["cell"]
    na_idx = [int(i) for i in np.where(syms == "Na")[0]]
    n_al = int((syms == "Al").sum())
    q = int(Q_CAT[cat_sym])
    if not na_idx:
        raise ValueError("no Na in structure — nothing to exchange")
    if len(na_idx) != n_al:
        raise ValueError("expected one Na per Al, found %d Na / %d Al"
                         % (len(na_idx), n_al))
    if n_al % q:
        raise ValueError("%s (q=%d) cannot charge-balance %d Al"
                         % (cat_sym, q, n_al))
    na_pos = struct["pos"][na_idx]
    if q == 1:
        groups = [(i,) for i in na_idx]
        start = na_pos.copy()
    else:
        D = mic_all(na_pos, na_pos, cell)
        np.fill_diagonal(D, np.inf)
        remaining = list(range(len(na_idx)))
        groups, starts = [], []
        while remaining:
            if len(remaining) == q:
                grp = list(remaining)
            else:
                sub = D[np.ix_(remaining, remaining)]
                seed = int(np.argmax(sub.min(axis=1)))      # most isolated
                order = [k for k in np.argsort(sub[seed]) if k != seed]
                grp = [remaining[seed]] + [remaining[int(k)]
                                           for k in order[:q - 1]]
            groups.append(tuple(na_idx[g] for g in sorted(grp)))
            starts.append(centroid_unwrapped(na_pos[sorted(grp)], cell))
            remaining = [r for r in remaining if r not in grp]
        start = wrap_to_cell(np.array(starts), cell)
    pre = precompute_fw(struct, al_sites=None, images=images)
    cat_pos, e = relax_cations(start, pre, cat_sym)
    return cat_pos, e, groups


def assemble_atoms(fw, al_sites, cat_pos, cat_sym='Na'):
    """ASE Atoms: framework with Al substituted at al_sites + cations appended
    LAST (the project-wide convention every eval/harvest relies on)."""
    from ase import Atoms
    syms = list(fw["syms"])
    for a in al_sites:
        syms[int(a)] = 'Al'
    cat_pos = np.asarray(cat_pos, float).reshape(-1, 3)
    return Atoms(symbols=syms + [cat_sym] * len(cat_pos),
                 positions=np.vstack([fw["pos"], cat_pos]),
                 cell=list(fw["cell"]), pbc=True)
