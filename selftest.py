#!/usr/bin/env python3
"""
selftest.py — zeolib verification against real repo data. Run after ANY zeolib
change:  python zeolib/selftest.py   (from Zeolites/, or anywhere)

Checks, in order:
  1. geometry MIC vs brute-force image search + the PBC-centroid regression
     case; general-cell (FAU rhombohedral) MIC vs ase.geometry.get_distances
  2. framework loader on the official MOR baseline (counts, orbits,
     Löwenstein); FAU fixture load (general cell) + t_rings
  3. CP2K input builders reproduce the staged na_placement_multicomp inputs
     BYTE-FOR-BYTE (the strongest guard against protocol drift); FIXED_ATOMS
     + rhombohedral renders (Foundations)
  4. CP2K parsers on a real 1500-Ry energy-force job; energy cross-checked
     against the shipped train_1500.xyz header; 3x3 cell parser vs the
     orthorhombic one on the cell_rebaseline_1500 fixture
  5. slurm/sbatch text conventions (mpirun-not-srun, OMP=1, throttle, timestamps)
  6. fileio LF guarantee + coords.inc round-trip
  7. maceenv model registry
  8. molecules guest templates vs the frozen v0/FAU copies
  9. placement void-grid/snap reproduces a shipped v0 SITE_INFO position
 10. constants per-framework combo map (charge balance, Al counts)
 11. provenance version stamping (git sha, dirty detection, stamp files)
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZROOT = os.path.dirname(HERE)
sys.path.insert(0, ZROOT)

from zeolib import constants, geometry, cp2k, slurm, fileio  # noqa: E402

_FAILS = []


def check(name, ok, detail=""):
    print("  %-58s %s" % (name, "PASS" if ok else "FAIL " + str(detail)))
    if not ok:
        _FAILS.append(name)


FAU_START_CELLPAR = [17.235, 17.235, 17.235, 60.0, 60.0, 60.0]  # Old CellOpt as-run


def fau_idealized_path(tmpdir):
    """
    Idealized all-Si FAU primitive cell as a temp extxyz — via
    framework.idealized_primitive, exactly the Foundations f0 seed step (the
    frozen 500-Ry-era Fau-All_Si.xyz is P1-broken; see that docstring).
    Returns None if spglib or the source file is unavailable.
    """
    try:
        import spglib  # noqa: F401
    except ImportError:
        return None
    from ase.io import write
    from zeolib import framework as fwm
    src = os.path.join(ZROOT, "FAU", "Old", "Old CellOpt", "Fau-All_Si.xyz")
    if not os.path.exists(src):
        return None
    ideal = fwm.idealized_primitive(src, cellpar=FAU_START_CELLPAR)
    p = os.path.join(tmpdir, "fau_ideal.xyz")
    write(p, ideal)
    return p


def test_geometry():
    print("[1] geometry")
    rng = np.random.default_rng(0)
    L = np.array([17.8, 20.7, 7.6])
    P = rng.uniform(-30, 60, size=(20, 3)) % L   # wrap: brute force below only scans +/-1 image
    Q = rng.uniform(-30, 60, size=(15, 3)) % L
    D = geometry.mic_all(P, Q, L)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1)
                       for k in (-1, 0, 1)], float) * L
    brute = np.min(np.linalg.norm(
        (P[:, None, None, :] - (Q[None, :, None, :] + shifts[None, None, :, :])),
        axis=3), axis=2)
    check("mic_all == brute-force over 27 images", np.allclose(D, brute, atol=1e-10))
    a, b = P[0], Q[0]
    check("mic_dist consistent with mic_all",
          abs(geometry.mic_dist(a, b, L) - D[0, 0]) < 1e-10)
    # PBC-centroid regression (the run_range_all.py mol_centroid bug):
    # diatomic straddling x=0 — naive mean lands mid-cell, unwrapped must not.
    mol = np.array([[0.1, 5.0, 3.0], [L[0] - 0.1, 5.0, 3.0]])
    cen = geometry.centroid_unwrapped(mol, L)
    check("centroid_unwrapped handles boundary-straddling pair",
          abs(cen[0] - 0.0) < 1e-9 and abs(cen[1] - 5.0) < 1e-12)
    naive = mol.mean(axis=0)
    check("  (naive mean is indeed wrong — guard is meaningful)",
          abs(naive[0] - L[0] / 2) < 0.2)
    # ── general-cell dispatch (FAU rhombohedral, Foundations 2026-07-09) ──
    check("diagonal (3,3) matrix routes to the box-lengths fast path",
          np.allclose(geometry.mic_all(P, Q, np.diag(L)), D, atol=1e-12))
    from ase.geometry.cell import cellpar_to_cell
    from ase.geometry import get_distances
    M = cellpar_to_cell([17.235, 17.235, 17.235, 60.0, 60.0, 60.0])
    P2 = rng.uniform(-20, 40, size=(25, 3))
    Q2 = rng.uniform(-20, 40, size=(20, 3))
    D2 = geometry.mic_all(P2, Q2, M)
    _, Dase = get_distances(P2, Q2, cell=M, pbc=True)
    check("general-cell mic_all == ase.geometry.get_distances (1e-9)",
          np.allclose(D2, Dase, atol=1e-9))
    check("general-cell mic_dist consistent with mic_all",
          abs(geometry.mic_dist(P2[0], Q2[0], M) - D2[0, 0]) < 1e-10)
    shift = 3 * M[0] - 2 * M[2] + M[1]
    check("general-cell MIC is wrap-invariant",
          abs(geometry.mic_dist(P2[1], Q2[1] + shift, M) - D2[1, 1]) < 1e-9)
    # round-only MIC is NOT exact at alpha=60 — the ±1 image search matters
    f = (P2[:, None, :] - Q2[None, :, :]) @ np.linalg.inv(M)
    Dround = np.linalg.norm((f - np.round(f)) @ M, axis=2)
    check("  (round-only MIC is indeed inexact here — guard is meaningful)",
          (Dround - D2).max() > 1e-6)
    mol2 = np.vstack([P2[2], P2[2] + M[0] + np.array([0.3, 0.0, 0.0])])
    cen2 = geometry.centroid_unwrapped(mol2, M)
    check("general-cell centroid_unwrapped handles a straddling pair",
          np.allclose(cen2, P2[2] + np.array([0.15, 0.0, 0.0]), atol=1e-9))
    w = geometry.perp_widths(M)
    check("perp_widths: rhombohedral widths equal, < a, > a/2",
          np.allclose(w, w[0]) and 8.6 < w[0] < 17.2)
    # cell_abc_angles: the CP2K &CELL pair, both cell forms
    Lo, ao = geometry.cell_abc_angles(np.array([17.8481, 20.6994, 7.5792]))
    check("cell_abc_angles: orthorhombic -> lengths kept, 90/90/90",
          np.allclose(Lo, [17.8481, 20.6994, 7.5792], atol=1e-12)
          and np.allclose(ao, 90.0, atol=1e-12))
    Lr, ar = geometry.cell_abc_angles(M)
    check("cell_abc_angles: FAU-like rhombohedral -> equal lengths, 60/60/60",
          np.allclose(Lr, Lr[0], atol=1e-9) and np.allclose(ar, 60.0, atol=1e-9))
    wr = geometry.wrap_to_cell(P2, M)
    fwr = wr @ np.linalg.inv(M)
    # molecule-aware wrapping: a diatomic straddling a face must stay whole
    cellw = np.array([10.0, 10.0, 10.0])
    di = np.array([[9.7, 5.0, 5.0], [10.4, 5.0, 5.0]])   # bond 0.7 across x=10
    naive = geometry.wrap_to_cell(di, cellw)
    kept = geometry.wrap_preserving_groups(di, cellw, groups=[[0, 1]])
    check("  (naive wrap does tear the molecule — guard is meaningful)",
          abs(np.linalg.norm(naive[1] - naive[0]) - 0.7) > 1.0)
    check("wrap_preserving_groups keeps a boundary-straddling molecule intact",
          abs(np.linalg.norm(kept[1] - kept[0]) - 0.7) < 1e-9
          and abs(geometry.mic_dist(kept[0], kept[1], cellw) - 0.7) < 1e-9)
    fr = (kept.mean(axis=0) / cellw)
    check("wrap_preserving_groups puts the group CENTROID inside the cell",
          bool(np.all(fr >= 0.0) and np.all(fr < 1.0)))
    mixed = np.vstack([np.array([[12.5, -1.0, 3.0]]), di])
    mw = geometry.wrap_preserving_groups(mixed, cellw, groups=[[1, 2]])
    check("wrap_preserving_groups wraps FREE atoms individually",
          bool(np.all(mw[0] >= 0.0) and np.all(mw[0] < 10.0)))

    # is_collapsed — the MLIP zero-separation-attractor guard (Stage-1a 2026-08-24).
    # Real numbers from ship_rank/Si15_std/best_geoms.extxyz: the two blown-up
    # candidates sat at 0.019 / 0.016 A; healthy ones at 1.583 / 1.584 A (Si-O).
    cellc = np.array([17.85, 20.70, 7.58])
    healthy = np.array([[0.0, 0.0, 0.0], [1.583, 0.0, 0.0], [5.0, 5.0, 2.0]])
    collapsed = np.array([[0.0, 0.0, 0.0], [0.019, 0.0, 0.0], [5.0, 5.0, 2.0]])
    check("is_collapsed FALSE on a healthy Si-O contact (1.583 A)",
          geometry.is_collapsed(healthy, cellc) is False)
    check("is_collapsed TRUE on the measured c00501 collapse (0.019 A)",
          geometry.is_collapsed(collapsed, cellc) is True)
    check("  (default floor sits between them — guard is meaningful)",
          1.583 > geometry.COLLAPSE_FLOOR_ANG > 0.019)
    check("is_collapsed respects an explicit floor",
          geometry.is_collapsed(healthy, cellc, floor=2.0) is True
          and geometry.is_collapsed(collapsed, cellc, floor=0.001) is False)
    check("is_collapsed honours MIC (pair split across a boundary)",
          geometry.is_collapsed(
              np.array([[0.01, 0.0, 0.0], [17.84, 0.0, 0.0], [5.0, 5.0, 2.0]]),
              cellc) is True)

    check("wrap_to_cell: fractionals in [0,1), MIC distances preserved",
          fwr.min() >= -1e-12 and fwr.max() < 1.0
          and np.allclose(geometry.mic_all(wr, Q2, M), D2, atol=1e-9))


def test_framework():
    print("[2] framework")
    from zeolib import framework as fwm
    try:
        import spglib  # noqa: F401
        sym = True
    except ImportError:
        sym = False
        print("  (spglib missing — orbit checks skipped)")
    fw = fwm.load_framework(symmetry=sym)
    check("MOR baseline: 144 atoms, 48 Si, 96 O",
          len(fw["syms"]) == 144 and len(fw["si_idx"]) == 48 and len(fw["o_idx"]) == 96)
    check("cell matches registered 1500-Ry baseline",
          np.allclose(fw["cell"], fwm.BASELINES["MOR"][1], atol=1e-6))
    ok = all(i in fw["loew"][j] for i in fw["loew"] for j in fw["loew"][i])
    check("Löwenstein graph symmetric", ok)
    if sym:
        sizes = sorted(len(v) for v in fw["by_type"].values())
        check("T-site orbits = 4 with sizes [8, 8, 16, 16]",
              len(fw["by_type"]) == 4 and sizes == [8, 8, 16, 16])
    tri = list(fw["loew"].items())[0]
    if tri[1]:
        nb = next(iter(tri[1]))
        check("loewenstein_ok rejects an Al-O-Al pair",
              not fwm.loewenstein_ok([tri[0], nb], fw["loew"]))
    al = fwm.random_al_placement(fw["si_idx"], fw["loew"], 3)
    check("random_al_placement returns Löwenstein-valid triple",
          al is not None and fwm.loewenstein_ok(al, fw["loew"]))
    # MOR t_rings sanity: MOR is famously a 5-ring framework
    r5 = fwm.t_rings(fw, 5)
    check("t_rings(MOR, 5): >0 rings, all valid Löwenstein-graph 5-cycles",
          len(r5) > 0 and all(
              len(set(r)) == 5
              and all(r[(k + 1) % 5] in fw["loew"][r[k]] for k in range(5))
              for r in r5))

    # ── registered FAU baseline (Foundations f0 / M1, 2026-07-10) ──
    # Present once f0 has run and the cell is registered; asserts the embedded
    # Lattice matches the registry and the general-cell loader resolves it.
    if "FAU" in fwm.BASELINES:
        rb = fwm.load_framework(baseline="FAU", symmetry=sym)
        ok = (len(rb["syms"]) == 144 and len(rb["si_idx"]) == 48
              and np.asarray(rb["cell"]).shape == (3, 3)
              and np.allclose(rb["cellmat"], fwm.BASELINES["FAU"][1], atol=1e-6))
        if sym:
            ok = ok and len(rb["by_type"]) == 1
        check("registered FAU baseline loads (general cell, single T-orbit)", ok)
    else:
        print("  (FAU not registered yet — f0/M1 pending; baseline pin skipped)")

    # ── FAU fixture (idealized general cell, Foundations 2026-07-10) ──
    if not sym:
        print("  (spglib missing — FAU fixture checks skipped)")
        return
    with tempfile.TemporaryDirectory() as td:
        fau_xyz = fau_idealized_path(td)
        if fau_xyz is None:
            print("  (FAU source xyz missing — FAU checks skipped)")
            return
        # the raw frozen file is P1-broken — the idealization is NECESSARY
        import spglib
        from ase.io import read as ase_read
        from ase.geometry.cell import cellpar_to_cell
        raw = ase_read(os.path.join(ZROOT, "FAU", "Old", "Old CellOpt",
                                    "Fau-All_Si.xyz"))
        Mraw = cellpar_to_cell(FAU_START_CELLPAR)
        raw.set_cell(Mraw); raw.set_pbc(True)
        ds_raw = spglib.get_symmetry_dataset(
            (Mraw, raw.get_scaled_positions(), raw.numbers), symprec=0.3)
        check("  (raw 500-Ry-era FAU file is P1 at symprec 0.3 — "
              "idealization is meaningful)", ds_raw.number == 1)
        fau = fwm.load_framework(xyz_path=fau_xyz, symmetry=True, symprec=0.05)
        check("FAU idealized: 144 atoms, 48 Si, 96 O, general (3,3) cell",
              len(fau["syms"]) == 144 and len(fau["si_idx"]) == 48
              and len(fau["o_idx"]) == 96
              and np.asarray(fau["cell"]).shape == (3, 3))
        check("FAU idealized: the single T-orbit at symprec 0.05",
              len(fau["by_type"]) == 1)
        # loader's Si-4-O assert passing IS the general-MIC connectivity check
        ok = all(i in fau["loew"][j] for i in fau["loew"] for j in fau["loew"][i])
        check("FAU Löwenstein graph symmetric, all T 4-connected",
              ok and all(len(v) == 4 for v in fau["loew"].values()))
        # bipartite T graph (even rings only) -> Löwenstein max 24 Al; Si:Al=2
        # (16 Al) is feasible — the f1 FAU-Si2 sampling assumption
        color = {}
        for s in sorted(fau["loew"]):
            if s in color:
                continue
            color[s] = 0
            stack = [s]
            while stack:
                u = stack.pop()
                for v in fau["loew"][u]:
                    if v not in color:
                        color[v] = 1 - color[u]
                        stack.append(v)
        bip = all(color[u] != color[v]
                  for u in fau["loew"] for v in fau["loew"][u])
        check("FAU T graph bipartite (Si:Al=2 Löwenstein-feasible)", bip)
        r6 = fwm.t_rings(fau, 6)
        r4 = fwm.t_rings(fau, 4)
        check("t_rings(FAU): 6-rings and 4-rings found, all valid cycles "
              "(%d / %d)" % (len(r6), len(r4)),
              len(r6) > 0 and len(r4) > 0 and all(
                  len(set(r)) == 6
                  and all(r[(k + 1) % 6] in fau["loew"][r[k]] for k in range(6))
                  for r in r6))
        perms = fwm.site_permutations(fau)
        one = fwm.enumerate_al_arrangements(fau, 1, perms=perms)
        check("FAU enumeration n=1: single class, g=48 (one T-orbit)",
              len(one) == 1 and one[0][1] == 48
              and len(perms) >= 48)


def test_enumeration():
    print("[2b] symmetry-distinct Al-arrangement enumeration")
    from zeolib import framework as fwm
    try:
        import spglib  # noqa: F401
    except ImportError:
        print("  (spglib missing — enumeration checks skipped)")
        return
    import itertools
    import random
    fw = fwm.load_framework()
    perms = fwm.site_permutations(fw)
    si = [int(i) for i in fw["si_idx"]]
    check("MOR site permutations: 16 ops incl. identity",
          len(perms) == 16 and any(all(m[i] == i for i in si) for m in perms))
    # n=1: classes must be exactly the T-orbits, degeneracies their sizes
    one = fwm.enumerate_al_arrangements(fw, 1, perms=perms)
    check("n=1 enumeration == T-orbits (4 classes, g=[8,8,16,16], sum 48)",
          len(one) == 4 and sorted(g for _, g in one) == [8, 8, 16, 16]
          and sum(g for _, g in one) == 48)
    # n=3 (Si15): exhaustive; sum of degeneracies must equal the brute-force
    # count of labeled Löwenstein-valid triples — validates dedupe AND g exactly
    three = fwm.enumerate_al_arrangements(fw, 3, perms=perms)
    brute = sum(1 for c in itertools.combinations(si, 3)
                if fwm.loewenstein_ok(c, fw["loew"]))
    check("n=3: sum(g) over classes == brute-force labeled count (%d)" % brute,
          sum(g for _, g in three) == brute)
    check("n=3: all class reps Löwenstein-valid, reps are canonical",
          all(fwm.loewenstein_ok(s, fw["loew"]) for s, _ in three)
          and all(fwm.canonical_arrangement(s, perms)[0] == s for s, _ in three))
    # a symmetry image canonicalizes back to its class rep
    rng = random.Random(7)
    s0, g0 = three[rng.randrange(len(three))]
    p = perms[rng.randrange(len(perms))]
    img = tuple(sorted(p[i] for i in s0))
    canon_img, g_img = fwm.canonical_arrangement(img, perms)
    check("symmetry image canonicalizes to same rep + same g",
          canon_img == s0 and g_img == g0)
    # invariants are symmetry-invariant
    inv0 = fwm.arrangement_invariants(fw, s0)
    inv1 = fwm.arrangement_invariants(fw, img)
    check("arrangement_invariants equal across a symmetry image", inv0 == inv1)
    # sampling: canonical dedupe + exclusion honored
    excl = {three[0][0]}
    samp = fwm.sample_al_arrangements(fw, 3, 25, perms=perms,
                                      rng=random.Random(1), exclude=excl)
    keys = [s for s, _ in samp]
    check("sample: 25 distinct canonical classes, exclude honored",
          len(samp) == 25 and len(set(keys)) == 25 and three[0][0] not in keys
          and all(fwm.canonical_arrangement(s, perms) == (s, g) for s, g in samp))


def test_cation():
    print("[2c] cation model (energy parity vs archived v1 mor_core.py)")
    import random
    from zeolib import framework as fwm
    from zeolib import cation
    # parity vs mor_core (v1-provenance copy, archived with the v1 chain
    # 2026-07-08 — imported as a frozen fixture, never duplicated).
    # mor_core overrides the embedded lattice with its 4-decimal CELL_ABC, so
    # the comparison framework is loaded with that same cell.
    # mor_core resolves RANGE (real UFF params) and all_Si.xyz relative to its
    # own file — both point one level short from the archive, and the RANGE
    # miss is a SILENT fallback to approximate params (2.9e-4 kcal off). So:
    # provide RANGE on sys.path first, pass the xyz explicitly below, and pin
    # that the real params actually loaded.
    sys.path.insert(0, os.path.join(ZROOT, "RANGE"))
    sys.path.insert(0, os.path.join(ZROOT, "MOR", "pipeline_archive",
                                    "stage1a_v1"))
    try:
        import mor_core
    except Exception as exc:
        print("  (mor_core not importable — parity skipped: %s)" % exc)
        return
    check("mor_core imported the REAL RANGE UFF params (no silent fallback)",
          "RANGE_go.utility" in sys.modules)
    fw = fwm.load_framework(cell=mor_core.CELL_ABC, symmetry=False)
    al = fwm.random_al_placement(fw["si_idx"], fw["loew"], 3,
                                 rng=random.Random(3))
    pre = cation.precompute_fw(fw, al)                      # exact images
    pre_leg = cation.precompute_fw(fw, al, images="legacy")  # mor_core parity
    start = cation.place_cations_near_al(fw, al)
    e0, g0 = cation.cation_energy_grad(start.flatten(), pre, "Na")
    # gradient sanity: analytic vs central finite difference (this catches the
    # mor_core factor-1/2 Coulomb-gradient bug the zeolib copy fixes)
    h, x, i = 1e-5, start.flatten().copy(), 4
    xp, xm = x.copy(), x.copy()
    xp[i] += h
    xm[i] -= h
    ep, _ = cation.cation_energy_grad(xp, pre, "Na")
    em, _ = cation.cation_energy_grad(xm, pre, "Na")
    check("analytic gradient matches finite difference",
          abs(g0[i] - (ep - em) / (2 * h)) < 1e-4 * max(1.0, abs(g0[i])))
    # explicit path: mor_core's own ../All-Si_cellopt default resolved from
    # pipeline/ — one level short from its archived home (frozen, not edited)
    mpos, msyms, msi, mo, msio, mloew = mor_core.load_framework(
        os.path.join(ZROOT, "MOR", "All-Si_cellopt", "all_Si.xyz"))
    mpre = mor_core.precompute_fw(mpos, msyms)
    mstart = mor_core.place_na_near_al(al, msio, mpos)
    check("place_cations_near_al == mor_core.place_na_near_al",
          np.allclose(start, mstart, atol=1e-10))
    meps, msig, mchg = mor_core.make_fw_params(al, mpre[0], mpre[3], mpre[4],
                                               mpre[5], mpre[6])
    me0, _ = mor_core.lj_cat_energy_grad(mstart.flatten(), mpre[2],
                                         meps, msig, mchg, "Na")
    e0_leg, _ = cation.cation_energy_grad(start.flatten(), pre_leg, "Na")
    check("legacy-images ENERGY parity with mor_core (|dE| < 1e-9 kcal)",
          abs(e0_leg - me0) < 1e-9, abs(e0_leg - me0))
    # exact images: energy must be WRAP-INVARIANT (the legacy ±1 expansion is
    # not — measured kcal-scale on MOR's short c axis, which is why "exact"
    # is the default for new work)
    mrel, meE = mor_core.relax_na_uff(al, mstart, mpre)
    zrel, zE = cation.relax_cations(start, pre, "Na")
    shifted = zrel.copy()
    shifted[0, 2] += fw["cell"][2]
    e_sh, _ = cation.cation_energy_grad(shifted.flatten(), pre, "Na")
    check("exact-images energy is wrap-invariant (<1e-9)",
          abs(e_sh - zE) < 1e-9, abs(e_sh - zE))
    e_leg_sh, _ = cation.cation_energy_grad(shifted.flatten(), pre_leg, "Na")
    e_leg_z, _ = cation.cation_energy_grad(zrel.flatten(), pre_leg, "Na")
    check("  (legacy images are indeed wrap-DEPENDENT — guard is meaningful)",
          abs(e_leg_sh - e_leg_z) > 0.1)
    # corrected gradient + full images -> equal-or-deeper minimum than
    # mor_core's (halved Coulomb gradient under-relaxed), scored consistently
    ze_at_m, _ = cation.cation_energy_grad(mrel.flatten(), pre, "Na")
    check("relax reaches equal-or-deeper minimum than mor_core's (dE=%.2f)"
          % (ze_at_m - zE), zE <= ze_at_m + 1e-6)
    # seed generation sanity
    seeds = cation.seed_cation_sets(fw, al, 4, rng=random.Random(11))
    dists = [cation.cation_set_distance(a, b, fw["cell"])
             for i, (a, _) in enumerate(seeds)
             for b, _ in [seeds[j] for j in range(i + 1, len(seeds))]]
    check("seed_cation_sets: >=2 distinct seeds, sorted by E, dedupe holds",
          len(seeds) >= 2
          and all(seeds[i][1] <= seeds[i + 1][1] for i in range(len(seeds) - 1))
          and all(d >= 0.75 for d in dists))
    at = cation.assemble_atoms(fw, al, seeds[0][0])
    ss = at.get_chemical_symbols()
    check("assemble_atoms: 3 Al substituted, 3 Na appended last",
          ss.count("Al") == 3 and ss[-3:] == ["Na"] * 3 and len(at) == 147)

    # ── exchange_na (Foundations f2, 2026-07-09) ──
    def struct_of(atoms):
        return dict(syms=np.array(atoms.get_chemical_symbols()),
                    pos=atoms.get_positions().copy(), cell=fw["cell"])

    st3 = struct_of(at)                       # 3 Al / 3 Na (Si15-like)
    agp, age, aggr = cation.exchange_na(st3, "Ag")
    fw_pos = at.get_positions()[:144]
    from zeolib.geometry import mic_all
    check("exchange_na Ag: 3 cations (1:1), singleton groups, >1.5 Å from fw",
          agp.shape == (3, 3) and len(aggr) == 3
          and all(len(g) == 1 for g in aggr)
          and mic_all(agp, fw_pos, fw["cell"]).min() > 1.5)
    bip, bie, bigr = cation.exchange_na(st3, "Bi")     # 3 Al -> 1 Bi
    check("exchange_na Bi on 3 Al: one cation from one 3-Na group",
          bip.shape == (1, 3) and len(bigr) == 1 and len(bigr[0]) == 3
          and sorted(bigr[0]) == [144, 145, 146])
    try:
        cation.exchange_na(st3, "Pb")                  # 3 Al: q=2 can't balance
        check("exchange_na rejects unbalanceable Pb on 3 Al", False)
    except ValueError:
        check("exchange_na rejects unbalanceable Pb on 3 Al", True)
    al4 = fwm.random_al_placement(fw["si_idx"], fw["loew"], 4,
                                  rng=random.Random(5))
    s4 = cation.seed_cation_sets(fw, al4, 1)
    at4 = cation.assemble_atoms(fw, al4, s4[0][0])
    st4 = struct_of(at4)
    pbp, pbe, pbgr = cation.exchange_na(st4, "Pb")
    na_all = sorted(i for g in pbgr for i in g)
    pbp2, pbe2, pbgr2 = cation.exchange_na(st4, "Pb")
    check("exchange_na Pb on 4 Al: 2 cations, groups partition the 4 Na, "
          "deterministic",
          pbp.shape == (2, 3) and na_all == [144, 145, 146, 147]
          and np.allclose(pbp, pbp2) and pbgr == pbgr2)

    # ── general-cell (FAU) energy/relax path ──
    with tempfile.TemporaryDirectory() as td:
        fau_xyz = fau_idealized_path(td)
        if fau_xyz is None:
            print("  (FAU fixture unavailable — general-cell cation checks "
                  "skipped)")
            return
        fau = fwm.load_framework(xyz_path=fau_xyz, symmetry=False)
        M = fau["cellmat"]
        alf = fwm.random_al_placement(fau["si_idx"], fau["loew"], 4,
                                      rng=random.Random(9))
        pref = cation.precompute_fw(fau, alf)
        startf = cation.place_cations_near_al(fau, alf)
        ef, gf = cation.cation_energy_grad(startf.flatten(), pref, "Na")
        relf, eref = cation.relax_cations(startf, pref, "Na")
        check("FAU general cell: energy finite, relax lowers it, wrapped",
              np.isfinite(ef) and np.isfinite(eref) and eref <= ef + 1e-6
              and np.all((relf @ np.linalg.inv(M)) > -1e-9)
              and np.all((relf @ np.linalg.inv(M)) < 1.0 + 1e-9))
        try:
            cation.precompute_fw(fau, alf, images="legacy")
            check("general cell rejects images='legacy'", False)
        except ValueError:
            check("general cell rejects images='legacy'", True)


def test_cp2k_generation():
    print("[3] cp2k input generation")
    import glob as g
    cell = [17.8481395059, 20.6994211544, 7.5791549307]
    elements = ("Si", "Al", "O", "Na")
    gen_cell = cp2k.cell_opt_input(cell, elements)
    gen_geo = cp2k.geo_opt_input(cell, elements)
    staged = sorted(g.glob(os.path.join(
        ZROOT, "MOR", "tests", "na_placement_multicomp", "n*_arr*", "cell-opt.inp")))
    if staged:
        ref = open(staged[0], newline="").read().replace("\r\n", "\n")
        check("cell_opt_input == staged multicomp cell-opt.inp (byte-parity)",
              gen_cell == ref, "first diff at char %d" %
              next((i for i, (x, y) in enumerate(zip(gen_cell, ref)) if x != y),
                   min(len(gen_cell), len(ref))))
        ref_g = open(os.path.join(os.path.dirname(staged[0]), "geo-opt.inp"),
                     newline="").read().replace("\r\n", "\n")
        check("geo_opt_input == staged multicomp geo-opt.inp (byte-parity)",
              gen_geo == ref_g)
    else:
        print("  (no staged multicomp inputs found — parity check skipped)")
    uks = cp2k.cell_opt_input(cell, ("Si", "Al", "O", "Ag", "I"), uks=True,
                              multiplicity=2, basis_rel="../../../../")
    check("UKS variant carries UKS + MULTIPLICITY + 4-up basis path",
          "    UKS\n    MULTIPLICITY 2\n" in uks
          and "BASIS_SET_FILE_NAME ../../../../BASIS" in uks
          and "GTH-PBE-q11" in uks and "GTH-PBE-q7" in uks)
    sp = cp2k.energy_force_input(cell, elements, restart_from="cell-opt-1.restart")
    check("energy_force_input: ENERGY_FORCE + stress print + EXT_RESTART, no ASPC",
          "RUN_TYPE ENERGY_FORCE" in sp and "&STRESS_TENSOR" in sp
          and "RESTART_POS T" in sp and "WF_INTERPOLATION" not in sp)
    for bad in ("Xx", "Fe"):
        try:
            cp2k.kind_blocks((bad,))
            check("kind_blocks rejects unknown element %s" % bad, False)
        except KeyError:
            check("kind_blocks rejects unknown element %s" % bad, True)
    # ── Foundations additions (2026-07-09) ──
    frozen = cp2k.geo_opt_input(cell, elements, restart_from=None,
                                fixed_atoms=(1, 147))
    blk = ("  &CONSTRAINT\n    &FIXED_ATOMS\n      LIST 1..147\n"
           "    &END FIXED_ATOMS\n  &END CONSTRAINT\n")
    check("geo_opt_input fixed_atoms renders the v0/FAU constraint block",
          blk in frozen and frozen.index(blk) < frozen.index("&END MOTION")
          and "&EXT_RESTART" not in frozen)
    check("  (default geo_opt_input carries no constraint)",
          "&CONSTRAINT" not in gen_geo)
    for badf in ((0, 5), (10, 3)):
        try:
            cp2k.geo_opt_input(cell, elements, fixed_atoms=badf)
            check("fixed_atoms rejects %r" % (badf,), False)
        except ValueError:
            check("fixed_atoms rejects %r" % (badf,), True)
    wfn = cp2k.geo_opt_input(cell, elements, restart_from=None,
                             wfn_restart="../screen_01/screen-RESTART.wfn")
    check("wfn_restart: RESTART guess + WFN_RESTART_FILE_NAME (v0 hand-off)",
          "    WFN_RESTART_FILE_NAME ../screen_01/screen-RESTART.wfn\n" in wfn
          and "SCF_GUESS RESTART" in wfn and "SCF_GUESS ATOMIC" not in wfn
          and "SCF_GUESS ATOMIC" in gen_geo)
    rho = cp2k.cell_opt_input([17.235, 17.235, 17.235], ("Si", "O"),
                              angles=(60.0, 60.0, 60.0),
                              symmetry="RHOMBOHEDRAL", keep_symmetry=True)
    check("rhombohedral cell block matches the FAU-era input form",
          "      ABC 17.2350 17.2350 17.2350\n" in rho
          and "      ALPHA_BETA_GAMMA 60.000 60.000 60.000\n" in rho
          and "      SYMMETRY RHOMBOHEDRAL\n" in rho
          and "KEEP_SYMMETRY TRUE" in rho and "KEEP_ANGLES TRUE" in rho)
    # ── SCF-rescue variant (2026-07-17, stage1a_v2 dft_run1 rescue) ──
    outer_blk = ("      &OUTER_SCF\n        EPS_SCF 1.0E-7\n"
                 "        MAX_SCF 60\n      &END OUTER_SCF\n")
    for name, txt in (
            ("cell_opt", cp2k.cell_opt_input(cell, elements, scf_outer=(50, 60))),
            ("geo_opt", cp2k.geo_opt_input(cell, elements, scf_outer=(50, 60))),
            ("energy_force", cp2k.energy_force_input(cell, elements,
                                                     scf_outer=(50, 60)))):
        check("scf_outer=(50,60) %s: inner MAX_SCF 50 + OUTER_SCF, eps unchanged"
              % name,
              "      MAX_SCF 50\n" in txt and outer_blk in txt
              and "MAX_SCF 3000" not in txt
              and txt.index("&END OT") < txt.index("&OUTER_SCF")
              and txt.count("EPS_SCF 1.0E-7") == 2)
    check("  (default inputs carry no OUTER_SCF, MAX_SCF 3000 intact)",
          "&OUTER_SCF" not in gen_geo and "&OUTER_SCF" not in gen_cell
          and "      MAX_SCF 3000\n" in gen_geo)
    resc = cp2k.geo_opt_input(cell, elements,
                              restart_from="cell-opt-1.restart",
                              wfn_restart="cell-opt-RESTART.wfn",
                              scf_outer=(50, 60))
    check("rescue geo_opt combines EXT_RESTART + wfn RESTART guess + OUTER_SCF",
          "RESTART_FILE_NAME cell-opt-1.restart" in resc
          and "WFN_RESTART_FILE_NAME cell-opt-RESTART.wfn" in resc
          and "SCF_GUESS RESTART" in resc and outer_blk in resc)
    # ── rescue_scf_text: post-hoc transform == builder scf_outer (2026-07-21,
    #    Foundations vacation supervisor — rescue without builder args) ──
    for name, plain, target in (
            ("cell_opt", cp2k.cell_opt_input(cell, elements),
             cp2k.cell_opt_input(cell, elements, scf_outer=(50, 60))),
            ("geo_opt", cp2k.geo_opt_input(cell, elements),
             cp2k.geo_opt_input(cell, elements, scf_outer=(50, 60))),
            ("energy_force", cp2k.energy_force_input(cell, elements),
             cp2k.energy_force_input(cell, elements, scf_outer=(50, 60)))):
        check("rescue_scf_text(%s) byte-equals builder scf_outer=(50,60)"
              % name, cp2k.rescue_scf_text(plain) == target)
    uks_plain = cp2k.geo_opt_input(cell, elements, uks=True, multiplicity=2,
                                   fixed_atoms=(1, 10), restart_from=None)
    uks_target = cp2k.geo_opt_input(cell, elements, uks=True, multiplicity=2,
                                    fixed_atoms=(1, 10), restart_from=None,
                                    scf_outer=(50, 60))
    # GEO_OPT convergence override (Foundations f3 screens, 2026-08-20):
    # loose for a geometry-producing step, tight by default for energies.
    g_loose = cp2k.geo_opt_input([17.0, 20.0, 7.5], ("Si", "O"),
                                 max_force="1.0E-3", rms_force="7.0E-4",
                                 max_dr="1.0E-2", rms_dr="7.0E-3")
    g_tight = cp2k.geo_opt_input([17.0, 20.0, 7.5], ("Si", "O"))
    check("geo_opt_input: convergence override applies to all four criteria",
          all(x in g_loose for x in ("MAX_FORCE 1.0E-3", "RMS_FORCE 7.0E-4",
                                     "MAX_DR 1.0E-2", "RMS_DR 7.0E-3")))
    check("geo_opt_input: tight defaults unchanged (production values)",
          all(x in g_tight for x in ("MAX_FORCE 1.0E-4", "RMS_FORCE 7.0E-5",
                                     "MAX_DR 1.0E-3", "RMS_DR 7.0E-4")))

    # SCF solver knobs (tests/scf_openshell, 2026-08-20). Defaults are pinned
    # byte-identical by the parity checks above; here we pin the opt-ins.
    d_def = cp2k.dft_section()
    d_diis = cp2k.dft_section(ot_minimizer="DIIS")
    d_fsi = cp2k.dft_section(ot_preconditioner="FULL_SINGLE_INVERSE")
    d_diag = cp2k.dft_section(diagonalization=True)
    check("dft_section: default still OT/FULL_ALL/CG with LINESEARCH",
          "PRECONDITIONER FULL_ALL" in d_def and "MINIMIZER CG" in d_def
          and "LINESEARCH 3PNT" in d_def and "&DIAGONALIZATION" not in d_def)
    check("dft_section: MINIMIZER DIIS drops the CG-only LINESEARCH",
          "MINIMIZER DIIS" in d_diis and "LINESEARCH" not in d_diis
          and "&OT T" in d_diis)
    check("dft_section: preconditioner override applied",
          "PRECONDITIONER FULL_SINGLE_INVERSE" in d_fsi
          and "MINIMIZER CG" in d_fsi)
    check("dft_section: diagonalization replaces OT with DIAG+MIXING",
          "&DIAGONALIZATION T" in d_diag and "BROYDEN_MIXING" in d_diag
          and "&OT T" not in d_diag and "EPS_SCF" in d_diag)
    check("dft_section: solver knobs do not disturb EPS_SCF/CUTOFF",
          all(("EPS_SCF %s" % cp2k.EPS_SCF) in x and "CUTOFF 1500" in x
              for x in (d_def, d_diis, d_fsi, d_diag)))

    # coords_file override (dft_run1 SCF rescue, 2026-08-19): a standalone
    # geo-opt from a rebuilt start must not clobber the job's original
    # coords.inc, and the DEFAULT must stay byte-identical (parity pins above).
    g_def = cp2k.geo_opt_input([17.0, 20.0, 7.5], ("Si", "Al", "O", "Na"))
    g_alt = cp2k.geo_opt_input([17.0, 20.0, 7.5], ("Si", "Al", "O", "Na"),
                               coords_file="coords_cellopt.inc")
    check("geo_opt_input coords_file: default coords.inc, override applied, "
          "exactly one line differs",
          "@INCLUDE 'coords.inc'" in g_def
          and "@INCLUDE 'coords_cellopt.inc'" in g_alt
          and sum(1 for x, y in zip(g_def.split("\n"), g_alt.split("\n"))
                  if x != y) == 1)
    check("rescue_scf_text UKS+FIXED_ATOMS geo_opt byte-parity",
          cp2k.rescue_scf_text(uks_plain) == uks_target)
    wfn_plain = cp2k.geo_opt_input(cell, elements, restart_from=None,
                                   wfn_restart="../screen/screen-RESTART.wfn")
    wfn_target = cp2k.geo_opt_input(cell, elements, restart_from=None,
                                    wfn_restart="../screen/screen-RESTART.wfn",
                                    scf_outer=(50, 60))
    check("rescue_scf_text wfn-restart geo_opt byte-parity",
          cp2k.rescue_scf_text(wfn_plain) == wfn_target)
    check("rescue_scf_text custom (30, 20) byte-parity",
          cp2k.rescue_scf_text(cp2k.geo_opt_input(cell, elements),
                               scf_outer=(30, 20))
          == cp2k.geo_opt_input(cell, elements, scf_outer=(30, 20)))
    try:
        cp2k.rescue_scf_text(cp2k.rescue_scf_text(
            cp2k.geo_opt_input(cell, elements)))
        twice = False
    except ValueError:
        twice = True
    check("rescue_scf_text refuses a second application", twice)


def test_cp2k_parsing():
    print("[4] cp2k output parsing (real 1500-Ry job)")
    d = os.path.join(ZROOT, "MOR", "tests", "na_training_set", "sp_arr00_na00_start")
    out = os.path.join(d, "energy-force.out")
    if not os.path.exists(out):
        print("  (fixture job missing — skipped)")
        return
    check("program_ended_ok on finished job", cp2k.program_ended_ok(out))
    # opt_completed accepts BOTH CP2K convergence banners (2026-07-22: the
    # FAU f1 cell-opts exit via L-BFGS's own criteria, MOR's via the standard
    # geometry criteria — see the docstring). SCF-only text must NOT pass.
    import tempfile as _tf
    for label, text, want in (
            ("standard banner",
             " *** GEOMETRY OPTIMIZATION COMPLETED ***\n", True),
            ("L-BFGS own-criteria banner",
             " * satisfied .... run CONVERGED!\n", True),
            ("SCF-converged text alone (must NOT count)",
             " *** SCF run converged in     2 steps ***\n", False),
            ("unconverged run", " step 12 of 3000\n", False)):
        with _tf.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
            f.write(text)
            p = f.name
        check("opt_completed: %s -> %s" % (label, want),
              cp2k.opt_completed(p) is want)
        os.remove(p)
    # run_time_seconds: the two independent sources must AGREE, and a
    # walltime-killed run must yield None rather than a partial duration
    # (walltime sizing reads these numbers as complete-run durations). [2026-08-24]
    md = os.path.join(ZROOT, "MFI", "AIMD_out", "md.out")
    if os.path.exists(md):
        t = cp2k.run_time_seconds(md)
        check("run_time_seconds reads the CP2K timing row (292233.2 s)",
              t is not None and abs(t - 292233.2) < 0.05, t)
        stamped = "".join(ln for ln in open(md, errors="replace")
                          if "PROGRAM STARTED AT" in ln or "PROGRAM ENDED AT" in ln)
        with _tf.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
            f.write(stamped)
            p2 = f.name
        t2 = cp2k.run_time_seconds(p2)
        check("  banner-timestamp fallback agrees with the timing row (<1.5 s)",
              t2 is not None and abs(t2 - t) < 1.5, t2)
        os.remove(p2)
    with _tf.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        f.write(" *** SCF run converged in     2 steps ***\n")
        p3 = f.name
    check("run_time_seconds -> None for a killed/running job (no partial time)",
          cp2k.run_time_seconds(p3) is None)
    os.remove(p3)
    e = cp2k.final_energy_ha(out)
    check("final_energy_ha plausible (-2000..-1500 Ha)",
          e is not None and -2000 < e < -1500, e)
    syms, pos = fileio.read_coords_inc(os.path.join(d, "coords.inc"))
    frc = cp2k.read_forces_au(d)
    check("forces frame parsed, one row per atom (%d)" % len(syms),
          frc is not None and len(frc) == len(syms))
    st = cp2k.read_stress_ase_ev_ang3(d)
    check("stress parsed: 9 components, |tr| < 1 eV/A^3",
          st is not None and len(st) == 9 and abs(st[0] + st[4] + st[8]) < 1.0)
    train = os.path.join(ZROOT, "MOR", "tests", "na_training_set", "train_1500.xyz")
    if os.path.exists(train) and e is not None:
        tag = "energy=%.8f" % (e * constants.HARTREE_TO_EV)
        found = any(tag in ln for ln in open(train) if "energy=" in ln)
        check("energy matches a train_1500.xyz frame header (%s)" % tag, found)
    else:
        print("  (train_1500.xyz missing — cross-check skipped)")
    # 3x3 cell parser vs the orthorhombic convenience on a real .cell print
    cf = os.path.join(ZROOT, "MOR", "tests", "cell_rebaseline_1500",
                      "cell-opt-1.cell")
    if os.path.exists(cf):
        abc = cp2k.read_last_cell_abc(cf)
        vec = cp2k.read_last_cell_vectors(cf)
        check("read_last_cell_vectors == diag(read_last_cell_abc) on an "
              "orthorhombic run",
              abc is not None and vec is not None
              and np.allclose(vec, np.diag(abc), atol=1e-8))
    else:
        print("  (cell_rebaseline_1500 .cell fixture missing — skipped)")


    # CP2K stress-print unit changed between versions (cp2k_image_parity,
    # 2026-08-20): 2022.1 prints [GPa], 2025.2/2026.1 print [bar]. The same
    # input run under both must parse to the SAME tensor; an unknown unit must
    # raise rather than silently yield None (which is what used to happen to
    # every 2022.1 harvest).
    ip = os.path.join(ZROOT, "MOR", "tests", "cp2k_image_parity")
    d22 = os.path.join(ip, "perlmutter_2022", "mor_si15_c00801")
    d25 = os.path.join(ip, "perlmutter_2025_2", "mor_si15_c00801")
    if os.path.isdir(d22) and os.path.isdir(d25):
        s22 = cp2k.read_stress_ase_ev_ang3(d22, pattern="energy-force*stress*")
        s25 = cp2k.read_stress_ase_ev_ang3(d25, pattern="energy-force*stress*")
        check("stress: GPa (2022.1) and bar (2025.2) prints parse equal (1e-7)",
              s22 is not None and s25 is not None
              and max(abs(x - y) for x, y in zip(s22, s25)) < 1e-7)
    else:
        print("  (cp2k_image_parity trees absent — stress-unit check skipped)")
    try:
        cp2k._stress_unit_probe = None
        bad = "STRESS| Analytical stress tensor [furlongs]\n"
        import tempfile
        td = tempfile.mkdtemp()
        with open(os.path.join(td, "x-stress-1_0.stress_tensor"), "w") as fh:
            fh.write("\n" + bad + "a\nb\nc\nd\n")
        cp2k.read_stress_ase_ev_ang3(td, pattern="*stress*")
        check("stress: unknown unit RAISES (no silent None)", False)
    except ValueError:
        check("stress: unknown unit RAISES (no silent None)", True)


def test_slurm():
    print("[5] slurm generation")
    sb = slurm.sbatch_text("t_x", [slurm.cp2k_run_line("cell-opt.inp"),
                                   slurm.cp2k_run_line("geo-opt.inp")])
    code_lines = [ln for ln in sb.splitlines() if not ln.lstrip().startswith("#")]
    check("sbatch: container mpirun, no srun launch, OMP=1, redirect not -o",
          "mpirun -np 32" in sb and not any("srun" in ln for ln in code_lines)
          and "OMP_NUM_THREADS=1" in sb and "> cell-opt.out 2>&1" in sb
          and slurm.CP2K_SIF in sb)
    sub = slurm.submit_script_text(["cfg_a", "cfg_b"])
    check("submit script: psub route + 50-throttle fallback + timestamps",
          "psub" in sub and str(slurm.MAX_CONCURRENT_JOBS) in sub
          and "date '+%F %T'" in sub and "submitted [cfg_a]" in sub)
    # Perlmutter profile — byte-parity vs a shipped wanderer-SP sbatch
    pm = slurm.sbatch_text(
        "wsparr15_na07",
        [slurm.cp2k_run_line("energy-force.inp", profile=slurm.PERLMUTTER)],
        profile=slurm.PERLMUTTER)
    ref_path = os.path.join(ZROOT, "MOR", "tests", "na_placement_3al",
                            "wsp_arr15_na07", "perlmutter.sbatch")
    if os.path.exists(ref_path):
        ref = open(ref_path, newline="").read().replace("\r\n", "\n")
        check("perlmutter sbatch == shipped wsp perlmutter.sbatch (byte-parity)",
              pm == ref)
    else:
        print("  (shipped perlmutter.sbatch missing — parity check skipped)")
    check("perlmutter sbatch: shifter+srun, no --account/--partition/--time-min",
          "srun shifter --entrypoint cp2k" in pm and "--account" not in pm
          and "--partition" not in pm and "--time-min" not in pm
          and slurm.PERLMUTTER_IMAGE in pm)
    alt = slurm.sbatch_text(
        "x", [], profile=slurm.PERLMUTTER, walltime="00:30:00",
        image="docker:cp2k/cp2k:2026.1")
    check("perlmutter sbatch: image= overrides the default, default unchanged",
          "--image docker:cp2k/cp2k:2026.1" in alt
          and slurm.PERLMUTTER_IMAGE not in alt
          and slurm.sbatch_text("x", [], profile=slurm.PERLMUTTER,
                                walltime="00:30:00") ==
              slurm.sbatch_text("x", [], profile=slurm.PERLMUTTER,
                                walltime="00:30:00", image=None))
    check("perlmutter sbatch: qos default 'regular', override to 'debug'",
          "--qos regular" in pm and "--qos debug" not in pm
          and "--qos debug" in slurm.sbatch_text(
              "x", [], profile=slurm.PERLMUTTER, qos="debug", walltime="00:10:00"))
    try:
        slurm.sbatch_text("x", [], profile=slurm.PERLMUTTER, walltime="3-00:00:00")
        check("perlmutter walltime cap (2 d) enforced", False)
    except ValueError:
        check("perlmutter walltime cap (2 d) enforced", True)
    check("pronghorn accepts long walltime (14 d cap)",
          "--time=13-00:00:00" in slurm.sbatch_text(
              "x", [], walltime="13-00:00:00"))
    psub = slurm.submit_script_text(["wsp_a"], profile=slurm.PERLMUTTER)
    check("perlmutter submit script: no throttle loop, shifterimg note, timestamps",
          "shifterimg pull" in psub and "squeue" not in psub and "psub" not in psub
          and "perlmutter.sbatch" in psub and "date '+%F %T'" in psub)
    # Perlmutter job-array bundling (one sbatch + manifest, 2026-07-17)
    man = slurm.array_manifest_text(["Si15/a01_s1", "Si15/a01_s2"])
    check("array manifest: one dir per line (1-based order), trailing newline",
          man == "Si15/a01_s1\nSi15/a01_s2\n")
    arr = slurm.array_sbatch_text(
        "f1_na", [slurm.cp2k_run_line("energy-force.inp",
                                      profile=slurm.PERLMUTTER)], 540)
    check("array sbatch: 1-based --array, shifter+srun, manifest sed lookup, "
          "fatal on missing line, no --account/--time-min",
          "--array 1-540\n" in arr and "srun shifter --entrypoint cp2k" in arr
          and 'sed -n "${SLURM_ARRAY_TASK_ID}p"' in arr
          and "FATAL: no manifest line" in arr and "exit 1" in arr
          and 'cd "${SLURM_SUBMIT_DIR}/${JOB_DIR}"' in arr
          and "OMP_NUM_THREADS=1" in arr and "--output cp2k_%A_%a.out" in arr
          and "--account" not in arr and "--time-min" not in arr
          and slurm.PERLMUTTER_IMAGE in arr)
    check("array sbatch: max_running renders %cap; n_jobs accepts the dir list",
          "--array 1-540%50" in slurm.array_sbatch_text(
              "x", [], 540, max_running=50)
          and "--array 1-2\n" in slurm.array_sbatch_text("x", [], ["a", "b"]))
    for label, fn in (
            ("array sbatch rejects pronghorn (no psub routing in an array)",
             lambda: slurm.array_sbatch_text("x", [], 5,
                                             profile=slurm.PRONGHORN)),
            ("array sbatch enforces the 2 d walltime cap",
             lambda: slurm.array_sbatch_text("x", [], 5,
                                             walltime="3-00:00:00")),
            ("array manifest rejects empty job_dirs",
             lambda: slurm.array_manifest_text([]))):
        try:
            fn()
            check(label, False)
        except ValueError:
            check(label, True)
    # write_csv_lf — csv defaults to CRLF even with newline="\n" on the handle
    import tempfile as _tf
    _p = os.path.join(_tf.mkdtemp(), "t.csv")
    fileio.write_csv_lf(_p, [dict(a=1, b="x"), dict(a=2, b="y,z")])
    _raw = open(_p, "rb").read()
    check("write_csv_lf: LF only, header first, quoting intact",
          _raw == b'a,b\n1,x\n2,"y,z"\n' and not fileio.has_crlf(_p))
    try:
        fileio.write_csv_lf(_p, [])
        check("write_csv_lf raises on no rows and no fieldnames", False)
    except ValueError:
        check("write_csv_lf raises on no rows and no fieldnames", True)
    # strip_wfn_restart — byte-parity with the no-handoff builder call
    _kw = dict(basis_rel="../../../", project="full-opt", max_iter=3000)
    _with = cp2k.geo_opt_input((10.0, 11.0, 12.0), ["Si", "O"],
                               wfn_restart="../screen_12r/screen-RESTART.wfn",
                               **_kw)
    _without = cp2k.geo_opt_input((10.0, 11.0, 12.0), ["Si", "O"], **_kw)
    check("strip_wfn_restart: byte-identical to wfn_restart=None",
          cp2k.strip_wfn_restart(_with) == _without
          and "SCF_GUESS ATOMIC" in _without)
    for label, fn in (
            ("strip_wfn_restart raises when there is no hand-off to strip",
             lambda: cp2k.strip_wfn_restart(_without)),):
        try:
            fn()
            check(label, False)
        except ValueError:
            check(label, True)
    # PACKED job array — several 32-rank CP2K jobs per node (2026-08-25,
    # Foundations Perlmutter migration; node-hour efficiency, see docstring)
    check("cp2k_run_line: srun_flags=None keeps the perlmutter line byte-identical",
          slurm.cp2k_run_line("x.inp", profile=slurm.PERLMUTTER, srun_flags=None)
          == "srun shifter --entrypoint cp2k -i x.inp > x.out 2>&1")
    grp = slurm.pack_groups(["d%d" % i for i in range(9)], 4)
    check("pack_groups: order-preserving chunks, short final group",
          grp == [["d0", "d1", "d2", "d3"], ["d4", "d5", "d6", "d7"], ["d8"]]
          and slurm.packed_array_manifest_text(grp)
          == "d0 d1 d2 d3\nd4 d5 d6 d7\nd8\n")
    pk = slurm.packed_array_sbatch_text("f3s", "screen.inp", grp, 4,
                                        walltime="12:00:00")
    check("packed array sbatch: node packed 4x32, --exact steps, group-fatal rc, "
          "no --account/--time-min",
          "--array 1-3\n" in pk and "--ntasks-per-node 128" in pk
          and "--exact --nodes 1 --ntasks 32 --cpus-per-task 2 "
              "--cpu-bind=cores --mem=110G shifter --entrypoint cp2k "
              "-i screen.inp > screen.out 2>&1" in pk
          and 'GROUP="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")"' in pk
          and "FATAL: no manifest line" in pk
          and 'wait "$p" || RC=1' in pk and "exit $RC" in pk
          and "OMP_NUM_THREADS=1" in pk
          and "--account" not in pk and "--time-min" not in pk
          and slurm.PERLMUTTER_IMAGE in pk)
    for label, fn in (
            ("packed array rejects pronghorn",
             lambda: slurm.packed_array_sbatch_text(
                 "x", "a.inp", 2, 4, profile=slurm.PRONGHORN)),
            ("packed array enforces the 2 d walltime cap",
             lambda: slurm.packed_array_sbatch_text(
                 "x", "a.inp", 2, 4, walltime="3-00:00:00")),
            ("packed array rejects per_node < 1",
             lambda: slurm.packed_array_sbatch_text("x", "a.inp", 2, 0)),
            ("packed manifest rejects a job dir containing whitespace",
             lambda: slurm.packed_array_manifest_text([["a b"]])),
            ("packed manifest rejects empty groups",
             lambda: slurm.packed_array_manifest_text([])),
            ("pack_groups rejects an empty job list",
             lambda: slurm.pack_groups([], 4))):
        try:
            fn()
            check(label, False)
        except ValueError:
            check(label, True)
    # copy-back puller (feedback_ship_copyback_script) — ships in every package
    cb = slurm.copy_back_script_text("tests/na_placement_multicomp")
    check("copy_back.sh: host+remote vars, restart excludes, no --delete, "
          "dest=script dir",
          cb.startswith("#!/usr/bin/env bash")
          and 'HOST="user@cluster.example.edu"' in cb
          and ('REMOTE="/scratch/user/MOR/'
               'tests/na_placement_multicomp"') in cb
          and "--exclude='*.wfn'" in cb and "--exclude='*.restart'" in cb
          and "--delete" not in cb
          and 'DEST="$(cd "$(dirname "$0")"' in cb)
    check("copy_back.sh: environment-adaptive (rsync branch + tar-over-ssh "
          "fallback, both excluding)",
          "command -v rsync" in cb and "rsync -avz" in cb
          and 'ssh "$HOST" "tar czf - -C ' + "'$REMOTE'" in cb
          and '| tar xzf - -C "$DEST"' in cb
          and cb.count("--exclude='*.wfn'") == 2)
    cb_abs = slurm.copy_back_script_text("/pscratch/x/pkg", profile=slurm.PERLMUTTER,
                                         extra_excludes=("*.cube",))
    check("copy_back.sh: absolute remote_dir + perlmutter host + extra exclude "
          "(both branches)",
          'HOST="user@perlmutter.example.gov"' in cb_abs
          and 'REMOTE="/pscratch/x/pkg"' in cb_abs
          and cb_abs.count("--exclude='*.cube'") == 2)
    try:
        slurm.copy_back_script_text("relative/path", profile=slurm.PERLMUTTER)
        check("copy_back.sh: base-less profile rejects relative remote_dir", False)
    except ValueError:
        check("copy_back.sh: base-less profile rejects relative remote_dir", True)
    # ship pusher (feedback_windows_transfer) — the upload counterpart
    sh = slurm.ship_script_text("tests/na_placement_multicomp")
    check("ship.sh: mkdir -p remote, rsync branch + tar-over-ssh fallback, "
          "pushes own dir, no excludes",
          sh.startswith("#!/usr/bin/env bash")
          and 'SRC="$(cd "$(dirname "$0")" && pwd)"' in sh
          and 'ssh "$HOST" "mkdir -p ' + "'$REMOTE'" in sh
          and 'rsync -avz "$SRC"/ "$HOST:$REMOTE/"' in sh
          and 'tar czf - -C "$SRC" . | ssh "$HOST" "tar xzf - -C ' + "'$REMOTE'" in sh
          and "--exclude" not in sh)
    try:
        slurm.ship_script_text("relative/path", profile=slurm.PERLMUTTER)
        check("ship.sh: base-less profile rejects relative remote_dir", False)
    except ValueError:
        check("ship.sh: base-less profile rejects relative remote_dir", True)


def test_fileio():
    print("[6] fileio")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.sh")
        fileio.write_lf(p, "#!/bin/bash\necho hi\n", executable=True)
        check("write_lf produces no CRLF on Windows", not fileio.has_crlf(p))
        pu = os.path.join(td, "u.txt")
        fileio.write_lf(pu, "Löwenstein → Å\n")
        check("write_lf writes UTF-8 regardless of platform default",
              open(pu, "rb").read() == "Löwenstein → Å\n".encode("utf-8"))
        ci = os.path.join(td, "coords.inc")
        fileio.write_coords_inc(ci, symbols=["Si", "O"],
                                positions=[(0.0, 1.5, 2.25), (3.0, 4.5, 6.75)])
        syms, pos = fileio.read_coords_inc(ci)
        check("coords.inc round-trip", syms == ["Si", "O"]
              and abs(pos[1][2] - 6.75) < 1e-12 and not fileio.has_crlf(ci))

        # --- multi-frame XYZ (Foundations communication/ compilation) ---
        traj = os.path.join(td, "geo-opt-pos-1.xyz")
        frame = "2\n i = %d, time = 0.000, E = -1.2345\nSi 0.0 0.0 %.1f\nO 1.0 1.0 1.0\n"
        fileio.write_lf(traj, "".join(frame % (k, k) for k in (1, 2, 3)))
        fr = fileio.read_xyz_frames(traj)
        check("read_xyz_frames: 3 CP2K trajectory frames parsed", len(fr) == 3)
        last = fileio.read_xyz_frames(traj, last_only=True)
        check("read_xyz_frames(last_only) == converged final frame",
              len(last) == 1 and abs(last[0]["positions"][0][2] - 3.0) < 1e-12)
        trunc = os.path.join(td, "killed-pos-1.xyz")
        fileio.write_lf(trunc, "2\n i = 1\nSi 0.0 0.0 0.0\n")
        try:
            fileio.read_xyz_frames(trunc)
            raised = False
        except ValueError:
            raised = True
        check("truncated (walltime-killed) trajectory RAISES, no partial geom",
              raised)

        ext = os.path.join(td, "compiled.xyz")
        fileio.write_extxyz(ext, fr, lattice=(10.0, 0, 0, 0, 11.0, 0, 0, 0, 12.0))
        back = fileio.read_xyz_frames(ext)
        head = open(ext).readlines()[1]
        check("write_extxyz round-trips through read_xyz_frames",
              len(back) == 3 and back[2]["symbols"] == ["Si", "O"]
              and abs(back[2]["positions"][0][2] - 3.0) < 1e-12)
        check("write_extxyz: Lattice emitted, CP2K comment quoted into info=",
              head.startswith('Lattice="10.000000 0.000000') and 'info="i = 1,' in head
              and not fileio.has_crlf(ext))
        # a re-emit must preserve BOTH cell and label, or compiled structure
        # sets silently lose their lattice on the second pass
        again = os.path.join(td, "compiled2.xyz")
        fileio.write_extxyz(again, back)
        check("extxyz re-emit preserves Lattice and info (round-trip stable)",
              open(again).readlines()[1] == head
              and fileio.read_xyz_frames(again)[0]["comment"].startswith("i = 1,")
              and abs(back[0]["lattice"][4] - 11.0) < 1e-9)
        from zeolib import framework as fwm
        base = fileio.read_xyz_frames(fwm.BASELINES["FAU"][0], last_only=True)
        check("read_xyz_frames on the registered FAU baseline: 144 atoms",
              len(base[0]["symbols"]) == 144)


def test_maceenv():
    print("[7] maceenv model registry (torch-free)")
    from zeolib import maceenv
    for name in ("na1500-std", "na1500-polar"):
        try:
            e = maceenv.resolve_model(name)
            check("%s resolves to an existing file in models/" % name,
                  os.path.isfile(e["path"]) and e["kind"] == "finetuned")
        except Exception as exc:
            check("%s resolves to an existing file in models/" % name, False, exc)
    e = maceenv.resolve_model("mp0-medium")
    check("foundation entry: alias, no path key",
          e.get("alias") == "medium" and "path" not in e)
    check("every registry entry names both a GPU and a CPU cluster env",
          all(m.get("env") and m.get("env_cpu")
              for m in maceenv.MODELS.values()))
    try:
        maceenv.resolve_model("no-such-model")
        check("unknown model name raises KeyError", False)
    except KeyError:
        check("unknown model name raises KeyError", True)
    try:
        maceenv.resolve_model("na1500-std", models_dir="/nonexistent")
        check("missing model file raises FileNotFoundError", False)
    except FileNotFoundError:
        check("missing model file raises FileNotFoundError", True)
    from ase import Atoms
    at = maceenv.prep_atoms(Atoms("Na"), "polar")
    at2 = maceenv.prep_atoms(Atoms("Na"), "standard")
    check("prep_atoms: polar info keys set, standard untouched",
          at.info.get("charge") == 0 and at.info.get("spin") == 1
          and "charge" not in at2.info)


def test_molecules():
    print("[8] molecules (guest templates vs frozen v0/FAU copies)")
    from zeolib import molecules
    check("guest set == constants.MULTIPLICITY keys",
          sorted(molecules.GUESTS) == sorted(constants.MULTIPLICITY)
          and sorted(molecules.MOL_RADIUS) == sorted(constants.MULTIPLICITY))
    elems, p = molecules.guest_positions("I2")
    check("I2 bond 2.670 Å", abs(np.linalg.norm(p[1] - p[0]) - 2.670) < 1e-9)
    _, p = molecules.guest_positions("HI")
    check("HI bond 1.609 Å", abs(np.linalg.norm(p[1] - p[0]) - 1.609) < 1e-9)
    e3, p3 = molecules.guest_positions("CH3I")
    check("CH3I: C-I 2.140 Å, elements C I H H H",
          abs(np.linalg.norm(p3[1] - p3[0]) - 2.140) < 1e-9
          and e3 == ["C", "I", "H", "H", "H"])
    at = molecules.guest_atoms("NO3")
    check("guest_atoms: ASE object, origin-centred template",
          len(at) == 4 and abs(at.get_positions()[0]).max() < 1e-12)
    check("guest_elements order (CH3I -> C, I, H)",
          molecules.guest_elements("CH3I") == ["C", "I", "H"])
    want_nat = {"I2": 2, "HI": 2, "CH3I": 5, "H2O": 3,
                "Cl2": 2, "NO2": 3, "NO3": 4}
    check("guest_natoms is per-ATOM, and != len(guest_elements) where it differs",
          all(molecules.guest_natoms(g) == n for g, n in want_nat.items())
          and all(molecules.guest_natoms(g) == len(molecules.guest_positions(g)[1])
                  for g in want_nat)
          and molecules.guest_natoms("NO3") != len(molecules.guest_elements("NO3")))
    # composition parity with the frozen FAU-era seed xyzs (overlapping names)
    bm = os.path.join(ZROOT, "FAU", "BindingEnergies", "BindingMolecules")
    if os.path.isdir(bm):
        ok, seen = True, 0
        for name in ("I2", "HI", "CH3I", "H2O", "Cl2", "NO2"):
            fp = os.path.join(bm, name + ".xyz")
            if not os.path.exists(fp):
                continue
            seen += 1
            n = int(open(fp).readline().split()[0])
            ok = ok and n == len(molecules.GUESTS[name])
        check("atom counts match frozen FAU BindingMolecules (%d compared)"
              % seen, ok and seen >= 4)
    else:
        print("  (FAU BindingMolecules missing — parity skipped)")


def test_placement():
    print("[9] placement (v0 void-grid port)")
    import re
    from zeolib import placement
    # toy determinism + clash floor on a 10 Å cube with one atom at origin
    cell = np.array([10.0, 10.0, 10.0])
    fw_pos = np.zeros((1, 3))
    grid = placement.build_void_grid(fw_pos, cell)
    pt, clr = placement.snap_to_void(np.array([5.0, 5.0, 5.0]), grid[0],
                                     grid[1], cell)
    check("toy snap: centre point returned with max clearance",
          pt is not None and np.allclose(pt, [5.0, 5.0, 5.0])
          and abs(clr - np.sqrt(75.0)) < 1e-9)
    rng = np.random.default_rng(0)
    res = placement.place_guest(np.array([5.0, 5.0, 5.0]), "I2", fw_pos, cell,
                                rng, grid=grid)
    check("toy place_guest ok: 2 I atoms, clash floor honored",
          res["status"] == "ok" and res["mol_pos"].shape == (2, 3)
          and placement.min_dist_mol_fw(res["mol_pos"], fw_pos, cell)
          >= placement.CLASH_DIST)
    dense = np.array([[x, y, z] for x in range(10) for y in range(10)
                      for z in range(10)], float)
    res2 = placement.place_guest(np.array([5.0, 5.0, 5.0]), "I2", dense, cell,
                                 rng)
    check("place_guest reports 'no void' in a dense framework",
          res2["status"] == "no void")
    # ── Foundations 2026-08-17: the pre-doomed-anchor defect and its fix ──
    # Two atoms straddle the target, leaving it 1.3 Å of clearance: accepted
    # by R_PROBE=1.2, but a guest with an atom AT its centroid (NO3) can then
    # never clear CLASH_DIST=1.6, however many orientations are tried.
    pocket = np.array([[5.0, 5.0, 3.7], [5.0, 5.0, 6.3]])
    pgrid = placement.build_void_grid(pocket, cell)
    tgt = np.array([5.0, 5.0, 5.0])
    r_v0 = placement.place_guest(tgt, "NO3", pocket, cell,
                                 np.random.default_rng(1), grid=pgrid)
    check("v0 defaults: centroid-occupied guest is pre-doomed at a 1.2-1.6 anchor",
          r_v0["status"] == "clash" and r_v0["clearance"] < placement.CLASH_DIST)
    r_fix = placement.place_guest(tgt, "NO3", pocket, cell,
                                  np.random.default_rng(1), grid=pgrid,
                                  r_probe=placement.CLASH_DIST, n_anchors=20)
    check("  r_probe=CLASH_DIST + n_anchors>1 recovers it",
          r_fix["status"] == "ok"
          and placement.min_dist_mol_fw(r_fix["mol_pos"], pocket, cell)
          >= placement.CLASH_DIST)
    r_pin = placement.place_guest(tgt, "NO3", pocket, cell,
                                  np.random.default_rng(1), grid=pgrid,
                                  r_probe=placement.CLASH_DIST, n_anchors=20,
                                  max_shift=0.4)
    check("  max_shift bounds the wander (sites stay distinct)",
          r_pin["status"] != "ok")
    a1 = placement.place_guest(tgt, "I2", pocket, cell,
                               np.random.default_rng(7), grid=pgrid)
    a2 = placement.place_guest(tgt, "I2", pocket, cell,
                               np.random.default_rng(7), grid=pgrid,
                               n_anchors=1, max_shift=None)
    check("  n_anchors=1/max_shift=None is byte-identical to the v0 path",
          a1["status"] == a2["status"]
          and np.allclose(a1["site"], a2["site"], atol=0))
    # ── reproduce a shipped v0 SITE_INFO (Ag_5 / I2 / site 01 = 12-ring) ──
    sdir = os.path.join(ZROOT, "MOR", "oldbinding", "Ag_5", "I2", "screen_01")
    fin = os.path.join(ZROOT, "MOR", "Al-MOR", "Ag_5", "fin-opt")
    if os.path.exists(os.path.join(sdir, "SITE_INFO")) and os.path.isdir(fin):
        info = open(os.path.join(sdir, "SITE_INFO")).read()
        want = [float(x) for x in
                re.search(r"position:\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)",
                          info).groups()]
        want_clr = float(re.search(r"clearance:\s+([\d.-]+)", info).group(1))
        lines = open(os.path.join(fin, "fin-opt-pos-1.xyz")).readlines()
        n = int(lines[0].split()[0])
        rows = [ln.split() for ln in lines[len(lines) - n:]]
        fwp = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in rows])
        m = re.search(r"ABC\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
                      open(os.path.join(fin, "fin-opt.inp")).read())
        cellv = np.array([float(g) for g in m.groups()])
        g2 = placement.build_void_grid(fwp, cellv)
        target = cellv / 2.0                       # 12-ring = cell centre
        pt2, clr2 = placement.snap_to_void(target, g2[0], g2[1], cellv)
        check("v0 SITE_INFO position reproduced (Ag_5/I2/12-ring)",
              pt2 is not None and np.allclose(pt2, want, atol=2e-4)
              and abs(clr2 - want_clr) < 2e-4,
              (pt2, want))
    else:
        print("  (v0 Ag_5 fixture missing — SITE_INFO reproduction skipped)")
    # ring_normal_sites geometry: the two anchors straddle the centroid
    from zeolib import framework as fwm
    fw = fwm.load_framework(symmetry=False)
    r5 = fwm.t_rings(fw, 5)
    pts = placement.ring_normal_sites(fw, r5[0], offset=2.0)
    from zeolib.geometry import unwrap
    cen = unwrap(fw["pos"][list(r5[0])], fw["cell"]).mean(axis=0)
    check("ring_normal_sites: ± anchors 4 Å apart, centred on the ring",
          pts.shape == (2, 3)
          and abs(np.linalg.norm(pts[0] - pts[1]) - 4.0) < 1e-9
          and np.allclose((pts[0] + pts[1]) / 2.0, cen, atol=1e-9))


def test_constants_combos():
    print("[10] constants per-framework combo map")
    from zeolib.cation import Q_CAT
    check("MOR alias unchanged: 16 combos", len(constants.COMBO_NAMES) == 16
          and constants.combo_names("MOR") == constants.COMBO_NAMES)
    check("FAU map: 17 combos", len(constants.combo_names("FAU")) == 17)
    mor_na = {"4.33": 9, "5": 8, "7": 6, "11": 4, "15": 3}
    fau_na = {"2": 16, "3": 12, "5": 8, "7": 6, "11": 4}
    check("n_al_for_ratio matches the historical Al counts",
          all(constants.n_al_for_ratio("MOR", r) == n for r, n in mor_na.items())
          and all(constants.n_al_for_ratio("FAU", r) == n
                  for r, n in fau_na.items()))
    bal = all(constants.n_al_for_ratio(f, r) % int(Q_CAT[c]) == 0
              for f, cm in constants.COMBOS_BY_FRAMEWORK.items()
              for c, rs in cm.items() for r in rs)
    check("every combo charge-balances (n_al mod q == 0)", bal)
    try:
        constants.n_al_for_ratio("MOR", "6")
        check("non-dividing ratio raises", False)
    except ValueError:
        check("non-dividing ratio raises", True)


def test_provenance():
    print("[11] provenance (zeolib version stamping)")
    from zeolib import provenance as prov

    info = prov.version_info()
    keys = {"zeolib_version", "vcs", "sha", "sha_full", "branch", "describe",
            "dirty", "source", "stamped_utc"}
    check("version_info: stable key set (manifest schema)",
          set(info) == keys, sorted(set(info) ^ keys))
    check("version_info: source is THIS zeolib", info["source"] == HERE, info["source"])

    in_repo = prov.is_repo()
    if in_repo:
        check("live checkout: vcs='git' and a 9-char sha",
              info["vcs"] == "git" and info["sha"] and len(info["sha"]) == 9,
              (info["vcs"], info["sha"]))
        check("sha is a prefix of sha_full",
              bool(info["sha_full"]) and info["sha_full"].startswith(info["sha"]),
              (info["sha"], info["sha_full"]))
        check("dirty is a real bool (never None) inside a repo",
              isinstance(info["dirty"], bool), info["dirty"])
    else:
        # A shipped copy has no .git. That must be VISIBLE, not faked clean.
        check("no repo: vcs='none' and sha is None",
              info["vcs"] == "none" and info["sha"] is None, info)
        check("no repo: dirty is None, never False ('no repo' != 'clean')",
              info["dirty"] is None, info["dirty"])

    line = prov.stamp_line(info)
    check("stamp_line one-liner names the version",
          line.startswith("zeolib ") and "\n" not in line, line)
    check("stamp_line marks a non-repo unmistakably",
          ("NO-GIT-REPO" in line) == (info["vcs"] != "git"), line)

    # A synthetic dirty/no-repo info must never render as a trustworthy sha
    faked = dict(info, vcs="none", sha=None, dirty=None)
    check("no-repo info cannot render a plausible sha",
          "NO-GIT-REPO" in prov.stamp_line(faked), prov.stamp_line(faked))
    dirty = dict(info, vcs="git", sha="deadbeef1", branch="main", dirty=True)
    check("dirty info renders DIRTY, not clean",
          "DIRTY" in prov.stamp_line(dirty), prov.stamp_line(dirty))

    # write_stamp / read_stamp round-trip, LF-clean (it travels to Linux)
    tmp = tempfile.mkdtemp()
    written = prov.write_stamp(tmp, extra={"package": "selftest-fixture"})
    stamp_path = os.path.join(tmp, prov.STAMP_NAME)
    check("write_stamp emits %s" % prov.STAMP_NAME, os.path.exists(stamp_path))
    check("stamp file is LF-only (ships to the cluster)",
          not fileio.has_crlf(stamp_path))
    back = prov.read_stamp(tmp)
    check("read_stamp round-trips write_stamp", back == written,
          (sorted(back.items()) != sorted(written.items())))
    check("extra context merged into the stamp",
          back.get("package") == "selftest-fixture", back.get("package"))

    # Missing stamp is FATAL, not an "unknown" dict (rule 7)
    empty = tempfile.mkdtemp()
    try:
        prov.read_stamp(empty)
        check("read_stamp on an unstamped dir raises", False, "returned instead")
    except FileNotFoundError:
        check("read_stamp on an unstamped dir raises", True)

    # require_clean is the loud path for results of record
    try:
        prov.require_clean(what="a selftest fixture")
        clean_ok = True
        err = ""
    except RuntimeError as exc:
        clean_ok = False
        err = str(exc)
    if in_repo and info["dirty"] is False:
        check("require_clean passes on a clean checkout", clean_ok, err)
    else:
        check("require_clean RAISES on a dirty tree / missing repo",
              not clean_ok, "did not raise")
        check("  ...and the message says what to do",
              ("Commit" in err or "not a git checkout" in err), err[:80])

    # A non-repo directory proves the no-git path even from a live checkout
    outside = tempfile.mkdtemp()
    check("is_repo False for a plain temp dir", not prov.is_repo(outside))
    check("git_sha None outside a repo", prov.git_sha(outside) is None)
    check("is_dirty None (not False) outside a repo",
          prov.is_dirty(outside) is None, prov.is_dirty(outside))


def main():
    print("zeolib selftest (root: %s)" % ZROOT)
    test_geometry()
    test_framework()
    test_enumeration()
    test_cation()
    test_cp2k_generation()
    test_cp2k_parsing()
    test_slurm()
    test_fileio()
    test_maceenv()
    test_molecules()
    test_placement()
    test_constants_combos()
    test_provenance()
    print()
    if _FAILS:
        print("FAILED: %d check(s): %s" % (len(_FAILS), "; ".join(_FAILS)))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
