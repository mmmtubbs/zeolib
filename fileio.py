"""
fileio.py — LF-safe file writing and coords.inc helpers.

EVERY file that ships to the cluster (.sh, .sbatch, .inp, .inc) MUST be written
through write_lf: generating them on the Windows PC with default newlines
produces CRLF, which breaks Pronghorn bash (`syntax error near $'{\\r'`) and
CP2K keyword parsing. (Mac writes LF, so the bug only bites from Windows.)
"""
import os


def write_lf(path, text, executable=False):
    """Write text with Unix LF line endings and UTF-8 encoding regardless of
    platform (Windows' cp1252 default both mangles multi-byte chars and
    crashes on e.g. '→' — bit the 2026-07-08 stage1a_v2 packaging run)."""
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write(text)
    if executable:
        os.chmod(path, 0o755)


def has_crlf(path):
    """True if the file contains any CRLF — use as a ship-time sanity check."""
    with open(path, "rb") as f:
        return b"\r\n" in f.read()


def write_coords_inc(path, atoms=None, symbols=None, positions=None):
    """
    Write a CP2K &COORD include file (LF). Pass either an ASE Atoms object or
    (symbols, positions). Format matches every existing setup script:
    '%-2s  %14.8f  %14.8f  %14.8f'.
    """
    if atoms is not None:
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
    lines = ["%-2s  %14.8f  %14.8f  %14.8f\n" % (s, p[0], p[1], p[2])
             for s, p in zip(symbols, positions)]
    write_lf(path, "".join(lines))


def write_csv_lf(path, rows, fieldnames=None):
    """
    Write a list of dicts as LF-only CSV.

    EXISTS BECAUSE `csv` DEFAULTS TO CRLF: the excel dialect's lineterminator
    is "\r\n", and it wins regardless of the file handle's newline= setting,
    so the natural-looking `open(p, "w", newline="\n")` + `csv.DictWriter`
    still produces CRLF. Found 2026-08-25 in the f3 Perlmutter migration, where
    a staged verify_reference.csv tripped a CRLF check; every Foundations
    result CSV written before that date has the same endings (harmless to
    csv.reader, which strips the \r, but it violates zeolib rule 6 and a shell
    tool reading the file gets a trailing \r glued to the last field).

    fieldnames defaults to the first row's keys, in insertion order.
    """
    import csv as _csv
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError("write_csv_lf: no rows and no fieldnames")
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, list(fieldnames), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_coords_inc(path):
    """Read a coords.inc -> (symbols list, positions list of (x,y,z) tuples)."""
    syms, pos = [], []
    for ln in open(path):
        p = ln.split()
        if len(p) >= 4:
            syms.append(p[0])
            pos.append((float(p[1]), float(p[2]), float(p[3])))
    return syms, pos


def read_xyz_frames(path, last_only=False):
    """
    Read a plain or extended XYZ file -> list of frames, each a dict
    {'symbols': [...], 'positions': [(x,y,z), ...], 'comment': str}.

    Handles the multi-frame CP2K trajectories (`<project>-pos-1.xyz`, one frame
    per optimiser step) as well as single-frame files. `last_only=True` returns
    just the final frame in a 1-element list — the converged geometry of a
    finished cell-opt/geo-opt, which is what any "compile the winners" step
    actually wants.

    No silent fallback (zeolib rule 7): a truncated trailing frame RAISES
    rather than returning a partial geometry, because a half-written
    trajectory is exactly what a walltime-killed job leaves behind.

    Provenance: Foundations 2026-08-17, communication/ structure compilation.
    """
    with open(path) as f:
        lines = f.readlines()
    frames, i, n_lines = [], 0, len(lines)
    while i < n_lines:
        if not lines[i].strip():
            i += 1
            continue
        try:
            nat = int(lines[i].split()[0])
        except (ValueError, IndexError):
            raise ValueError("%s: line %d is not an XYZ atom count: %r"
                             % (path, i + 1, lines[i][:60]))
        comment = lines[i + 1].rstrip("\n") if i + 1 < n_lines else ""
        body = lines[i + 2:i + 2 + nat]
        if len(body) < nat:
            raise ValueError("%s: truncated frame at line %d (wanted %d atoms, "
                             "got %d)" % (path, i + 1, nat, len(body)))
        syms, pos = [], []
        for ln in body:
            p = ln.split()
            syms.append(p[0])
            pos.append((float(p[1]), float(p[2]), float(p[3])))
        fr = {"symbols": syms, "positions": pos, "comment": comment,
              "raw_comment": comment}
        # Extended-XYZ round-trip: lift Lattice="..." and info="..." back out,
        # so write_extxyz(read_xyz_frames(f)) reproduces f. Without this a
        # re-emit silently DROPS the cell and re-wraps the whole header as
        # free text (Foundations 2026-08-17).
        if 'Lattice="' in comment:
            try:
                fr["lattice"] = tuple(
                    float(v) for v in
                    comment.split('Lattice="')[1].split('"')[0].split())
            except (ValueError, IndexError):
                raise ValueError("%s: malformed Lattice= in %r"
                                 % (path, comment[:80]))
        if 'info="' in comment:
            fr["comment"] = comment.split('info="')[1].rsplit('"', 1)[0]
        frames.append(fr)
        i += 2 + nat
    if not frames:
        raise ValueError("%s: no XYZ frames found" % path)
    return frames[-1:] if last_only else frames


def write_extxyz(path, frames, lattice=None):
    """
    Write frames to ONE multi-frame extended-XYZ file — the format OVITO, ASE
    and VMD read directly, so a compiled candidates/winners file opens as a
    frame-by-frame flipbook for review.

    `frames` are dicts with 'symbols' and 'positions' (as returned by
    read_xyz_frames), plus optional 'comment' (free text) and 'lattice'
    (row-major 3x3 as 9 numbers, Å). The `lattice` argument supplies a default
    for frames that carry none; a per-frame 'lattice' wins.

    Free text goes into a quoted info="..." key rather than bare into the
    comment line: raw CP2K trajectory headers ("i = 1, time = ..., E = ...")
    contain '=' and ',' and would otherwise break extended-XYZ key=value
    parsing. Written LF via write_lf.

    Provenance: Foundations 2026-08-17, communication/ structure compilation.
    """
    out = []
    for fr in frames:
        lat = fr.get("lattice", lattice)
        head = ""
        if lat is not None:
            head += 'Lattice="%s" ' % " ".join("%.6f" % v for v in lat)
        head += "Properties=species:S:1:pos:R:3"
        info = (fr.get("comment") or "").replace('"', "'").strip()
        if info:
            head += ' info="%s"' % info
        out.append("%d\n%s\n" % (len(fr["symbols"]), head))
        for s, p in zip(fr["symbols"], fr["positions"]):
            out.append("%-2s  %14.8f  %14.8f  %14.8f\n" % (s, p[0], p[1], p[2]))
    write_lf(path, "".join(out))
