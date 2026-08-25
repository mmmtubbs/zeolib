"""
zeolib — the shared zeolite-pipeline library for the Zeolites project.

One canonical home for everything that used to be re-implemented per job/test:
framework loading (T-site orbits, Löwenstein graph), minimum-image geometry,
CP2K input generation + output parsing, SLURM/sbatch/submit-script generation,
LF-safe file writing, unit conversions, and the MACE environment bootstrap.

Import pattern (from a script anywhere under Zeolites/, adjust the ".."s):

    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..")))  # -> Zeolites/
    from zeolib import cp2k, framework, geometry, slurm, fileio, constants

Rules of the road (see zeolib/README.md and Zeolites/CLAUDE.md):
  * New scripts import from here instead of copying helpers.
  * Anything generated for the cluster is written with fileio.write_lf (CRLF kills bash/CP2K).
  * Behaviour changes require `python zeolib/selftest.py` to pass.
  * Completed tests / archives are provenance — never retrofitted to zeolib.

Submodules are loaded lazily so that light modules (constants, cp2k, slurm,
fileio) work in environments without ase/spglib/scipy.
"""

__version__ = "0.1.0"

_SUBMODULES = ("constants", "geometry", "framework", "cp2k", "slurm",
               "fileio", "maceenv")


def __getattr__(name):
    if name in _SUBMODULES:
        import importlib
        mod = importlib.import_module("." + name, __name__)
        globals()[name] = mod
        return mod
    raise AttributeError("module 'zeolib' has no attribute %r" % name)
