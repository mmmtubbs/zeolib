"""
maceenv.py — MACE environment bootstrap, model registry, calculator factory.

Supersedes MOR/mace_utils.py and mor_core.ensure_mace_env (whose ../mace_env
fallback went stale when the env moved to ~/mace_env_MOR_pipeline — audit
2026-07-01). On the cluster, envs (`mace`, `mace-polar`, `mace-polar-gpu`) are
conda-activated by the sbatch script instead; this module is mainly for
desktop-side runs. Reminder: the Intel Mac desktop has NO torch wheel — MACE
inference/training runs on Pronghorn; this bootstrap is for environments where
a mace env exists.

Model registry (2026-07-07): Zeolites/models/ is the canonical home for
fine-tuned .model files (moved out of MOR/tests/finetune_1500/ — living code
must not reach into frozen test dirs). Foundation models are registered by
their mace-native names and resolved from ~/.cache/mace by mace itself.
Scripts select any model by ONE registry name via get_calc(name, ...);
resolve_model() is import-safe without torch (selftest runs on the desktop).
"""
import os
import sys

_ZEOLITES_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_ENV_CANDIDATES = [
    os.environ.get("MACE_PYTHON", ""),                                # explicit override
    os.path.expanduser("~/mace_env_MOR_pipeline/bin/python"),         # current home (2026-07)
]

# ── Model registry ──────────────────────────────────────────────────────────
# name -> dict(kind, arch, file|alias, env, env_cpu, note)
#   kind    : "finetuned" (file in MODELS_DIR) | "foundation" (mace resolves alias)
#   arch    : "standard" (mace_mp / MACECalculator) | "polar" (mace_polar; needs
#             prep_atoms info keys and the mace-polar env)
#   env     : conda env expected on the cluster for GPU runs
#   env_cpu : conda env for CPU runs on the cluster (the GPU envs pin a
#             cuda-11.8 torch for the P100/driver-470 nodes; CPU jobs use the
#             plain envs validated in tests/mace_polar_smoketest). Both env
#             fields are documentation for sbatch generators, not enforced.
# Provenance per model: Zeolites/models/README.md.
MODELS_DIR = os.environ.get("ZEOLIB_MODELS_DIR",
                            os.path.join(_ZEOLITES_ROOT, "models"))
MODELS = {
    "na1500-std":   dict(kind="finetuned", arch="standard",
                         file="mace_na1500_std.model",
                         env="mace-gpu", env_cpu="mace",
                         note="mp-0 medium fine-tuned on train_1500.xyz (Na-MOR Si15)"),
    "na1500-polar": dict(kind="finetuned", arch="polar",
                         file="mace_na1500_polar.model",
                         env="mace-polar-gpu", env_cpu="mace-polar",
                         note="polar-1-m fine-tuned on train_1500.xyz (Na-MOR Si15)"),
    "mp0-medium":   dict(kind="foundation", arch="standard",
                         alias="medium", env="mace-gpu", env_cpu="mace",
                         note="untrained MACE-MP-0 medium foundation"),
    "polar-1-m":    dict(kind="foundation", arch="polar",
                         alias="polar-1-m", env="mace-polar-gpu",
                         env_cpu="mace-polar",
                         note="untrained MACE-POLAR-1 medium foundation"),
}


def resolve_model(name, models_dir=None):
    """
    Registry lookup WITHOUT importing torch/mace (safe on the torch-less desktop).
    Returns the entry dict plus 'name' and, for finetuned models, 'path'
    (raises if the file is missing so a bad deploy fails at setup, not mid-run).
    models_dir overrides MODELS_DIR (which itself honors $ZEOLIB_MODELS_DIR).
    """
    if name not in MODELS:
        raise KeyError("unknown model %r — registry has: %s"
                       % (name, ", ".join(sorted(MODELS))))
    entry = dict(MODELS[name], name=name)
    if entry["kind"] == "finetuned":
        path = os.path.join(models_dir or MODELS_DIR, entry["file"])
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "model %r not found at %s (set ZEOLIB_MODELS_DIR or pass "
                "models_dir; canonical home is Zeolites/models/)" % (name, path))
        entry["path"] = path
    return entry


def prep_atoms(at, arch):
    """Attach the info keys a polar-arch model requires (neutral closed shell,
    no field). Harmless no-op for arch='standard'. Returns `at` for chaining."""
    if arch == "polar":
        at.info["charge"] = 0
        at.info["spin"] = 1
        at.info["external_field"] = [0.0, 0.0, 0.0]
    return at


def get_calc(name, device="cpu", dtype="float32", models_dir=None):
    """
    ASE calculator for any registry model (fine-tuned or foundation, standard
    or polar) by name. Call ensure_mace_env() first. For arch='polar' models,
    pass each Atoms through prep_atoms(at, 'polar') before scoring.
    """
    entry = resolve_model(name, models_dir=models_dir)
    if entry["kind"] == "finetuned":
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=[entry["path"]], device=device,
                              default_dtype=dtype)
    if entry["arch"] == "polar":
        from mace.calculators import mace_polar
        return mace_polar(model=entry["alias"], device=device, default_dtype=dtype)
    from mace.calculators import mace_mp
    return mace_mp(model=entry["alias"], device=device, default_dtype=dtype,
                   dispersion=False)


def ensure_mace_env():
    """
    If mace is not importable, re-exec the current script with the first
    existing candidate env python. Call at the very top of a script, before
    heavy imports and before setting OMP/MKL env vars.
    """
    try:
        import mace  # noqa: F401
        return
    except ImportError:
        pass
    for py in _ENV_CANDIDATES:
        if py and os.path.exists(py):
            os.execv(py, [py] + sys.argv)
    sys.exit("ERROR: mace not importable and no mace env found "
             "(tried MACE_PYTHON, ~/mace_env_MOR_pipeline). ")


def get_mace_calc(model="medium", dtype="float32", device="cpu", dispersion=False):
    """MACE-MP ASE calculator (legacy signature — prefer get_calc(name)).
    Call ensure_mace_env() first."""
    from mace.calculators import mace_mp
    return mace_mp(model=model, dispersion=dispersion,
                   default_dtype=dtype, device=device)
