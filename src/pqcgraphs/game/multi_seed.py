"""Multi-seed experiment runner.

Runs an experiment function for each of N seeds, saves per-seed results
to a structured JSON with the aggregated mean/CI payload appended. Clears
JAX caches between seeds so the VRAM footprint stays bounded regardless
of sweep length.

Usage
-----
    from pqcgraphs.experiments import exp_d1_tfim_scaling
    from pqcgraphs.game.multi_seed import run_seeds

    run_seeds(
        exp_d1_tfim_scaling.run,
        seeds=(0, 1, 2, 3, 4),
        kwargs=dict(n_values=(4, 6, 8), n_iters=15),
        out_path="results/multi_seed_d1.json",
    )

The runner is deliberately sequential rather than multiprocessing: even a
single Nash run can hold ~7 GB of JAX XLA cache, so two concurrent runs
on an 8 GB GPU is not viable. Between-seed wall-clock is ~15 min per D1
n-value, ~2 min per F6 run.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable


def _clear_all_caches() -> None:
    """Best-effort cache purge — JAX global + project-level lru_caches + gc."""
    try:
        import jax
        jax.clear_caches()
    except Exception:  # noqa: BLE001
        pass

    # Project-level lru_caches keyed on structural identity.
    for modpath, attr in (
        ("pqcgraphs.gpu.qfim_effdim", "_cached_qfim_fn"),
        ("pqcgraphs.gpu.theta_optimizer", "_cached_energy_grad_fn"),
        ("pqcgraphs.objectives.performance", "_cached_energy_fn"),
    ):
        try:
            import importlib
            mod = importlib.import_module(modpath)
            getattr(mod, attr).cache_clear()
        except Exception:  # noqa: BLE001
            pass

    gc.collect()


def run_seeds(
    experiment_run: Callable[..., Dict[str, Any]],
    *,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    kwargs: Dict[str, Any] = None,
    out_path: str = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run `experiment_run(seed=..., **kwargs)` for every seed.

    Each seed's full output is kept verbatim under `per_seed[seed]`. The
    caller can post-process these into per-row medians / CIs with
    `scripts.plotting.stats`; we deliberately do NOT aggregate here so
    that the raw data stays intact for re-plotting with different
    statistical choices later.
    """
    kwargs = dict(kwargs or {})
    # Prevent each sub-run from overwriting the canonical single-seed
    # output at `kwargs.get("out_path")` — multi_seed writes its OWN file.
    kwargs.pop("out_path", None)

    results: Dict[str, Any] = {
        "name": f"multi_seed::{experiment_run.__module__}.{experiment_run.__name__}",
        "seeds": list(seeds),
        "kwargs": {k: v for k, v in kwargs.items() if _json_safe(v)},
        "per_seed": {},
        "timing": {},
    }

    overall_t0 = time.perf_counter()
    for seed in seeds:
        _clear_all_caches()
        if verbose:
            print(f"[multi_seed] seed = {seed}", flush=True)
        t0 = time.perf_counter()
        seed_kwargs = {**kwargs, "seed": int(seed)}
        # Force the underlying experiment to write to a unique temporary
        # path so multiple parallel invocations don't collide — but since
        # we're sequential, we just point at per-seed files to keep them
        # for forensics.
        tmp_out = Path(f"results/.multi_seed_tmp_{experiment_run.__name__}_seed{seed}.json")
        seed_kwargs["out_path"] = tmp_out
        r = experiment_run(**seed_kwargs)
        dt = time.perf_counter() - t0
        results["per_seed"][str(seed)] = r
        results["timing"][str(seed)] = dt
        if verbose:
            print(f"    ... done in {dt:.1f}s", flush=True)

    results["total_wall_s"] = time.perf_counter() - overall_t0

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, default=str))
    return results


def _json_safe(v) -> bool:
    """Check that `v` will json-serialize cleanly."""
    try:
        json.dumps(v, default=str)
        return True
    except TypeError:
        return False


__all__ = ["run_seeds"]
