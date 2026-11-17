"""Recorded runs: saving, loading, preparing for training.

A run is a dict of arrays, all with the same first dimension (one row per control step k):
    X       (N, 18)  state before the step
    U       (N, 4)   command actually sent (controller output + E)
    X_next  (N, 18)  state after the step
    E       (N, 4)   excitation noise contained in U
    T       (N,)     time at the start of the step [s]
plus `meta`, a dict describing the run (reference, disturbance, seeds).

Train/test split is by run, never by shuffling rows: neighbouring rows are almost identical,
so a random split would test on the training data.
"""

import json
from pathlib import Path

import numpy as np

ARRAY_KEYS = ("X", "U", "X_next", "E", "T")


def save_run(path: str | Path, run: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: run[key] for key in ARRAY_KEYS}
    np.savez(path, **arrays, meta=json.dumps(run["meta"]))


def load_run(path: str | Path) -> dict:
    with np.load(path) as f:
        run = {key: f[key] for key in ARRAY_KEYS}
        run["meta"] = json.loads(str(f["meta"]))
    return run


def drop_startup(run: dict, t_min: float = 2.0) -> dict:
    """Remove the first t_min seconds (rotor spin-up, see docs/decisions.md #4)."""
    keep = run["T"] >= t_min
    out = {key: run[key][keep] for key in ARRAY_KEYS}
    out["meta"] = run["meta"]
    return out


def stack_runs(runs: list[dict]) -> dict:
    """Concatenate runs into one set of one-step tuples (for training on (x, u) -> x_next).

    The result is not a continuous trajectory anymore: use single runs for k-step rollouts.
    """
    out = {key: np.concatenate([r[key] for r in runs]) for key in ARRAY_KEYS}
    out["meta"] = [r["meta"] for r in runs]
    return out
