"""Open-loop k-step prediction of nominal vs hybrid on the test runs (docs/decisions.md #16).

Each model is rolled out from every measured state of a test run with the recorded commands,
for up to HORIZON steps, and compared with the measured states. k = 10-30 (0.1-0.3 s) is the
range BPTT will differentiate through later.

Attitude is a check, not a result: the residual only corrects translation (#12), so nominal
and hybrid must predict the same attitude.

Cross test: each residual model also on the other conditions' data. Shows what an offline
residual does when the situation changes (motivation for online adaptation).

Usage:
    pixi run python scripts/eval_model.py      # table on the console, results/figures/eval_model.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adaptive_rl.data import drop_startup, load_run
from adaptive_rl.dynamics import hybrid, nominal
from adaptive_rl.dynamics.fit import load_model

CONDITIONS = ("nominal", "heavy", "wind")
TEST_RUNS = ("test_circle", "test_hover")
HORIZON = 50
TABLE_K = (1, 10, 30)
CROSS_K = 30


def errors(model, run: dict) -> dict:
    """k-step errors (k = 1..HORIZON) of the nominal (model=None) or hybrid model on one run."""
    mass = run["meta"]["nominal_mass"]
    dt = 1.0 / run["meta"]["control_freq"]
    if model is None:
        step_fn = lambda x, u: nominal.step(x, u, mass, dt)  # noqa: E731
    else:
        step_fn = lambda x, u: hybrid.step(model, x, u, mass, dt)  # noqa: E731
    err = hybrid.rollout_errors(step_fn, run["X"], run["U"], run["X_next"], HORIZON)
    return {key: np.asarray(value) for key, value in err.items()}


def print_table(condition: str, run_name: str, err_nom: dict, err_hyb: dict) -> None:
    print(f"\n{condition} / {run_name}  (mean over starts, [p90]; ratio = hybrid / nominal)")
    print("   k |  position nominal -> hybrid [cm]     ratio |  velocity nominal -> hybrid [cm/s]  ratio | attitude [deg]")
    for k in TABLE_K:
        j = k - 1  # column j = error after j + 1 steps
        cells = []
        for key in ("p", "v"):
            n, h = 100 * err_nom[key][:, j], 100 * err_hyb[key][:, j]
            cells.append(
                f"{n.mean():6.2f} [{np.percentile(n, 90):6.2f}] -> {h.mean():6.2f} [{np.percentile(h, 90):6.2f}]"
                f"  {h.mean() / n.mean():5.2f}"
            )
        attitude = f"{err_nom['R'][:, j].mean():.2f} -> {err_hyb['R'][:, j].mean():.2f}"
        print(f"  {k:2d} | {cells[0]} | {cells[1]} | {attitude}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/data")
    parser.add_argument("--models", default="results/models")
    parser.add_argument("--out", default="results/figures/eval_model.png")
    args = parser.parse_args()

    models = {c: load_model(Path(args.models) / f"{c}.npz")[0] for c in CONDITIONS}
    runs = {
        (c, r): drop_startup(load_run(Path(args.data) / c / f"{r}.npz")) for c in CONDITIONS for r in TEST_RUNS
    }

    # Each model on its own condition: table + figure.
    fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(13, 7), sharex=True)
    k_axis = np.arange(1, HORIZON + 1)
    for col, condition in enumerate(CONDITIONS):
        for run_name in TEST_RUNS:
            err_nom = errors(None, runs[condition, run_name])
            err_hyb = errors(models[condition], runs[condition, run_name])
            assert np.allclose(err_nom["R"], err_hyb["R"]), "residual must not change the attitude (#12)"
            print_table(condition, run_name, err_nom, err_hyb)

            style = "-" if run_name == "test_circle" else "--"
            for row, (key, unit) in enumerate((("p", "position error [cm]"), ("v", "velocity error [cm/s]"))):
                ax = axes[row, col]
                ax.plot(k_axis, 100 * err_nom[key].mean(axis=0), style, color="tab:gray", label=f"nominal, {run_name}")
                ax.plot(k_axis, 100 * err_hyb[key].mean(axis=0), style, color="tab:blue", label=f"hybrid, {run_name}")
                ax.set_ylabel(unit)
        axes[0, col].set_title(condition)
        axes[1, col].set_xlabel("prediction steps k (10 ms each)")
    for ax in axes.flat:
        ax.axvspan(10, 30, color="0.92", zorder=0)  # range relevant for BPTT
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Open-loop k-step prediction error, mean over all start points")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\n-> {out}")

    # Cross test: rows = residual trained on, columns = data condition (test_circle).
    print(f"\nCross test: position error at k = {CROSS_K} [cm], test_circle, rows = residual trained on")
    print("               " + "".join(f"{c:>10s}" for c in CONDITIONS))
    for trained_on in (None, *CONDITIONS):
        label = "none (nominal)" if trained_on is None else trained_on
        cells = []
        for data_condition in CONDITIONS:
            model = None if trained_on is None else models[trained_on]
            # Same horizon (= same start windows) as the tables, so the numbers match exactly.
            err = errors(model, runs[data_condition, "test_circle"])
            cells.append(f"{100 * err['p'][:, CROSS_K - 1].mean():10.2f}")
        print(f"{label:15s}" + "".join(cells))


if __name__ == "__main__":
    main()
