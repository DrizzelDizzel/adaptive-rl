"""Fit one residual model per condition on its training run (docs/decisions.md #12, #14).

Validation: the last 20 % of the training run as one contiguous block (a random row split
would put a near-identical neighbour of every validation row into the training set).
The test runs are not touched here; they are eval_model.py's job.

Usage:
    pixi run python scripts/train_model.py        # writes results/models/<condition>.npz
"""

import argparse
from pathlib import Path

import optax
from flax import nnx

from adaptive_rl.data import drop_startup, load_run
from adaptive_rl.dynamics.fit import fit, save_model, training_data
from adaptive_rl.dynamics.residual import ResidualMLP

CONDITIONS = ("nominal", "heavy", "wind")
VAL_FRACTION = 0.2
LEARNING_RATE = 1e-2  # as LotF
MAX_STEPS = 500  # best step measured at 40-130 for lr 1e-2 (#14)
SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/data")
    parser.add_argument("--out", default="results/models")
    args = parser.parse_args()

    for condition in CONDITIONS:
        run = drop_startup(load_run(Path(args.data) / condition / "train.npz"))
        F, Y = training_data(run)
        n_train = int((1.0 - VAL_FRACTION) * len(F))
        print(f"{condition}: {n_train} train rows, {len(F) - n_train} val rows")

        model = ResidualMLP(rngs=nnx.Rngs(SEED))
        optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)
        result = fit(model, optimizer, (F[:n_train], Y[:n_train]), (F[n_train:], Y[n_train:]), max_steps=MAX_STEPS)

        meta = dict(
            condition=condition,
            data=str(Path(args.data) / condition / "train.npz"),
            n_train=n_train,
            n_val=len(F) - n_train,
            learning_rate=LEARNING_RATE,
            max_steps=MAX_STEPS,
            best_step=result["best_step"],
            seed=SEED,
            run_meta=run["meta"],
        )
        path = Path(args.out) / f"{condition}.npz"
        save_model(path, model, meta, result["history"])
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
