"""Fitting the residual model: training data, loss, one update step, offline fit, save/load.

    run (X, U, X_next) --training_data--> (F, Y) = (features, target a_res)
    loss = MSE(model(F), Y) + LAMBDA_REG * sum_layers ||W||_2
    train_step: one Adam update on (F, Y)              <- reused later for online learning
    fit:        train_step in a loop, keeps the weights with the best validation loss

Settings and deviations from LotF: docs/decisions.md #14.

Optax in two sentences: an optimizer is a pure function (grads, opt_state) -> (updates,
new opt_state); Adam's opt_state holds the running mean and variance of the gradients.
nnx.Optimizer wraps that state next to the model, so optimizer.update(model, grads) applies
one Adam step to the model's Param arrays in place.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from adaptive_rl.dynamics.residual import ResidualMLP, features, target

LAMBDA_REG = 1e-3  # weight of the spectral-norm penalty, as LotF


def training_data(run: dict) -> tuple[jax.Array, jax.Array]:
    """Network inputs F (N, 16) and targets Y (N, 3) for every row of a recorded run.

    Targets are recomputed from (X, U, X_next) on every call instead of stored with the run:
    they depend on the nominal model, and a stored target would silently go stale when
    nominal.step changes. jax.vmap turns the single-sample functions into batched ones.
    """
    mass = run["meta"]["nominal_mass"]  # the model only ever knows the nominal mass (#8)
    dt = 1.0 / run["meta"]["control_freq"]
    X, U, X_next = (jnp.asarray(run[key]) for key in ("X", "U", "X_next"))
    F = jax.vmap(features, in_axes=(0, 0, None))(X, U, mass)
    Y = jax.vmap(target, in_axes=(0, 0, 0, None, None))(X, U, X_next, mass, dt)
    return F, Y


def mse(model: ResidualMLP, F: jax.Array, Y: jax.Array) -> jax.Array:
    """Mean squared error of a_res [(m/s^2)^2], averaged over samples and the 3 axes."""
    return jnp.mean((model(F) - Y) ** 2)


def spectral_penalty(model: ResidualMLP) -> jax.Array:
    """Sum over all layers of the largest singular value of the weight matrix (as LotF).

    Bounds how steeply the net can change with its input (its Lipschitz constant), so the
    learned a_res stays smooth between and beyond the training samples.
    """
    layers = (model.layer1, model.layer2, model.out)
    return sum(jnp.linalg.norm(layer.kernel[...], ord=2) for layer in layers)


def loss(model: ResidualMLP, F: jax.Array, Y: jax.Array) -> jax.Array:
    return mse(model, F, Y) + LAMBDA_REG * spectral_penalty(model)


@nnx.jit
def train_step(model: ResidualMLP, optimizer: nnx.Optimizer, F: jax.Array, Y: jax.Array) -> jax.Array:
    """One Adam step on the loss over (F, Y), in place. Returns the loss before the step.

    nnx.jit compiles the whole step (forward, backward, Adam) once per input shape; later
    calls with arrays of the same shape reuse the compiled code.
    nnx.value_and_grad differentiates w.r.t. the model's Param arrays only.
    """
    value, grads = nnx.value_and_grad(loss)(model, F, Y)
    optimizer.update(model, grads)
    return value


def fit(
    model: ResidualMLP,
    optimizer: nnx.Optimizer,
    train: tuple[jax.Array, jax.Array],
    val: tuple[jax.Array, jax.Array],
    max_steps: int = 500,
    eval_every: int = 10,
    print_every: int = 50,
) -> dict:
    """Full-batch training with early stopping by selection: train max_steps, evaluate every
    eval_every steps, and leave the model at the weights with the lowest validation MSE.

    Full batch: every step sees all training rows, so one step = one epoch and the result is
    deterministic for a given seed.
    Returns the history (arrays over evaluation points, step 0 = untrained, i.e. a_res = 0)
    and the chosen step.

    A Python loop around the jitted step, so evaluation, printing and keeping the best weights
    are plain Python. Faster alternative: the whole loop in jax.lax.scan (compiled once, as
    LotF), at the price of doing those three inside the compiled function (#15).
    """
    F_train, Y_train = train
    F_val, Y_val = val
    eval_mse = nnx.jit(mse)

    history = {"step": [], "train_mse": [], "val_mse": [], "penalty": []}
    best_val, best_step, best_model = np.inf, 0, nnx.clone(model)

    for step in range(max_steps + 1):
        if step > 0:
            train_step(model, optimizer, F_train, Y_train)
        if step % eval_every != 0:
            continue

        val_mse = float(eval_mse(model, F_val, Y_val))
        history["step"].append(step)
        history["train_mse"].append(float(eval_mse(model, F_train, Y_train)))
        history["val_mse"].append(val_mse)
        history["penalty"].append(float(spectral_penalty(model)))
        if val_mse < best_val:
            # nnx.clone copies the weights; keeping a plain reference would follow the model.
            best_val, best_step, best_model = val_mse, step, nnx.clone(model)

        if step % print_every == 0:
            print(
                f"  step {step:4d} | train {history['train_mse'][-1]:.4f} | val {val_mse:.4f} | "
                f"penalty {history['penalty'][-1]:.2f}"
            )

    nnx.update(model, nnx.state(best_model, nnx.Param))
    print(f"  best step {best_step} | val {best_val:.4f}")
    return {"history": {key: np.asarray(v) for key, v in history.items()}, "best_step": best_step}


def save_model(path: str | Path, model: ResidualMLP, meta: dict, history: dict | None = None) -> None:
    """Weights as flat arrays ("layer1/kernel", ...), meta as JSON, optional history arrays.

    meta is extended by what load_model needs to rebuild the network (width, activation).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta, width=model.layer1.out_features, activation=model.activation.__name__)
    weights = {
        "/".join(key): np.asarray(param[...]) for key, param in nnx.to_flat_state(nnx.state(model, nnx.Param))
    }
    hist = {f"history/{key}": value for key, value in (history or {}).items()}
    np.savez(path, **weights, **hist, meta=json.dumps(meta))


def load_model(path: str | Path) -> tuple[ResidualMLP, dict]:
    """Rebuild a ResidualMLP from save_model's file. Returns (model, meta).

    The network is created with fresh random weights first (nnx needs a concrete object),
    then every Param is overwritten with the stored array.
    """
    with np.load(path) as f:
        meta = json.loads(str(f["meta"]))
        activation = getattr(nnx, meta["activation"])  # e.g. "relu" -> nnx.relu
        model = ResidualMLP(width=meta["width"], activation=activation, rngs=nnx.Rngs(0))
        # nnx.state returns the model's own Param objects, so writing into them updates the model.
        for key, param in nnx.to_flat_state(nnx.state(model, nnx.Param)):
            param[...] = jnp.asarray(f["/".join(key)])
    return model, meta


def load_history(path: str | Path) -> dict:
    """The training history stored by save_model (empty dict if none)."""
    with np.load(path) as f:
        return {key.removeprefix("history/"): f[key] for key in f.files if key.startswith("history/")}
