"""Tests for fit.py: what breaks silently (save/load format, keeping the best weights rather
than the last ones). Whether the fit is good is eval_model.py's job."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from adaptive_rl.dynamics.fit import fit, load_history, load_model, mse, save_model
from adaptive_rl.dynamics.residual import N_FEATURES, ResidualMLP


def test_save_load_roundtrip(tmp_path):
    """Loaded model gives the same predictions; meta and history survive."""
    model = ResidualMLP(width=32, rngs=nnx.Rngs(0))
    # Non-zero output layer, otherwise both models would predict 0 and the test proves nothing.
    model.out.kernel[...] = jax.random.normal(jax.random.key(1), model.out.kernel[...].shape)
    history = {"step": np.arange(3), "val_mse": np.array([3.0, 2.0, 1.0])}
    save_model(tmp_path / "m.npz", model, {"condition": "heavy"}, history)

    loaded, meta = load_model(tmp_path / "m.npz")
    F = jax.random.normal(jax.random.key(2), (5, N_FEATURES))
    np.testing.assert_array_equal(loaded(F), model(F))
    assert meta["condition"] == "heavy" and meta["width"] == 32 and meta["activation"] == "relu"
    np.testing.assert_array_equal(load_history(tmp_path / "m.npz")["val_mse"], history["val_mse"])


def test_fit_leaves_model_at_best_validation_step():
    """After fit, the model's val MSE must be the minimum of the history, not the last value.
    Train and val disagree on purpose, so the val loss rises again while training goes on."""
    F_train = jax.random.normal(jax.random.key(0), (64, N_FEATURES))
    F_val = jax.random.normal(jax.random.key(1), (64, N_FEATURES))
    Y_train, Y_val = jnp.ones((64, 3)), -jnp.ones((64, 3))  # learning train hurts val

    model = ResidualMLP(width=16, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
    result = fit(model, optimizer, (F_train, Y_train), (F_val, Y_val), max_steps=30, eval_every=10)

    val = result["history"]["val_mse"]
    assert val[-1] > val.min()  # the setup really overfits, otherwise the test proves nothing
    np.testing.assert_allclose(float(mse(model, F_val, Y_val)), val.min(), rtol=1e-6)
