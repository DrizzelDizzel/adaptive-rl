"""Hybrid model: nominal physics plus learned residual acceleration.

    x_next = nominal.step(x, u, a_res = ResidualMLP(features(x, u)))

Only the translational part is corrected; rotation stays nominal (#6, #12).
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from adaptive_rl.dynamics import nominal
from adaptive_rl.dynamics.residual import ResidualMLP, features
from adaptive_rl.state import unpack_state


def step(
    model: ResidualMLP,
    x: jax.Array,
    u: jax.Array,
    mass: float,
    dt: float,
    grad_through_residual: bool = False,
) -> jax.Array:
    """One control step of the hybrid model. Same arguments as nominal.step plus the network.

    grad_through_residual (#11): which Jacobians d x_next / d(x, u) a gradient through this
    step sees. The forward value is the same in both cases.
      False (default, as LotF): a_res is a constant for differentiation, so the Jacobians are
        exactly those of nominal.step at the same (x, u) (a_res enters additively).
        jax.lax.stop_gradient returns its input unchanged and makes jax.grad treat it as a
        constant; same result as LotF's custom_jvp, without a custom derivative rule.
      True: the chain rule also runs through the network, d a_res / d(x, u).
    Training the residual (fit.py) calls the network directly and is not affected.
    """
    a_res = model(features(x, u, mass))
    if not grad_through_residual:
        a_res = jax.lax.stop_gradient(a_res)
    return nominal.step(x, u, mass, dt, a_res=a_res)


def rollout(step_fn: Callable[[jax.Array, jax.Array], jax.Array], x0: jax.Array, us: jax.Array) -> jax.Array:
    """Open-loop prediction: start at x0, apply the recorded commands us (K, 4).

    Returns xs (K, 18) with xs[j] = predicted state after j + 1 steps, i.e. xs[j] is compared
    with X_next[k0 + j] of the recorded run (k0 = index of x0).

    step_fn(x, u) -> x_next, e.g. `lambda x, u: nominal.step(x, u, mass, dt)` or the same with
    hybrid.step, so nominal and hybrid are rolled out by the same code.

    jax.lax.scan is the loop over k: the carry x is passed from step to step (the model's z^-1),
    and the per-step outputs are stacked. Unlike a Python for-loop it is traced once, so jit
    compiles one step instead of K copies, and grad can differentiate through all K steps.
    """

    def body(x, u):
        x_next = step_fn(x, u)
        return x_next, x_next  # (carry for the next step, output of this step)

    _, xs = jax.lax.scan(body, x0, us)
    return xs


def rollout_errors(
    step_fn: Callable[[jax.Array, jax.Array], jax.Array],
    X: jax.Array,
    U: jax.Array,
    X_next: jax.Array,
    horizon: int,
) -> dict[str, jax.Array]:
    """k-step prediction errors of a model on a recorded run, from every possible start.

    For each start k0 = 0 .. N - horizon: roll out from the measured X[k0] with the recorded
    commands U[k0 : k0 + horizon] and compare with X_next[k0 : k0 + horizon].
    Returns arrays of shape (n_starts, horizon); column j is the error after j + 1 steps:
        "p" position error |p_pred - p| [m], "v" velocity error [m/s],
        "R" attitude error, angle of the rotation R_pred^T R [deg].

    Overlapping windows are correlated, which does not bias their mean, it only means there are
    fewer independent windows than starts. jax.vmap runs all starts as one batch: one compiled
    rollout instead of a Python loop over ~1700 starts.
    """
    # Recorded runs arrive as numpy arrays; numpy cannot be indexed with a traced k0 under jit.
    X, U, X_next = jnp.asarray(X), jnp.asarray(U), jnp.asarray(X_next)
    starts = jnp.arange(X.shape[0] - horizon + 1)

    def errors_from(k0):
        us = jax.lax.dynamic_slice_in_dim(U, k0, horizon)  # U[k0 : k0 + horizon] for a traced k0
        xs_true = jax.lax.dynamic_slice_in_dim(X_next, k0, horizon)
        xs_pred = rollout(step_fn, X[k0], us)
        p, R, v, _ = jax.vmap(unpack_state)(xs_pred)
        p_true, R_true, v_true, _ = jax.vmap(unpack_state)(xs_true)
        # trace(R_pred^T R_true) = sum of elementwise products; = 1 + 2 cos(angle).
        cos_angle = (jnp.einsum("kij,kij->k", R, R_true) - 1.0) / 2.0
        return {
            "p": jnp.linalg.norm(p - p_true, axis=-1),
            "v": jnp.linalg.norm(v - v_true, axis=-1),
            "R": jnp.degrees(jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0))),  # clip: rounding > 1
        }

    return jax.jit(jax.vmap(errors_from))(starts)
