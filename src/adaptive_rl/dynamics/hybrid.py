"""Hybrid model: nominal physics plus learned residual acceleration.

    x_next = nominal.step(x, u, a_res = ResidualMLP(features(x, u)))

Only the translational part is corrected; rotation stays nominal (#6, #12).
"""

from collections.abc import Callable

import jax

from adaptive_rl.dynamics import nominal
from adaptive_rl.dynamics.residual import ResidualMLP, features


def step(model: ResidualMLP, x: jax.Array, u: jax.Array, mass: float, dt: float) -> jax.Array:
    """One control step of the hybrid model. Same arguments as nominal.step plus the network."""
    a_res = model(features(x, u, mass))
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
