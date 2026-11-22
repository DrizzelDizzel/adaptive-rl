"""Tests for policy.py and the first-order window gradient: what breaks silently (policy at
start == baseline, alignment of states and references in the window, straight-through T2)."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from adaptive_rl.controllers import CascadedController
from adaptive_rl.dynamics import nominal
from adaptive_rl.pipelines.first_order.gradient import make_optimizer, stage_cost, update, window_loss
from adaptive_rl.policy import PolicyMLP, act, inputs
from adaptive_rl.reference import Circle

MASS, DT, H = 0.0434, 0.01, 8
BASELINE = CascadedController(MASS)
REF = Circle(center=(0.0, 0.0, 1.5), radius=0.5, period=4.0)


def references(n: int, t0: float = 0.3):
    P, V, A = zip(*[REF(t0 + k * DT) for k in range(n)])
    return jnp.array(P), jnp.array(V), jnp.array(A)


def start_state():
    p, v, _ = REF(0.3)
    return jnp.concatenate([jnp.asarray(p) + 0.05, jnp.eye(3).ravel(), jnp.asarray(v), jnp.zeros(3)])


def trained_like_policy():
    """Non-zero output layer, otherwise the correction is 0 and several checks prove nothing."""
    policy = PolicyMLP(rngs=nnx.Rngs(0))
    policy.out.kernel[...] = 0.1 * jax.random.normal(jax.random.key(1), policy.out.kernel[...].shape)
    return policy


def test_untrained_policy_is_the_baseline():
    policy = PolicyMLP(rngs=nnx.Rngs(0))
    x, refs = start_state(), references(1)
    ref = (refs[0][0], refs[1][0], refs[2][0])
    np.testing.assert_array_equal(act(policy, BASELINE, x, ref), BASELINE(x, ref))


def test_window_on_model_data_equals_model_rollout():
    """If the 'measured' states come from the model itself, the T2 window (measured values,
    model derivatives) must give the same loss and gradient as differentiating a pure model
    rollout (T1). Catches off-by-one between states and references and a wrong straight-through."""
    policy = trained_like_policy()
    refs = references(H + 1)

    def model_rollout_loss(policy):
        x, costs = start_state(), []
        for j in range(H):
            ref = (refs[0][j], refs[1][j], refs[2][j])
            ref_next = (refs[0][j + 1], refs[1][j + 1], refs[2][j + 1])
            out = policy(inputs(x, ref))
            x = nominal.step(x, act(policy, BASELINE, x, ref), MASS, DT)
            costs.append(stage_cost(x, ref_next, out))
        return jnp.mean(jnp.stack(costs))

    # "Measured" states = the model's own closed-loop trajectory under this policy.
    X = [start_state()]
    for j in range(H):
        X.append(nominal.step(X[-1], act(policy, BASELINE, X[-1], (refs[0][j], refs[1][j], refs[2][j])), MASS, DT))
    X = jnp.stack(X)

    loss_t2, grad_t2 = nnx.value_and_grad(window_loss)(policy, BASELINE, X, refs, DT)
    loss_t1, grad_t1 = nnx.value_and_grad(model_rollout_loss)(policy)
    np.testing.assert_allclose(loss_t2, loss_t1, rtol=1e-5)
    for a, b in zip(jax.tree.leaves(grad_t2), jax.tree.leaves(grad_t1)):
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-7)


def test_update_from_zero_init_raises_thrust_when_below_reference():
    """Drone 5 cm below the reference over the whole window (as with a heavier drone): one
    update must make the correction push upwards (more thrust). Checks sign conventions end to
    end and that the zero output layer does not block learning.
    (The window loss VALUE cannot drop: under T2 it is computed on the measured states.)"""
    policy = PolicyMLP(rngs=nnx.Rngs(0))
    refs = references(H + 1)
    below = jnp.array([0.0, 0.0, 0.05])
    X = jnp.stack(
        [jnp.concatenate([refs[0][j] - below, jnp.eye(3).ravel(), refs[1][j], jnp.zeros(3)]) for j in range(H + 1)]
    )
    optimizer = make_optimizer(policy, learning_rate=1e-1, max_grad_norm=10.0)
    _, grad_norm = update(policy, optimizer, BASELINE, X, refs, DT)

    x0, ref0 = X[0], (refs[0][0], refs[1][0], refs[2][0])
    thrust_correction = (act(policy, BASELINE, x0, ref0) - BASELINE(x0, ref0))[3]
    assert grad_norm > 0
    assert thrust_correction > 0
