"""Tests for policy.py and the first-order rollout gradient: what breaks silently (policy at
start == baseline, alignment of states and references in the rollout, online model updates
reaching the jitted gradient, sign of the adaptation)."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from adaptive_rl.controllers import CascadedController
from adaptive_rl.dynamics import nominal
from adaptive_rl.dynamics.residual import ResidualMLP
from adaptive_rl.pipelines.first_order.gradient import make_optimizer, rollout_loss, stage_cost, update
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


def test_rollout_loss_equals_explicit_nominal_rollout():
    """With an untrained residual (a_res = 0) the scanned hybrid rollout must give the same loss
    and gradient as a plain Python loop over nominal.step. Catches off-by-one between states and
    references (cost of x_{j+1} against ref_{j+1}) and a residual that is not zero at start."""
    policy, residual = trained_like_policy(), ResidualMLP(rngs=nnx.Rngs(0))
    refs = references(H + 1)

    def explicit_loss(policy):
        x, costs = start_state(), []
        for j in range(H):
            ref = (refs[0][j], refs[1][j], refs[2][j])
            ref_next = (refs[0][j + 1], refs[1][j + 1], refs[2][j + 1])
            out = policy(inputs(x, ref))
            x = nominal.step(x, act(policy, BASELINE, x, ref), MASS, DT)
            costs.append(stage_cost(x, ref_next, out))
        return jnp.mean(jnp.stack(costs))

    loss, grad = nnx.value_and_grad(rollout_loss)(policy, residual, BASELINE, start_state(), refs, DT)
    loss_ref, grad_ref = nnx.value_and_grad(explicit_loss)(policy)
    np.testing.assert_allclose(loss, loss_ref, rtol=1e-5)
    for a, b in zip(jax.tree.leaves(grad), jax.tree.leaves(grad_ref)):
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-7)


def test_jitted_update_sees_residual_changes_and_switch():
    """The JIT trap: after the residual's weights change (as an online model update does), the
    compiled update must use the new weights, not the ones of its first call. And the switch
    grad_through_residual must reach hybrid.step (different gradient once d a_res/d(x,u) != 0)."""
    policy, residual = trained_like_policy(), ResidualMLP(rngs=nnx.Rngs(0))
    optimizer = make_optimizer(policy, learning_rate=0.0, max_grad_norm=1e9)  # policy stays fixed
    x0, refs = start_state(), references(H + 1)

    loss_before, _ = update(policy, optimizer, residual, BASELINE, x0, refs, DT)
    residual.out.kernel[...] = 0.1 * jax.random.normal(jax.random.key(2), residual.out.kernel[...].shape)
    loss_after, grad_j1 = update(policy, optimizer, residual, BASELINE, x0, refs, DT)
    _, grad_j2 = update(policy, optimizer, residual, BASELINE, x0, refs, DT, grad_through_residual=True)

    assert not np.isclose(loss_before, loss_after)
    assert not np.isclose(grad_j1, grad_j2)


def test_update_raises_thrust_when_model_predicts_sinking():
    """Drone exactly on the reference, but the residual says it is heavier (a_res = -0.23 g in
    z, as +30 % mass): the rollout predicts sinking, so one update must make the correction push
    upwards (more thrust). Checks sign conventions end to end, the core mechanism (the model's
    prediction drives the policy) and that the zero output layer does not block learning."""
    policy, residual = PolicyMLP(rngs=nnx.Rngs(0)), ResidualMLP(rngs=nnx.Rngs(0))
    residual.out.bias[...] = jnp.array([0.0, 0.0, -0.23 * 9.81])
    refs = references(H + 1)
    x0 = jnp.concatenate([refs[0][0], jnp.eye(3).ravel(), refs[1][0], jnp.zeros(3)])
    optimizer = make_optimizer(policy, learning_rate=1e-1, max_grad_norm=10.0)
    _, grad_norm = update(policy, optimizer, residual, BASELINE, x0, refs, DT)

    ref0 = (refs[0][0], refs[1][0], refs[2][0])
    thrust_correction = (act(policy, BASELINE, x0, ref0) - BASELINE(x0, ref0))[3]
    assert grad_norm > 0
    assert thrust_correction > 0
