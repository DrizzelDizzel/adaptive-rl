"""First-order policy gradient: BPTT through a rollout of the hybrid model (docs/decisions.md #17, #11).

In every control step, from the state x_0 just measured, the current policy flies H steps
inside the model, and the loss is the tracking cost of that imagined flight:

    x^_0     = x_0 (measured, independent of theta)
    u_j      = act(theta, x^_j, ref_j)
    x^_{j+1} = hybrid.step(residual, x^_j, u_j)
    loss     = mean_j cost(x^_{j+1}, ref_{j+1}, policy output_j)

The plant never needs an episode: only x_0 comes from the real flight. The model decides
where the drone is predicted to go, so a residual that has learned a disturbance makes the
gradient correct the drift before it shows up in the measurements.

Jacobians (#11): with grad_through_residual=False (default, as LotF) BPTT differentiates the
nominal model only; the residual still enters through the predicted states x^_j at which cost
and Jacobians are evaluated. True also differentiates through the network (option).
"""

from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from adaptive_rl.controllers import CascadedController
from adaptive_rl.dynamics import hybrid
from adaptive_rl.dynamics.residual import ResidualMLP
from adaptive_rl.policy import PolicyMLP, act, inputs
from adaptive_rl.state import unpack_state

# Stage cost weights as LotF (traj_tracking_state_env._get_reward), action term on the
# correction instead of u - u_hover (#20, P4). W_CORRECTION = 0.01, not LotF's 0.1: LotF
# penalises thrust in N, ours is in units of m g, so the same number would be ~6x stronger.
W_POS, W_VEL, W_CORRECTION = 1.0, 1.0, 0.01


def smooth_l1(e: jax.Array) -> jax.Array:
    """Huber function of the vector norm, delta = 1, as LotF's smooth_l1: 0.5 |e|^2 for |e| < 1,
    linear beyond. Our errors are well below 1, so in practice this is 0.5 |e|^2.
    The 1e-6 keeps the norm's derivative finite at e = 0 (as in LotF)."""
    norm = jnp.linalg.norm(e + 1e-6)
    quadratic = jnp.minimum(norm, 1.0)
    return 0.5 * quadratic**2 + (norm - quadratic)


def stage_cost(x_next: jax.Array, ref_next, policy_output: jax.Array) -> jax.Array:
    """Cost of one step: position and velocity error after the step [m, m/s], plus the size of
    the correction (network output, normalised units: rad/s and fractions of hover thrust)."""
    p, _, v, _ = unpack_state(x_next)
    p_ref, v_ref, _ = ref_next
    return W_POS * smooth_l1(p - p_ref) + W_VEL * smooth_l1(v - v_ref) + W_CORRECTION * smooth_l1(policy_output)


def rollout_loss(
    policy: PolicyMLP,
    residual: ResidualMLP,
    baseline: CascadedController,
    x0: jax.Array,
    refs,
    dt: float,
    grad_through_residual: bool = False,
) -> jax.Array:
    """Mean stage cost of an H-step rollout in the hybrid model. x0: (18,) measured state;
    refs: (P_ref, V_ref, A_ref), each (H+1, 3), the reference at the time of x^_0 .. x^_H."""
    P_ref, V_ref, A_ref = refs

    def body(x, step_data):
        p_ref, v_ref, a_ref, p_ref_next, v_ref_next, a_ref_next = step_data
        ref, ref_next = (p_ref, v_ref, a_ref), (p_ref_next, v_ref_next, a_ref_next)

        u = act(policy, baseline, x, ref)
        x_next = hybrid.step(residual, x, u, baseline.mass, dt, grad_through_residual)
        cost = stage_cost(x_next, ref_next, policy(inputs(x, ref)))
        return x_next, cost

    # lax.scan over j = 0 .. H-1 (as hybrid.rollout): carry x^_j, output cost_j.
    step_data = (P_ref[:-1], V_ref[:-1], A_ref[:-1], P_ref[1:], V_ref[1:], A_ref[1:])
    _, costs = jax.lax.scan(body, x0, step_data)
    return jnp.mean(costs)


def make_optimizer(policy: PolicyMLP, learning_rate: float, max_grad_norm: float) -> nnx.Optimizer:
    """Plain SGD with a fixed step size (P5): no Adam, whose per-parameter normalisation would
    hide the difference between a biased (FO) and a noisy (ZO) gradient. Gradient clipping by
    global norm as a safety net against single exploding BPTT gradients."""
    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.sgd(learning_rate))
    return nnx.Optimizer(policy, tx, wrt=nnx.Param)


@partial(nnx.jit, static_argnames=("baseline", "dt", "grad_through_residual"))
def update(
    policy: PolicyMLP,
    optimizer: nnx.Optimizer,
    residual: ResidualMLP,
    baseline: CascadedController,
    x0: jax.Array,
    refs,
    dt: float,
    grad_through_residual: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """One gradient step on the rollout loss, in place. Returns (loss, gradient norm before
    clipping). The gradient is taken w.r.t. the policy only (first argument).

    residual is a traced argument on purpose: nnx.jit passes its weights in as arrays on every
    call, so online model updates are seen. Captured in a closure instead, jit would bake the
    weights of the first call in as constants and silently ignore later updates.
    baseline, dt and the switch are static: fixed for a run, compiled in as constants."""
    loss, grads = nnx.value_and_grad(rollout_loss)(policy, residual, baseline, x0, refs, dt, grad_through_residual)
    optimizer.update(policy, grads)
    return loss, optax.tree.norm(grads)
