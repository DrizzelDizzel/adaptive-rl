"""First-order policy gradient along the measured trajectory (T2, J1; docs/decisions.md #11, #20).

Window = the last H control steps that really happened: measured states x_0 .. x_H and the
references at those times. The loss is the tracking cost of that window as a function of the
policy weights theta:

    x~_0 = x_0 (measured, independent of theta)
    u_j  = act(theta, x~_j, ref_j)
    x~_{j+1} = x_{j+1, measured} + f(x~_j, u_j) - stop_gradient(f(x~_j, u_j))
    loss = mean_j cost(x~_{j+1}, ref_{j+1}, policy output_j)

The straight-through line: its VALUE is the measured state (f - stop_gradient(f) = 0), its
DERIVATIVE is that of the model f (stop_gradient has none). So the cost is evaluated on real
states, while BPTT propagates d x / d theta through the model's Jacobians A = df/dx and
B = df/du, including the closed-loop feedback through the baseline in act(). f is the nominal
model (J1, as LotF); J2 would use hybrid.step(..., grad_through_residual=True) instead.

The window was flown with slightly older weights (up to H updates ago); the loss re-evaluates
the actions with the current ones. With small steps the difference is assumed negligible.
"""

from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from adaptive_rl.controllers import CascadedController
from adaptive_rl.dynamics import nominal
from adaptive_rl.policy import PolicyMLP, act, inputs
from adaptive_rl.state import unpack_state

# Stage cost weights as LotF (traj_tracking_state_env._get_reward), action term on the
# correction instead of u - u_hover (#20, P4).
W_POS, W_VEL, W_CORRECTION = 1.0, 1.0, 0.1


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


def window_loss(policy: PolicyMLP, baseline: CascadedController, X: jax.Array, refs, dt: float) -> jax.Array:
    """Mean stage cost over the window. X: (H+1, 18) measured states; refs: (P_ref, V_ref,
    A_ref), each (H+1, 3), the reference at the time of each state."""
    P_ref, V_ref, A_ref = refs

    def body(x_tilde, step_data):
        x_next_measured, p_ref, v_ref, a_ref, p_ref_next, v_ref_next, a_ref_next = step_data
        ref, ref_next = (p_ref, v_ref, a_ref), (p_ref_next, v_ref_next, a_ref_next)

        u = act(policy, baseline, x_tilde, ref)
        x_model = nominal.step(x_tilde, u, baseline.mass, dt)
        x_next = x_next_measured + (x_model - jax.lax.stop_gradient(x_model))  # straight-through

        cost = stage_cost(x_next, ref_next, policy(inputs(x_tilde, ref)))
        return x_next, cost

    # lax.scan over j = 0 .. H-1 (as hybrid.rollout): carry x~_j, output cost_j.
    step_data = (X[1:], P_ref[:-1], V_ref[:-1], A_ref[:-1], P_ref[1:], V_ref[1:], A_ref[1:])
    _, costs = jax.lax.scan(body, X[0], step_data)
    return jnp.mean(costs)


def make_optimizer(policy: PolicyMLP, learning_rate: float, max_grad_norm: float) -> nnx.Optimizer:
    """Plain SGD with a fixed step size (P5): no Adam, whose per-parameter normalisation would
    hide the difference between a biased (FO) and a noisy (ZO) gradient. Gradient clipping by
    global norm as a safety net against single exploding BPTT gradients."""
    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.sgd(learning_rate))
    return nnx.Optimizer(policy, tx, wrt=nnx.Param)


@partial(nnx.jit, static_argnames=("baseline", "dt"))
def update(
    policy: PolicyMLP, optimizer: nnx.Optimizer, baseline: CascadedController, X: jax.Array, refs, dt: float
) -> tuple[jax.Array, jax.Array]:
    """One gradient step on the window loss, in place. Returns (loss, gradient norm before
    clipping). baseline and dt are static: fixed for a run, compiled in as constants."""
    loss, grads = nnx.value_and_grad(window_loss)(policy, baseline, X, refs, dt)
    optimizer.update(policy, grads)
    return loss, optax.tree.norm(grads)
