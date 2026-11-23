"""Residual policy: the baseline controller plus a learned correction (docs/decisions.md #20).

    u = baseline(x, ref) + scale * PolicyMLP(inputs(x, ref))

The output layer starts at exactly zero, so the untrained policy IS the baseline controller:
stable from the first control step, which matters because there are no resets (P1).
Inputs are tracking errors, not absolute positions, so the policy is not tied to one
trajectory (P2). Small and smooth (2 x 32, tanh): the zeroth-order gradient gets noisier with
every parameter, and BPTT multiplies the policy's derivatives over the rollout (P3).
"""

import jax
import jax.numpy as jnp
from flax import nnx

from adaptive_rl.constants import GRAVITY
from adaptive_rl.controllers import CascadedController
from adaptive_rl.state import unpack_state

N_INPUTS = 18

# Fixed physical scales (as #12): every input and output is O(1) for typical flight.
POS_ERROR_SCALE = 0.1  # [m]
VEL_ERROR_SCALE = 0.5  # [m/s]


def inputs(x: jax.Array, ref: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
    """Network input (18,): position error (3), velocity error (3), R (9, row-major), a_ref (3).

    No absolute position (tracking errors suffice and do not tie the policy to one place, as
    #12), no measured omega (the nominal model ignores it as well, #6).
    """
    p, R, v, _ = unpack_state(x)
    p_ref, v_ref, a_ref = ref
    return jnp.concatenate(
        [(p - p_ref) / POS_ERROR_SCALE, (v - v_ref) / VEL_ERROR_SCALE, R.reshape(9), a_ref / GRAVITY]
    )


class PolicyMLP(nnx.Module):
    """18 -> width -> width -> 4, tanh. Output in normalised units (see act)."""

    def __init__(self, *, width: int = 32, rngs: nnx.Rngs):
        self.layer1 = nnx.Linear(N_INPUTS, width, rngs=rngs)
        self.layer2 = nnx.Linear(width, width, rngs=rngs)
        # Zero output layer: untrained correction = 0 -> policy == baseline. Gradients still
        # reach it at once (its gradient is the hidden activation, not zero).
        self.out = nnx.Linear(
            width, 4, kernel_init=nnx.initializers.zeros, bias_init=nnx.initializers.zeros, rngs=rngs
        )

    def __call__(self, z: jax.Array) -> jax.Array:
        h = jnp.tanh(self.layer1(z))
        h = jnp.tanh(self.layer2(h))
        return self.out(h)


def correction(policy: PolicyMLP, x: jax.Array, ref, mass: float) -> jax.Array:
    """The learned part of u [rad/s, rad/s, rad/s, N]. Rates in rad/s, thrust in units of the
    nominal hover thrust m g, so a network output of 0.1 means +10 % thrust."""
    out = policy(inputs(x, ref))
    return out * jnp.array([1.0, 1.0, 1.0, mass * GRAVITY])


def act(policy: PolicyMLP, baseline: CascadedController, x: jax.Array, ref) -> jax.Array:
    """u = baseline(x, ref) + correction. The baseline knows the nominal mass and its gains.

    Pure function of (policy weights, x, ref): the same code runs in the control loop and
    inside the BPTT rollout, so the gradient sees exactly the policy that flies.
    """
    return baseline(x, ref) + correction(policy, x, ref, baseline.mass)
