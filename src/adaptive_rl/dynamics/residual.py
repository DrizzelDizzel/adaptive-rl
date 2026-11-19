"""Residual model: a small MLP for the acceleration the nominal model misses.

    a_res = ResidualMLP(features(x, u, mass))     [m/s^2], world frame, shape (3,)

hybrid.py adds a_res inside nominal.step. Design and deviations from LotF: docs/decisions.md #12.

flax.nnx in two sentences: a Module is a plain Python object that holds its weights as
attributes (like PyTorch). JAX transformations only work on pure functions of arrays, so
nnx.jit / nnx.grad split the object into "structure" and "weight arrays", transform, and put
it back together (nnx.split / nnx.merge do the same by hand).
"""

import jax
import jax.numpy as jnp
from flax import nnx

from adaptive_rl.constants import GRAVITY
from adaptive_rl.dynamics import nominal
from adaptive_rl.state import unpack_state

N_FEATURES = 16


def features(x: jax.Array, u: jax.Array, mass: float) -> jax.Array:
    """Network input (16,): R (9, row-major), v (3), omega_cmd (3), thrust / (m g) (1).

    No position (#12): the disturbances so far do not depend on place, and a position input
    would let the net memorise where the training trajectory flew. #13 revisits this for a
    position-dependent disturbance; it is a one-line change here.
    Fixed physical scales instead of data statistics: every entry is O(1) already (R in
    [-1, 1], v and omega_cmd ~ 1, thrust ~ 1 at hover). Stays valid when the data
    distribution shifts online, nothing to store or update.
    """
    _, R, v, _ = unpack_state(x)
    return jnp.concatenate([R.reshape(9), v, u[0:3], u[3:4] / (mass * GRAVITY)])


def target(x: jax.Array, u: jax.Array, x_next: jax.Array, mass: float, dt: float) -> jax.Array:
    """Training target: the a_res that makes nominal.step hit the measured velocity exactly.

    nominal.step adds a_res * dt to v, so a_res = (v_next_measured - v_next_nominal) / dt.
    Computed through nominal.step itself, so the nominal model is defined in one place only.
    """
    v_next_nominal = unpack_state(nominal.step(x, u, mass, dt))[2]
    v_next_measured = unpack_state(x_next)[2]
    return (v_next_measured - v_next_nominal) / dt


class ResidualMLP(nnx.Module):
    """16 -> width -> width -> 3, as LotF (2 x 128, ReLU).

    The output layer starts at exactly zero (weights and bias), so an untrained model gives
    a_res = 0 and hybrid == nominal. Gradients still flow: the output layer gets them at once,
    the hidden layers from the second update on.
    """

    def __init__(self, *, width: int = 128, activation=nnx.relu, rngs: nnx.Rngs):
        # Same initialisation as LotF's hidden layers (variance_scaling, fan_avg, normal).
        init = nnx.initializers.variance_scaling(1.0, "fan_avg", "normal")
        self.layer1 = nnx.Linear(N_FEATURES, width, kernel_init=init, rngs=rngs)
        self.layer2 = nnx.Linear(width, width, kernel_init=init, rngs=rngs)
        self.out = nnx.Linear(
            width, 3, kernel_init=nnx.initializers.zeros, bias_init=nnx.initializers.zeros, rngs=rngs
        )
        # Argument instead of fixed ReLU: if the policy gradient later flows through this net
        # (#11), ReLU's piecewise-constant derivative may need to become a smooth one.
        self.activation = activation

    def __call__(self, f: jax.Array) -> jax.Array:
        h = self.activation(self.layer1(f))
        h = self.activation(self.layer2(h))
        return self.out(h)
