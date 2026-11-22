"""Baseline controllers. Interface shared with every policy:

    u = controller(x, ref)

x:   flat state (p, R, v, omega), layout in state.py
ref: (p_ref, v_ref, a_ref) in the world frame, from reference.py
u:   [omega_x, omega_y, omega_z, thrust] (body rates [rad/s], collective thrust [N])

Written in JAX (jnp), not numpy: the residual policy (policy.py) adds a learned correction to
this controller's output, and the first-order policy gradient needs d u / d x through it
(the closed-loop feedback, docs/decisions.md P6 in #20). Accepts numpy inputs as well and
returns a jax array; float32, like the simulator.
"""

from functools import partial

import jax
import jax.numpy as jnp

from adaptive_rl.constants import GRAVITY
from adaptive_rl.state import unpack_state

E3 = jnp.array([0.0, 0.0, 1.0])


def vee(S: jax.Array) -> jax.Array:
    """Inverse of the hat map: skew-symmetric 3x3 matrix -> vector."""
    return jnp.array([S[2, 1], S[0, 2], S[1, 0]])


@partial(jax.jit, static_argnames=("mass", "kp", "kd", "k_att"))
def cascaded_control(
    x: jax.Array,
    ref: tuple[jax.Array, jax.Array, jax.Array],
    mass: float,
    kp: float,
    kd: float,
    k_att: float,
) -> jax.Array:
    """The control law of CascadedController as a pure function (see the class for the idea).

    jax.jit compiles it once; the gains are static (compile-time constants), the arrays are
    traced. Pure and in jnp, so jax.grad / jax.jacfwd can differentiate through it.
    """
    p, R, v, _ = unpack_state(x)
    p_ref, v_ref, a_ref = ref

    # 1) Desired force in world frame [N]
    a_des = a_ref + kp * (p_ref - p) + kd * (v_ref - v)
    f_des = mass * (a_des + GRAVITY * E3)

    # 2) Collective thrust: only the component along the current body z axis is useful
    b3 = R[:, 2]  # body z axis expressed in world frame
    thrust = f_des @ b3

    # 3) Desired attitude: body z along f_des, body x as close to world x as possible (yaw 0)
    b3_des = f_des / jnp.linalg.norm(f_des)
    b2_des = jnp.cross(b3_des, jnp.array([1.0, 0.0, 0.0]))
    b2_des = b2_des / jnp.linalg.norm(b2_des)
    b1_des = jnp.cross(b2_des, b3_des)
    R_des = jnp.column_stack([b1_des, b2_des, b3_des])

    # 4) Attitude error (body frame) -> body-rate command
    e_R = 0.5 * vee(R_des.T @ R - R.T @ R_des)
    omega_cmd = -k_att * e_R

    return jnp.concatenate([omega_cmd, thrust[None]])


class CascadedController:
    """Position -> attitude -> body rates. The innermost loop (rates -> torques) is Crazyflow's.

    Outer loop (translation): PD on position/velocity plus feedforward acceleration gives the
    force vector the drone must produce in the world frame.
    Middle loop (attitude): rotate the thrust axis (body z) onto that force vector, yaw = 0,
    and command body rates proportional to the attitude error (geometric control on SO(3),
    no Euler angles -> no singularities).

    Stateless: no integrator. Consequence: constant disturbances (drag, wrong mass, external
    force) leave a steady-state tracking error. Acceptable for data collection.
    """

    def __init__(self, mass: float, kp: float = 6.0, kd: float = 4.0, k_att: float = 8.0):
        self.mass = mass  # nominal mass [kg]
        self.kp = kp  # [1/s^2]
        self.kd = kd  # [1/s]
        self.k_att = k_att  # [1/s] attitude error -> body rate

    def __call__(self, x: jax.Array, ref: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        return cascaded_control(x, ref, self.mass, self.kp, self.kd, self.k_att)
