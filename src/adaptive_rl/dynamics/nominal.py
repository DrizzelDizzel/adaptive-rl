"""Nominal (analytical) dynamics model: x_next = f_nom(x, u). Pure JAX.

Same model as the LotF surrogate (`simplified_dyn`), docs/decisions.md #6:
    translation: point mass, thrust along body z, gravity; acceleration held constant over dt
    rotation:    the body rates are the commanded rates, no rate dynamics, no delay
    omega_next := omega_cmd  (omega in x is not used)

Known mismatch to the plant, on purpose: Crazyflow's rate loop and rotors (omega lags behind
omega_cmd), thrust lag, drag, and any disturbance. The residual model (hybrid.py) learns the
translational part of it through `a_res`; the rotational part stays (see #6 for numbers).

Written for a single sample (x: (18,), u: (4,)). Batching (jax.vmap) and compiling (jax.jit)
are left to the caller, so the physics reads as on paper.
"""

import jax
import jax.numpy as jnp

from adaptive_rl.constants import GRAVITY
from adaptive_rl.state import pack_state, unpack_state

G_VEC = jnp.array([0.0, 0.0, -GRAVITY])  # world frame, z up


def hat(w: jax.Array) -> jax.Array:
    """Vector -> skew-symmetric matrix, so that hat(w) @ a = cross(w, a)."""
    return jnp.array(
        [
            [0.0, -w[2], w[1]],
            [w[2], 0.0, -w[0]],
            [-w[1], w[0], 0.0],
        ]
    )


def so3_exp(phi: jax.Array) -> jax.Array:
    """Rotation matrix exp(hat(phi)) for a rotation vector phi [rad] (Rodrigues' formula).

        exp(K) = I + sin(theta)/theta * K + (1 - cos(theta))/theta^2 * K @ K,  theta = |phi|

    Two numerical details, both invisible in the forward value but fatal for training:
    - theta = sqrt(phi @ phi) has an infinite derivative at phi = 0 (hovering, omega_cmd = 0),
      and jnp.where differentiates BOTH branches (the unused one is multiplied by 0, but
      0 * NaN = NaN). So the sqrt only ever sees a safe value ("double where"), and near 0 the
      coefficients come from their Taylor series instead.
    - 1 - cos(theta) loses most digits in float32 for small theta (typical |omega| * dt ~ 5e-3).
      2 sin^2(theta/2) is the same quantity without that cancellation.
    """
    theta2 = phi @ phi
    small = theta2 < 1e-6
    theta2_safe = jnp.where(small, 1.0, theta2)
    theta = jnp.sqrt(theta2_safe)

    a = jnp.where(small, 1.0 - theta2 / 6.0, jnp.sin(theta) / theta)
    b = jnp.where(small, 0.5 - theta2 / 24.0, 2.0 * jnp.sin(0.5 * theta) ** 2 / theta2_safe)

    K = hat(phi)
    return jnp.eye(3) + a * K + b * (K @ K)


def step(x: jax.Array, u: jax.Array, mass: float, dt: float, a_res: jax.Array | float = 0.0) -> jax.Array:
    """One control step of the nominal model.

    Args:
        x: state (18,), layout in state.py.
        u: command (4,) = [omega_x, omega_y, omega_z (body, rad/s), thrust (N)], held over dt.
        mass: nominal mass [kg] (plant.nominal_mass), never the true one (decision #8).
        dt: control period [s].
        a_res: extra acceleration [m/s^2], world frame, held over dt. 0 for the pure nominal
            model; hybrid.py passes the residual network's prediction here.

    Returns:
        x_next (18,).
    """
    p, R, v, _ = unpack_state(x)
    omega_cmd, thrust = u[0:3], u[3]

    # Translation. The acceleration uses the attitude at the start of the step and is held
    # constant over dt (as in LotF; their RK4 gives exactly this for constant a).
    b3 = R[:, 2]  # thrust axis = body z, expressed in world frame
    a = thrust / mass * b3 + G_VEC + a_res
    p_next = p + v * dt + 0.5 * a * dt**2
    v_next = v + a * dt

    # Rotation: omega is constant in the BODY frame over dt, so the increment multiplies from
    # the right (R @ exp), not from the left (that would be a rotation about world axes).
    R_next = R @ so3_exp(omega_cmd * dt)

    return pack_state(p_next, R_next, v_next, omega_cmd)
