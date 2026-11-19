"""Tests for the nominal model: signs, frames and gradients, i.e. what breaks silently.

Not tested here: agreement with the plant. That mismatch is a result (eval_model.py), not a bug.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from adaptive_rl.constants import GRAVITY
from adaptive_rl.dynamics.nominal import hat, so3_exp, step
from adaptive_rl.state import pack_state, unpack_state

MASS = 0.03  # [kg], any positive value
DT = 0.01


def level_state(p=(0.0, 0.0, 1.0), v=(0.0, 0.0, 0.0), R=np.eye(3)):
    return pack_state(jnp.asarray(p), jnp.asarray(R), jnp.asarray(v), jnp.zeros(3))


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    p, v, w = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
    R = Rotation.random(random_state=1).as_matrix()
    p2, R2, v2, w2 = unpack_state(pack_state(p, R, v, w))
    for a, b in [(p, p2), (R, R2), (v, v2), (w, w2)]:
        np.testing.assert_allclose(b, a, atol=1e-6)
    # Same layout as plant.py: R flattened row-major.
    np.testing.assert_allclose(pack_state(p, R, v, w)[3:12], R.ravel(), atol=1e-6)


def test_hover_is_equilibrium():
    """Level, thrust = m g, zero rates -> nothing changes. Catches the sign/direction of gravity."""
    x = level_state()
    u = jnp.array([0.0, 0.0, 0.0, MASS * GRAVITY])
    np.testing.assert_allclose(step(x, u, MASS, DT), x, atol=1e-6)


def test_free_fall():
    x = level_state()
    x_next = step(x, jnp.zeros(4), MASS, DT)
    p, _, v, _ = unpack_state(x_next)
    np.testing.assert_allclose(v, [0.0, 0.0, -GRAVITY * DT], atol=1e-6)
    np.testing.assert_allclose(p, [0.0, 0.0, 1.0 - 0.5 * GRAVITY * DT**2], atol=1e-6)


def test_thrust_acts_along_body_z():
    """Rolled by 90 deg: body z points along world -y, so thrust accelerates along -y."""
    R = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    x = level_state(R=R)
    x_next = step(x, jnp.array([0.0, 0.0, 0.0, MASS * 1.0]), MASS, DT)  # thrust / m = 1 m/s^2
    _, _, v, _ = unpack_state(x_next)
    np.testing.assert_allclose(v, [0.0, -1.0 * DT, -GRAVITY * DT], atol=1e-6)


def test_rates_are_body_frame():
    """Yawed by 90 deg, then a pure roll rate: R_next = R @ Rx, not Rx @ R."""
    R0 = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    roll_rate, dt = 2.0, 0.1  # large angle, so the two conventions differ clearly
    x_next = step(level_state(R=R0), jnp.array([roll_rate, 0.0, 0.0, 0.0]), MASS, dt)
    _, R1, _, omega = unpack_state(x_next)

    Rx = Rotation.from_rotvec([roll_rate * dt, 0.0, 0.0]).as_matrix()
    np.testing.assert_allclose(R1, R0 @ Rx, atol=1e-6)
    assert not np.allclose(R0 @ Rx, Rx @ R0, atol=1e-3)  # the test can tell them apart
    np.testing.assert_allclose(omega, [roll_rate, 0.0, 0.0])  # omega_next := omega_cmd


def test_so3_exp_matches_scipy_and_stays_on_so3():
    rng = np.random.default_rng(2)
    for phi in [rng.normal(size=3), 1e-4 * rng.normal(size=3), np.zeros(3)]:
        R = np.asarray(so3_exp(jnp.asarray(phi)))
        np.testing.assert_allclose(R, Rotation.from_rotvec(phi).as_matrix(), atol=1e-6)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_gradient_at_zero_rate_is_finite_and_correct():
    """omega_cmd = 0 (hover) is the most common input. d R_next / d omega_cmd there must not be
    NaN (sqrt at 0), and must equal R @ hat(e_i) * dt."""
    R0 = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix()
    x = level_state(R=R0)

    def R_next(omega_cmd):
        u = jnp.concatenate([omega_cmd, jnp.array([MASS * GRAVITY])])
        return unpack_state(step(x, u, MASS, DT))[1]

    J = jax.jacfwd(R_next)(jnp.zeros(3))  # (3, 3, 3): d R_next[a, b] / d omega[i]
    assert np.all(np.isfinite(J))
    for i in range(3):
        np.testing.assert_allclose(J[:, :, i], R0 @ hat(jnp.eye(3)[i]) * DT, atol=1e-6)

    # Reverse mode (what BPTT uses) as well.
    g = jax.grad(lambda w: R_next(w).sum())(jnp.zeros(3))
    assert np.all(np.isfinite(g))


def test_residual_acceleration_enters_like_a_force():
    x = level_state()
    u = jnp.array([0.0, 0.0, 0.0, MASS * GRAVITY])
    a_res = jnp.array([0.5, -0.2, 0.1])
    _, _, v, _ = unpack_state(step(x, u, MASS, DT, a_res=a_res))
    np.testing.assert_allclose(v, a_res * DT, atol=1e-6)
