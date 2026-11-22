"""Tests for the Crazyflow wrapper. Slow-ish (JIT compilation), so one Plant per module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from adaptive_rl.constants import GRAVITY
from adaptive_rl.controllers import CascadedController
from adaptive_rl.plant import Plant
from adaptive_rl.reference import Circle
from adaptive_rl.state import STATE_DIM, unpack_state

P0 = (0.0, 0.0, 2.0)


@pytest.fixture(scope="module")
def plant():
    p = Plant(control_freq=100, p0=P0)
    yield p
    p.close()


def test_reset_returns_initial_state(plant):
    x = plant.reset()
    p, R, v, omega = unpack_state(x)
    assert x.shape == (STATE_DIM,)
    np.testing.assert_allclose(p, P0)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(v, 0.0)
    np.testing.assert_allclose(omega, 0.0)


def test_reset_restores_state_after_flying(plant):
    # Checks that the modified data was stored as Crazyflow's default_data.
    x0 = plant.reset()
    for _ in range(20):
        plant.step(np.array([0.5, -0.5, 1.0, 0.3]))
    np.testing.assert_allclose(plant.reset(), x0)


def test_rotation_stays_orthonormal(plant):
    plant.reset()
    for _ in range(50):
        x = plant.step(np.array([0.5, -0.3, 1.0, plant.nominal_mass * GRAVITY]))
    _, R, _, _ = unpack_state(x)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-5)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-5)


def test_yaw_rate_command_turns_about_body_z(plant):
    # 1 rad/s about body z for 1 s. The inner rate loop lags, so the yaw angle ends slightly
    # below 1 rad; roll and pitch stay level.
    plant.reset()
    for _ in range(100):
        x = plant.step(np.array([0.0, 0.0, 1.0, plant.nominal_mass * GRAVITY]))
    _, R, _, omega = unpack_state(x)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    assert 0.85 < yaw < 1.0
    assert omega[2] == pytest.approx(1.0, abs=0.02)
    np.testing.assert_allclose(R[:, 2], [0.0, 0.0, 1.0], atol=1e-3)  # body z still vertical


def test_controller_hovers(plant):
    ref = Circle(center=P0, radius=0.0)
    controller = CascadedController(plant.nominal_mass)
    x = plant.reset()
    for k in range(300):
        x = plant.step(controller(x, ref(k * plant.dt)))
    np.testing.assert_allclose(x[0:3], P0, atol=0.02)


def test_controller_tracks_circle(plant):
    # The plant starts at P0, the circle's start point is center + (r, 0, 0).
    ref = Circle(center=(P0[0] - 0.5, P0[1], P0[2]), radius=0.5, period=4.0)
    controller = CascadedController(plant.nominal_mass)
    x = plant.reset()
    errors = []
    for k in range(600):
        r = ref(k * plant.dt)
        x = plant.step(controller(x, r))
        errors.append(np.linalg.norm(x[0:3] - ref((k + 1) * plant.dt)[0]))
    assert max(errors[200:]) < 0.03  # after the 2 s start-up transient


def disturbance_effect(plant, **disturbance) -> np.ndarray:
    """Velocity change over 0.1 s caused by a disturbance, relative to an identical
    undisturbed run. Differencing cancels drag and the rotor spin-up after reset()."""
    hover = np.array([0.0, 0.0, 0.0, plant.nominal_mass * GRAVITY])
    v_end = []
    for dist in ({}, disturbance):
        plant.reset()
        for _ in range(100):
            plant.step(hover)
        plant.set_disturbance(**dist)
        for _ in range(10):
            x = plant.step(hover)
        v_end.append(x[12:15])
    return (v_end[1] - v_end[0]) / (10 * plant.dt)


def test_constant_force_accelerates_along_world_axis(plant):
    a = disturbance_effect(plant, force=(0.05, 0.0, 0.0))
    np.testing.assert_allclose(a, [0.05 / plant.nominal_mass, 0.0, 0.0], rtol=0.05, atol=1e-3)


def test_heavier_drone_sinks_and_nominal_mass_unchanged(plant):
    m = plant.nominal_mass
    a = disturbance_effect(plant, mass=1.3 * m)
    # m_true * a = m * g - m_true * g  ->  a = g (1/1.3 - 1)
    np.testing.assert_allclose(a, [0.0, 0.0, GRAVITY * (1 / 1.3 - 1)], rtol=0.05, atol=1e-3)
    assert plant.nominal_mass == m


def test_reset_removes_disturbance(plant):
    x0 = plant.reset()
    plant.set_disturbance(mass=2 * plant.nominal_mass, force=(0.1, 0.0, 0.0))
    plant.step(np.zeros(4))
    np.testing.assert_allclose(plant.reset(), x0)
    hover = np.array([0.0, 0.0, 0.0, plant.nominal_mass * GRAVITY])
    for _ in range(100):
        x = plant.step(hover)
    assert abs(x[12]) < 1e-6  # no residual force along x


def test_true_step_matches_step_and_is_differentiable(plant):
    """true_step from a snapshot must reproduce step() (same physics, same x layout incl. the
    jnp quaternion conversion), leave the plant untouched, and give a finite Jacobian."""
    x = plant.reset()
    controller = CascadedController(plant.nominal_mass)
    reference = Circle(center=P0, radius=0.5, period=4.0)
    for k in range(50):  # spin the rotors up, so the hidden states are not trivial
        x = plant.step(controller(x, reference(k * plant.dt)))
    u = controller(x, reference(50 * plant.dt))

    snap = plant.snapshot()
    _, x_true_step = plant.true_step(snap, jnp.asarray(u, dtype=jnp.float32))
    np.testing.assert_allclose(plant.snapshot().states.pos, snap.states.pos)  # plant untouched
    x_step = plant.step(u)
    np.testing.assert_allclose(x_true_step, x_step, atol=1e-5)

    J = jax.jacfwd(lambda u: plant.true_step(snap, u)[1])(jnp.asarray(u, dtype=jnp.float32))
    assert J.shape == (STATE_DIM, 4)
    assert np.all(np.isfinite(J))
    assert J[14, 3] > 0  # more thrust -> more upward velocity (v_z = x[14])
