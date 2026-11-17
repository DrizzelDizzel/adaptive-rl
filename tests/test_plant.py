"""Tests for the Crazyflow wrapper. Slow-ish (JIT compilation), so one Plant per module."""

import numpy as np
import pytest

from adaptive_rl.controllers import GRAVITY, CascadedController
from adaptive_rl.plant import STATE_DIM, Plant, unpack_state
from adaptive_rl.reference import Circle

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
        x = plant.step(np.array([0.5, -0.3, 1.0, plant.mass * GRAVITY]))
    _, R, _, _ = unpack_state(x)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-5)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-5)


def test_yaw_rate_command_turns_about_body_z(plant):
    # 1 rad/s about body z for 1 s. The inner rate loop lags, so the yaw angle ends slightly
    # below 1 rad; roll and pitch stay level.
    plant.reset()
    for _ in range(100):
        x = plant.step(np.array([0.0, 0.0, 1.0, plant.mass * GRAVITY]))
    _, R, _, omega = unpack_state(x)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    assert 0.85 < yaw < 1.0
    assert omega[2] == pytest.approx(1.0, abs=0.02)
    np.testing.assert_allclose(R[:, 2], [0.0, 0.0, 1.0], atol=1e-3)  # body z still vertical


def test_controller_hovers(plant):
    ref = Circle(center=P0, radius=0.0)
    controller = CascadedController(plant.mass)
    x = plant.reset()
    for k in range(300):
        x = plant.step(controller(x, ref(k * plant.dt)))
    np.testing.assert_allclose(x[0:3], P0, atol=0.02)


def test_controller_tracks_circle(plant):
    # The plant starts at P0, the circle's start point is center + (r, 0, 0).
    ref = Circle(center=(P0[0] - 0.5, P0[1], P0[2]), radius=0.5, period=4.0)
    controller = CascadedController(plant.mass)
    x = plant.reset()
    errors = []
    for k in range(600):
        r = ref(k * plant.dt)
        x = plant.step(controller(x, r))
        errors.append(np.linalg.norm(x[0:3] - ref((k + 1) * plant.dt)[0]))
    assert max(errors[200:]) < 0.03  # after the 2 s start-up transient
