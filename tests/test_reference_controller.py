"""Pure-numpy tests for the reference and the controller (no simulator)."""

import numpy as np

from adaptive_rl.controllers import GRAVITY, CascadedController, vee
from adaptive_rl.excitation import HeldNoise
from adaptive_rl.reference import Circle, SumOfSines


def hat(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def test_vee_inverts_hat():
    w = np.array([0.3, -1.2, 2.5])
    np.testing.assert_allclose(vee(hat(w)), w)


def test_circle_velocity_and_acceleration_are_derivatives():
    ref, h = Circle(radius=0.5, period=4.0), 1e-5
    for t in [0.0, 0.7, 2.3]:
        p_minus, v_minus, _ = ref(t - h)
        p_plus, v_plus, _ = ref(t + h)
        _, v, a = ref(t)
        np.testing.assert_allclose((p_plus - p_minus) / (2 * h), v, atol=1e-6)
        np.testing.assert_allclose((v_plus - v_minus) / (2 * h), a, atol=1e-6)


def test_controller_at_hover_equilibrium_commands_hover_thrust():
    mass = 0.0434
    p_ref = np.array([0.0, 0.0, 1.0])
    x = np.concatenate([p_ref, np.eye(3).ravel(), np.zeros(3), np.zeros(3)])
    u = CascadedController(mass)(x, (p_ref, np.zeros(3), np.zeros(3)))
    np.testing.assert_allclose(u, [0.0, 0.0, 0.0, mass * GRAVITY], atol=1e-12)


def test_controller_tilts_towards_position_error():
    # Target in +x: the drone must pitch so that body z tilts towards +x, i.e. a positive
    # rotation about body y.
    mass = 0.0434
    x = np.concatenate([np.zeros(3), np.eye(3).ravel(), np.zeros(3), np.zeros(3)])
    u = CascadedController(mass)(x, (np.array([0.1, 0.0, 0.0]), np.zeros(3), np.zeros(3)))
    assert u[1] > 0.0
    assert abs(u[0]) < 1e-12 and abs(u[2]) < 1e-12


def test_sum_of_sines_derivatives_and_bounds():
    ref, h = SumOfSines(center=(0.0, 0.0, 1.5), amplitude=(0.8, 0.8, 0.3), seed=3), 1e-5
    for t in [0.0, 5.3, 41.0]:
        p_minus, v_minus, _ = ref(t - h)
        p_plus, v_plus, _ = ref(t + h)
        p, v, a = ref(t)
        np.testing.assert_allclose((p_plus - p_minus) / (2 * h), v, atol=1e-6)
        np.testing.assert_allclose((v_plus - v_minus) / (2 * h), a, atol=1e-6)
        assert np.all(np.abs(p - [0.0, 0.0, 1.5]) <= [0.8, 0.8, 0.3])


def test_sum_of_sines_same_seed_same_trajectory():
    np.testing.assert_allclose(SumOfSines(seed=7)(3.0)[0], SumOfSines(seed=7)(3.0)[0])
    assert not np.allclose(SumOfSines(seed=7)(3.0)[0], SumOfSines(seed=8)(3.0)[0])


def test_held_noise_is_held_and_reproducible():
    noise = HeldNoise(std_rate=0.5, std_thrust=0.04, hold_steps=10, seed=0)
    e = np.array([noise(k) for k in range(30)])
    assert np.all(e[0:10] == e[0]) and np.all(e[10:20] == e[10])
    assert not np.allclose(e[0], e[10])
    again = HeldNoise(std_rate=0.5, std_thrust=0.04, hold_steps=10, seed=0)
    np.testing.assert_array_equal(again(0), e[0])
