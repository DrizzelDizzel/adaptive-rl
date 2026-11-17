"""Reference trajectories: time t -> desired position, velocity, acceleration (world frame).

A reference only knows time. It knows nothing about the drone or the controller.
"""

import numpy as np


class Circle:
    """Horizontal circle at constant height, flown counter-clockwise (seen from above).

    radius = 0 gives hover at `center`.
    """

    def __init__(
        self,
        center: tuple[float, float, float] = (0.0, 0.0, 1.0),
        radius: float = 0.5,  # [m]
        period: float = 4.0,  # [s] time for one lap
    ):
        self.center = center
        self.radius = radius
        self.period = period

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = 2.0 * np.pi / self.period  # angular speed [rad/s]
        c, s = np.cos(w * t), np.sin(w * t)
        r = self.radius
        p = np.asarray(self.center) + np.array([r * c, r * s, 0.0])
        v = np.array([-r * w * s, r * w * c, 0.0])
        a = np.array([-r * w**2 * c, -r * w**2 * s, 0.0])  # centripetal, points to center
        return p, v, a

    def start_position(self) -> tuple[float, float, float]:
        """Position at t = 0, so the plant can start on the trajectory."""
        p, _, _ = self(0.0)
        return tuple(float(pi) for pi in p)


class SumOfSines:
    """Smooth random trajectory for training data: per axis a sum of sines.

    p(t) = center + sum_i A_i sin(2 pi f_i t + phi_i), with random frequencies and phases from
    a fixed seed (same seed -> same trajectory). The amplitude per axis is split evenly over the
    sines, so |p - center| <= amplitude per axis and |a| <= amplitude * (2 pi f_max)^2.
    """

    def __init__(
        self,
        center: tuple[float, float, float] = (0.0, 0.0, 1.5),
        amplitude: tuple[float, float, float] = (0.8, 0.8, 0.3),  # [m] per axis
        freq_range: tuple[float, float] = (0.05, 0.3),  # [Hz]
        n_sines: int = 3,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.center = np.asarray(center, dtype=float)
        self.amp = np.asarray(amplitude, dtype=float)[:, None] / n_sines  # (3, n_sines)
        self.omega = 2.0 * np.pi * rng.uniform(*freq_range, size=(3, n_sines))  # [rad/s]
        self.phase = rng.uniform(0.0, 2.0 * np.pi, size=(3, n_sines))

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arg = self.omega * t + self.phase
        p = self.center + (self.amp * np.sin(arg)).sum(axis=1)
        v = (self.amp * self.omega * np.cos(arg)).sum(axis=1)
        a = (-self.amp * self.omega**2 * np.sin(arg)).sum(axis=1)
        return p, v, a

    def start_position(self) -> tuple[float, float, float]:
        p, _, _ = self(0.0)
        return tuple(float(pi) for pi in p)
