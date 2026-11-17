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
