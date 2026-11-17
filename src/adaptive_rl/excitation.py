"""Excitation noise added to the controller output during data collection.

Lives on the computer side of the plant boundary: the same code would run with a real drone.
Placeholder until a stochastic policy brings its own exploration noise.
"""

import numpy as np


class HeldNoise:
    """Gaussian noise on u = [omega_x, omega_y, omega_z, thrust], held for `hold_steps` steps.

    Holding matters: the drone's inner loops (time constants 30–70 ms) filter out noise that
    changes every 10 ms. Held for 100 ms, the noise actually reaches the state.
    """

    def __init__(
        self,
        std_rate: float,  # [rad/s]
        std_thrust: float,  # [N]
        hold_steps: int = 10,
        seed: int = 0,
    ):
        self.std = np.array([std_rate, std_rate, std_rate, std_thrust])
        self.hold_steps = hold_steps
        self._rng = np.random.default_rng(seed)
        self._e = np.zeros(4)

    def __call__(self, k: int) -> np.ndarray:
        """Noise for control step k. Call once per step, with k = 0, 1, 2, ..."""
        if k % self.hold_steps == 0:
            self._e = self.std * self._rng.standard_normal(4)
        return self._e.copy()
