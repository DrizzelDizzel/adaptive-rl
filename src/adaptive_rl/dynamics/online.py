"""Online model update: a sliding window of the newest real transitions (docs/decisions.md #22).

    every control step:  (x, u, x+) --sample--> (f, y) --> window (oldest sample drops out)
                         fit.train_step(residual, optimizer, window.F, window.Y)   (Adam, warm)

The window has a fixed number of rows, so the jitted train_step always sees arrays of the same
shape and compiles once. (jit compiles one program per input shape; a growing buffer would
recompile in every control step.) Row order does not matter for the MSE, so new samples simply
overwrite the oldest row in turn (ring buffer) instead of shifting the whole array.
"""

import jax
import numpy as np

from adaptive_rl.dynamics.residual import N_FEATURES, features, target


@jax.jit
def sample(x: jax.Array, u: jax.Array, x_next: jax.Array, mass: float, dt: float) -> tuple[jax.Array, jax.Array]:
    """One transition as network input f (16,) and training target y = a_res (3,), the same
    functions as offline (fit.training_data), so online and offline learn the same quantity."""
    return features(x, u, mass), target(x, u, x_next, mass, dt)


class SlidingWindow:
    """The newest `size` samples (f, y). Before it is full, rows that were never written are
    zeros and must not be trained on; callers wait for `full`."""

    def __init__(self, size: int):
        self.F = np.zeros((size, N_FEATURES), dtype=np.float32)
        self.Y = np.zeros((size, 3), dtype=np.float32)
        self.count = 0  # samples added so far; the next one goes to row count % size

    def add(self, f, y) -> None:
        row = self.count % len(self.F)
        self.F[row], self.Y[row] = f, y
        self.count += 1

    @property
    def full(self) -> bool:
        return self.count >= len(self.F)
