"""Tests for online.py: what breaks silently (the window holding the wrong samples after it
wraps around, or claiming to be full while unwritten zero rows are still in it)."""

import numpy as np

from adaptive_rl.dynamics.online import SlidingWindow


def test_window_holds_exactly_the_newest_samples():
    window = SlidingWindow(size=5)
    for i in range(4):
        window.add(np.full(16, i), np.full(3, i))
    assert not window.full

    for i in range(4, 12):
        window.add(np.full(16, i), np.full(3, i))
    assert window.full
    # Samples 7 .. 11 in some row order (order does not matter for the MSE).
    assert sorted(window.F[:, 0]) == [7, 8, 9, 10, 11]
    np.testing.assert_array_equal(window.F[:, 0], window.Y[:, 0])  # f and y of a sample stay in the same row
